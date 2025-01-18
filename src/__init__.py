from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import sys
from pathlib import Path
import os
from PyPDF2 import PdfWriter,PdfReader
from io import StringIO, BytesIO
import PyPDF2
import requests
sys.path.append(Path(os.curdir).parent)

# If your model is in a local directory
MODEL_PATH = Path("distilbert-classification/")  # Adjust this path to where your model is actually saved
load_model = DistilBertForSequenceClassification.from_pretrained(str(MODEL_PATH))
load_tokenizer = DistilBertTokenizer.from_pretrained(str(MODEL_PATH))


class pdfClassifier:
    def __init__(self, model_path: str = str(Path("distilbert-classification/")), tokenizer_path:str = str(Path("distilbert-classification/"))) -> None:
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.key_pairs = {'cable': 0, 'fuses':1, 'lighting':2, 'others':3}
    def get_text(self,url :str):
        pdf_io_bytes = ""
        retries = 3 # maximum number of retries      
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=5.0)
                pdf_io_bytes = BytesIO(response.content)
                pdf_text = PyPDF2.PdfReader(pdf_io_bytes).pages[0].extract_text()
                return url + pdf_text,True
            except Exception as e:
                if attempt == retries - 1:
                    return url,False
    def classify_link(self,url:str):
        try:
            text,bool_pdf_worked = self.get_text(url)
            tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            logits = torch.nn.functional.softmax(logits, dim=1).tolist()[0]
            logits = [round(l,4)*100 for l in logits]
            result = [('cable', logits[0]), ('fuses',logits[1]), ('lighting',logits[2]), ('others',logits[3])]
            result = sorted(result, key = lambda x: x[-1],reverse=True)
            return result,bool_pdf_worked
        except Exception as e:
            return repr(e)
    

pdfLinkClassifier = pdfClassifier()
    