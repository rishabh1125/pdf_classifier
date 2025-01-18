import streamlit as st
import sys
import os
from pathlib import Path
sys.path.append(os.path.join(Path(__file__).parent.parent))
from src import pdfLinkClassifier 
import pandas as pd
st.title("PDF Classification")

st.write("Add pdf url to classify it.")

pdf_url = st.text_input("Enter PDF URL")

if pdf_url:
    if st.button("Classify PDF"):
        st.write("Classification in progress...")
        # Display the prediction results
        result = pdfLinkClassifier.classify_link(pdf_url)
        prediction, link_text= result
        if not link_text:
            st.write("Could not extract pdf link... Using url endpoint for conversion.")
        st.subheader("Classification Results:")
        st.write(f"Predicted : {prediction[0][0]}")
        prediction = pd.DataFrame(prediction, columns=['product', 'confidence (in %)'])
        # Visualize the results with a bar chart
        st.bar_chart(prediction, x='product', y='confidence (in %)')
    else:
        st.warning("")