# Project Title: PDF Classifier of Construction Item documents

This project is a Streamlit app that allows users to input a link to a PDF document. The app extracts text from the first page of the PDF and uses it for classification in the construction industry, specifically for electrical items. The classifier categorizes the input into four classes: lights, fuses, cable, and others.


## File Structure
- `distilbert-classification/`: Contains the model files including `config.json`, `special_tokens_map.json`, and the model weights.
- `src/`: Contains the main code for the PDF classifier.
  - `__init__.py`: The main implementation of the PDF classifier using DistilBert.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `.gitattributes`: Configures Git LFS for large files.

## Steps to Replicate the Project
1. **Clone the Repository**: 
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   ```

3. **Install Required Packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Model**: Ensure that the model files are placed in the `distilbert-classification/` directory.

5. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

## Running the Streamlit App
- After running the above command, open your web browser and navigate to `http://localhost:8501` to access the application.

## Usage
- Input a PDF link in the provided field, and the classifier will return the predicted categories along with their probabilities.

## Note
- Ensure that the necessary permissions are granted for accessing the PDF files from the provided links.
