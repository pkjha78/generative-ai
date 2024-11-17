# Analyzing Medical Reports with BioBERT
from sentence_transformers import SentenceTransformer
import torch
import PyPDF2
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from dotenv import load_dotenv
import os

def extract_text_from_pdf(pdf_path):
    # ... (same as before)
    """Extracts text from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """

    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def preprocess_text(text):
    # ... (same as before)
    """Preprocesses text for BioBERT.

    Args:
        text (str): The text to preprocess.

    Returns:
        list: Tokenized text.
    """

    # Tokenization and preprocessing steps might vary depending on the specific task and model
    # Here's a basic example:
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    #tokenized_text = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    tokenized_text = tokenizer(text, truncation=True, max_length=512, padding=True, return_tensors="pt")
    return tokenized_text

def analyze_text(text):
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Split the text into sentences
    sentences = text.split('.')

    # Process each sentence individually
    all_outputs = []
    for sentence in sentences:
        inputs = tokenizer(sentence, truncation=True, padding=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            all_outputs.append(outputs)

    # Combine the outputs from different sentences
    # Here, we'll average the logits
    all_logits = torch.cat([output.logits for output in all_outputs], dim=0)
    average_logits = torch.mean(all_logits, dim=0)
    probabilities = torch.softmax(average_logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1)

    return predicted_class

def summarize_text(text):
    # Use a summarization model like BART or T5
    summarizer = pipeline("summarization")
    summary = summarizer(text)
    return summary[0]['summary_text']

if __name__ == "__main__":
    load_dotenv(dotenv_path="../../../.env", override=True)
    # ... (extract text and tokenize as before)
    pdf_path = "FinalDiagnosticReport517445.pdf"
    text = extract_text_from_pdf(pdf_path)
    tokenized_text = preprocess_text(text)
    # Load a pre-trained NER model
    #    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Analyze the text using BioBERT
    analysis_results = analyze_text(text)

    print("Analysis Results:")
    print(analysis_results)

    model = pipeline("token-classification", model="emilyalsentzer/Bio_ClinicalBERT")
    # Perform NER
    ner_results = model(text)

    # Extract relevant entities (e.g., diseases, medications)
    entities = []
    for token, tag in zip(tokenized_text.tokens(), ner_results[0]):
        if tag.startswith("B-") or tag.startswith("I-"):
            entities.append(token)

    # Use the extracted entities and the original text for further analysis or summarization
    # ... (e.g., filter relevant information, use BioBERT for deeper understanding)

    summary = summarize_text(text)
    print("Summary:")
    print(summary)