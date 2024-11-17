import PyPDF2
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import torch

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def preprocess_text(text):
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    #tokenized_text = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    tokenized_text = tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=512)
    return tokenized_text

def analyze_text(text):
    model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    sentences = text.split('.')  # Split text into sentences

    all_outputs = []
    for sentence in sentences:
        tokenized_text = preprocess_text(sentence)
        with torch.no_grad():
            outputs = model(**tokenized_text)
            all_outputs.append(outputs)  # Store outputs from each sentence

    # Combine the outputs from different sentences (modify based on your task)
    all_logits = torch.cat([output.logits for output in all_outputs], dim=0)
    average_logits = torch.mean(all_logits, dim=0)
    probabilities = torch.softmax(average_logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1)

    # Interpret the predictions based on your specific task
    # ...
     # Assuming a predefined mapping of class indices to labels
    class_labels = ["normal", "abnormal", "critical"]

    predicted_class_index = predicted_class.item()
    predicted_class_label = class_labels[predicted_class_index]

    return predicted_class_label

if __name__ == "__main__":
    load_dotenv(dotenv_path="../../../.env", override=True)
    pdf_path = "FinalDiagnosticReport517445.pdf"
    text = extract_text_from_pdf(pdf_path)
    analyze_text(text)