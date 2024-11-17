# Imports
import PyPDF2
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import Dataset
import torch

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF document. Handles potential errors.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """

    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
            return text
    except (FileNotFoundError, PyPDF2.errors.PdfReaderError) as e:
        print(f"Error extracting text from PDF: {e}")
        return ""  # Handle errors gracefully (e.g., return empty string)

def tokenize_function(examples):
    """
    Tokenizes a single data point (example) with padding and truncation.

    Args:
        examples (dict): A dictionary containing the text data.
            - 'report' (str): The medical report text.

    Returns:
        dict: A dictionary with the tokenized data.
            - 'input_ids' (list): The tokenized input IDs.
            - 'attention_mask' (list): The attention mask for padding.
    """

    #return tokenizer(examples['report'], padding="max_length", truncation=True)
    return tokenizer(examples['report'], padding="max_length", truncation=True, max_length=512)  # Set desired max_length

def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

        start_scores, end_scores = outputs.start_logits, outputs.end_logits
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        answer = context[start_idx:end_idx+1]
    return answer

def prepare_data(data):
    labels = []
    reports = []
    for entry in data:
        labels.append(entry["diagnosis"])  # Assuming "diagnosis" holds labels
        reports.append(entry["report"])
    return {"report": reports, "label": labels}

if __name__ == "__main__":

    # Define paths and model name
    pdf_path = "FinalDiagnosticReport517445.pdf"
    model_name = "emilyalsentzer/Bio_ClinicalBERT"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)

    # Prepare dataset (assuming you have a list or DataFrame with 'report' and 'diagnosis' columns)
    data = [
        {"report": text, "diagnosis": "Complete Blood Count (CBC)"},  # Replace with your actual data
        {"report": text, "diagnosis": "Erythrocyte Sedimentation Rate (ESR)"},  # Add more data points
        {"report": text, "diagnosis": "HbA1C (Glycosylated Haemoglobin)"},  
        {"report": text, "diagnosis": "Blood Group ABO & Rh Typing"},
        {"report": text, "diagnosis": "Glucose Fasting (BSF)"},  
        {"report": text, "diagnosis": "Liver Function Test (LFT)"},
        {"report": text, "diagnosis": "Kidney Function Test (KFT)"},  
        {"report": text, "diagnosis": "Lipid Profile"},
        {"report": text, "diagnosis": "Calcium"},  
        {"report": text, "diagnosis": "Vitamin B12 / Cyanocobalamin"},
        {"report": text, "diagnosis": "Vitamin D 25 Hydroxy"},  
        {"report": text, "diagnosis": "Thyroid Profile Total"},
        {"report": text, "diagnosis": "Urine Routine and Microscopic Examination"},  
        {"report": text, "diagnosis": "Glucose Post Prandial (BSPP)"},
        {"report": text, "diagnosis": "2D ECHOCARDIOGRAPHY"},  
        {"report": text, "diagnosis": "X RAY CHEST"},
    ]

    # Create a Pandas DataFrame
    df = pd.DataFrame(data, columns=["report", "diagnosis"])

    # Create Hugging Face Dataset
    dataset = Dataset.from_pandas(df)

    # Tokenize the dataset in batches for efficiency
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Save the tokenized dataset
    path_to_save_tokenized_dataset = "../../../src/storage"  # Use forward slash for paths
    tokenized_dataset.save_to_disk(path_to_save_tokenized_dataset)  # Corrected typo

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",  # Output directory for checkpoints etc.
        per_device_train_batch_size=8,  # Adjust batch size as needed
        num_train_epochs=3,  # Adjust training epochs
    )

    # Prepare data for training
    prepared_data = prepare_data(data)
    
    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(prepared_data["label"])))  # Set num_labels based on unique labels

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    # Train the model
    trainer.train()

    # ... (Save the trained model for future use)

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    # Example usage:
    question = "What are the symptoms of COVID-19?"
    context = "COVID-19 symptoms typically include fever, cough, and shortness of breath. Some people may also experience fatigue, muscle aches, headache, loss of taste or smell, sore throat, congestion, runny nose, nausea, vomiting, or diarrhea."

    answer = answer_question(question, context)
    print(answer)
