#from transformers import AutoTokenizer, pipeline,  AutoModelForTokenClassification
#tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#nerpipeline = pipeline('ner', model=model, tokenizer=tokenizer)
#text = "The patient presented with fever, cough, and shortness of breath."
#nerpipeline(text)

from transformers import pipeline

ner_model_name = "allenai/scibert_scivocab_uncased"  # Or "dslim/bert-base-cased-finetuned-ner"

nerpipeline = pipeline('ner', model=ner_model_name)
text = "The patient presented with fever, cough, and shortness of breath."
results = nerpipeline(text)

print(results)