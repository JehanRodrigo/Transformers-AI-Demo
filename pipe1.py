from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

result = classifier("I've been waiting for a HuggingFace course my whole life.")

print(result)
