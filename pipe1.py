from transformers import pipeline

classifier = pipeline("sentiment-analysis")

result = classifier("I'v been waiting for a HuggingFace course my whole life.")

print(result)
