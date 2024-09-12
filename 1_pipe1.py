from transformers import pipeline

classifier = pipeline("sentiment-analysis")
# model="" Not mandatory, but it is recommended to specify the model to use. If not specified, the default model is used.

result = classifier("I've been waiting for a HuggingFace course my whole life.")

print(result)
