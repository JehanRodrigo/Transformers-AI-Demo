from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# AutoTokenizer, AutoModelForSequenceClassification are classes that areavailable in the transformers library

classifier = pipeline("sentiment-analysis")

result = classifier("I've been waiting for a HuggingFace course my whole life.")

print(result)




# creating instances 
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# default model that is used by the pipeline("sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
# from_pretrained() method is a class method that is used to instantiate a model from a pre-trained model configuration.

tokenizer = AutoTokenizer.from_pretrained(model_name)
# from_pretrained() method is a class method that is used to instantiate a model from a pre-trained model configuration.

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

result = classifier("I've been waiting for a HuggingFace course my whole life.")

print(result) 
# The output of the two print statements must be the same.


#Tokernizer basically tokenizes the input text and converts it into a format (mathematical representation) that the model can understand.


sequence = "Using transformers is easy"
result = tokenizer(sequence) # call the tokenizer directly
print(result)

tokens = tokenizer.tokenize(sequence) # gives our tokens back
print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens) # gives our ids back
print(ids)

decorded_string = tokenizer.decode(ids) # gives our original string back
print(decorded_string)
