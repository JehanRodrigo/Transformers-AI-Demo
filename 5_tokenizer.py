from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# default model that is used by the pipeline("sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
#calls the model

tokenizer = AutoTokenizer.from_pretrained(model_name)
# from_pretrained() method is a class method that is used to instantiate a model from a pre-trained model configuration.

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


# {'input_ids': [101, 2478, 19081, 2003, 3733, 102], 'attention_mask': [1, 1, 1, 1, 1, 1]}
# 101 and 102 marks the beginning and end of the sequence
# 'attention_mask' is used to tell the model which tokens to pay attention to and which to ignore using 1 and 0 respectively
# ['using', 'transformers', 'is', 'easy']
#[2478, 19081, 2003, 3733]
#using transformers is easy