from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

Xtrain = ["I've been waiting for a HuggingFace course my whole life.", "Python is great!"]
#now we use multiple sentences

res = classifier(Xtrain) #here we feed the list of sentences to the pipeline through classifier
print(res)


batch = tokenizer(Xtrain, padding=True, truncation=True, max_length=512, return_tensors="pt")
# Here we feed the data to the tokenizer
# padding=True will pad the sentences to the same length
# return_tensors="pt" will return tensors in pytorch format
print(batch)

with torch.no_grad():
    outputs = model(**batch)
    print(outputs)

    predictions = F.softmax(outputs.logits, dim=-1)
    print(predictions)
    
labels = torch.argmax(predictions, dim=-1)
print(labels)




""" Output:
[{'label': 'POSITIVE', 'score': 0.9598048329353333}, {'label': 'POSITIVE', 'score': 0.9998615980148315}]
{'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102],
        [  101, 18750,  2003,  2307,   999,   102,     0,     0,     0,     0,
             0,     0,     0,     0,     0,     0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}  
SequenceClassifierOutput(loss=None, logits=tensor([[-1.5607,  1.6123],
        [-4.2745,  4.6111]]), hidden_states=None, attentions=None)
tensor([[4.0195e-02, 9.5980e-01],
        [1.3835e-04, 9.9986e-01]])
tensor([1, 1])"""