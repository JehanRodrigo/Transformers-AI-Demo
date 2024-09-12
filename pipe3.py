from transformers import pipeline

classifier = pipeline("zero-shot-classification")

res = classifier(
"This is a course about the Transformers library", 
candidate_labels=["education", "politics", "business"],
)

print(res)

# model.safetensors:  43%|▍| 703M/1.63G [01:52<02:23, 6.48MB/s