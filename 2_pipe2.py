from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
"In this course, we will teach you how to", 
max_length=50, 
num_return_sequences=2,
)

print(res)

# [{'generated_text': "In this course, we will teach you how to use the internet to connect a variety of different services. We're not going to give a tutorial on the how to write code with Javascript and we'll teach you how to use the internet as an interface"}, {'generated_text': 'In this course, we will teach you how to use tools like OpenWiz to explore and learn more about working with the most advanced tools in the world.\n\n\n\n\nAs of the 2014 Spring 2016 edition, we are now 100%'}