from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

from datasets import load_dataset

# Load the pre-trained model and tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

model = GPT2LMHeadModel.from_pretrained('gpt2')

# Load your custom dataset or sample dataset

# Dataset should include input texts where completion is needed

dataset = load_dataset('csv', data_files='Fine Tuning a Model\data\medical_completion_dataset.csv')

# Tokenize the data

def tokenize_function(examples):

    return tokenizer(examples['Input'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up training arguments

training_args = TrainingArguments(

    output_dir='./results',          # output directory

    evaluation_strategy="epoch",     # evaluate at each epoch

    per_device_train_batch_size=16,  # batch size for training

    per_device_eval_batch_size=16,   # batch size for evaluation

    num_train_epochs=3,              # number of training epochs

    weight_decay=0.01,               # strength of weight decay

)

# Trainer setup for fine-tuning

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=tokenized_datasets['train'],

    eval_dataset=tokenized_datasets['validation'],

)

# Fine-tune the model

trainer.train()

# Save the fine-tuned model

model.save_pretrained('./fine_tuned_gpt2')

tokenizer.save_pretrained('./fine_tuned_gpt2')
