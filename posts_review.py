>>> import pandas as pd
... from datasets import Dataset
... from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
... 
... data_path = "language_data.csv"
... df = pd.read_csv(data_path)
... 
... dataset = Dataset.from_pandas(df)
... 
... violating_words_path = "violating_words.csv"
... violating_df = pd.read_csv(violating_words_path)
... VIOLATING_WORDS = violating_df['word'].tolist()
... 
... # BERT tokenizer
... tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
... 
... def tokenize_function(examples):
...     return tokenizer(examples['text'], padding=True, truncation=True)
... 
... # apply the tokenizer to the entire dataset
... tokenized_datasets = dataset.map(tokenize_function, batched=True)
... 
... # split the dataset into training and evaluation sets (80% training, 20% evaluation)
... train_dataset = tokenized_datasets.train_test_split(test_size=0.8)['train']
... eval_dataset = tokenized_datasets.train_test_split(test_size=0.2)['test']
... 
... # text classification
... model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
... 
... # set up training arguments
... training_args = TrainingArguments(
...     output_dir="./results",         
...     evaluation_strategy="epoch",     
...     learning_rate=2e-5,              
...     per_device_train_batch_size=8,    
...     per_device_eval_batch_size=8,     
...     num_train_epochs=3,              
...     weight_decay=0.01,               
...     logging_dir='./logs',      
... )
... 
... trainer = Trainer(
...     model=model, 
...     args=training_args, 
...     train_dataset=train_dataset, 
...     eval_dataset=eval_dataset
... )
... 
... trainer.train()
... 
... # calculate the ratio of violating words in a post
... def check_violating_content_ratio(post, violating_words=VIOLATING_WORDS):
...     words = post.lower().split()
...     violating_count = sum(1 for word in words if word in violating_words)
...     return violating_count / len(words) if len(words) > 0 else 0 
... 
... # check for violations and decide if it should be deleted or flagged for review
def process_post(post):
   
    inputs = tokenizer(post, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1).item() 

    if prediction == 1:
        violating_ratio = check_violating_content_ratio(post)
        
        if violating_ratio > 0.5:  #  if the ratio of violating words is over 50%, delete the post
            return "Post deleted due to high violating content ratio."
        else:  # flag for manual review
            return "Post flagged for manual review."

    else:
        return "Post is safe."
