Python 3.12.2 (tags/v3.12.2:6abddd9, Feb  6 2024, 21:26:36) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
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
... # 加载预训练的BERT分词器
... tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
... 
... def tokenize_function(examples):
...     return tokenizer(examples['text'], padding=True, truncation=True)
... 
... # 应用分词器到整个数据集
... tokenized_datasets = dataset.map(tokenize_function, batched=True)
... 
... # 将数据集分为训练集和验证集（80%训练，20%验证）
... train_dataset = tokenized_datasets.train_test_split(test_size=0.8)['train']
... eval_dataset = tokenized_datasets.train_test_split(test_size=0.2)['test']
... 
... # 加载预训练BERT模型进行文本分类
... model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
... 
... # 设置训练参数
... training_args = TrainingArguments(
...     output_dir="./results",           # 输出目录
...     evaluation_strategy="epoch",      # 每个epoch评估一次
...     learning_rate=2e-5,               # 学习率
...     per_device_train_batch_size=8,    # 每个设备训练的批量大小
...     per_device_eval_batch_size=8,     # 每个设备评估的批量大小
...     num_train_epochs=3,               # 训练的epoch数量
...     weight_decay=0.01,                # 权重衰减
...     logging_dir='./logs',             # 日志目录
... )
... 
... # 使用Trainer API进行训练
... trainer = Trainer(
...     model=model, 
...     args=training_args, 
...     train_dataset=train_dataset, 
...     eval_dataset=eval_dataset
... )
... 
... # 训练模型
... trainer.train()
... 
... # 计算帖子中的违规词汇比例
... def check_violating_content_ratio(post, violating_words=VIOLATING_WORDS):
...     words = post.lower().split()
...     violating_count = sum(1 for word in words if word in violating_words)
...     return violating_count / len(words) if len(words) > 0 else 0 
... 
... # 处理预测：如果帖子为违规，则检查违规词汇比例
def process_post(post):
   
    inputs = tokenizer(post, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1).item() 

    if prediction == 1:  # 如果帖子被判定为违规
        violating_ratio = check_violating_content_ratio(post)
        
        if violating_ratio > 0.3:  # 如果违规词汇比例超过30%，直接删除
            return "Post deleted due to high violating content ratio."
        else:  # 否则，触发人工审核
            return "Post flagged for manual review."

    else:  # 如果帖子未违规，正常通过
        return "Post is safe."

# 测试
test_posts = [
    "I love this event, it’s amazing!",
    "This event is a fraud, I hate it!",
    "This is the worst scam ever!",
    "I had a great time, thanks for organizing it!"
]

for post in test_posts:
    result = process_post(post)
    print(f"Post: {post}\nResult: {result}\n")
KeyboardInterrupt
