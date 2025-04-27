from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

# 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 2分类: O 和 NAME

# 标签映射
label_map = {0: 'O', 1: 'NAME'}

# 训练数据
train_texts = [
    "张三",
    "北京市朝阳医院",
    "This is normal text",
    "Patient ID: 123456",
    "The hospital is located in Chicago",
    "My favorite color is blue",
    "李四",
    "电话: 123-456-7890"
]
train_labels = [
    [1],  # 张三 -> NAME
    [1],  # 北京市朝阳医院 -> NAME
    [0, 0, 0, 0, 0],  # "This" -> O, "is" -> O, etc.
    [0, 0, 0, 0],  # "Patient" -> O, "ID" -> O, etc.
    [0, 0, 0, 0, 0, 0, 0],  # "The" -> O, "hospital" -> O, etc.
    [0, 0, 0, 0],  # "My" -> O, "favorite" -> O, etc.
    [1],  # 李四 -> NAME
    [0, 0, 0]   # "电话" -> O, "123-456-7890" -> O
]

# 测试数据
test_texts = [
    "John Doe",
    "I love reading books",
    "Medical Record Number: 789123",
    "It is a sunny day",
    "深圳市人民医院"
]

# 训练数据处理
train_encodings = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt", is_split_into_words=False)

# 计算最大序列长度
max_length = max(len(encoding) for encoding in train_encodings['input_ids'])

# 填充标签使其和token长度一致
train_labels_padded = []
for label in train_labels:
    label_padded = label + [0] * (max_length - len(label))  # Padding
    train_labels_padded.append(label_padded)

train_labels_padded = torch.tensor(train_labels_padded)

# 简单训练1-2个epoch（示范用）
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 训练过程
model.train()
for epoch in range(2):  # 训练2轮
    for i in range(len(train_texts)):
        inputs = {key: val[i:i+1] for key, val in train_encodings.items()}  # batch size = 1
        labels = train_labels_padded[i:i+1]
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch+1} Loss:", loss.item())

# 测试
model.eval()
with torch.no_grad():
    for text in test_texts:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)

        # 获取tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        print(f"Text: {text}")
        for token, pred in zip(tokens, predictions[0]):
            label = label_map[pred.item()]
            print(f"Token: {token} --> {label}")
        print()
