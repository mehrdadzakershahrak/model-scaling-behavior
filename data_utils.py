import pandas as pd
from sklearn.model_selection import train_test_split
import torch  # Add this import
from transformers import GPT2Tokenizer

def load_and_preprocess_data(dataset_name='wikitext', max_length=128):
    # Load dataset based on name
    if dataset_name == 'wikitext':
        data = load_wikitext()
    elif dataset_name == 'c4':
        data = load_c4()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Tokenize data
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    train_encodings = tokenizer(train_data['text'].tolist(), truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(test_data['text'].tolist(), truncation=True, padding=True, max_length=max_length)
    
    # Convert to PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_encodings['input_ids']),
        torch.tensor(train_encodings['attention_mask'])
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_encodings['input_ids']),
        torch.tensor(test_encodings['attention_mask'])
    )
    
    return train_dataset, test_dataset

def load_wikitext():
    # Placeholder for loading WikiText dataset
    # You would need to implement the actual loading logic
    pass

def load_c4():
    # Placeholder for loading C4 dataset
    # You would need to implement the actual loading logic
    pass

# Function to create unlabeled dataset
def create_unlabeled_dataset(dataset, unlabeled_ratio=0.8):
    unlabeled_size = int(len(dataset) * unlabeled_ratio)
    unlabeled_dataset, _ = torch.utils.data.random_split(dataset, [unlabeled_size, len(dataset) - unlabeled_size])
    return unlabeled_dataset