import pandas as pd
from sklearn.model_selection import train_test_split
import torch  # Add this import

def load_and_preprocess_data():
    # Simplified implementation for demonstration
    data = pd.DataFrame({
        'text': ['Sample text 1', 'Sample text 2', 'Sample text 3'],
        'label': [0, 1, 2]
    })
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Convert to PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(train_data.index.values),
        torch.tensor(train_data['label'].values)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(test_data.index.values),
        torch.tensor(test_data['label'].values)
    )
    
    ground_truth = test_data['label'].values
    
    return train_dataset, test_dataset, ground_truth

# Remove unused functions