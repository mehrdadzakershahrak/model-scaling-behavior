import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def load_models():
    # Load pretrained models of different sizes
    weak_model = GPT2LMHeadModel.from_pretrained('gpt2')
    intermediate_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    strong_model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    
    return weak_model, intermediate_model, strong_model

def finetune_model(model, train_data, num_epochs=2, batch_size=32, learning_rate=5e-5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for batch in torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True):
            inputs, labels = batch
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model

def generate_weak_labels(weak_model, data):
    weak_model.eval()
    weak_labels = []
    
    with torch.no_grad():
        for item in data:
            inputs = item['input_ids'].unsqueeze(0)
            outputs = weak_model(inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=-1).squeeze().item()
            weak_labels.append(predicted_label)
    
    return weak_labels