import torch
from transformers import GPT2LMHeadModel, GPT2Config

def load_models():
    weak_model = GPT2LMHeadModel.from_pretrained('gpt2-small')
    intermediate_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    strong_model = GPT2LMHeadModel.from_pretrained('gpt2-large')
    
    return weak_model, intermediate_model, strong_model

def finetune_model(model, train_data, num_epochs=2, batch_size=32, learning_rate=1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(num_epochs):
        for batch in torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True):
            inputs, masks = batch
            outputs = model(inputs, attention_mask=masks, labels=inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return model

def generate_weak_labels(weak_model, data):
    weak_model.eval()
    weak_labels = []
    
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(data, batch_size=32):
            inputs, masks = batch
            outputs = weak_model(inputs, attention_mask=masks)
            weak_labels.extend(outputs.logits.argmax(dim=-1).tolist())
    
    return weak_labels

def finetune_model_with_aux_conf(model, train_data, weak_labels, num_epochs=2, batch_size=32, learning_rate=1e-4, alpha_max=0.75):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(num_epochs):
        for batch, weak_label_batch in zip(torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True),
                                           torch.utils.data.DataLoader(weak_labels, batch_size=batch_size, shuffle=True)):
            inputs, masks = batch
            outputs = model(inputs, attention_mask=masks, labels=inputs)
            main_loss = outputs.loss
            
            # Calculate auxiliary confidence loss
            logits = outputs.logits
            weak_probs = torch.softmax(logits, dim=-1)
            aux_conf_loss = -torch.mean(torch.sum(weak_probs * torch.log(weak_probs + 1e-10), dim=-1))
            
            # Combine losses
            alpha = min(epoch / num_epochs * alpha_max, alpha_max)
            total_loss = (1 - alpha) * main_loss + alpha * aux_conf_loss
            
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return model

def generative_finetuning(model, unlabeled_data, num_epochs=2, batch_size=16, learning_rate=5e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(num_epochs):
        for batch in torch.utils.data.DataLoader(unlabeled_data, batch_size=batch_size, shuffle=True):
            inputs, masks = batch
            
            # Generate synthetic continuations
            with torch.no_grad():
                continuations = model.generate(inputs, attention_mask=masks, max_length=inputs.size(1) * 2, do_sample=True)
            
            # Concatenate original inputs with generated continuations
            extended_inputs = torch.cat([inputs, continuations[:, inputs.size(1):]], dim=1)
            extended_masks = torch.ones_like(extended_inputs)
            
            # Train on extended inputs
            outputs = model(extended_inputs, attention_mask=extended_masks, labels=extended_inputs)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return model