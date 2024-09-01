from models import finetune_model, generate_weak_labels

def run_baseline_experiment(weak_model, strong_model, train_data, test_data, ground_truth):
    # Finetune the strong model on the train data
    finetuned_strong_model = finetune_model(strong_model, train_data)
    
    # Evaluate the finetuned strong model
    baseline_accuracy = evaluate_model(finetuned_strong_model, test_data, ground_truth)
    
    return [baseline_accuracy]

def run_aux_conf_experiment(weak_model, strong_model, train_data, test_data, ground_truth):
    # Generate weak labels using the weak model
    weak_labels = generate_weak_labels(weak_model, train_data)
    
    # Finetune the strong model with auxiliary confidence loss
    finetuned_strong_model = finetune_model_with_aux_conf(strong_model, train_data, weak_labels)
    
    # Evaluate the finetuned strong model
    aux_conf_accuracy = evaluate_model(finetuned_strong_model, test_data, ground_truth)
    
    return [aux_conf_accuracy]

def run_bootstrapping_experiment(weak_model, intermediate_model, strong_model, train_data, test_data, ground_truth):
    # Generate weak labels using the weak model
    weak_labels = generate_weak_labels(weak_model, train_data)
    
    # Finetune the intermediate model with weak labels
    finetuned_intermediate = finetune_model(intermediate_model, train_data, labels=weak_labels)
    
    # Generate intermediate labels
    intermediate_labels = generate_weak_labels(finetuned_intermediate, train_data)
    
    # Finetune the strong model with intermediate labels
    finetuned_strong_model = finetune_model(strong_model, train_data, labels=intermediate_labels)
    
    # Evaluate the finetuned strong model
    bootstrap_accuracy = evaluate_model(finetuned_strong_model, test_data, ground_truth)
    
    return [bootstrap_accuracy]

def evaluate_model(model, test_data, ground_truth):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for inputs, labels in test_data:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    accuracy = correct_predictions / total_predictions
    return accuracy

def finetune_model_with_aux_conf(model, train_data, weak_labels, num_epochs=2, batch_size=32, learning_rate=5e-5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for batch, weak_batch in zip(torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True),
                                     torch.utils.data.DataLoader(weak_labels, batch_size=batch_size, shuffle=True)):
            inputs, labels = batch
            weak_label = weak_batch
            
            outputs = model(inputs, labels=labels)
            main_loss = outputs.loss
            
            # Calculate auxiliary confidence loss
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            aux_conf_loss = torch.mean(torch.abs(probs.max(dim=-1)[0] - weak_label))
            
            # Combine losses
            total_loss = main_loss + aux_conf_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
    return model