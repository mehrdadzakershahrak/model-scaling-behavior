from models import finetune_model, generate_weak_labels, finetune_model_with_aux_conf, generative_finetuning
import torch

def calculate_perplexity(model, data):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(data, batch_size=32):
            inputs, masks = batch
            outputs = model(inputs, attention_mask=masks, labels=inputs)
            total_loss += outputs.loss.item() * inputs.size(1)
            total_tokens += inputs.size(1)
    
    return torch.exp(torch.tensor(total_loss / total_tokens))

def run_baseline_experiment(weak_model, strong_model, train_data, test_data):
    # Fine-tune weak and strong models
    finetuned_weak = finetune_model(weak_model, train_data)
    finetuned_strong = finetune_model(strong_model, train_data)
    
    # Evaluate models
    weak_perplexity = calculate_perplexity(finetuned_weak, test_data)
    strong_perplexity = calculate_perplexity(finetuned_strong, test_data)
    
    return [weak_perplexity, (weak_perplexity + strong_perplexity) / 2, strong_perplexity]

def run_aux_conf_experiment(weak_model, strong_model, train_data, test_data):
    # Generate weak labels
    weak_labels = generate_weak_labels(weak_model, train_data)
    
    # Fine-tune models with auxiliary confidence loss
    finetuned_weak = finetune_model_with_aux_conf(weak_model, train_data, weak_labels)
    finetuned_strong = finetune_model_with_aux_conf(strong_model, train_data, weak_labels)
    
    # Evaluate models
    weak_perplexity = calculate_perplexity(finetuned_weak, test_data)
    strong_perplexity = calculate_perplexity(finetuned_strong, test_data)
    
    return [weak_perplexity, (weak_perplexity + strong_perplexity) / 2, strong_perplexity]

def run_bootstrapping_experiment(weak_model, intermediate_model, strong_model, train_data, test_data):
    # Generate weak labels
    weak_labels = generate_weak_labels(weak_model, train_data)
    
    # Fine-tune intermediate model with weak labels
    finetuned_intermediate = finetune_model_with_aux_conf(intermediate_model, train_data, weak_labels)
    
    # Generate better labels using finetuned intermediate model
    better_labels = generate_weak_labels(finetuned_intermediate, train_data)
    
    # Fine-tune strong model with better labels
    finetuned_strong = finetune_model_with_aux_conf(strong_model, train_data, better_labels)
    
    # Evaluate models
    weak_perplexity = calculate_perplexity(weak_model, test_data)
    intermediate_perplexity = calculate_perplexity(finetuned_intermediate, test_data)
    strong_perplexity = calculate_perplexity(finetuned_strong, test_data)
    
    return [weak_perplexity, intermediate_perplexity, strong_perplexity]

def run_generative_finetuning_experiment(weak_model, strong_model, train_data, test_data, unlabeled_data):
    # Generative fine-tuning on unlabeled data
    gen_finetuned_weak = generative_finetuning(weak_model, unlabeled_data)
    gen_finetuned_strong = generative_finetuning(strong_model, unlabeled_data)
    
    # Fine-tune on labeled data
    finetuned_weak = finetune_model(gen_finetuned_weak, train_data)
    finetuned_strong = finetune_model(gen_finetuned_strong, train_data)
    
    # Evaluate models
    weak_perplexity = calculate_perplexity(finetuned_weak, test_data)
    strong_perplexity = calculate_perplexity(finetuned_strong, test_data)
    
    return [weak_perplexity, (weak_perplexity + strong_perplexity) / 2, strong_perplexity]