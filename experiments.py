from models import finetune_model, generate_weak_labels, finetune_model_with_aux_conf, generative_finetuning
from sklearn.metrics import accuracy_score

def run_baseline_experiment(weak_model, strong_model, train_data, test_data, ground_truth):
    # Fine-tune weak and strong models
    finetuned_weak = finetune_model(weak_model, train_data)
    finetuned_strong = finetune_model(strong_model, train_data)
    
    # Evaluate models
    weak_preds = generate_weak_labels(finetuned_weak, test_data)
    strong_preds = generate_weak_labels(finetuned_strong, test_data)
    
    weak_acc = accuracy_score(ground_truth, weak_preds)
    strong_acc = accuracy_score(ground_truth, strong_preds)
    
    return [weak_acc, (weak_acc + strong_acc) / 2, strong_acc]

def run_aux_conf_experiment(weak_model, strong_model, train_data, test_data, ground_truth):
    # Generate weak labels
    weak_labels = generate_weak_labels(weak_model, train_data)
    
    # Fine-tune models with auxiliary confidence loss
    finetuned_weak = finetune_model_with_aux_conf(weak_model, train_data, weak_labels)
    finetuned_strong = finetune_model_with_aux_conf(strong_model, train_data, weak_labels)
    
    # Evaluate models
    weak_preds = generate_weak_labels(finetuned_weak, test_data)
    strong_preds = generate_weak_labels(finetuned_strong, test_data)
    
    weak_acc = accuracy_score(ground_truth, weak_preds)
    strong_acc = accuracy_score(ground_truth, strong_preds)
    
    return [weak_acc, (weak_acc + strong_acc) / 2, strong_acc]

def run_bootstrapping_experiment(weak_model, intermediate_model, strong_model, train_data, test_data, ground_truth):
    # Generate weak labels
    weak_labels = generate_weak_labels(weak_model, train_data)
    
    # Fine-tune intermediate model with weak labels
    finetuned_intermediate = finetune_model_with_aux_conf(intermediate_model, train_data, weak_labels)
    
    # Generate better labels using finetuned intermediate model
    better_labels = generate_weak_labels(finetuned_intermediate, train_data)
    
    # Fine-tune strong model with better labels
    finetuned_strong = finetune_model_with_aux_conf(strong_model, train_data, better_labels)
    
    # Evaluate models
    weak_preds = generate_weak_labels(weak_model, test_data)
    intermediate_preds = generate_weak_labels(finetuned_intermediate, test_data)
    strong_preds = generate_weak_labels(finetuned_strong, test_data)
    
    weak_acc = accuracy_score(ground_truth, weak_preds)
    intermediate_acc = accuracy_score(ground_truth, intermediate_preds)
    strong_acc = accuracy_score(ground_truth, strong_preds)
    
    return [weak_acc, intermediate_acc, strong_acc]

def run_generative_finetuning_experiment(weak_model, strong_model, train_data, test_data, ground_truth, unlabeled_data):
    # Generative fine-tuning on unlabeled data
    gen_finetuned_weak = generative_finetuning(weak_model, unlabeled_data)
    gen_finetuned_strong = generative_finetuning(strong_model, unlabeled_data)
    
    # Fine-tune on labeled data
    finetuned_weak = finetune_model(gen_finetuned_weak, train_data)
    finetuned_strong = finetune_model(gen_finetuned_strong, train_data)
    
    # Evaluate models
    weak_preds = generate_weak_labels(finetuned_weak, test_data)
    strong_preds = generate_weak_labels(finetuned_strong, test_data)
    
    weak_acc = accuracy_score(ground_truth, weak_preds)
    strong_acc = accuracy_score(ground_truth, strong_preds)
    
    return [weak_acc, (weak_acc + strong_acc) / 2, strong_acc]