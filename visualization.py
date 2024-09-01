import matplotlib.pyplot as plt
import torch  # Add this import

def plot_scaling_behavior(baseline_results, aux_conf_results, bootstrap_results, gen_finetuning_results):
    plt.figure(figsize=(12, 8))
    
    model_sizes = ['Small', 'Medium', 'Large']
    
    plt.plot(model_sizes, baseline_results, marker='o', label='Baseline')
    plt.plot(model_sizes, aux_conf_results, marker='s', label='Auxiliary Confidence Loss')
    plt.plot(model_sizes, bootstrap_results, marker='^', label='Bootstrapping')
    plt.plot(model_sizes, gen_finetuning_results, marker='D', label='Generative Fine-tuning')
    
    plt.xlabel('Model Size')
    plt.ylabel('Accuracy')
    plt.title('Scaling Behavior of Different Training Approaches')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('scaling_behavior.png')
    plt.close()

def plot_error_distribution(model, test_data, ground_truth):
    model.eval()
    errors = []
    
    with torch.no_grad():
        for item, true_label in zip(test_data, ground_truth):
            inputs = item['input_ids'].unsqueeze(0)
            outputs = model(inputs)
            logits = outputs.logits
            predicted_label = torch.argmax(logits, dim=-1).squeeze().item()
            
            error = abs(predicted_label - true_label)
            errors.append(error)
    
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=20, edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution of Strong Model')
    
    plt.savefig('error_distribution.png')
    plt.close()