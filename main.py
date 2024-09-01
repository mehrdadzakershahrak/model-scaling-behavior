from models import load_models
from data_utils import load_and_preprocess_data
from experiments import (
    run_baseline_experiment,
    run_aux_conf_experiment,
    run_bootstrapping_experiment,
    run_generative_finetuning_experiment
)
from visualization import plot_scaling_behavior, plot_error_distribution

if __name__ == "__main__":
    # Load models
    weak_model, intermediate_model, strong_model = load_models()
    
    # Load and preprocess data
    train_data, test_data, ground_truth, unlabeled_data = load_and_preprocess_data()
    
    # Run experiments
    baseline_results = run_baseline_experiment(weak_model, strong_model, train_data, test_data, ground_truth)
    aux_conf_results = run_aux_conf_experiment(weak_model, strong_model, train_data, test_data, ground_truth)
    bootstrap_results = run_bootstrapping_experiment(weak_model, intermediate_model, strong_model, train_data, test_data, ground_truth)
    gen_finetuning_results = run_generative_finetuning_experiment(weak_model, strong_model, train_data, test_data, ground_truth, unlabeled_data)
    
    # Visualize results
    plot_scaling_behavior(baseline_results, aux_conf_results, bootstrap_results, gen_finetuning_results)
    plot_error_distribution(strong_model, test_data, ground_truth)

    # Print results
    print("Baseline Results:", baseline_results)
    print("Auxiliary Confidence Loss Results:", aux_conf_results)
    print("Bootstrapping Results:", bootstrap_results)
    print("Generative Fine-tuning Results:", gen_finetuning_results)
