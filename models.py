import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from openai import OpenAI

def load_models():
    # Load pretrained models of different sizes using OpenAI API
    client = OpenAI()
    weak_model = client.models.retrieve("text-davinci-001")
    intermediate_model = client.models.retrieve("text-davinci-002")
    strong_model = client.models.retrieve("text-davinci-003")
    
    return weak_model, intermediate_model, strong_model

def finetune_model(model, train_data, num_epochs=2, batch_size=32, learning_rate=1e-4):
    client = OpenAI()
    
    # Prepare the training data in the format required by OpenAI's API
    formatted_data = [
        {"prompt": item['text'], "completion": str(item['label'])}
        for item in train_data
    ]
    
    # Create a fine-tuning job
    response = client.fine_tunes.create(
        model=model.id,
        training_file=formatted_data,
        n_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate_multiplier=learning_rate
    )
    
    # Wait for the fine-tuning to complete
    fine_tune_job = client.fine_tunes.retrieve(response.id)
    while fine_tune_job.status != "succeeded":
        fine_tune_job = client.fine_tunes.retrieve(response.id)
    
    # Return the fine-tuned model
    return client.models.retrieve(fine_tune_job.fine_tuned_model)

def generate_weak_labels(weak_model, data):
    client = OpenAI()
    weak_labels = []
    
    for item in data:
        response = client.completions.create(
            model=weak_model.id,
            prompt=item['text'],
            max_tokens=1,
            n=1,
            stop=None,
            temperature=0.5
        )
        weak_label = int(response.choices[0].text.strip())
        weak_labels.append(weak_label)
    
    return weak_labels

def finetune_model_with_aux_conf(model, train_data, weak_labels, num_epochs=2, batch_size=32, learning_rate=1e-4, alpha_max=0.75):
    client = OpenAI()
    
    # Prepare the training data with auxiliary confidence loss
    formatted_data = []
    for item, weak_label in zip(train_data, weak_labels):
        alpha = min(item['confidence'], alpha_max)
        prompt = f"{item['text']} Confidence: {alpha}"
        completion = f"{item['label']} Weak label: {weak_label}"
        formatted_data.append({"prompt": prompt, "completion": completion})
    
    # Create a fine-tuning job
    response = client.fine_tunes.create(
        model=model.id,
        training_file=formatted_data,
        n_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate_multiplier=learning_rate
    )
    
    # Wait for the fine-tuning to complete
    fine_tune_job = client.fine_tunes.retrieve(response.id)
    while fine_tune_job.status != "succeeded":
        fine_tune_job = client.fine_tunes.retrieve(response.id)
    
    # Return the fine-tuned model
    return client.models.retrieve(fine_tune_job.fine_tuned_model)

def generative_finetuning(model, unlabeled_data, num_epochs=2, batch_size=16, learning_rate=5e-5):
    client = OpenAI()
    
    # Prepare the unlabeled data for generative fine-tuning
    formatted_data = [
        {"prompt": item['text'], "completion": ""}
        for item in unlabeled_data
    ]
    
    # Create a fine-tuning job
    response = client.fine_tunes.create(
        model=model.id,
        training_file=formatted_data,
        n_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate_multiplier=learning_rate
    )
    
    # Wait for the fine-tuning to complete
    fine_tune_job = client.fine_tunes.retrieve(response.id)
    while fine_tune_job.status != "succeeded":
        fine_tune_job = client.fine_tunes.retrieve(response.id)
    
    # Return the fine-tuned model
    return client.models.retrieve(fine_tune_job.fine_tuned_model)