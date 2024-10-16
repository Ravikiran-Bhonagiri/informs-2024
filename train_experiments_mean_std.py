import json
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import datetime
# Import your dataset and models
from Custom_dataset import VideoAnalyticsDataset
from experiment_configs import experiment_configs
from Models import (
    NumericalOnlyModel,
    CategoricalOnlyModel,
    NLPOnlyModel,
    NumericalCategoricalModel,
    NumericalNLPModel,
    CategoricalNLPModel,
    MultiInputModel,
    MultiInputModelRes,
    WideAndDeepModel
)

from build_dictionary import build_word_dictionary

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

build_word_dictionary('baseline_6_all_features.csv', ['video_description', 'transcribe_text' ,'generated_vlm_text','llm_response','processed_response'], 'word_dictionary.json')

# Paths
data_file_path = 'data_normalized.csv'  # Update with your data file path
word_dict_path = 'word_dictionary.json'  # Update with your word dictionary path

# Load word tokenizer if NLP features are used
need_word_tokenizer = any(config['nlp_cols'] for config in experiment_configs)
if need_word_tokenizer:
    with open(word_dict_path, 'r') as f:
        word_tokenizer = json.load(f)
    vocab_size = len(word_tokenizer)
else:
    word_tokenizer = None
    vocab_size = None

# Training parameters
batch_size = 512
learning_rate = 0.003
num_epochs = 1000  # Adjust as needed

# Output directory for saving models and logs
output_base_dir = 'experiment_results'
os.makedirs(output_base_dir, exist_ok=True)
class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        return torch.mean(torch.abs((targets - predictions) / (targets + self.epsilon))) * 100

# Optionally, combine MSE and MAPE
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, epsilon=1e-8):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.mape = MAPELoss(epsilon)
        self.alpha = alpha

    def forward(self, predictions, targets):
        return self.alpha * self.mse(predictions, targets) + (1 - self.alpha) * self.mape(predictions, targets)
# Function to calculate MAPE
def calculate_mape(y_true, y_pred, epsilon=1e-8):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

# Function to calculate RMSE
def calculate_rmse(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Function to train and validate the model
def train_and_validate(model, train_loader, val_loader, test_loader, model_name, experiment_dir):
    num_epochs = 1000  # Adjust as needed
    learning_rate = 0.003  # Adjust as needed

    # Criteria and optimizer
    #criterion_regression = nn.MSELoss()
    # Example
    criterion_regression = CombinedLoss(alpha=.33).to(device)
    criterion_cross_entropy = nn.CrossEntropyLoss()  # For author_id_encoded
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize dictionaries to store MSE values per epoch
    train_mse_history = {'comment_count': [], 'heart_count': [], 'play_count': [], 'share_count': []}
    val_mse_history = {'comment_count': [], 'heart_count': [], 'play_count': [], 'share_count': []}

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        # Initialize variables to accumulate MSE values
        train_mse = {'comment_count': 0, 'heart_count': 0, 'play_count': 0, 'share_count': 0}
        num_train_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Prepare inputs and labels
            inputs = {}
            if 'numerical_features' in batch:
                inputs['numerical_features'] = batch['numerical_features'].to(device)
            if 'categorical_features' in batch:
                inputs['categorical_features'] = batch['categorical_features'].to(device)
            if 'nlp_tokens' in batch:
                nlp_tokens = batch['nlp_tokens'].to(device)
                inputs['nlp_tokens_list'] = [nlp_tokens[:, i, :] for i in range(nlp_tokens.size(1))]

            labels = batch['labels'].to(device)

            #author_id_true = labels[:, 4].long()
            comment_count_true = labels[:, 0]
            heart_count_true = labels[:, 1]
            play_count_true = labels[:, 2]
            share_count_true = labels[:, 3]

            # Forward pass
            outputs = model(**inputs)
            #author_id_pred = outputs['author_id_pred'].squeeze()
            comment_count_pred = outputs['comment_count_pred'].squeeze()
            heart_count_pred = outputs['heart_count_pred'].squeeze()
            play_count_pred = outputs['play_count_pred'].squeeze()
            share_count_pred = outputs['share_count_pred'].squeeze()

            # Compute losses
            #loss_author_id = criterion_cross_entropy(author_id_pred, author_id_true)  # CrossEntropy for author ID
            loss_comment_count = criterion_regression(comment_count_pred, comment_count_true)
            loss_heart_count = criterion_regression(heart_count_pred, heart_count_true)
            loss_play_count = criterion_regression(play_count_pred, play_count_true)
            loss_share_count = criterion_regression(share_count_pred, share_count_true)

            #total_batch_loss = (loss_author_id + 1*loss_comment_count + 1*loss_heart_count + 1*loss_play_count + 1*loss_share_count)
            total_batch_loss = (1*loss_comment_count + 1*loss_heart_count + 1*loss_play_count + 1*loss_share_count)
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()

            # Accumulate MSE values
            train_mse['comment_count'] += loss_comment_count.item()
            train_mse['heart_count'] += loss_heart_count.item()
            train_mse['play_count'] += loss_play_count.item()
            train_mse['share_count'] += loss_share_count.item()
            num_train_batches += 1

        # Average MSE over batches
        for key in train_mse:
            train_mse[key] /= num_train_batches
            train_mse_history[key].append(train_mse[key])

        # Validation
        model.eval()
        val_loss = 0
        val_mse = {'comment_count': 0, 'heart_count': 0, 'play_count': 0, 'share_count': 0}
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                # Prepare inputs and labels
                inputs = {}
                if 'numerical_features' in batch:
                    inputs['numerical_features'] = batch['numerical_features'].to(device)
                if 'categorical_features' in batch:
                    inputs['categorical_features'] = batch['categorical_features'].to(device)
                if 'nlp_tokens' in batch:
                    nlp_tokens = batch['nlp_tokens'].to(device)
                    inputs['nlp_tokens_list'] = [nlp_tokens[:, i, :] for i in range(nlp_tokens.size(1))]

                labels = batch['labels'].to(device)
                comment_count_true = labels[:, 0]
                heart_count_true = labels[:, 1]
                play_count_true = labels[:, 2]
                share_count_true = labels[:, 3]

                # Forward pass
                outputs = model(**inputs)
                comment_count_pred = outputs['comment_count_pred'].squeeze()
                heart_count_pred = outputs['heart_count_pred'].squeeze()
                play_count_pred = outputs['play_count_pred'].squeeze()
                share_count_pred = outputs['share_count_pred'].squeeze()

                # Compute losses
                loss_comment_count = criterion_regression(comment_count_pred, comment_count_true)
                loss_heart_count = criterion_regression(heart_count_pred, heart_count_true)
                loss_play_count = criterion_regression(play_count_pred, play_count_true)
                loss_share_count = criterion_regression(share_count_pred, share_count_true)

                total_batch_loss = (1*loss_comment_count + 1*loss_heart_count +
                                    1*loss_play_count + 1*loss_share_count)
                val_loss += total_batch_loss.item()

                # Accumulate MSE values
                val_mse['comment_count'] += loss_comment_count.item()
                val_mse['heart_count'] += loss_heart_count.item()
                val_mse['play_count'] += loss_play_count.item()
                val_mse['share_count'] += loss_share_count.item()
                num_val_batches += 1

        # Average MSE over batches
        for key in val_mse:
            val_mse[key] /= num_val_batches
            val_mse_history[key].append(val_mse[key])

        # Checkpoint the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Load the best model
    model.load_state_dict(best_model_state)

    # Testing
    model.eval()
    test_mse = {'comment_count': 0, 'heart_count': 0, 'play_count': 0, 'share_count': 0}
    num_test_batches = 0

    # Initialize lists to collect true and predicted values for metrics
    true_values = {'comment_count': [], 'heart_count': [], 'play_count': [], 'share_count': []}
    pred_values = {'comment_count': [], 'heart_count': [], 'play_count': [], 'share_count': []}

    with torch.no_grad():
        for batch in test_loader:
            # Prepare inputs and labels
            inputs = {}
            if 'numerical_features' in batch:
                inputs['numerical_features'] = batch['numerical_features'].to(device)
            if 'categorical_features' in batch:
                inputs['categorical_features'] = batch['categorical_features'].to(device)
            if 'nlp_tokens' in batch:
                nlp_tokens = batch['nlp_tokens'].to(device)
                inputs['nlp_tokens_list'] = [nlp_tokens[:, i, :] for i in range(nlp_tokens.size(1))]

            labels = batch['labels'].to(device)
            comment_count_true = labels[:, 0]
            heart_count_true = labels[:, 1]
            play_count_true = labels[:, 2]
            share_count_true = labels[:, 3]

            # Forward pass
            outputs = model(**inputs)
            comment_count_pred = outputs['comment_count_pred'].squeeze()
            heart_count_pred = outputs['heart_count_pred'].squeeze()
            play_count_pred = outputs['play_count_pred'].squeeze()
            share_count_pred = outputs['share_count_pred'].squeeze()

            # Compute losses
            loss_comment_count = criterion_regression(comment_count_pred, comment_count_true)
            loss_heart_count = criterion_regression(heart_count_pred, heart_count_true)
            loss_play_count = criterion_regression(play_count_pred, play_count_true)
            loss_share_count = criterion_regression(share_count_pred, share_count_true)

            # Accumulate MSE values
            test_mse['comment_count'] += loss_comment_count.item()
            test_mse['heart_count'] += loss_heart_count.item()
            test_mse['play_count'] += loss_play_count.item()
            test_mse['share_count'] += loss_share_count.item()
            num_test_batches += 1

            # Collect true and predicted values
            true_values['comment_count'].extend(comment_count_true.cpu().numpy())
            true_values['heart_count'].extend(heart_count_true.cpu().numpy())
            true_values['play_count'].extend(play_count_true.cpu().numpy())
            true_values['share_count'].extend(share_count_true.cpu().numpy())

            pred_values['comment_count'].extend(comment_count_pred.cpu().numpy())
            pred_values['heart_count'].extend(heart_count_pred.cpu().numpy())
            pred_values['play_count'].extend(play_count_pred.cpu().numpy())
            pred_values['share_count'].extend(share_count_pred.cpu().numpy())

    # Average MSE over batches and convert to Python float
    for key in test_mse:
        test_mse[key] /= num_test_batches
        test_mse[key] = float(test_mse[key])  # Convert to Python float

    # Compute RMSE and MAPE for test set
    test_rmse = {}
    test_mape = {}
    for key in test_mse:
        y_true = np.array(true_values[key])
        y_pred = np.array(pred_values[key])
        test_rmse_value = np.sqrt(mean_squared_error(y_true, y_pred))
        test_mape_value = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        # Convert to Python float
        test_rmse[key] = float(test_rmse_value)
        test_mape[key] = float(test_mape_value)

    # Save the MSE curves
    save_mse_curves(train_mse_history, val_mse_history, model_name, experiment_dir)

    # Save test metrics
    test_metrics = {
        'MSE': test_mse,
        'RMSE': test_rmse,
        'MAPE': test_mape
    }
    metrics_path = os.path.join(experiment_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)

    return test_metrics

def save_mse_curves(train_mse_history, val_mse_history, model_name, experiment_dir):
    epochs = range(1, len(train_mse_history['comment_count']) + 1)

    for target in ['comment_count', 'heart_count', 'play_count', 'share_count']:
        plt.figure()
        plt.plot(epochs, train_mse_history[target], label='Train MSE')
        plt.plot(epochs, val_mse_history[target], label='Validation MSE')
        plt.title(f'{model_name} - {target} MSE over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plot_path = os.path.join(experiment_dir, f'{target}_mse_curve.png')
        plt.savefig(plot_path)
        plt.close()

        # Save the data as CSV
        mse_data = pd.DataFrame({
            'Epoch': epochs,
            'Train MSE': train_mse_history[target],
            'Validation MSE': val_mse_history[target]
        })
        csv_path = os.path.join(experiment_dir, f'{target}_mse_data.csv')
        mse_data.to_csv(csv_path, index=False)

# Function to compute and save aggregated metrics
def compute_and_save_metrics(run_metrics, experiment_name):
    # Convert list of dicts to dict of lists
    metrics_dict = {}
    for key in run_metrics[0]['MSE'].keys():
        metrics_dict[key] = {
            'MSE': [run['MSE'][key] for run in run_metrics],
            'RMSE': [run['RMSE'][key] for run in run_metrics],
            'MAPE': [run['MAPE'][key] for run in run_metrics]
        }

    # Compute mean and std
    metrics_mean = {}
    metrics_std = {}
    for key in metrics_dict:
        metrics_mean[key] = {
            'MSE': np.mean(metrics_dict[key]['MSE']),
            'RMSE': np.mean(metrics_dict[key]['RMSE']),
            'MAPE': np.mean(metrics_dict[key]['MAPE'])
        }
        metrics_std[key] = {
            'MSE': np.std(metrics_dict[key]['MSE']),
            'RMSE': np.std(metrics_dict[key]['RMSE']),
            'MAPE': np.std(metrics_dict[key]['MAPE'])
        }

    # Save to CSV
    rows = []
    for key in metrics_mean:
        rows.append({
            'Metric': key,
            'MSE Mean': metrics_mean[key]['MSE'],
            'MSE Std': metrics_std[key]['MSE'],
            'RMSE Mean': metrics_mean[key]['RMSE'],
            'RMSE Std': metrics_std[key]['RMSE'],
            'MAPE Mean': metrics_mean[key]['MAPE'],
            'MAPE Std': metrics_std[key]['MAPE']
        })
    df = pd.DataFrame(rows)
    results_dir = os.path.join('experiment_results', f"{experiment_name}_results")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'aggregated_metrics.csv')
    df.to_csv(csv_path, index=False)

# Main training loop over experiments
import datetime
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
for config in experiment_configs:
    experiment_name = config['name']
    run_metrics = []  # List to store metrics from each run

    for run in range(1):  # Run each experiment 10 times
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = os.path.join(output_base_dir, f"{experiment_name}_run{run+1}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        print(f"\nStarting experiment: {experiment_name}, Run: {run+1}")

        # Prepare dataset
        dataset = VideoAnalyticsDataset(
            file_path=data_file_path,
            numerical_cols=config['numerical_cols'],
            categorical_cols=config['categorical_cols'],
            nlp_cols=config['nlp_cols'],
            label_cols=['video_comment_count', 'video_heart_count', 'video_play_count', 'video_share_count', 'author_id_encoded'],
            max_length=256,
            word_tokenizer=word_tokenizer if config['nlp_cols'] else None
        )

        # Split dataset into training, validation, and test sets (70%, 15%, 15%)
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn)

        # Prepare model
        model_params = config.get('model_params', {})
        output_sizes = {
            'comment_count_pred': 1,
            'heart_count_pred': 1,
            'play_count_pred': 1,
            'share_count_pred': 1
        }

        if config['model_class'].__name__ == 'NumericalOnlyModel':
            model = config['model_class'](
                numerical_feature_size=len(config['numerical_cols']),
                output_sizes=output_sizes,
                **model_params
            )
        elif config['model_class'].__name__ == 'CategoricalOnlyModel':
            categorical_feature_sizes = [dataset.get_categorical_feature_size(col) for col in config['categorical_cols']]
            model = config['model_class'](
                categorical_feature_sizes=categorical_feature_sizes,
                output_sizes=output_sizes,
                **model_params
            )
        elif config['model_class'].__name__ == 'NLPOnlyModel':
            num_nlp_inputs = len(config['nlp_cols'])
            model = config['model_class'](
                vocab_size=vocab_size,
                output_sizes=output_sizes,
                num_nlp_inputs=num_nlp_inputs,
                device=device,
                **model_params
            )
        elif config['model_class'].__name__ == 'NumericalCategoricalModel':
            categorical_feature_sizes = [dataset.get_categorical_feature_size(col) for col in config['categorical_cols']]
            model = config['model_class'](
                numerical_feature_size=len(config['numerical_cols']),
                categorical_feature_sizes=categorical_feature_sizes,
                output_sizes=output_sizes,
                **model_params
            )
        elif config['model_class'].__name__ == 'NumericalNLPModel':
            num_nlp_inputs = len(config['nlp_cols'])
            model = config['model_class'](
                numerical_feature_size=len(config['numerical_cols']),
                vocab_size=vocab_size,
                output_sizes=output_sizes,
                num_nlp_inputs=num_nlp_inputs,
                device=device,
                **model_params
            )
        elif config['model_class'].__name__ == 'CategoricalNLPModel':
            categorical_feature_sizes = [dataset.get_categorical_feature_size(col) for col in config['categorical_cols']]
            num_nlp_inputs = len(config['nlp_cols'])
            model = config['model_class'](
                categorical_feature_sizes=categorical_feature_sizes,
                vocab_size=vocab_size,
                output_sizes=output_sizes,
                num_nlp_inputs=num_nlp_inputs,
                device=device,
                **model_params
            )
        elif config['model_class'].__name__ == 'MultiInputModelRes':
            categorical_feature_sizes = [dataset.get_categorical_feature_size(col) for col in config['categorical_cols']]
            nlp_feature_configs = config['model_params'].get('nlp_feature_configs', [])
            model = config['model_class'](
                vocab_size=vocab_size,
                embed_size=config['model_params'].get('embed_size', 256),
                numerical_feature_size=len(config['numerical_cols']),
                categorical_feature_sizes=categorical_feature_sizes,
                nlp_feature_configs=nlp_feature_configs,
                shared_layer_sizes=config['model_params'].get('shared_layer_sizes', [256, 128]),
                output_sizes=output_sizes,
                device=device,
                max_length=config['model_params'].get('max_length', 256),
                dropout=config['model_params'].get('dropout', 0.1)
            )
        elif config['model_class'].__name__ == 'MultiInputModel':
            categorical_feature_sizes = [dataset.get_categorical_feature_size(col) for col in config['categorical_cols']]
            nlp_feature_configs = config['model_params'].get('nlp_feature_configs', [])
            model = config['model_class'](
                vocab_size=vocab_size,
                embed_size=config['model_params'].get('embed_size', 256),
                numerical_feature_size=len(config['numerical_cols']),
                categorical_feature_sizes=categorical_feature_sizes,
                nlp_feature_configs=nlp_feature_configs,
                shared_layer_sizes=config['model_params'].get('shared_layer_sizes', [256, 128]),
                output_sizes=output_sizes,
                device=device,
                max_length=config['model_params'].get('max_length', 256),
                dropout=config['model_params'].get('dropout', 0.1)
            )
        elif config['model_class'].__name__ == 'WideAndDeepModel':
            categorical_feature_sizes = [dataset.get_categorical_feature_size(col) for col in config['categorical_cols']]
            nlp_feature_config = config['model_params'].get('nlp_feature_config', [])
            model = config['model_class'](
                vocab_size=vocab_size,
                embed_size=config['model_params'].get('embed_size', 256),
                numerical_feature_size=len(config['numerical_cols']),
                categorical_feature_sizes=categorical_feature_sizes,
                nlp_feature_config=nlp_feature_config,
                deep_layer_sizes=config['model_params'].get('deep_layer_sizes', [256, 128]),
                output_sizes=output_sizes,
                device=device,
                max_length=config['model_params'].get('max_length', 256),
                dropout=config['model_params'].get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unsupported model class: {config['model_class'].__name__}")

        model.to(device)


        # Wrap the model with DataParallel if multiple GPUs are available
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for training.")
            model = nn.DataParallel(model)
        else:
            print("Using a single GPU for training.")
        # Train and validate the model
        metrics = train_and_validate(model, train_loader, val_loader, test_loader, config['name'], experiment_dir)

        # Store metrics from this run
        run_metrics.append(metrics)

    # After all runs, compute mean and std
    compute_and_save_metrics(run_metrics, experiment_name)

print("All experiments completed.")
