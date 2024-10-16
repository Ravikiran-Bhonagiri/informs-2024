import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json

import re
import emoji
from sklearn.preprocessing import LabelEncoder

class VideoAnalyticsDataset(Dataset):
    def __init__(self, file_path, numerical_cols=None, categorical_cols=None, nlp_cols=None, label_cols=None, max_length=512, save_word_dict=True, word_dict_path='word_dictionary.json', word_tokenizer=None):
        """
        Initializes the dataset by loading the data file and processing NLP columns.

        Args:
        - file_path (str): Path to the data file.
        - numerical_cols (list of str): List of numerical feature column names.
        - categorical_cols (list of str): List of categorical feature column names.
        - nlp_cols (list of str): List of NLP (text) feature column names.
        - label_cols (list of str): List of label column names.
        - max_length (int): Maximum length for tokenized sequences.
        - save_word_dict (bool): Whether to save the word dictionary to a file.
        - word_dict_path (str): Path to save the word dictionary.
        - word_tokenizer (dict): Pre-built word tokenizer dictionary. If provided, it will be used instead of building a new one.
        """
        # Load data
        self.data = pd.read_csv(file_path)
        
        # Set default columns
        if numerical_cols is None:
            numerical_cols = []
        if categorical_cols is None:
            categorical_cols = []
        if nlp_cols is None:
            nlp_cols = []
        if label_cols is None:
            label_cols = []
        
        self.max_length = max_length
        self.nlp_cols = nlp_cols
        self.categorical_cols = categorical_cols  # Store categorical column names
        self.save_word_dict = save_word_dict
        self.word_dict_path = word_dict_path
        
        # Initialize or load word tokenizer
        if word_tokenizer is not None:
            self.word_tokenizer = word_tokenizer
            self.build_word_tokenizer = False
        else:
            self.word_tokenizer = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
            self.build_word_tokenizer = True
        
        # Extract data
        self.numerical_data = self.data[numerical_cols].astype(np.float32).values if numerical_cols else np.array([])
        self.categorical_data = self.data[categorical_cols].astype(str).values if categorical_cols else np.array([])
        self.labels = self.data[label_cols].astype(np.float32).values if label_cols else np.array([])
        
        # Process categorical features
        self.label_encoders = []
        if categorical_cols:
            for i in range(self.categorical_data.shape[1]):
                le = LabelEncoder()
                self.categorical_data[:, i] = le.fit_transform(self.categorical_data[:, i])
                self.label_encoders.append(le)
            self.categorical_data = self.categorical_data.astype(np.int64)
        
        # Process NLP columns
        self.tokenized_nlp_data = []
        if nlp_cols:
            for idx in range(len(self.data)):
                tokenized_columns = []
                for col in nlp_cols:
                    text = self.data[col].iloc[idx]
                    cleaned_text = self.clean_text(text)
                    tokens = self.tokenize_text(cleaned_text)
                    tokenized_columns.append(tokens)
                self.tokenized_nlp_data.append(tokenized_columns)
            self.tokenized_nlp_data = np.array(self.tokenized_nlp_data)  # Shape: (num_samples, num_nlp_columns, max_length)
        else:
            self.tokenized_nlp_data = np.array([])
        
        # Save the word dictionary if required and if we built it
        if self.save_word_dict and self.build_word_tokenizer:
            with open(self.word_dict_path, 'w') as f:
                json.dump(self.word_tokenizer, f)
    
    def clean_text(self, text):
        """
        Cleans the text by removing emojis and extra spaces.
        """
        if pd.isna(text):
            return ''
        # Remove emojis
        text_no_emoji = emoji.replace_emoji(text, replace="")
        # Remove extra spaces
        cleaned_text = re.sub(r'\s+', ' ', text_no_emoji).strip()
        return cleaned_text

    def get_categorical_feature_size(self, col_name):
        idx = self.categorical_cols.index(col_name)
        le = self.label_encoders[idx]
        return len(le.classes_)
    
    def collate_fn(self, batch):
        batch_items = {}
        keys = batch[0].keys()
        for key in keys:
            if key == 'nlp_tokens':
                # Handle nlp_tokens specially
                nlp_tokens_list = [sample[key] for sample in batch]
                # Stack along batch dimension
                batch_items[key] = torch.stack(nlp_tokens_list, dim=0)  # Shape: (batch_size, num_nlp_inputs, seq_length)
            else:
                batch_items[key] = torch.stack([sample[key] for sample in batch])
        return batch_items

    def tokenize_text(self, text):
        """
        Tokenizes the text, updates the word tokenizer, and returns the tokenized sequence.
        """
        if not text:
            return [self.word_tokenizer['<pad>']] * self.max_length  # Return a padded sequence if text is empty
        
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Update the word tokenizer only if we are building it
        if self.build_word_tokenizer:
            for token in tokens:
                if token not in self.word_tokenizer:
                    self.word_tokenizer[token] = len(self.word_tokenizer)
        
        # Convert tokens to indices
        token_indices = [self.word_tokenizer.get(token, self.word_tokenizer['<unk>']) for token in tokens]
        
        # Create tokenized sequence with <start> and <end> tokens
        tokenized_sequence = [self.word_tokenizer['<start>']] + token_indices + [self.word_tokenizer['<end>']]
        
        # Pad or truncate the sequence to max_length
        if len(tokenized_sequence) < self.max_length:
            tokenized_sequence += [self.word_tokenizer['<pad>']] * (self.max_length - len(tokenized_sequence))
        else:
            tokenized_sequence = tokenized_sequence[:self.max_length]
        
        return tokenized_sequence
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        items = {}
        if self.numerical_data.size > 0:
            numerical_features = torch.tensor(self.numerical_data[idx], dtype=torch.float32)
            items['numerical_features'] = numerical_features
        if self.categorical_data.size > 0:
            categorical_features = torch.tensor(self.categorical_data[idx], dtype=torch.long)
            items['categorical_features'] = categorical_features
        if self.tokenized_nlp_data.size > 0:
            nlp_tokens_list = []
            for col_idx in range(len(self.nlp_cols)):
                nlp_tokens = torch.tensor(self.tokenized_nlp_data[idx, col_idx], dtype=torch.long)
                nlp_tokens_list.append(nlp_tokens)
            # Stack nlp_tokens_list to form (num_nlp_inputs, seq_length)
            nlp_tokens = torch.stack(nlp_tokens_list, dim=0)  # Shape: (num_nlp_inputs, seq_length)
            items['nlp_tokens'] = nlp_tokens
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        items['labels'] = labels
        return items

class VideoAnalyticsDataset2(Dataset):
    def __init__(self, file_path, normalize=True, save_stats=True, stats_file='normalization_stats.json'):
        """
        Initializes the dataset by loading the Excel file and extracting relevant data.
        Args:
        - file_path (str): Path to the Excel file with the data.
        - normalize (bool): Whether to normalize the continuous features.
        - save_stats (bool): Whether to save the mean and std values for normalization.
        - stats_file (str): Path to save the normalization stats (mean, std).
        """
        # Load the data from the Excel file
        self.data = pd.read_excel(file_path)
        #, 'author_total_video_count', 'author_total_heart_count'
        # Extract author-related features
        self.author_features = self.data[['author_follower_count', 'author_following_count']].values
        
        # Extract tokenized LLM responses
        self.llm_tokenized = self.data['llm_tokenized'].apply(lambda x: eval(x)).values  # Convert string representation of lists back to lists

        # Extract output labels: author_id_encoded and video-related counts
        self.author_id_encoded = self.data['author_id_encoded'].values
        self.video_comment_count = self.data['video_comment_count'].values
        self.video_heart_count = self.data['video_heart_count'].values
        self.video_play_count = self.data['video_play_count'].values
        self.video_share_count = self.data['video_share_count'].values

        # Normalize continuous features if required
        if normalize:
            self.author_features, author_stats = self._normalize_features(self.author_features)
            self.video_comment_count, comment_stats = self._normalize_values(self.video_comment_count)
            self.video_heart_count, heart_stats = self._normalize_values(self.video_heart_count)
            self.video_play_count, play_stats = self._normalize_values(self.video_play_count)
            self.video_share_count, share_stats = self._normalize_values(self.video_share_count)

            # Save the mean and std values for later use
            if save_stats:
                normalization_stats = {
                    "author_features": author_stats,
                    "video_comment_count": comment_stats,
                    "video_heart_count": heart_stats,
                    "video_play_count": play_stats,
                    "video_share_count": share_stats
                }
                with open(stats_file, 'w') as f:
                    json.dump(normalization_stats, f)

    def _normalize_features(self, features):
        """
        Normalize a set of features using mean and standard deviation.
        Args:
        - features (np.ndarray): The features to normalize.
        Returns:
        - normalized_features (np.ndarray): The normalized features.
        - stats (dict): Mean and std values for the features.
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Avoid division by zero if std is zero
        normalized_features = (features - mean) / std
        return normalized_features, {"mean": mean.tolist(), "std": std.tolist()}

    def _normalize_values(self, values):
        """
        Normalize a set of values using mean and standard deviation.
        Args:
        - values (np.ndarray): The values to normalize.
        Returns:
        - normalized_values (np.ndarray): The normalized values.
        - stats (dict): Mean and std values for the values.
        """
        mean = np.mean(values)
        std = np.std(values)
        std = std if std != 0 else 1  # Avoid division by zero if std is zero
        normalized_values = (values - mean) / std
        return normalized_values, {"mean": mean, "std": std}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the input features and target outputs for the given index.
        Args:
        - idx (int): Index of the sample.
        
        Returns:
        - author_features (Tensor): Tensor containing normalized author follower, following, and total video counts.
        - llm_tokenized (Tensor): Tensor containing tokenized LLM responses.
        - outputs (Tensor): Tensor containing normalized video-related counts and author_id_encoded.
        """
        # Convert author-related features and tokenized LLM response to tensors
        author_features = torch.tensor(self.author_features[idx], dtype=torch.float32)
        llm_tokenized = torch.tensor(self.llm_tokenized[idx], dtype=torch.long)
        
        # Create the output tensor: normalized video counts and author_id_encoded
        outputs = torch.tensor([self.author_id_encoded[idx],
                                self.video_comment_count[idx],
                                self.video_heart_count[idx],
                                self.video_play_count[idx],
                                self.video_share_count[idx]], dtype=torch.float32)

        return author_features, llm_tokenized, outputs

class VideoAnalyticsDataset3(Dataset):
    def __init__(self, file_path, normalize=True, save_stats=True, stats_file='scaling_stats.json', min_value=0, max_value=1):
        """
        Initializes the dataset by loading the Excel file and extracting relevant data.
        Args:
        - file_path (str): Path to the Excel file with the data.
        - normalize (bool): Whether to normalize the continuous features using Min-Max scaling.
        - save_stats (bool): Whether to save the min and max values for scaling.
        - stats_file (str): Path to save the scaling stats (min, max).
        - min_value (float): Minimum value for Min-Max scaling (default: 0).
        - max_value (float): Maximum value for Min-Max scaling (default: 1).
        """
        # Load the data from the Excel file
        self.data = pd.read_excel(file_path)

        # Extract author-related features
        self.author_features = self.data[['author_follower_count', 'author_following_count', 'author_total_video_count']].values
        
        # Extract tokenized LLM responses
        self.llm_tokenized = self.data['llm_tokenized'].apply(lambda x: eval(x)).values  # Convert string representation of lists back to lists

        # Extract output labels: author_id_encoded and video-related counts
        self.author_id_encoded = self.data['author_id_encoded'].values
        self.video_comment_count = self.data['video_comment_count'].values
        self.video_heart_count = self.data['video_heart_count'].values
        self.video_play_count = self.data['video_play_count'].values
        self.video_share_count = self.data['video_share_count'].values

        # Normalize continuous features using Min-Max scaling if required
        if normalize:
            self.author_features, author_stats = self._min_max_scale(self.author_features, min_value, max_value)
            self.video_comment_count, comment_stats = self._min_max_scale(self.video_comment_count, min_value, max_value)
            self.video_heart_count, heart_stats = self._min_max_scale(self.video_heart_count, min_value, max_value)
            self.video_play_count, play_stats = self._min_max_scale(self.video_play_count, min_value, max_value)
            self.video_share_count, share_stats = self._min_max_scale(self.video_share_count, min_value, max_value)

            # Save the min and max values for later use
            if save_stats:
                scaling_stats = {
                    "author_features": author_stats,
                    "video_comment_count": comment_stats,
                    "video_heart_count": heart_stats,
                    "video_play_count": play_stats,
                    "video_share_count": share_stats
                }
                with open(stats_file, 'w') as f:
                    json.dump(scaling_stats, f)

    def _min_max_scale(self, values, min_value, max_value):
        """
        Apply Min-Max scaling to a set of values.
        Args:
        - values (np.ndarray): The values to scale.
        - min_value (float): Minimum value for scaling.
        - max_value (float): Maximum value for scaling.
        Returns:
        - scaled_values (np.ndarray): The scaled values.
        - stats (dict): Min and max values for the scaling.
        """
        min_val = np.min(values, axis=0)
        max_val = np.max(values, axis=0)
        scaled_values = (values - min_val) / (max_val - min_val + 1e-8) * (max_value - min_value) + min_value  # Prevent division by zero
        return scaled_values, {"min": min_val.tolist(), "max": max_val.tolist()}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the input features and target outputs for the given index.
        Args:
        - idx (int): Index of the sample.
        
        Returns:
        - author_features (Tensor): Tensor containing scaled author follower, following, and total video counts.
        - llm_tokenized (Tensor): Tensor containing tokenized LLM responses.
        - outputs (Tensor): Tensor containing scaled video-related counts and author_id_encoded.
        """
        # Convert author-related features and tokenized LLM response to tensors
        author_features = torch.tensor(self.author_features[idx], dtype=torch.float32)
        llm_tokenized = torch.tensor(self.llm_tokenized[idx], dtype=torch.long)
        
        # Create the output tensor: scaled video counts and author_id_encoded
        outputs = torch.tensor([self.author_id_encoded[idx],
                                self.video_comment_count[idx],
                                self.video_heart_count[idx],
                                self.video_play_count[idx],
                                self.video_share_count[idx]], dtype=torch.float32)

        return author_features, llm_tokenized, outputs


class VideoAnalyticsDerivedDataset_BaselineModel(Dataset):
    def __init__(self, file_path, normalize=True, save_stats=True, stats_file='normalization_stats.json'):
        """
        Initializes the dataset by loading the Excel file and extracting relevant data.
        Args:
        - file_path (str): Path to the Excel file with the data.
        - normalize (bool): Whether to normalize the continuous features.
        - save_stats (bool): Whether to save the mean and std values for normalization.
        - stats_file (str): Path to save the normalization stats (mean, std).
        """
        # Load the data from the Excel file
        self.data = pd.read_excel(file_path)

        # Drop the year, month, day, hour, and dayofweek columns as you are using time embeddings for them
        self.data = self.data.drop(columns=['year', 'month', 'day', 'hour', 'dayofweek'])

        # Separate continuous and categorical features (to be normalized selectively)
        continuous_features = ['author_follower_count', 'author_following_count', 'author_total_heart_count', 'author_total_video_count']
        derived_features = ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'hour_sin', 'hour_cos']
        categorical_features = [
            'author_id_A1', 'author_id_A10', 'author_id_A11', 'author_id_A12', 'author_id_A13', 'author_id_A14', 'author_id_A15',
            'author_id_A2', 'author_id_A3', 'author_id_A4', 'author_id_A5', 'author_id_A6', 'author_id_A7', 'author_id_A8', 'author_id_A9',
            'video_definition_360p', 'video_definition_480p', 'video_definition_540p', 'video_definition_720p'
        ]

        # Extract features to be normalized
        self.continuous_features = self.data[continuous_features].values

        # Extract features that won't be normalized (categorical and one-hot encoded)
        self.derived_features = self.data[derived_features].values
        self.categorical_features = self.data[categorical_features].values

        # Extract output labels: author_id_encoded and video-related counts (targets)
        self.author_id_encoded = self.data['author_id_encoded'].values
        self.video_comment_count = self.data['video_comment_count'].values
        self.video_heart_count = self.data['video_heart_count'].values
        self.video_play_count = self.data['video_play_count'].values
        self.video_share_count = self.data['video_share_count'].values

        # Normalize continuous features if required
        if normalize:
            self.continuous_features, author_stats = self._normalize_features(self.continuous_features)
            self.video_comment_count, comment_stats = self._normalize_values(self.video_comment_count)
            self.video_heart_count, heart_stats = self._normalize_values(self.video_heart_count)
            self.video_play_count, play_stats = self._normalize_values(self.video_play_count)
            self.video_share_count, share_stats = self._normalize_values(self.video_share_count)

            # Save the mean and std values for later use
            if save_stats:
                normalization_stats = {
                    "continuous_features": author_stats,
                    "video_comment_count": comment_stats,
                    "video_heart_count": heart_stats,
                    "video_play_count": play_stats,
                    "video_share_count": share_stats
                }
                with open(stats_file, 'w') as f:
                    json.dump(normalization_stats, f)

    def _normalize_features(self, features):
        """
        Normalize a set of features using mean and standard deviation.
        Args:
        - features (np.ndarray): The features to normalize.
        Returns:
        - normalized_features (np.ndarray): The normalized features.
        - stats (dict): Mean and std values for the features.
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Avoid division by zero if std is zero
        normalized_features = (features - mean) / std
        return normalized_features, {"mean": mean.tolist(), "std": std.tolist()}

    def _normalize_values(self, values):
        """
        Normalize a set of values using mean and standard deviation.
        Args:
        - values (np.ndarray): The values to normalize.
        Returns:
        - normalized_values (np.ndarray): The normalized values.
        - stats (dict): Mean and std values for the values.
        """
        mean = np.mean(values)
        std = np.std(values)
        std = std if std != 0 else 1  # Avoid division by zero if std is zero
        normalized_values = (values - mean) / std
        return normalized_values, {"mean": mean, "std": std}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the input features and target outputs for the given index.
        Args:
        - idx (int): Index of the sample.
        
        Returns:
        - author_features (Tensor): Tensor containing scaled author and derived features.
        - outputs (Tensor): Tensor containing scaled video-related counts and author_id_encoded.
        """
        # Convert continuous, derived, and categorical features to tensors
        continuous_features = torch.tensor(self.continuous_features[idx], dtype=torch.float32)
        derived_features = torch.tensor(self.derived_features[idx], dtype=torch.float32)
        categorical_features = torch.tensor(self.categorical_features[idx], dtype=torch.float32)

        # Concatenate all features (continuous, derived, categorical) into one tensor
        author_features = torch.cat([continuous_features, derived_features, categorical_features], dim=0)

        # Create the output tensor: scaled video counts and author_id_encoded
        outputs = torch.tensor([self.author_id_encoded[idx],
                                self.video_comment_count[idx],
                                self.video_heart_count[idx],
                                self.video_play_count[idx],
                                self.video_share_count[idx]], dtype=torch.float32)

        return author_features, outputs


class VideoAnalyticsDerivedDataset_Baseline_LLM(Dataset):
    def __init__(self, file_path, normalize=True, save_stats=True, stats_file='normalization_stats.json'):
        """
        Initializes the dataset by loading the Excel file and extracting relevant data.
        Args:
        - file_path (str): Path to the Excel file with the data.
        - normalize (bool): Whether to normalize the continuous features.
        - save_stats (bool): Whether to save the mean and std values for normalization.
        - stats_file (str): Path to save the normalization stats (mean, std).
        """
        # Load the data from the Excel file
        self.data = pd.read_excel(file_path)

        # Drop the year, month, day, hour, and dayofweek columns as you are using time embeddings for them
        self.data = self.data.drop(columns=['year', 'month', 'day', 'hour', 'dayofweek'])

        # Separate continuous and categorical features (to be normalized selectively)
        continuous_features = ['author_follower_count', 'author_following_count', 'author_total_heart_count', 'author_total_video_count']
        derived_features = ['month_sin', 'month_cos', 'dayofweek_sin', 'dayofweek_cos', 'hour_sin', 'hour_cos']
        categorical_features = [
            'author_id_A1', 'author_id_A10', 'author_id_A11', 'author_id_A12', 'author_id_A13', 'author_id_A14', 'author_id_A15',
            'author_id_A2', 'author_id_A3', 'author_id_A4', 'author_id_A5', 'author_id_A6', 'author_id_A7', 'author_id_A8', 'author_id_A9',
            'video_definition_360p', 'video_definition_480p', 'video_definition_540p', 'video_definition_720p'
        ]

        # LLM tokenized data (Assuming it is a string representation of lists in the file, hence `eval`)
        self.llm_tokenized = self.data['llm_tokenized'].apply(lambda x: eval(x)).values

        # Extract features to be normalized
        self.continuous_features = self.data[continuous_features].values

        # Extract features that won't be normalized (categorical and one-hot encoded)
        self.derived_features = self.data[derived_features].values
        self.categorical_features = self.data[categorical_features].values

        # Extract output labels: author_id_encoded and video-related counts (targets)
        self.author_id_encoded = self.data['author_id_encoded'].values
        self.video_comment_count = self.data['video_comment_count'].values
        self.video_heart_count = self.data['video_heart_count'].values
        self.video_play_count = self.data['video_play_count'].values
        self.video_share_count = self.data['video_share_count'].values

        # Normalize continuous features if required
        if normalize:
            self.continuous_features, author_stats = self._normalize_features(self.continuous_features)
            self.video_comment_count, comment_stats = self._normalize_values(self.video_comment_count)
            self.video_heart_count, heart_stats = self._normalize_values(self.video_heart_count)
            self.video_play_count, play_stats = self._normalize_values(self.video_play_count)
            self.video_share_count, share_stats = self._normalize_values(self.video_share_count)

            # Save the mean and std values for later use
            if save_stats:
                normalization_stats = {
                    "continuous_features": author_stats,
                    "video_comment_count": comment_stats,
                    "video_heart_count": heart_stats,
                    "video_play_count": play_stats,
                    "video_share_count": share_stats
                }
                with open(stats_file, 'w') as f:
                    json.dump(normalization_stats, f)

    def _normalize_features(self, features):
        """
        Normalize a set of features using mean and standard deviation.
        Args:
        - features (np.ndarray): The features to normalize.
        Returns:
        - normalized_features (np.ndarray): The normalized features.
        - stats (dict): Mean and std values for the features.
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Avoid division by zero if std is zero
        normalized_features = (features - mean) / std
        return normalized_features, {"mean": mean.tolist(), "std": std.tolist()}

    def _normalize_values(self, values):
        """
        Normalize a set of values using mean and standard deviation.
        Args:
        - values (np.ndarray): The values to normalize.
        Returns:
        - normalized_values (np.ndarray): The normalized values.
        - stats (dict): Mean and std values for the values.
        """
        mean = np.mean(values)
        std = np.std(values)
        std = std if std != 0 else 1  # Avoid division by zero if std is zero
        normalized_values = (values - mean) / std
        return normalized_values, {"mean": mean, "std": std}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the input features and target outputs for the given index.
        Args:
        - idx (int): Index of the sample.
        
        Returns:
        - author_features (Tensor): Tensor containing scaled author and derived features.
        - llm_tokenized (Tensor): Tensor containing tokenized LLM responses.
        - outputs (Tensor): Tensor containing scaled video-related counts and author_id_encoded.
        """
        # Convert continuous, derived, and categorical features to tensors
        continuous_features = torch.tensor(self.continuous_features[idx], dtype=torch.float32)
        derived_features = torch.tensor(self.derived_features[idx], dtype=torch.float32)
        categorical_features = torch.tensor(self.categorical_features[idx], dtype=torch.float32)

        # Concatenate all features (continuous, derived, categorical) into one tensor
        author_features = torch.cat([continuous_features, derived_features, categorical_features], dim=0)

        # LLM tokenized features
        llm_tokenized = torch.tensor(self.llm_tokenized[idx], dtype=torch.long)

        # Create the output tensor: scaled video counts and author_id_encoded
        outputs = torch.tensor([self.author_id_encoded[idx],
                                self.video_comment_count[idx],
                                self.video_heart_count[idx],
                                self.video_play_count[idx],
                                self.video_share_count[idx]], dtype=torch.float32)

        return author_features, llm_tokenized, outputs


class VideoAnalyticsDatasetBERT(Dataset):
    def __init__(self, file_path, normalize=True, save_stats=True, stats_file='normalization_stats.json'):
        """
        Initializes the dataset by loading the Excel file and extracting relevant data.
        Args:
        - file_path (str): Path to the Excel file with the data.
        - normalize (bool): Whether to normalize the continuous features.
        - save_stats (bool): Whether to save the mean and std values for normalization.
        - stats_file (str): Path to save the normalization stats (mean, std).
        """
        # Load the data from the Excel file
        self.data = pd.read_excel(file_path)

        # Extract author-related features
        self.author_features = self.data[['author_follower_count', 'author_following_count', 'author_total_video_count']].values
        
        # Extract tokenized LLM responses
        self.llm_tokenized = self.data['llm_tokenized'].apply(lambda x: eval(x)).values  # Convert string representation of lists back to lists

        # Extract output labels: author_id_encoded and video-related counts
        self.author_id_encoded = self.data['author_id_encoded'].values
        self.video_comment_count = self.data['video_comment_count'].values
        self.video_heart_count = self.data['video_heart_count'].values
        self.video_play_count = self.data['video_play_count'].values
        self.video_share_count = self.data['video_share_count'].values

        # Normalize continuous features if required
        if normalize:
            self.author_features, author_stats = self._normalize_features(self.author_features)
            self.video_comment_count, comment_stats = self._normalize_values(self.video_comment_count)
            self.video_heart_count, heart_stats = self._normalize_values(self.video_heart_count)
            self.video_play_count, play_stats = self._normalize_values(self.video_play_count)
            self.video_share_count, share_stats = self._normalize_values(self.video_share_count)

            # Save the mean and std values for later use
            if save_stats:
                normalization_stats = {
                    "author_features": author_stats,
                    "video_comment_count": comment_stats,
                    "video_heart_count": heart_stats,
                    "video_play_count": play_stats,
                    "video_share_count": share_stats
                }
                with open(stats_file, 'w') as f:
                    json.dump(normalization_stats, f)

    def _normalize_features(self, features):
        """
        Normalize a set of features using mean and standard deviation.
        Args:
        - features (np.ndarray): The features to normalize.
        Returns:
        - normalized_features (np.ndarray): The normalized features.
        - stats (dict): Mean and std values for the features.
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Avoid division by zero if std is zero
        normalized_features = (features - mean) / std
        return normalized_features, {"mean": mean.tolist(), "std": std.tolist()}

    def _normalize_values(self, values):
        """
        Normalize a set of values using mean and standard deviation.
        Args:
        - values (np.ndarray): The values to normalize.
        Returns:
        - normalized_values (np.ndarray): The normalized values.
        - stats (dict): Mean and std values for the values.
        """
        mean = np.mean(values)
        std = np.std(values)
        std = std if std != 0 else 1  # Avoid division by zero if std is zero
        normalized_values = (values - mean) / std
        return normalized_values, {"mean": mean, "std": std}

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the input features and target outputs for the given index.
        Args:
        - idx (int): Index of the sample.
        
        Returns:
        - author_features (Tensor): Tensor containing normalized author follower, following, and total video counts.
        - input_ids (Tensor): Tensor containing tokenized BERT input IDs.
        - attention_mask (Tensor): Tensor containing the attention mask for BERT.
        - outputs (Tensor): Tensor containing normalized video-related counts and author_id_encoded.
        """
        # Convert author-related features to tensors
        author_features = torch.tensor(self.author_features[idx], dtype=torch.float32)
        
        # Convert tokenized LLM responses to tensors (input_ids and attention_mask)
        llm_tokenized = self.llm_tokenized[idx]
        input_ids = torch.tensor(llm_tokenized, dtype=torch.long)

        # Generate the attention mask (1 for valid tokens, 0 for padding)
        attention_mask = (input_ids != 0).long()

        # Create the output tensor: normalized video counts and author_id_encoded
        outputs = torch.tensor([self.author_id_encoded[idx],
                                self.video_comment_count[idx],
                                self.video_heart_count[idx],
                                self.video_play_count[idx],
                                self.video_share_count[idx]], dtype=torch.float32)

        return author_features, input_ids, attention_mask, outputs
# Example usage:
#if __name__ == "__main__":
#    # Path to the Excel file
#    file_path = 'C:/Users/Owner/OneDrive/Desktop/Video_Analytics/merged_training_data_with_llm_and_author_categorized.xlsx'
#    
##    # Create the dataset
 #   dataset = VideoAnalyticsDataset(file_path)
    
#    # Example: Get the first sample
#    author_features, llm_tokenized, outputs = dataset[0]
    
#    print("Author Features:", author_features)
#    print("LLM Tokenized:", llm_tokenized)
#    print("Outputs (author_id_encoded, video_comment_count, video_heart_count, video_play_count, video_share_count):", outputs)