import pandas as pd
import numpy as np
#import tensorflow as tf
from collections import Counter
from torchvision import transforms

import torch
from torch import optim
from torch import nn
from torch import optim
import torch.nn.functional as F

#from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
import random

import matplotlib.pyplot as plt

import seaborn as sns
from transformers import BertModel
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler

torch.manual_seed(0)

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias = False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias = False)
        
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        self.attention = None
        self.after_attention_layer = None
        self.after_linear_layer = None
        
    def forward(self, values, keys, query):
        N = query.shape[0]  
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys  = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        e = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        # Attention(Q, K, V)
        attention = torch.softmax(e / (self.embed_size**(1/2)), dim = 3)
        self.attention = attention
        
        #print(attention.shape)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len,self.heads*self.head_dim)
        #print(out.shape)
        self.after_attention_layer = out
        out = self.fc_out(out)
        self.after_linear_layer = out
        #print(out.shape)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        
        self.attention = SelfAttention(embed_size, heads)    
        self.norm1 = nn.LayerNorm(embed_size) 
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(nn.Linear(embed_size, forward_expansion*embed_size),
                                          nn.ReLU(),
                                          nn.Linear(forward_expansion*embed_size, embed_size))
        self.dropout = nn.Dropout(dropout)
            
    def forward(self, values, keys, query):
        
        attention = self.attention(values, keys, query)
    
        x = self.dropout(self.norm1(attention + query))
        
        #sns.heatmap(x[0].detach().numpy(), cmap="YlGnBu", ax=ax5)
        
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        
        return out

class Encoder(nn.Module):
                                                                                                #max sentence length
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_lenght, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(vocab_size, embed_size, padding_idx = 0)
        self.position_embedding = nn.Embedding(max_lenght, embed_size, padding_idx = 0)
        
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc_out1 = nn.Linear(embed_size,128)
        
        self.relu = nn.PReLU()
        
        self.fc_out2 = nn.Linear(128,1)
        
        self.result = None
        #self.fc_out3 = nn.Linear(64,1)
        
    def forward(self, x):
        
        N , seq_lenght = x.shape
        
        positions = torch.arange(0, seq_lenght).expand(N, seq_lenght).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out)

        out = self.fc_out1(out.mean(dim=1))
        self.result = self.fc_out2(self.relu(out))
        out = self.fc_out2(self.relu(out))

        return out

class EncoderWithAuthorFeatures(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, batch_size):
        super(EncoderWithAuthorFeatures, self).__init__()
        self.batch_size = batch_size
        self.embed_size = embed_size
        self.device = device

        # Transformer-related layers
        self.word_embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embed_size, padding_idx=0)
        self.layers = nn.ModuleList([TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.fc_out1 = nn.Linear(embed_size, 128)
        self.relu = nn.PReLU()

        # Neural network for processing Author Features (author_follower_count, author_following_count, author_total_video_count)
        self.author_fc1 = nn.Linear(3, 64)  # Author features have 3 inputs
        self.author_fc2 = nn.Linear(64, 128)
        self.author_fc3 = nn.Linear(128, 128)  # Additional layer for author features
        self.author_fc4 = nn.Linear(128, 128)  # Deeper layer for author features processing
        self.author_relu = nn.Mish()

        # Transformer-only output layers (no shared prediction heads)

        # Independent Prediction Networks for each task
        # Author ID Prediction Head
        self.author_id_fc1 = nn.Linear(256, 128)
        self.author_id_fc2 = nn.Linear(128, 64)
        self.author_id_fc3 = nn.Linear(64, 15)  # Predicting 15 classes for author_id_encoded

        # Comment Count Prediction Head
        self.comment_fc1 = nn.Linear(256, 128)
        self.comment_fc2 = nn.Linear(128, 64)
        self.comment_fc3 = nn.Linear(64, 1)  # Predicting comment count

        # Heart Count Prediction Head
        self.heart_fc1 = nn.Linear(256, 128)
        self.heart_fc2 = nn.Linear(128, 64)
        self.heart_fc3 = nn.Linear(64, 1)  # Predicting heart count

        # Play Count Prediction Head
        self.play_fc1 = nn.Linear(256, 128)
        self.play_fc2 = nn.Linear(128, 64)
        self.play_fc3 = nn.Linear(64, 1)  # Predicting play count

        # Share Count Prediction Head
        self.share_fc1 = nn.Linear(256, 128)
        self.share_fc2 = nn.Linear(128, 64)
        self.share_fc3 = nn.Linear(64, 1)  # Predicting share count

    def forward(self, x, author_features):
        """
        x: Tokenized input (batch_size, seq_length)
        author_features: Author input features (batch_size, 3)
        """
        N, seq_length = x.shape

        # Positional encoding and word embedding
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        # Pass through transformer layers
        for layer in self.layers:
            out = layer(out, out, out)

        # Get the latent representation from the Transformer
        out = self.fc_out1(out.mean(dim=1))
        transformer_latent = self.relu(out)

        # Process author features through a deeper neural network
        author_out = self.author_relu(self.author_fc1(author_features))
        author_out = self.author_relu(self.author_fc2(author_out))
        author_out = self.author_relu(self.author_fc3(author_out))
        author_out = self.author_relu(self.author_fc4(author_out))  # Additional layer for author features

        # Concatenate Transformer latent space with processed author features
        combined_out = torch.cat((transformer_latent, author_out), dim=1)  # Shape: (batch_size, 256)

        # Author ID Prediction Head
        author_id_out = self.relu(self.author_id_fc1(combined_out))
        author_id_out = self.relu(self.author_id_fc2(author_id_out))
        author_id_pred = self.author_id_fc3(author_id_out)  # Logits for author_id_encoded

        # Comment Count Prediction Head
        comment_out = self.relu(self.comment_fc1(combined_out))
        comment_out = self.relu(self.comment_fc2(comment_out))
        comment_count_pred = self.comment_fc3(comment_out)  # Regression for comment count

        # Heart Count Prediction Head
        heart_out = self.relu(self.heart_fc1(combined_out))
        heart_out = self.relu(self.heart_fc2(heart_out))
        heart_count_pred = self.heart_fc3(heart_out)  # Regression for heart count

        # Play Count Prediction Head
        play_out = self.relu(self.play_fc1(combined_out))
        play_out = self.relu(self.play_fc2(play_out))
        play_count_pred = self.play_fc3(play_out)  # Regression for play count

        # Share Count Prediction Head
        share_out = self.relu(self.share_fc1(combined_out))
        share_out = self.relu(self.share_fc2(share_out))
        share_count_pred = self.share_fc3(share_out)  # Regression for share count

        # Return a dictionary of the predictions
        return {
            'author_id_pred': author_id_pred,
            'comment_count_pred': comment_count_pred,
            'heart_count_pred': heart_count_pred,
            'play_count_pred': play_count_pred,
            'share_count_pred': share_count_pred
        }

# Define TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: (seq_length, batch_size, embed_size)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(attn_output + x)
        forward_output = self.feed_forward(x)
        x = self.norm2(forward_output + x)
        return x

# Define TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, max_length, device):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_length, embed_size)
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(self.device)

        x = x + self.position_embedding(positions)
        x = self.dropout(x)

        # Transpose for MultiheadAttention: (seq_length, batch_size, embed_size)
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, mask)

        # Transpose back to (batch_size, seq_length, embed_size)
        x = x.transpose(0, 1)

        # Aggregate over the sequence dimension
        x = x.mean(dim=1)  # Shape: (batch_size, embed_size)
        return x

# Define TransformerEncoderWrapper
class TransformerEncoderWrapper(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length, device):
        super(TransformerEncoderWrapper, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.transformer_encoder = TransformerEncoder(
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_length=max_length,
            device=device,
        )

    def forward(self, x):
        embedded = self.embedding(x)  # x: (batch_size, seq_length)
        out = self.transformer_encoder(embedded)  # out: (batch_size, embed_size)
        return out

# Define EmbeddingAveragePooling
class EmbeddingAveragePooling(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout):
        super(EmbeddingAveragePooling, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_size)
        x = self.dropout(embedded)
        x = x.mean(dim=1)  # (batch_size, embed_size)
        return x
# Define WideAndDeepModel
class WideAndDeepModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        numerical_feature_size,
        categorical_feature_sizes,
        nlp_feature_config,
        deep_layer_sizes,
        output_sizes,
        device,
        max_length=512,
        dropout=0.1,
    ):
        super(WideAndDeepModel, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)

        # Wide Part: Linear model
        self.wide_input_size = numerical_feature_size + sum(categorical_feature_sizes)
        self.wide_linear = nn.Linear(self.wide_input_size, sum(output_sizes.values()))

        # Embeddings for Categorical Features
        self.categorical_embeddings = nn.ModuleList()
        self.categorical_embed_sizes = []
        for size in categorical_feature_sizes:
            embed_dim = min(50, (size + 1) // 2)  # Rule of thumb for embedding size
            self.categorical_embeddings.append(nn.Embedding(size, embed_dim))
            self.categorical_embed_sizes.append(embed_dim)

        # Numerical Features Network (Deep Part)
        if numerical_feature_size > 0:
            self.numerical_deep = nn.Sequential(
                nn.Linear(numerical_feature_size, deep_layer_sizes[0]),
                nn.ReLU(),
                nn.BatchNorm1d(deep_layer_sizes[0]),
                nn.Dropout(dropout),
            )
            numerical_output_size = deep_layer_sizes[0]
        else:
            self.numerical_deep = None
            numerical_output_size = 0

        # NLP Features Network (Deep Part)
        self.nlp_models = nn.ModuleList()
        self.nlp_output_sizes = []
        for nlp_config in nlp_feature_config:
            nlp_type = nlp_config['type']
            if nlp_type == 'transformer':
                nlp_model = TransformerEncoderWrapper(
                    vocab_size=vocab_size,
                    embed_size=embed_size,
                    num_layers=nlp_config.get('num_layers', 2),
                    heads=nlp_config.get('heads', 8),
                    forward_expansion=nlp_config.get('forward_expansion', 4),
                    dropout=dropout,
                    max_length=max_length,
                    device=device,
                )
                nlp_output_size = embed_size
            else:
                # Simple embedding + pooling
                nlp_model = EmbeddingAveragePooling(
                    vocab_size=vocab_size,
                    embed_size=embed_size,
                    dropout=dropout,
                )
                nlp_output_size = embed_size
            self.nlp_models.append(nlp_model)
            self.nlp_output_sizes.append(nlp_output_size)

        # Deep Part: Final layers
        total_deep_input_size = sum(self.categorical_embed_sizes) + numerical_output_size + sum(self.nlp_output_sizes)
        deep_layers = []
        input_size = total_deep_input_size
        for layer_size in deep_layer_sizes:
            deep_layers.append(nn.Linear(input_size, layer_size))
            deep_layers.append(nn.ReLU())
            deep_layers.append(nn.BatchNorm1d(layer_size))
            deep_layers.append(nn.Dropout(dropout))
            input_size = layer_size
        self.deep_network = nn.Sequential(*deep_layers)
        self.deep_output_size = input_size  # Last layer size

        # Output Heads
        self.output_heads = nn.ModuleDict()
        for target_name, output_size in output_sizes.items():
            self.output_heads[target_name] = nn.Linear(self.deep_output_size + sum(output_sizes.values()), output_size)

    def forward(self, numerical_features, categorical_features, nlp_tokens_list):
        batch_size = numerical_features.size(0)

        # Wide Part
        wide_features = []

        if numerical_features.size(1) > 0:
            wide_numerical = numerical_features.to(self.device)
            wide_features.append(wide_numerical)

        if categorical_features.size(1) > 0:
            categorical_features = categorical_features.to(self.device)
            # Convert categorical features to one-hot encoding for wide part
            categorical_one_hot = []
            for i in range(categorical_features.size(1)):
                size = self.categorical_embeddings[i].num_embeddings
                cat_one_hot = F.one_hot(categorical_features[:, i], num_classes=size).float()
                categorical_one_hot.append(cat_one_hot)
            wide_categorical = torch.cat(categorical_one_hot, dim=1)
            wide_features.append(wide_categorical)

        if wide_features:
            wide_input = torch.cat(wide_features, dim=1)
            wide_output = self.wide_linear(wide_input)
        else:
            wide_output = torch.zeros(batch_size, sum(self.output_heads[target_name].out_features for target_name in self.output_heads), device=self.device)

        # Deep Part
        deep_features = []

        # Numerical Features
        if self.numerical_deep is not None:
            numerical_features = numerical_features.to(self.device)
            num_out = self.numerical_deep(numerical_features)
            deep_features.append(num_out)

        # Categorical Embeddings
        if categorical_features.size(1) > 0:
            cat_embeds = []
            for i, embed in enumerate(self.categorical_embeddings):
                cat_feature = categorical_features[:, i]
                cat_embed = embed(cat_feature)
                cat_embeds.append(cat_embed)
            cat_embeds = torch.cat(cat_embeds, dim=1)
            deep_features.append(cat_embeds)

        # NLP Features
        for i, nlp_model in enumerate(self.nlp_models):
            tokens = nlp_tokens_list[i].to(self.device)  # Shape: (batch_size, seq_length)
            nlp_out = nlp_model(tokens)  # Should return (batch_size, nlp_output_size)
            deep_features.append(nlp_out)

        # Concatenate all deep features
        deep_input = torch.cat(deep_features, dim=1)
        deep_output = self.deep_network(deep_input)

        # Combine Wide and Deep Parts
        combined_input = torch.cat([deep_output, wide_output], dim=1)

        # Output Heads
        outputs = {}
        offset = 0
        for target_name, head in self.output_heads.items():
            output_size = head.out_features
            output = head(combined_input)
            outputs[target_name] = output
            offset += output_size

        return outputs

# Define TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: (seq_length, batch_size, embed_size)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(attn_output + x)
        forward_output = self.feed_forward(x)
        x = self.norm2(forward_output + x)
        return x

# Define TransformerEncoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, max_length, device):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device

        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_length, embed_size)
        N, seq_length, embed_dim = x.shape
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(self.device)

        out = x + self.position_embedding(positions)
        out = self.dropout(out)

        # Transpose for MultiheadAttention: (seq_length, batch_size, embed_size)
        out = out.transpose(0, 1)

        for layer in self.layers:
            out = layer(out, mask)

        # Transpose back to (batch_size, seq_length, embed_size)
        out = out.transpose(0, 1)

        # Aggregate over the sequence dimension
        out = out.mean(dim=1)  # Shape: (batch_size, embed_size)
        return out

# Define TransformerEncoderWrapper
class TransformerEncoderWrapper(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, max_length, device):
        super(TransformerEncoderWrapper, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.transformer_encoder = TransformerEncoder(
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_length=max_length,
            device=device,
        )

    def forward(self, x):
        embedded = self.embedding(x)  # x: (batch_size, seq_length)
        out = self.transformer_encoder(embedded)  # out: (batch_size, embed_size)
        return out

# Define LSTMEncoder
class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, device):
        super(LSTMEncoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_size)
        output, (hidden, cell) = self.lstm(embedded)
        return hidden[-1]  # Return the last layer's hidden state, shape: (batch_size, hidden_size)

# Define EmbeddingAveragePooling
class EmbeddingAveragePooling(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout):
        super(EmbeddingAveragePooling, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_size)
        x = self.dropout(embedded)
        x = x.mean(dim=1)  # (batch_size, embed_size)
        return x
class ResidualBlock(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
            nn.ReLU(),
            nn.BatchNorm1d(input_size),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return x + self.network(x)
# Define MultiInputModel
class MultiInputModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        numerical_feature_size,
        categorical_feature_sizes,
        nlp_feature_configs,
        shared_layer_sizes,
        output_sizes,
        device,
        max_length=512,
        dropout=0.1,
    ):
        super(MultiInputModel, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)

        # Numerical Features Processor
        if numerical_feature_size > 0:
            self.numerical_processor = nn.Sequential(
                nn.Linear(numerical_feature_size, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.LeakyReLU(negative_slope=0.01),
                nn.Dropout(p=0.2),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.PReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )
            self.numerical_output_size = 64
        else:
            self.numerical_processor = None
            self.numerical_output_size = 0

        # Categorical Features Processor
        if categorical_feature_sizes:
            self.categorical_embeddings = nn.ModuleList()
            self.categorical_embed_sizes = []
            for size in categorical_feature_sizes:
                embed_dim = min(50, (size + 1) // 2)  # Rule of thumb
                self.categorical_embeddings.append(nn.Embedding(size, embed_dim))
                self.categorical_embed_sizes.append(embed_dim)
            total_categorical_embed_size = sum(self.categorical_embed_sizes)
            self.categorical_processor = nn.Sequential(
                nn.Linear(total_categorical_embed_size, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Dropout(dropout),
            )
            self.categorical_output_size = 64
        else:
            self.categorical_embeddings = None
            self.categorical_processor = None
            self.categorical_output_size = 0
        self.relu = nn.PReLU()
        # NLP Features Processor
        self.nlp_processors = nn.ModuleList()
        self.nlp_output_sizes = []
        for nlp_config in nlp_feature_configs:
            nlp_type = nlp_config['type']
            if nlp_type == 'transformer':
                nlp_processor = TransformerEncoderWrapper(
                    vocab_size=vocab_size,
                    embed_size=embed_size,
                    num_layers=nlp_config.get('num_layers', 1),
                    heads=nlp_config.get('heads', 16),
                    forward_expansion=nlp_config.get('forward_expansion', 4),
                    dropout=dropout,
                    max_length=max_length,
                    device=device,
                )
                nlp_output_size = embed_size  # Output of transformer encoder
            elif nlp_type == 'lstm':
                nlp_processor = LSTMEncoder(
                    vocab_size=vocab_size,
                    embed_size=embed_size,
                    hidden_size=nlp_config.get('hidden_size', 128),
                    num_layers=nlp_config.get('num_layers', 1),
                    dropout=dropout,
                    device=device,
                )
                nlp_output_size = nlp_config.get('hidden_size', 128)
            else:
                # Default to simple embedding + pooling
                nlp_processor = EmbeddingAveragePooling(
                    vocab_size=vocab_size,
                    embed_size=embed_size,
                    dropout=dropout,
                )
                nlp_output_size = embed_size
            self.nlp_processors.append(nlp_processor)
            self.nlp_output_sizes.append(nlp_output_size)

        # Shared Layers
        total_input_size = self.numerical_output_size + self.categorical_output_size + sum(self.nlp_output_sizes)
        shared_layers = []
        input_size = total_input_size
        for layer_size in shared_layer_sizes:
            shared_layers.append(nn.Linear(input_size, layer_size))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.BatchNorm1d(layer_size))
            shared_layers.append(nn.Dropout(dropout))
            input_size = layer_size
        self.shared_network = nn.Sequential(*shared_layers)

        # Output Heads
        self.output_heads = nn.ModuleDict()
        for target_name, output_size in output_sizes.items():
            self.output_heads[target_name] = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU()  # Ensure positive output if using log
            )

        # Separate prediction networks for each task
        self.author_id_fc1 = nn.Linear(192, 128)
        self.author_id_fc2 = nn.Linear(128, 15)  # Predicting 15 classes for author_id_encoded


    def forward(self, numerical_features, categorical_features, nlp_tokens_list):
        features = []

        # Process Numerical Features
        if self.numerical_processor is not None:
            numerical_features = numerical_features.to(self.device)
            num_out = self.numerical_processor(numerical_features)
            features.append(num_out)

        # Process Categorical Features
        if self.categorical_embeddings is not None:
            categorical_features = categorical_features.to(self.device)
            cat_embeds = []
            for i, embed in enumerate(self.categorical_embeddings):
                cat_feature = categorical_features[:, i]
                cat_embed = embed(cat_feature)
                cat_embeds.append(cat_embed)
            cat_embeds = torch.cat(cat_embeds, dim=1)
            cat_out = self.categorical_processor(cat_embeds)
            features.append(cat_out)

        # Process NLP Features
        for i, nlp_processor in enumerate(self.nlp_processors):
            tokens = nlp_tokens_list[i].to(self.device)  # Shape: (batch_size, seq_length)
            nlp_out = nlp_processor(tokens)  # Should return (batch_size, nlp_output_size)
            features.append(nlp_out)

        # Combine All Features
        combined_features = torch.cat(features, dim=1)

        # Pass Through Shared Layers
        shared_output = self.shared_network(combined_features)

        # Output Heads
        # Output Heads
        outputs = {}
        for target_name, head in self.output_heads.items():
            outputs[target_name] = head(shared_output)
            # Apply log transformation
            outputs[target_name] = torch.log(outputs[target_name] + 1e-8)
    
        # Author ID Prediction Head
        author_id_out = self.relu(self.author_id_fc1(combined_features))
        author_id_pred = self.author_id_fc2(author_id_out)  # Logits for author_id_encoded
        outputs['author_id_pred'] = author_id_pred
        return outputs
class NumericalProcessor(nn.Module):
    def __init__(self, input_size, dropout=0.1):
        super(NumericalProcessor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(negative_slope=0.01),
            #nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.Tanh(),
            #nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.PReLU(),
            #nn.BatchNorm1d(64),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.network(x)

class CategoricalProcessor(nn.Module):
    def __init__(self, categorical_feature_sizes, dropout=0.1):
        super(CategoricalProcessor, self).__init__()
        self.embeddings = nn.ModuleList()
        self.embed_sizes = []
        for size in categorical_feature_sizes:
            embed_dim = min(50, (size + 1) // 2)  # Rule of thumb
            self.embeddings.append(nn.Embedding(size, embed_dim))
            self.embed_sizes.append(embed_dim)
        
        total_embed_size = sum(self.embed_sizes)
        self.network = nn.Sequential(
            nn.Linear(total_embed_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        embed_list = []
        for i, embed in enumerate(self.embeddings):
            embed_feature = embed(x[:, i])
            embed_list.append(embed_feature)
        concatenated = torch.cat(embed_list, dim=1)
        return self.network(concatenated)
class MultiInputModelRes(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_size,
        numerical_feature_size,
        categorical_feature_sizes,
        nlp_feature_configs,
        shared_layer_sizes,
        output_sizes,
        device,
        max_length=512,
        dropout=0.1,
    ):
        super(MultiInputModelRes, self).__init__()
        self.device = device
        self.dropout = nn.Dropout(dropout)

        # Initialize feature processors
        self.numerical_processor = NumericalProcessor(numerical_feature_size, dropout) if numerical_feature_size > 0 else None
        self.categorical_processor = CategoricalProcessor(categorical_feature_sizes, dropout) if categorical_feature_sizes else None
        
        self.nlp_processors = nn.ModuleList()
        self.nlp_output_sizes = []
        for nlp_config in nlp_feature_configs:
            nlp_type = nlp_config['type']
            if nlp_type == 'transformer':
                nlp_processor = TransformerEncoderWrapper(
                    vocab_size=vocab_size,
                    embed_size=embed_size,
                    num_layers=nlp_config.get('num_layers', 1),
                    heads=nlp_config.get('heads', 8),
                    forward_expansion=nlp_config.get('forward_expansion', 4),
                    dropout=dropout,
                    max_length=max_length,
                    device=device,
                )
                nlp_output_size = embed_size  # Output of transformer encoder
            elif nlp_type == 'lstm':
                nlp_processor = LSTMEncoder(
                    vocab_size=vocab_size,
                    embed_size=embed_size,
                    hidden_size=nlp_config.get('hidden_size', 128),
                    num_layers=nlp_config.get('num_layers', 1),
                    dropout=dropout,
                    device=device,
                )
                nlp_output_size = nlp_config.get('hidden_size', 128) * 2  # Bidirectional
            else:
                # Default to simple embedding + pooling
                nlp_processor = EmbeddingAveragePooling(
                    vocab_size=vocab_size,
                    embed_size=embed_size,
                    dropout=dropout,
                )
                nlp_output_size = embed_size
            self.nlp_processors.append(nlp_processor)
            self.nlp_output_sizes.append(nlp_output_size)

        # Calculate total input size
        total_input_size = 0
        if self.numerical_processor:
            total_input_size += 64
        if self.categorical_processor:
            total_input_size += 64
        total_input_size += sum(self.nlp_output_sizes)

        # Shared Layers with Residual Blocks
        shared_layers = []
        input_size = total_input_size
        for layer_size in shared_layer_sizes:
            shared_layers.append(nn.Linear(input_size, layer_size))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.BatchNorm1d(layer_size))
            shared_layers.append(nn.Dropout(dropout))
            # Add a residual block
            shared_layers.append(ResidualBlock(layer_size, layer_size // 2, dropout))
            input_size = layer_size
        self.shared_network = nn.Sequential(*shared_layers)

        # Output Heads
        self.output_heads = nn.ModuleDict()
        for target_name, output_size in output_sizes.items():
            self.output_heads[target_name] = nn.Sequential(
                nn.Linear(input_size, output_size),
                nn.ReLU()  # Ensure positive output if using log
            )
    
    def forward(self, numerical_features, categorical_features, nlp_tokens_list):
        features = []

        # Process Numerical Features
        if self.numerical_processor is not None:
            numerical_features = numerical_features.to(self.device)
            num_out = self.numerical_processor(numerical_features)
            features.append(num_out)

        # Process Categorical Features
        if self.categorical_processor is not None:
            categorical_features = categorical_features.to(self.device)
            cat_out = self.categorical_processor(categorical_features)
            features.append(cat_out)

        # Process NLP Features
        for i, nlp_processor in enumerate(self.nlp_processors):
            tokens = nlp_tokens_list[i].to(self.device)  # Shape: (batch_size, seq_length)
            nlp_out = nlp_processor(tokens)  # Should return (batch_size, nlp_output_size)
            features.append(nlp_out)

        # Combine All Features
        combined_features = torch.cat(features, dim=1)

        # Pass Through Shared Layers
        shared_output = self.shared_network(combined_features)

        # Output Heads
        outputs = {}
        for target_name, head in self.output_heads.items():
            outputs[target_name] = head(shared_output)
            # Apply log transformation
            outputs[target_name] = torch.log(outputs[target_name] + 1e-8)

        return outputs


class NumericalOnlyModel(nn.Module):
    def __init__(self, numerical_feature_size, output_sizes, hidden_sizes=[128, 64], dropout=0.1):
        super(NumericalOnlyModel, self).__init__()
        layers = []
        input_size = numerical_feature_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        self.network = nn.Sequential(*layers)
        
        # Output Heads
        self.output_heads = nn.ModuleDict()
        for target_name, output_size in output_sizes.items():
            self.output_heads[target_name] = nn.Linear(input_size, output_size)
    
    def forward(self, numerical_features):
        features = self.network(numerical_features)
        outputs = {}
        for target_name, head in self.output_heads.items():
            outputs[target_name] = head(features)
        return outputs

class CategoricalOnlyModel(nn.Module):
    def __init__(self, categorical_feature_sizes, output_sizes, embed_dim=16, hidden_sizes=[128, 64], dropout=0.1):
        super(CategoricalOnlyModel, self).__init__()
        self.embeddings = nn.ModuleList()
        for size in categorical_feature_sizes:
            self.embeddings.append(nn.Embedding(size, embed_dim))
        
        total_embed_size = embed_dim * len(categorical_feature_sizes)
        layers = []
        input_size = total_embed_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_size = hidden_size
        self.network = nn.Sequential(*layers)
        
        # Output Heads
        self.output_heads = nn.ModuleDict()
        for target_name, output_size in output_sizes.items():
            self.output_heads[target_name] = nn.Linear(input_size, output_size)
    
    def forward(self, categorical_features):
        embeds = []
        for i, embedding in enumerate(self.embeddings):
            embed = embedding(categorical_features[:, i])
            embeds.append(embed)
        features = torch.cat(embeds, dim=1)
        features = self.network(features)
        outputs = {}
        for target_name, head in self.output_heads.items():
            outputs[target_name] = head(features)
        return outputs

class NLPOnlyModel(nn.Module):
    def __init__(self, vocab_size, embed_size, output_sizes, nlp_encoder_type='lstm',
                 hidden_size=128, num_layers=2, dropout=0.1, num_nlp_inputs=1, device='cpu'):
        super(NLPOnlyModel, self).__init__()
        self.device = device
        self.num_nlp_inputs = num_nlp_inputs

        # Embeddings and Encoders for each NLP input
        self.embeddings = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.encoder_output_sizes = []

        for _ in range(num_nlp_inputs):
            embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.embeddings.append(embedding)

            if nlp_encoder_type == 'lstm':
                encoder = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                                  batch_first=True, dropout=dropout if num_layers > 1 else 0)
                self.encoder_output_sizes.append(hidden_size)
            elif nlp_encoder_type == 'transformer':
                encoder = TransformerEncoder(
                    vocab_size=vocab_size,
                    embed_size=embed_size,
                    num_layers=num_layers,
                    heads=8,
                    forward_expansion=4,
                    dropout=dropout,
                    max_length=512,
                    device=device,
                )
                self.encoder_output_sizes.append(embed_size)
            else:
                raise ValueError("Invalid nlp_encoder_type. Choose 'lstm' or 'transformer'.")
            self.encoders.append(encoder)

        total_output_size = sum(self.encoder_output_sizes)

        # Output Heads
        self.output_heads = nn.ModuleDict()
        for target_name, output_size in output_sizes.items():
            self.output_heads[target_name] = nn.Linear(total_output_size, output_size)

    def forward(self, nlp_tokens_list):
        features_list = []
        for i in range(self.num_nlp_inputs):
            tokens = nlp_tokens_list[i].to(self.device)
            embedded = self.embeddings[i](tokens)
            if isinstance(self.encoders[i], nn.LSTM):
                output, (hidden, cell) = self.encoders[i](embedded)
                features = hidden[-1]
            else:
                features = self.encoders[i](embedded)
            features_list.append(features)
        combined_features = torch.cat(features_list, dim=1)
        outputs = {}
        for target_name, head in self.output_heads.items():
            outputs[target_name] = head(combined_features)
        return outputs

class NumericalCategoricalModel(nn.Module):
    def __init__(self, numerical_feature_size, categorical_feature_sizes, output_sizes, hidden_sizes=[128, 64], dropout=0.1):
        super(NumericalCategoricalModel, self).__init__()
        # Numerical Processor
        self.numerical_processor = nn.Sequential(
            nn.Linear(numerical_feature_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Categorical Embeddings
        self.embeddings = nn.ModuleList()
        embed_dim = 16  # Adjust as needed
        for size in categorical_feature_sizes:
            self.embeddings.append(nn.Embedding(size, embed_dim))
        total_embed_size = embed_dim * len(categorical_feature_sizes)
        
        # Categorical Processor
        self.categorical_processor = nn.Sequential(
            nn.Linear(total_embed_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined Processor
        combined_input_size = hidden_sizes[0] * 2
        self.combined_processor = nn.Sequential(
            nn.Linear(combined_input_size, hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output Heads
        self.output_heads = nn.ModuleDict()
        for target_name, output_size in output_sizes.items():
            self.output_heads[target_name] = nn.Linear(hidden_sizes[1], output_size)
    
    def forward(self, numerical_features, categorical_features):
        num_out = self.numerical_processor(numerical_features)
        
        embeds = []
        for i, embedding in enumerate(self.embeddings):
            embed = embedding(categorical_features[:, i])
            embeds.append(embed)
        cat_features = torch.cat(embeds, dim=1)
        cat_out = self.categorical_processor(cat_features)
        
        combined = torch.cat([num_out, cat_out], dim=1)
        features = self.combined_processor(combined)
        
        outputs = {}
        for target_name, head in self.output_heads.items():
            outputs[target_name] = head(features)
        return outputs

# TransformerBlock Definition
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        # Self-attention
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, dropout=dropout)
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        # Feed Forward Network
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x: (batch_size, seq_length, embed_size)
        # Transpose x to (seq_length, batch_size, embed_size)
        x = x.transpose(0, 1)
        
        # Self-attention expects (seq_length, batch_size, embed_size)
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        
        # Residual connection and LayerNorm
        x = x + attn_output
        x = self.norm1(x)
        
        # Feed Forward Network
        forward_output = self.feed_forward(x)
        
        # Second residual connection and LayerNorm
        x = x + forward_output
        x = self.norm2(x)
        
        # Transpose back to (batch_size, seq_length, embed_size)
        out = x.transpose(0, 1)
        return out

# TransformerEncoder Definition
class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, forward_expansion, dropout, max_length, device):
        super(TransformerEncoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        
        self.position_embedding = nn.Embedding(max_length, embed_size)
        
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        N, seq_length, embed_dim = x.shape
        positions = torch.arange(0, seq_length).unsqueeze(0).expand(N, seq_length).to(self.device)
        
        out = x + self.position_embedding(positions)
        out = self.dropout(out)
        
        for layer in self.layers:
            out = layer(out, mask)
        
        # Aggregate over the sequence dimension
        out = out.mean(dim=1)  # Shape: (batch_size, embed_size)
        return out

# NumericalNLPModel Definition
class NumericalNLPModel(nn.Module): 
    def __init__(self, numerical_feature_size, vocab_size, embed_size, output_sizes,
                 nlp_encoder_type='lstm', hidden_size=128, num_layers=2,
                 hidden_sizes=[128, 64], dropout=0.1, num_nlp_inputs=1, device='cpu'):
        super(NumericalNLPModel, self).__init__()
        self.device = device
        self.num_nlp_inputs = num_nlp_inputs

        # Numerical Processor
        self.numerical_processor = nn.Sequential(
            nn.Linear(numerical_feature_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Embeddings and Encoders for each NLP input
        self.embeddings = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.encoder_output_sizes = []

        for _ in range(num_nlp_inputs):
            embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.embeddings.append(embedding)

            if nlp_encoder_type == 'lstm':
                encoder = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                                  batch_first=True, dropout=dropout)
                self.encoder_output_sizes.append(hidden_size)
            elif nlp_encoder_type == 'transformer':
                encoder = TransformerEncoder(
                    embed_size=embed_size,
                    num_layers=num_layers,
                    heads=8,
                    forward_expansion=4,
                    dropout=dropout,
                    max_length=512,
                    device=device,
                )
                self.encoder_output_sizes.append(embed_size)
            else:
                raise ValueError("Invalid nlp_encoder_type. Choose 'lstm' or 'transformer'.")
            self.encoders.append(encoder)

        total_nlp_output_size = sum(self.encoder_output_sizes)

        # Combined Processor
        combined_input_size = hidden_sizes[0] + total_nlp_output_size
        self.combined_processor = nn.Sequential(
            nn.Linear(combined_input_size, hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output Heads
        self.output_heads = nn.ModuleDict()
        for target_name, output_size in output_sizes.items():
            self.output_heads[target_name] = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, numerical_features, nlp_tokens_list):
        numerical_features = numerical_features.to(self.device)
        num_out = self.numerical_processor(numerical_features)

        nlp_features_list = []
        for i in range(self.num_nlp_inputs):
            tokens = nlp_tokens_list[i].to(self.device)  # tokens shape: (batch_size, seq_length)
            embedded = self.embeddings[i](tokens)  # embedded shape: (batch_size, seq_length, embed_size)
            if isinstance(self.encoders[i], nn.LSTM):
                output, (hidden, cell) = self.encoders[i](embedded)  # LSTM output
                nlp_out = hidden[-1]  # Get the last layer's hidden state
            else:
                nlp_out = self.encoders[i](embedded)  # Transformer output
            nlp_features_list.append(nlp_out)
        combined_nlp_features = torch.cat(nlp_features_list, dim=1)

        combined = torch.cat([num_out, combined_nlp_features], dim=1)
        features = self.combined_processor(combined)

        outputs = {}
        for target_name, head in self.output_heads.items():
            outputs[target_name] = head(features)
        return outputs

class CategoricalNLPModel(nn.Module):
    def __init__(self, categorical_feature_sizes, vocab_size, embed_size, output_sizes,
                 nlp_encoder_type='lstm', hidden_size=128, num_layers=2,
                 hidden_sizes=[128, 64], dropout=0.1, num_nlp_inputs=1, device='cpu'):
        super(CategoricalNLPModel, self).__init__()
        self.device = device
        self.num_nlp_inputs = num_nlp_inputs

        # Categorical Embeddings
        self.categorical_embeddings = nn.ModuleList()
        embed_dim = 16  # Adjust as needed
        for size in categorical_feature_sizes:
            self.categorical_embeddings.append(nn.Embedding(size, embed_dim))
        total_embed_size = embed_dim * len(categorical_feature_sizes)

        # Categorical Processor
        self.categorical_processor = nn.Sequential(
            nn.Linear(total_embed_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Embeddings and Encoders for each NLP input
        self.embeddings = nn.ModuleList()
        self.encoders = nn.ModuleList()
        self.encoder_output_sizes = []

        for _ in range(num_nlp_inputs):
            embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            self.embeddings.append(embedding)

            if nlp_encoder_type == 'lstm':
                encoder = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                                  batch_first=True, dropout=dropout)
                self.encoder_output_sizes.append(hidden_size)
            elif nlp_encoder_type == 'transformer':
                encoder = TransformerEncoder(
                    embed_size=embed_size,
                    num_layers=num_layers,
                    heads=8,
                    forward_expansion=4,
                    dropout=dropout,
                    max_length=512,
                    device=device,
                )
                self.encoder_output_sizes.append(embed_size)
            else:
                raise ValueError("Invalid nlp_encoder_type. Choose 'lstm' or 'transformer'.")
            self.encoders.append(encoder)

        total_nlp_output_size = sum(self.encoder_output_sizes)

        # Combined Processor
        combined_input_size = hidden_sizes[0] + total_nlp_output_size
        self.combined_processor = nn.Sequential(
            nn.Linear(combined_input_size, hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Output Heads
        self.output_heads = nn.ModuleDict()
        for target_name, output_size in output_sizes.items():
            self.output_heads[target_name] = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, categorical_features, nlp_tokens_list):
        # Process Categorical Features
        categorical_features = categorical_features.to(self.device)
        embeds = []
        for i, embedding in enumerate(self.categorical_embeddings):
            embed = embedding(categorical_features[:, i])
            embeds.append(embed)
        cat_features = torch.cat(embeds, dim=1)
        cat_out = self.categorical_processor(cat_features)

        # Process NLP Features
        nlp_features_list = []
        for i in range(self.num_nlp_inputs):
            tokens = nlp_tokens_list[i].to(self.device)
            embedded = self.embeddings[i](tokens)
            if isinstance(self.encoders[i], nn.LSTM):
                output, (hidden, cell) = self.encoders[i](embedded)
                nlp_out = hidden[-1]
            else:
                nlp_out = self.encoders[i](embedded)
            nlp_features_list.append(nlp_out)
        combined_nlp_features = torch.cat(nlp_features_list, dim=1)

        # Combine and Process
        combined = torch.cat([cat_out, combined_nlp_features], dim=1)
        features = self.combined_processor(combined)

        outputs = {}
        for target_name, head in self.output_heads.items():
            outputs[target_name] = head(features)
        return outputs



class EncoderWithBERTAndAuthorFeatures(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', num_layers=1, heads=8, device='cuda', dropout=0.1, freeze_bert=True):
        super(EncoderWithBERTAndAuthorFeatures, self).__init__()
        self.device = device

        # Pretrained BERT model
        self.bert = BertModel.from_pretrained(bert_model_name)
        bert_hidden_size = self.bert.config.hidden_size  # BERT hidden size (typically 768 for base BERT)

        # Option to freeze BERT parameters
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Fully connected layer to reduce BERT's output size to match with author features (128)
        self.fc_out1 = nn.Linear(bert_hidden_size, 128)
        self.relu = nn.PReLU()

        # Neural network for processing Author Features (author_follower_count, author_following_count, author_total_video_count)
        self.author_fc1 = nn.Linear(3, 128)  # Author features are projected to 128 dimensions
        self.author_fc2 = nn.Linear(128, 128)
        self.author_fc3 = nn.Linear(128, 128)
        self.author_fc4 = nn.Linear(128, 128)
        self.author_relu = nn.Mish()

        # Self-Attention Layer to learn interaction between BERT latent space and author features
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=heads, dropout=dropout)

        # Independent Prediction Networks for each task
        # Author ID Prediction Head
        self.author_id_fc1 = nn.Linear(256, 128)
        self.author_id_fc2 = nn.Linear(128, 64)
        self.author_id_fc3 = nn.Linear(64, 15)  # Predicting 15 classes for author_id_encoded

        # Comment Count Prediction Head
        self.comment_fc1 = nn.Linear(256, 128)
        self.comment_fc2 = nn.Linear(128, 64)
        self.comment_fc3 = nn.Linear(64, 1)  # Predicting comment count

        # Heart Count Prediction Head
        self.heart_fc1 = nn.Linear(256, 128)
        self.heart_fc2 = nn.Linear(128, 64)
        self.heart_fc3 = nn.Linear(64, 1)  # Predicting heart count

        # Play Count Prediction Head
        self.play_fc1 = nn.Linear(256, 128)
        self.play_fc2 = nn.Linear(128, 64)
        self.play_fc3 = nn.Linear(64, 1)  # Predicting play count

        # Share Count Prediction Head
        self.share_fc1 = nn.Linear(256, 128)
        self.share_fc2 = nn.Linear(128, 64)
        self.share_fc3 = nn.Linear(64, 1)  # Predicting share count

    def forward(self, input_ids, attention_mask, author_features):
        """
        input_ids: Tokenized input for BERT (batch_size, seq_length)
        attention_mask: Attention mask for BERT (batch_size, seq_length)
        author_features: Author input features (batch_size, 3)
        """
        # Pass the input through the BERT model
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_pooled_output = bert_output.pooler_output  # Shape: (batch_size, 768)
        
        # Reduce BERT output to match the 128-dimensional space
        transformer_latent = self.fc_out1(bert_pooled_output)  # Shape: (batch_size, 128)
        transformer_latent = self.relu(transformer_latent)

        # Process author features through a deeper neural network
        author_out = self.author_relu(self.author_fc1(author_features))  # Shape: (batch_size, 128)
        author_out = self.author_relu(self.author_fc2(author_out))
        author_out = self.author_relu(self.author_fc3(author_out))
        author_out = self.author_relu(self.author_fc4(author_out))  # Shape: (batch_size, 128)

        # Self-Attention between transformer latent and author latent
        #transformer_latent = transformer_latent.unsqueeze(1)  # (batch_size, 1, 128)
        #author_out = author_out.unsqueeze(1)  # (batch_size, 1, 128)

        # Perform attention: Query is the transformer latent, and key and value are the author features
        #attended_out, _ = self.attention(transformer_latent, author_out, author_out)
        #attended_out = attended_out.squeeze(1)  # Shape: (batch_size, 128)

        # Concatenation between transformer latent and author latent
        transformer_latent = transformer_latent.unsqueeze(1)  # Shape: (batch_size, 1, 128)
        author_out = author_out.unsqueeze(1)  # Shape: (batch_size, 1, 128)

        # Concatenate along the feature dimension
        attended_out = torch.cat((transformer_latent, author_out), dim=2)  # Shape: (batch_size, 1, 256)
        attended_out = attended_out.squeeze(1)  # Shape: (batch_size, 256)

        # Use attended output for prediction heads
        # Author ID Prediction Head
        author_id_out = self.relu(self.author_id_fc1(attended_out))
        author_id_out = self.relu(self.author_id_fc2(author_id_out))
        author_id_pred = self.author_id_fc3(author_id_out)  # Logits for author_id_encoded

        # Comment Count Prediction Head
        comment_out = self.relu(self.comment_fc1(attended_out))
        comment_out = self.relu(self.comment_fc2(comment_out))
        comment_count_pred = self.comment_fc3(comment_out)  # Regression for comment count

        # Heart Count Prediction Head
        heart_out = self.relu(self.heart_fc1(attended_out))
        heart_out = self.relu(self.heart_fc2(heart_out))
        heart_count_pred = self.heart_fc3(heart_out)  # Regression for heart count

        # Play Count Prediction Head
        play_out = self.relu(self.play_fc1(attended_out))
        play_out = self.relu(self.play_fc2(play_out))
        play_count_pred = self.play_fc3(play_out)  # Regression for play count

        # Share Count Prediction Head
        share_out = self.relu(self.share_fc1(attended_out))
        share_out = self.relu(self.share_fc2(share_out))
        share_count_pred = self.share_fc3(share_out)  # Regression for share count

        # Return a dictionary of the predictions
        return {
            'author_id_pred': author_id_pred,
            'comment_count_pred': comment_count_pred,
            'heart_count_pred': heart_count_pred,
            'play_count_pred': play_count_pred,
            'share_count_pred': share_count_pred
        }