# build_word_dictionary.py

import pandas as pd
import re
import json
import emoji
import argparse

def clean_text(text):
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

def tokenize_text(text, word_tokenizer, build_word_tokenizer=True):
    """
    Tokenizes the text, updates the word tokenizer, and returns the tokenized sequence.
    """
    if not text:
        return []
    
    tokens = re.findall(r'\b\w+\b', text.lower())
    
    # Update the word tokenizer only if we are building it
    if build_word_tokenizer:
        for token in tokens:
            if token not in word_tokenizer:
                word_tokenizer[token] = len(word_tokenizer)
    
    return tokens

def build_word_dictionary(file_path, nlp_cols, output_path='word_dictionary.json'):
    """
    Builds a word dictionary from the specified NLP columns in the CSV file.

    Args:
    - file_path (str): Path to the CSV file.
    - nlp_cols (list of str): List of NLP (text) column names to process.
    - output_path (str): Path to save the word dictionary JSON file.
    """
    # Initialize word tokenizer with special tokens
    word_tokenizer = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}

    # Load data
    data = pd.read_csv(file_path)
    
    # Check if all NLP columns exist in the data
    missing_cols = [col for col in nlp_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"The following NLP columns are missing in the data: {missing_cols}")
    
    # Iterate over the data and process NLP columns
    for idx in range(len(data)):
        for col in nlp_cols:
            text = data[col].iloc[idx]
            cleaned_text = clean_text(text)
            tokenize_text(cleaned_text, word_tokenizer, build_word_tokenizer=True)
    
    # Save the word dictionary to a JSON file
    with open(output_path, 'w') as f:
        json.dump(word_tokenizer, f)
    
    print(f"Word dictionary saved to '{output_path}'. Total words: {len(word_tokenizer)}")

if __name__ == "__main__":
    build_word_dictionary('baseline_6_all_features.csv', ['video_description', 'transcribe_text' ,'generated_vlm_text','llm_response','processed_response'], 'word_dictionary.json')
