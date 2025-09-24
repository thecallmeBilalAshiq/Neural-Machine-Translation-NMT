# -*- coding: utf-8 -*-
import os
import zipfile
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
from collections import Counter
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
import torch
from torch.utils.data import DataLoader, TensorDataset

# Install necessary libraries
# pip install nltk tensorflow_text transliterate torchtext

# GitHub repository clone
os.system('git clone https://github.com/amir9ume/urdu_ghazals_rekhta')

# List the files in the cloned repository
os.listdir('urdu_ghazals_rekhta')

# Extract the dataset
with zipfile.ZipFile('urdu_ghazals_rekhta/dataset/dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('urdu_ghazals_rekhta/dataset/extracted')

# Define function to load Urdu and Roman Urdu pairs
def load_urdu_roman_pairs(urdu_dir, roman_urdu_dir):
    urdu_poems = []
    roman_urdu_poems = []
    for urdu_file in os.listdir(urdu_dir):
        if urdu_file == '.DS_Store':
            continue
        urdu_file_path = os.path.join(urdu_dir, urdu_file)
        roman_urdu_file_path = os.path.join(roman_urdu_dir, urdu_file)

        if os.path.exists(urdu_file_path) and os.path.exists(roman_urdu_file_path):
            with open(urdu_file_path, 'r', encoding='utf-8') as urdu_f:
                urdu_poem = urdu_f.read().strip()

            with open(roman_urdu_file_path, 'r', encoding='utf-8') as roman_f:
                roman_urdu_poem = roman_f.read().strip()

            urdu_poems.append(urdu_poem)
            roman_urdu_poems.append(roman_urdu_poem)

    return urdu_poems, roman_urdu_poems

# Load all poems
urdu_poems, roman_urdu_poems = load_urdu_roman_pairs(
    'urdu_ghazals_rekhta/dataset/extracted/dataset/faiz-ahmad-faiz/ur',
    'urdu_ghazals_rekhta/dataset/extracted/dataset/faiz-ahmad-faiz/en'
)

# Sample processing and BPE
def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            pairs[pair] += freq
    return pairs

def bpe_tokenize(corpus, num_merges):
    vocab = Counter([' '.join(word) + ' </w>' for word in corpus])
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        new_vocab = {}
        bigram = ' '.join(best_pair)
        replacement = ''.join(best_pair)
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = vocab[word]
        vocab = new_vocab
    tokenized_corpus = []
    for word in corpus:
        tokens = word.split()
        tokenized_corpus.append(tokens)
    return vocab, tokenized_corpus

corpus = ["garmi e shauq e nazaraa ka asar to dekho", "faiz ahmad faiz ghazals"]
bpe_vocab, bpe_tokenized_corpus = bpe_tokenize(corpus, num_merges=10)

def build_vocab(text_data, min_freq=1):
    word_counts = Counter()
    for sentence in text_data:
        words = sentence.split()  # Tokenize sentence into words
        word_counts.update(words)
    vocab = {word: idx for idx, (word, count) in enumerate(word_counts.items()) if count >= min_freq}
    vocab['<unk>'] = len(vocab)
    vocab['<pad>'] = len(vocab)
    return vocab

# Build vocabularies
urdu_vocab = build_vocab(urdu_poems)
roman_urdu_vocab = build_vocab(roman_urdu_poems)

# Convert sentences to indices
def sentence_to_indices_bpe(sentence, vocab, max_length=50):
    tokens = sentence
    indices = [vocab.get(word, vocab['<unk>']) for word in tokens]
    return indices[:max_length] + [vocab['<pad>']] * max(0, max_length - len(indices))

# Tokenize dataset using BPE
urdu_bpe_tokenized = [sentence_to_indices_bpe(poem, urdu_vocab, max_length=50) for poem in urdu_poems]
roman_urdu_bpe_tokenized = [sentence_to_indices_bpe(poem, roman_urdu_vocab, max_length=50) for poem in roman_urdu_poems]

# Convert to PyTorch tensors
urdu_tensor_bpe = torch.tensor(urdu_bpe_tokenized)
roman_urdu_tensor_bpe = torch.tensor(roman_urdu_bpe_tokenized)

# Create DataLoader
train_dataset_bpe = TensorDataset(urdu_tensor_bpe, roman_urdu_tensor_bpe)
train_loader_bpe = DataLoader(train_dataset_bpe, batch_size=32, shuffle=True)

# Define model architecture
class Seq2Seq(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.src_embedding = nn.Embedding(input_dim, emb_dim)
        self.trg_embedding = nn.Embedding(output_dim, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=n_layers, dropout=dropout, bidirectional=True)
        self.decoder = nn.LSTM(emb_dim, hidden_dim * 2, num_layers=n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg):
        src = src.permute(1, 0)
        trg = trg.permute(1, 0)
        embedded_src = self.dropout(self.src_embedding(src))
        embedded_trg = self.dropout(self.trg_embedding(trg))
        encoder_outputs, (hidden, cell) = self.encoder(embedded_src)
        hidden = torch.cat((hidden[0:self.encoder.num_layers, :, :], hidden[self.encoder.num_layers:, :, :]), dim=2)
        cell = torch.cat((cell[0:self.encoder.num_layers, :, :], cell[self.encoder.num_layers:, :, :]), dim=2)
        decoder_outputs, _ = self.decoder(embedded_trg, (hidden, cell))
        predictions = self.fc_out(decoder_outputs.view(-1, decoder_outputs.shape[-1]))
        predictions = predictions.view(decoder_outputs.shape[0], decoder_outputs.shape[1], -1)
        predictions = predictions.permute(1, 0, 2)
        return predictions

# Initialize model
input_dim = len(urdu_vocab)
output_dim = len(roman_urdu_vocab)
emb_dim = 256
hidden_dim = 512
n_layers = 2
dropout = 0.5
model = Seq2Seq(input_dim, output_dim, emb_dim, hidden_dim, n_layers, dropout)

# Optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(ignore_index=urdu_vocab['<pad>'])

# Training loop
def train(model, train_loader, optimizer, criterion, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for src_batch, trg_batch in train_loader:
            optimizer.zero_grad()
            output = model(src_batch, trg_batch)
            loss = criterion(output.reshape(-1, output.shape[-1]), trg_batch.reshape(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# Train the model
train(model, train_loader_bpe, optimizer, criterion, num_epochs=10)
