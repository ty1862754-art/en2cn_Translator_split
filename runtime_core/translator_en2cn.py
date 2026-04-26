#!/usr/bin/env python
# coding: utf-8

# Data Preparation: English-to-Chinese Translator Data

from model.transformer import build_transformer
from tokenization import PrepareData, MaskBatch
import json
import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore') # Filtering warnings


# init parameters
PAD = 0  # padding word-id
UNK = 1  # unknown word-id


# DEBUG = True    # Debug / Learning Purposes.
DEBUG = False # Build the model, better with GPU CUDA enabled.

def get_config(debug=True):
    if debug:
        return{
            'lr': 1e-2,
            'batch_size': 64,
            'num_epochs': 2,
            'n_layer': 3,
            'h_num': 8,
            'd_model': 128, # Dimensions of the embeddings in the Transformer
            'd_ff': 256, # Dimensions of the feedforward layer in the Transformer
            'dropout': 0.1,
            'seq_len': 60, # max length
            'train_file': 'data/en-cn/train_mini.txt',
            'dev_file': 'data/en-cn/dev_mini.txt',
            'save_file': 'save/models/model.pt',
            'checkpoint_file': 'save/models/checkpoint.pt',
            'best_model_file': 'save/models/best_model.pt',
            'loss_history_file': 'save/models/loss_history.json',
            'loss_curve_file': 'save/models/loss_curve.png'
        }
    else:
        return{
            'lr': 1e-4,
            'batch_size': 64,
            'num_epochs': 30,
            'n_layer': 6,
            'h_num': 8,
            'd_model': 256, # Dimensions of the embeddings in the Transformer
            'd_ff': 1024, # Dimensions of the feedforward layer in the Transformer
            'dropout': 0.1,
            'seq_len': 60, # max length
            'train_file': 'data/en-cn/train.txt',
            'dev_file': 'data/en-cn/dev.txt',
            'save_file': 'save/models/model.pt',
            'checkpoint_file': 'save/models/checkpoint.pt',
            'best_model_file': 'save/models/best_model.pt',
            'loss_history_file': 'save/models/loss_history.json',
            'loss_curve_file': 'save/models/loss_curve.png'
        }


def get_model(config, vocab_src_len, vocab_tgt_len):
    # Loading model using the 'build_transformer' function.
    # We will use the lengths of the source language and target language vocabularies, the 'seq_len', and the dimensionality of the embeddings
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'], 
                              config['n_layer'], config['h_num'], config['dropout'], config['d_ff'])
    return model



# get config
config = get_config(DEBUG) # Retrieving config settings

# Setting up device to run on GPU to train faster
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")

# Data Preprocessing
data = PrepareData(config['train_file'], config['dev_file'], config['batch_size'], UNK, PAD)
src_vocab_size = len(data.en_word_dict); print(f"src_vocab_size {src_vocab_size}")
tgt_vocab_size = len(data.cn_word_dict); print(f"tgt_vocab_size {tgt_vocab_size}")

# Model
model = get_model(config, src_vocab_size, tgt_vocab_size).to(device)



# Initializing CrossEntropyLoss function for training
# We ignore padding tokens when computing loss, as they are not relevant for the learning process
# We also apply label_smoothing to prevent overfitting
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps = 1e-9)


def save_checkpoint(checkpoint_path, model, optimizer, epoch, global_step, loss_value, best_loss, loss_history):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'loss': float(loss_value),
        'best_loss': float(best_loss),
        'loss_history': loss_history,
    }
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint_if_exists(checkpoint_path, model, optimizer, device):
    if not os.path.exists(checkpoint_path):
        return 0, 0, float('inf'), float('inf'), []

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    initial_epoch = checkpoint.get('epoch', -1) + 1
    global_step = checkpoint.get('global_step', 0)
    last_loss = checkpoint.get('loss', float('inf'))
    best_loss = checkpoint.get('best_loss', last_loss)
    loss_history = checkpoint.get('loss_history', [])
    # Backward compatibility: old checkpoint may not have loss history.
    if not loss_history and checkpoint.get('epoch', None) is not None and np.isfinite(last_loss):
        loss_history = [{'epoch': int(checkpoint['epoch']), 'loss': float(last_loss)}]
    print(f"Loaded checkpoint from {checkpoint_path} (resume at epoch {initial_epoch})")
    return initial_epoch, global_step, last_loss, best_loss, loss_history


def save_loss_artifacts(loss_history, history_path, curve_path):
    if not loss_history:
        return

    os.makedirs(os.path.dirname(history_path), exist_ok=True)

    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(loss_history, f, ensure_ascii=False, indent=2)

    epochs = [item['epoch'] for item in loss_history]
    losses = [item['loss'] for item in loss_history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker='o', linewidth=1.8)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(curve_path, dpi=160)
    plt.close()



def casual_mask(size):
    # Creating a square matrix of dimensions 'size x size' filled with ones
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0


# Define function to obtain the most probable next token
def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    # Retrieving the indices from the start and end of sequences of the target tokens
    bos_id = tokenizer_tgt.get('BOS')
    eos_id = tokenizer_tgt.get('EOS')
    
    # Computing the output of the encoder for the source sequence
    encoder_output = model.encode(source, source_mask)
    # Initializing the decoder input with the Start of Sentence token
    decoder_input = torch.empty(1,1).fill_(bos_id).type_as(source).to(device)
    
    # Looping until the 'max_len', maximum length, is reached
    while True:
        if decoder_input.size(1) == max_len:
            break
            
        # Building a mask for the decoder input
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        # Calculating the output of the decoder
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # Applying the projection layer to get the probabilities for the next token
        prob = model.project(out[:, -1])
        
        # Selecting token with the highest probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1). type_as(source).fill_(next_word.item()).to(device)], dim=1)
        
        # If the next token is an End of Sentence token, we finish the loop
        if next_word == eos_id:
            break
            
    return decoder_input.squeeze(0) # Sequence of tokens generated by the decoder



# Defining function to evaluate the model on the validation dataset
# num_examples = 2, two examples per run
def run_validation(model, data, tokenizer_tgt, max_len, device, print_msg, num_examples=4):
    model.eval() # Setting model to evaluation mode
    count = 0 # Initializing counter to keep track of how many examples have been processed
    
    console_width = 80 # Fixed witdh for printed messages
    
    # Creating evaluation loop
    with torch.no_grad(): # Ensuring that no gradients are computed during this process
        for i, batch in enumerate(data.dev_data):
            count += 1
            encoder_input = batch.src.to(device)
            encoder_mask = batch.src_mask.to(device)
            
            # Ensuring that the batch_size of the validation set is 1
            assert encoder_input.size(0) ==  1, 'Batch size must be 1 for validation.'
            
            # Applying the 'greedy_decode' function to get the model's output for the source text of the input batch
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            # Retrieving source and target texts from the batch
            source_text = " ".join([data.en_index_dict[w] for w in data.dev_en[i]])
            target_text = " ".join([data.cn_index_dict[w] for w in data.dev_cn[i]])

            # save all in the translation list
            model_out_text = []
            # convert id to Chinese, skip 'BOS' 0.
            print(model_out)
            for j in range(1, model_out.size(0)):
                sym = data.cn_index_dict[model_out[j].item()]
                if sym != 'EOS':
                    model_out_text.append(sym)
                else:
                    break

            # Printing results
            print_msg('-'*console_width)
            print_msg(f'SOURCE: {source_text}')
            print_msg(f'TARGET: {target_text}')
            print_msg(f'PREDICTED: {model_out_text}')
            
            # After two examples, we break the loop
            if count == num_examples:
                break


# Training model
print(">>>>>>> start train")
train_start = time.time()

# Initializing epoch and global step variables
initial_epoch, global_step, last_loss, best_loss, loss_history = load_checkpoint_if_exists(
    config['checkpoint_file'], model, optimizer, device
)

# Iterating over each epoch from the 'initial_epoch' variable up to the number of epochs informed in the config
for epoch in range(initial_epoch, config['num_epochs']):
    # Initializing an iterator over the training dataloader
    # We also use tqdm to display a progress bar
    batch_iterator = tqdm(data.train_data, desc = f'Processing epoch {epoch:02d}')
    epoch_loss_sum = 0.0
    epoch_step_count = 0
    
    # For each batch...
    for batch in batch_iterator:
        model.train() # Train the model
        
        # Loading input data and masks onto the GPU
        encoder_input = batch.src.to(device)
        decoder_input = batch.tgt.to(device)
        encoder_mask = batch.src_mask.to(device)
        decoder_mask = batch.tgt_mask.to(device)
        # print(encoder_input[0], encoder_mask[0], decoder_input[0], decoder_mask[0])
        # print(encoder_input.shape, encoder_mask.shape, decoder_input.shape, decoder_mask.shape)

        # Running tensors through the Transformer
        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = model.project(decoder_output)
        
        # Loading the target labels onto the GPU
        label = batch.tgt_y.to(device)
        
        # Computing loss between model's output and true labels
        loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))
        
        # Updating progress bar
        batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
        last_loss = loss.item()
        epoch_loss_sum += last_loss
        epoch_step_count += 1
        
        # Performing backpropagation
        loss.backward()
        
        # Updating parameters based on the gradients
        optimizer.step()
        
        # Clearing the gradients to prepare for the next batch
        optimizer.zero_grad()
        
        global_step += 1 # Updating global step count

    # to evaluate model performance
    if epoch % 5 == 0:
        run_validation(model, data, data.cn_word_dict, config['seq_len'], device, lambda msg: batch_iterator.write(msg))

    epoch_loss = epoch_loss_sum / max(epoch_step_count, 1)
    loss_history.append({'epoch': int(epoch), 'loss': float(epoch_loss)})
    print(f"Epoch {epoch:02d} average loss: {epoch_loss:.6f}")

    # Save best model based on the minimum epoch-ending loss.
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        os.makedirs(os.path.dirname(config['best_model_file']), exist_ok=True)
        torch.save(model.state_dict(), config['best_model_file'])
        print(f"Saved best model to {config['best_model_file']} (loss={best_loss:.6f})")

    # Save full checkpoint after each epoch.
    save_checkpoint(config['checkpoint_file'], model, optimizer, epoch, global_step, epoch_loss, best_loss, loss_history)

print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")

# Save final model weights from the last epoch.
os.makedirs(os.path.dirname(config['save_file']), exist_ok=True)
torch.save(model.state_dict(), config['save_file'])
print(f"Saved final weights to {config['save_file']}")

save_loss_artifacts(loss_history, config['loss_history_file'], config['loss_curve_file'])
print(f"Saved loss history to {config['loss_history_file']}")
print(f"Saved loss curve to {config['loss_curve_file']}")




