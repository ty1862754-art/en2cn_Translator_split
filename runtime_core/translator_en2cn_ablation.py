#!/usr/bin/env python
# coding: utf-8

# 超参数消融实验脚本 (1.4)
# 支持命令行参数: --n_layer, --h_num, --d_model, --name

from model.transformer import build_transformer
from tokenization import PrepareData, casual_mask
import json
import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import jieba
from nltk.translate.bleu_score import corpus_bleu
from torch.optim.lr_scheduler import LambdaLR
import math
import argparse

import warnings
warnings.filterwarnings('ignore')

PAD = 0
UNK = 1
DEBUG = False


def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Ablation Study')
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--h_num', type=int, default=8)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--name', type=str, default='ablation_default')
    parser.add_argument('--epochs', type=int, default=50)
    return parser.parse_args()


def get_config():
    args = parse_args()

    d_ff = 1024 if args.d_model == 256 else args.d_model * 4
    save_dir = f"save/models-ablation/{args.name}/"

    return {
        'peak_lr': 3e-4,
        'warmup_epochs': 5,
        'batch_size': 64,
        'num_epochs': args.epochs,
        'n_layer': args.n_layer,
        'h_num': args.h_num,
        'd_model': args.d_model,
        'd_ff': d_ff,
        'dropout': 0.1,
        'seq_len': 60,
        'train_file': 'data/en-cn/train.txt',
        'dev_file': 'data/en-cn/dev.txt',
        'save_file': f'{save_dir}model.pt',
        'checkpoint_file': f'{save_dir}checkpoint.pt',
        'best_model_file': f'{save_dir}best_model.pt',
        'loss_history_file': f'{save_dir}loss_history.json',
        'loss_curve_file': f'{save_dir}loss_curve.png',
        'experiment_name': args.name,
    }


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'],
                              config['n_layer'], config['h_num'], config['dropout'], config['d_ff'])
    return model


config = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device {device}")
print(f"Ablation: {config['experiment_name']} | n_layer={config['n_layer']} h_num={config['h_num']} d_model={config['d_model']} d_ff={config['d_ff']}")

data = PrepareData(config['train_file'], config['dev_file'], config['batch_size'], UNK, PAD)
src_vocab_size = len(data.en_word_dict); print(f"src_vocab_size {src_vocab_size}")
tgt_vocab_size = len(data.cn_word_dict); print(f"tgt_vocab_size {tgt_vocab_size}")

model = get_model(config, src_vocab_size, tgt_vocab_size).to(device)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['peak_lr'], eps=1e-9)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


warmup_steps = config['warmup_epochs'] * len(data.train_data)
total_steps = config['num_epochs'] * len(data.train_data)
scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
print(f"Cosine Warmup: peak_lr={config['peak_lr']}, warmup_steps={warmup_steps}, total_steps={total_steps}")


def save_checkpoint(checkpoint_path, model, optimizer, epoch, global_step, loss_value, best_loss, loss_history):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'loss': loss_value,
        'best_loss': best_loss,
        'loss_history': loss_history,
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint_if_exists(checkpoint_path, model, optimizer, device):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        last_loss = checkpoint['loss']
        best_loss = checkpoint['best_loss']
        loss_history = checkpoint['loss_history']
        print(f"Loaded checkpoint from {checkpoint_path} (resume at epoch {epoch})")
        return epoch, global_step, last_loss, best_loss, loss_history
    return 0, 0, float('inf'), float('inf'), []


def save_loss_artifacts(loss_history, loss_history_file, loss_curve_file):
    with open(loss_history_file, 'w', encoding='utf-8') as f:
        json.dump(loss_history, f, ensure_ascii=False, indent=2)
    epochs = [h['epoch'] for h in loss_history]
    losses = [h['loss'] for h in loss_history]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss Curve ({config["experiment_name"]})')
    plt.grid(True, alpha=0.3)
    plt.savefig(loss_curve_file, dpi=150, bbox_inches='tight')
    plt.close()


def save_experiment_config(config, final_val_bleu=None, final_test_bleu=None):
    config_path = f"save/models-ablation/{config['experiment_name']}/config.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    exp_config = {
        'experiment_name': config.get('experiment_name', ''),
        'n_layer': config.get('n_layer'),
        'h_num': config.get('h_num'),
        'd_model': config.get('d_model'),
        'd_ff': config.get('d_ff'),
        'num_epochs': config.get('num_epochs', 50),
        'final_val_bleu': final_val_bleu,
        'final_test_bleu': final_test_bleu,
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(exp_config, f, ensure_ascii=False, indent=2)
    print(f"Experiment config saved to {config_path}")


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    bos_id = tokenizer_tgt.get('BOS', 2)
    eos_id = tokenizer_tgt.get('EOS', 3)
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(bos_id).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )
        if next_word == eos_id:
            break
    return decoder_input.squeeze(0)


def run_validation(model, data, tokenizer_tgt, max_len, device, print_msg, num_examples=4):
    model.eval()
    console_width = 80
    all_preds = []
    all_refs = []

    with torch.no_grad():
        for i in range(len(data.dev_cn)):
            en_ids = data.dev_en[i]
            src = torch.tensor([en_ids]).to(device)
            src_mask = (src != PAD).unsqueeze(1).unsqueeze(1).to(device)

            model_out = greedy_decode(model, src, src_mask, data.cn_word_dict, max_len, device)

            pred_sent_str = ""
            for j in range(1, model_out.size(0)):
                sym = data.cn_index_dict[model_out[j].item()]
                if sym == 'EOS': break
                pred_sent_str += sym

            pred_tokenized = list(jieba.cut(pred_sent_str))
            all_preds.append(pred_tokenized)

            ref_words = [data.cn_index_dict[w] for w in data.dev_cn[i] if data.cn_index_dict[w] not in ['BOS', 'EOS', 'PAD']]
            ref_sent_str = "".join(ref_words)
            ref_tokenized = list(jieba.cut(ref_sent_str))
            all_refs.append([ref_tokenized])

            if i < num_examples:
                source_text = " ".join([data.en_index_dict[w] for w in data.dev_en[i] if data.en_index_dict[w] not in ['BOS', 'EOS', 'PAD']])
                print_msg('-' * console_width)
                print_msg(f'SOURCE: {source_text}')
                print_msg(f'TARGET: {ref_sent_str}')
                print_msg(f'PREDICTED: {pred_sent_str}')

    if not all_preds:
        score = 0.0
    else:
        score = corpus_bleu(all_refs, all_preds)

    print_msg('=' * console_width)
    print_msg(f'>>>> VALIDATION BLEU SCORE: {score:.4f}')
    print_msg('=' * console_width)
    return score


# ============================================================
# Training
# ============================================================
print(f">>>>>>> start ablation train: {config['experiment_name']}")
train_start = time.time()

initial_epoch, global_step, last_loss, best_loss, loss_history = load_checkpoint_if_exists(
    config['checkpoint_file'], model, optimizer, device
)

for epoch in range(initial_epoch, config['num_epochs']):
    batch_iterator = tqdm(data.train_data, desc=f'Processing epoch {epoch:02d}')
    epoch_loss_sum = 0.0
    epoch_step_count = 0

    for batch in batch_iterator:
        model.train()

        encoder_input = batch.src.to(device)
        decoder_input = batch.tgt.to(device)
        encoder_mask = batch.src_mask.to(device)
        decoder_mask = batch.tgt_mask.to(device)

        encoder_output = model.encode(encoder_input, encoder_mask)
        decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
        proj_output = model.project(decoder_output)

        label = batch.tgt_y.to(device)
        loss = loss_fn(proj_output.view(-1, tgt_vocab_size), label.view(-1))

        batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})
        last_loss = loss.item()
        epoch_loss_sum += last_loss
        epoch_step_count += 1

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    if epoch > 0 and epoch % 10 == 0:
        val_bleu = run_validation(model, data, data.cn_word_dict, config['seq_len'], device, lambda msg: batch_iterator.write(msg))

    epoch_loss = epoch_loss_sum / max(epoch_step_count, 1)
    loss_history.append({'epoch': int(epoch), 'loss': float(epoch_loss)})
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:02d} average loss: {epoch_loss:.6f} | lr: {current_lr:.2e}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        os.makedirs(os.path.dirname(config['best_model_file']), exist_ok=True)
        torch.save(model.state_dict(), config['best_model_file'])
        print(f"Saved best model to {config['best_model_file']} (loss={best_loss:.6f})")

    save_checkpoint(config['checkpoint_file'], model, optimizer, epoch, global_step, epoch_loss, best_loss, loss_history)
    save_loss_artifacts(loss_history, config['loss_history_file'], config['loss_curve_file'])

    if epoch > 0 and epoch % 10 == 0:
        intermediate_model_file = os.path.join(os.path.dirname(config['save_file']), f'model_epoch_{epoch:02d}.pt')
        torch.save(model.state_dict(), intermediate_model_file)
        print(f"Saved intermediate model for evaluation to {intermediate_model_file}")

print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")

os.makedirs(os.path.dirname(config['save_file']), exist_ok=True)
torch.save(model.state_dict(), config['save_file'])
print(f"Saved final weights to {config['save_file']}")
print(f"Final training artifacts saved to {config['loss_history_file']} and {config['loss_curve_file']}")
save_experiment_config(config)
