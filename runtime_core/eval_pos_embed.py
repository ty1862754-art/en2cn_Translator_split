#!/usr/bin/env python
# coding: utf-8

# BLEU Evaluation for Positional Embedding Experiment (Learnable PE vs Sinusoidal PE)
import os
import sys
import torch
import jieba
import evaluate
from tqdm import tqdm
from nltk import word_tokenize

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.transformer import build_transformer
from tokenization import PrepareData

PAD = 0
UNK = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    bos_id = tokenizer_tgt.get('BOS')
    eos_id = tokenizer_tgt.get('EOS')
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(bos_id).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = (torch.triu(torch.ones(1, decoder_input.size(1), decoder_input.size(1)), diagonal=1) == 0).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        if next_word == eos_id:
            break
    return decoder_input.squeeze(0)


def evaluate_model(experiment_name, model_dir, pe_type, test_file, output_file):
    print(f"\n{'='*60}")
    print(f"Evaluating: {experiment_name} (pe_type={pe_type})")
    print(f"{'='*60}")

    # Load data to get vocabulary
    train_file = 'data/en-cn/train.txt'
    dev_file = 'data/en-cn/dev.txt'
    data = PrepareData(train_file, dev_file, 64, UNK, PAD)
    src_vocab_size = len(data.en_word_dict)
    tgt_vocab_size = len(data.cn_word_dict)

    # Build model with correct pe_type
    model = build_transformer(src_vocab_size, tgt_vocab_size, 60, 60, 256, 6, 8, 0.1, 1024, pe_type=pe_type).to(device)

    model_path = os.path.join(model_dir, 'best_model.pt')
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, 'model.pt')
    print(f"Loading model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully!")

    # Translate test set
    print("Translating test set...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(test_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        lines = f_in.readlines()
        for line in tqdm(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            en_text = parts[0]
            en_tokens = ["BOS"] + word_tokenize(en_text.lower()) + ["EOS"]
            en_ids = [data.en_word_dict.get(w, UNK) for w in en_tokens]
            src = torch.tensor([en_ids]).to(device)
            src_mask = (src != PAD).unsqueeze(1).unsqueeze(1).to(device)
            with torch.no_grad():
                model_out = greedy_decode(model, src, src_mask, data.cn_word_dict, 60, device)
            pred_sent = ""
            for j in range(1, model_out.size(0)):
                sym = data.cn_index_dict[model_out[j].item()]
                if sym == 'EOS':
                    break
                pred_sent += sym
            words = jieba.cut(pred_sent, cut_all=False)
            f_out.write(" ".join(words) + '\n')

    print(f"Translations saved to: {output_file}")

    # BLEU evaluation
    def zh_tokenize(s):
        s = s.strip()
        return " ".join(jieba.cut(s, cut_all=False)) if s else ""

    def read_preds(path):
        preds = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    preds.append(line.strip())
        return preds

    def read_refs(path):
        refs = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                refs.append([zh_tokenize(parts[1])])
        return refs

    preds = read_preds(output_file)
    refs = read_refs(test_file)

    n = min(len(preds), len(refs))
    preds = preds[:n]
    refs = refs[:n]

    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=preds, references=refs)

    print(f"\nBLEU score for {experiment_name}: {results['bleu']:.4f}")
    print(f"  precisions: {[f'{p:.4f}' for p in results['precisions']]}")
    print(f"  brevity_penalty: {results['brevity_penalty']:.4f}")

    # Print some examples
    print("\nSample translations:")
    for i in range(min(5, n)):
        print(f"  [{i}] PRED: {preds[i]}")
        print(f"  [{i}] REF : {refs[i][0]}")
        print()

    return results['bleu']


if __name__ == '__main__':
    test_file = 'data/en-cn/test.txt'

    # 1. Learnable PE
    learnable_bleu = evaluate_model(
        experiment_name='learnable_pe',
        model_dir='save/models-posembed/learnable_pe',
        pe_type='learnable',
        test_file=test_file,
        output_file='save/predictions_segmented_learnable_pe.txt'
    )

    # 2. Baseline Sinusoidal PE (experiment B)
    print("\n" + "="*60)
    print("Note: Baseline (Sinusoidal PE) result is from experiment B:")
    print("  cosine_warmup_lr3e-4_warm5 -> test BLEU = 0.2323")
    print("="*60)

    print(f"\n{'='*60}")
    print(f"SUMMARY: Positional Embedding Comparison")
    print(f"{'='*60}")
    print(f"  Sinusoidal PE (baseline):  0.2323")
    print(f"  Learnable PE:              {learnable_bleu:.4f}")
    print(f"{'='*60}")
