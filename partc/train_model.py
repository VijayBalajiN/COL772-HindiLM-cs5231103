import torch.nn as nn
import torch
import os
import json
# YOUR TOKENIZER AND MODEL from PART A AND PART B RESPECTIVELY
# If you wish to change their code, please do so in their respective files under parta/ and partb/ directories.
from partb.bpe_tokenizer import BPETokenizer
from parta.model import LanguageModel, collate_fn

# You can also create additional files in this directory and import them here if needed.
# For example, the line below import a dummy function from utils.py file.
from .utils import dummy_function  # Replace with actual utility functions as needed

# You can structure your code as you see fit as long as the CLI works as specified.
# Finally, treat this as your FINAL MODEL TRAINING SCRIPT. Do not perform hyperparameter tuning here.
# You can create separate scripts for hyperparameter tuning if needed.

device = torch.device("cpu")


def calc_bpc(logits, input_ids, attention_mask, bpe_tokenizer):
    batch_size, seq_len, vocab_size = logits.shape
    avg_loss = []
    no_toks = []
    no_chars = []
    for seq in range(batch_size):
        seq_logits = logits[seq]
        seq_input_ids = input_ids[seq]
        seq_attention_mask = attention_mask[seq]
        seq_loss = 0
        curr_chars = 0
        valid_tokens = 0
        for i in range(seq_len-1):
            if seq_attention_mask[i+1] == 0:
                break
            token_logits = seq_logits[i]
            token_id = int(seq_input_ids[i+1].item())
            token_prob = torch.softmax(token_logits, dim=0)[token_id]
            token_loss = -torch.log2(token_prob)
            seq_loss += token_loss.item()
            curr_chars += len(bpe_tokenizer.vocab[token_id])
            valid_tokens += 1
        if valid_tokens == 0:
            continue
        no_chars.append(curr_chars)
        avg_loss.append(seq_loss / valid_tokens)
        no_toks.append(valid_tokens)
    num = 0
    denom = 0
    for i in range(len(avg_loss)):
        num += avg_loss[i]*no_toks[i]
        denom += no_chars[i]
    return num/denom

        
        
    
    

def batch_input(tokenized_texts, batch_size):
    batched_tok_texts = []
    curr_batch = []
    for i in range(len(tokenized_texts)):
        curr_batch.append(tokenized_texts[i])
        if i and (i+1)%batch_size == 0:
            batched_tok_texts.append((i//batch_size, curr_batch))
            curr_batch = []
    if curr_batch:
        batched_tok_texts.append((len(tokenized_texts)//batch_size, curr_batch))
    return batched_tok_texts


def train_loop(model, batched_tok_train, loss_fn, optim):
    model.train()
    for (batch, input_ids) in batched_tok_train:
        tensor_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        collated_input = collate_fn({'input_ids': tensor_input_ids})
        collated_input['input_ids'] = collated_input['input_ids'].to(device)
        collated_input['attention_mask'] = collated_input['attention_mask'].to(device)
        output_logits = model(collated_input['input_ids'], collated_input['attention_mask'])
        shifted_logits = output_logits[:, :-1, :]
        shifted_targets = collated_input['input_ids'][:, 1:]
        loss = loss_fn(shifted_logits.reshape(-1, shifted_logits.shape[-1]), shifted_targets.reshape(-1))
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (batch+1)%40 == 0:
            bpc = calc_bpc(output_logits, collated_input['input_ids'], collated_input['attention_mask'], bpe_tokenizer)
            print("Batch:", batch+1, "Loss:", loss.item(), "BPC:", bpc)
            
def val_loop(model, batched_tok_val, loss_fn):
    model.eval()
    total_loss = 0
    total_bpc = 0
    count = 0
    with torch.no_grad():
        for (batch, input_ids) in batched_tok_val:
            tensor_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
            collated_input = collate_fn({'input_ids': tensor_input_ids})
            collated_input['input_ids'] = collated_input['input_ids'].to(device)
            collated_input['attention_mask'] = collated_input['attention_mask'].to(device)
            output_logits = model(collated_input['input_ids'], collated_input['attention_mask'])
            shifted_logits = output_logits[:, :-1, :]
            shifted_targets = collated_input['input_ids'][:, 1:]
            loss = loss_fn(shifted_logits.reshape(-1, shifted_logits.shape[-1]), shifted_targets.reshape(-1))
            bpc = calc_bpc(output_logits, collated_input['input_ids'], collated_input['attention_mask'], bpe_tokenizer)
            total_loss += loss.item()
            total_bpc += bpc
            count += 1
    avg_val_loss = total_loss/count
    avg_val_bpc = total_bpc/count
    print("Validation Loss:", avg_val_loss, "Validation BPC:", avg_val_bpc)
    return avg_val_loss, avg_val_bpc
        
    
import time
def main(args):
    # raise NotImplementedError("This is a placeholder for the training script. Please implement the training logic here.")
    global bpe_tokenizer, device
    start_time = time.time()

    bpe_tokenizer = BPETokenizer()
    bpe_tokenizer.load(args.tokenizer_path)
    train_texts = []
    with open(args.train_path, "r") as f:
        for line in f:
            train_texts.append(line.strip())
    val_texts = []
    with open(args.valid_path, "r") as f:
        for line in f:
            val_texts.append(line.strip())
    print("read data in", time.time() - start_time, "seconds")

    train_cache_path = os.path.join(os.path.dirname(os.path.abspath(args.train_path)), "encoded_train.json")
    val_cache_path = os.path.join(os.path.dirname(os.path.abspath(args.valid_path)), "encoded_val.json")

    if os.path.exists(train_cache_path):
        with open(train_cache_path, "r") as f:
            tokenized_train = json.load(f)
        print(f"Loaded cached train encodings from {train_cache_path}")
    else:
        tokenized_train = [bpe_tokenizer.encode(text) for text in train_texts]
        with open(train_cache_path, "w") as f:
            json.dump(tokenized_train, f)
        print(f"Saved train encodings to {train_cache_path}")

    if os.path.exists(val_cache_path):
        with open(val_cache_path, "r") as f:
            tokenized_val = json.load(f)
        print(f"Loaded cached valid encodings from {val_cache_path}")
    else:
        tokenized_val = [bpe_tokenizer.encode(text) for text in val_texts]
        with open(val_cache_path, "w") as f:
            json.dump(tokenized_val, f)
        print(f"Saved valid encodings to {val_cache_path}")

    print("tokenized data in", time.time() - start_time, "seconds")
    config = {                              #directly setting from case 2 in data of parta
                "d_model": 256,
                "n_heads": 8,
                "d_head": 32,
                "n_layers": 4,
                "vocab_size": bpe_tokenizer.get_vocab_size(),
                "mode": "standard",
                "tau": 1.5,
                "batch_size": 32,
                "lr": 1e-4
            }
    model = LanguageModel(config)
    model.set_weights_randomly()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "mps":
        # Attention is memory-heavy on MPS; keep batch conservative to avoid OOM.
        config["batch_size"] = min(config["batch_size"], 32)

    model = model.to(device)
    print(f"Using device: {device}")
    print("initialized model in", time.time() - start_time, "seconds")
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    batched_tok_train = batch_input(tokenized_train, config["batch_size"])
    batched_tok_val = batch_input(tokenized_val, config["batch_size"])
    l = [[len(x) for x in batched_tok_train[0][1]]]
    print(*l)
    print(f"batch size = {config['batch_size']}, number of batches in train = {len(batched_tok_train)}, number of batches in val = {len(batched_tok_val)}")
    max_train_seconds = 5 * 60 * 60
    epoch0_start_time = None
    best_val_bpc = float("inf")

    for epoch in range(80):
        current_epoch_start = time.time()
        if epoch0_start_time is None:
            epoch0_start_time = current_epoch_start
        elif current_epoch_start - epoch0_start_time >= max_train_seconds:
            break

        print("--------------Epoch:", epoch, "--------------")
        train_loop(model, batched_tok_train, loss_fn, optimizer)
        _, val_bpc = val_loop(model, batched_tok_val, loss_fn)
        #save the model
        os.makedirs(args.output_model_path, exist_ok=True)
        torch.save(model.state_dict(), args.output_model_path + "/model" + str(epoch) + ".pt")

        if val_bpc < best_val_bpc:
            best_val_bpc = val_bpc
        elif epoch > 20:
            print("Stopping training: Validation BPC did not improve this epoch.")
            break

    torch.save(model.state_dict(), args.output_model_path + "/model.pt")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model on the given dataset.')
    parser.add_argument('--train_path', type=str, required=True, help='Path to the train dataset')
    parser.add_argument('--valid_path', type=str, required=True, help='Path to the valid dataset')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to the tokenizer')
    parser.add_argument('--output_model_path', type=str, default='checkpoints', help='Directory to save checkpoints')

    args = parser.parse_args()
    main(args)
