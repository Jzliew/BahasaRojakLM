import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from datasets import load_from_disk, DatasetDict
from tqdm import tqdm
import argparse

from model import *  # Your model definitions

def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT MLM model with XLM tokenizer")

    parser.add_argument('--model', type=str, default="MixedXLM",
                        help='Type of model to train')

    parser.add_argument('--dataset_path', type=str, default="./data/bpe_tokenized",
                        help='Path to the dataset folder')
    parser.add_argument('--tokenizer_path', type=str, default='./models/bpe_tokenizer.json',
                        help='Path to the tokenizer JSON file')
    parser.add_argument('--result_dir', type=str, default='./result/bert_test/',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--patience', type=int, default=20000, help='Early stopping patience')
    parser.add_argument('--validation_interval', type=int, default=5000, help='Validation interval (steps)')
    parser.add_argument('--log_step', type=int, default=500, help='Logging step interval')
    parser.add_argument('--best_model_interval', type=int, default=50, help='Interval for saving best model checkpoints')
    parser.add_argument('--pad_token_id', type=int, default=3, help='PAD token ID')
    parser.add_argument('--mask_token_id', type=int, default=4, help='MASK token ID')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Vocabulary size')
    
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')

    
    parser.add_argument("--d_model", type=int, default=512 , help='hidden state size of model')
    parser.add_argument("--n_layers", type=int, default=6 , help='number of transformer layers in model')
    parser.add_argument("--heads", type=int, default=8, help='number of attention heads in MAH')
    parser.add_argument("--dropout", type=float, default=0.1, help='droupout ratio')
    
    
    

    args = parser.parse_args()
    return args

def tokenize_vocab(vocab, tokenizer):
    return [tokenizer.encode(word).ids for word in tqdm(vocab)]
    
def get_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 3)
    else:
        gpu_memory = 0.0
    return gpu_memory

def mask_tokens(inputs, mask_token_id=4, vocab_size=50000, mlm_probability=0.15):
    LOSS_IGNORE = -100
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability, device=inputs.device)
    special_tokens_mask = (inputs == 3)  # PAD_TOKEN_ID = 3
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    pred_mask = torch.bernoulli(probability_matrix).bool()
    labels[~pred_mask] = LOSS_IGNORE

    _x_real = inputs[pred_mask]
    _x_rand = torch.randint(low=0, high=vocab_size, size=_x_real.shape, device=inputs.device)
    _x_mask = torch.full_like(_x_real, mask_token_id)

    probs = torch.multinomial(torch.tensor([0.8, 0.1, 0.1], device=inputs.device), len(_x_real), replacement=True)
    _x = _x_mask * (probs == 0) + _x_real * (probs == 1) + _x_rand * (probs == 2)
    inputs = inputs.masked_scatter(pred_mask, _x)

    return inputs, labels



def train():
    args = parse_args()

    # Device setup
    device = args.device

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    full_dataset = load_from_disk(args.dataset_path)
    split_dataset = full_dataset.train_test_split(test_size=0.01)
    tokenized_dataset = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Create DataLoaders
    train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=args.batch_size, shuffle=True)
    validation_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=args.batch_size)

    # Initialize model, optimizer, scaler, etc.
    
    # Assume vocab_size is passed as an argument or computed from tokenizer or dataset
    vocab_size = args.vocab_size

    if args.model == 'BERT':
        # Initialize model with parameters from args
        bert_model = BERT(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            heads=args.heads,
            dropout=args.dropout
        )
        model = BERTLM(bert_model, vocab_size)
    elif args.model == 'MixedXLM':
        # Load the vocab sets from the JSON file
        input_file = "./models/vocab_sets_v3.json"
        with open(input_file, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        bpe_tokenizer = Tokenizer.from_file(args.tokenizer_path)


        # Extract vocab sets from the loaded dictionary
        ms_vocab_words = vocab_dict["ms_vocab_words"]
        eng_vocab_words = vocab_dict["eng_vocab_words"]
        chinese_words = vocab_dict["chinese_words"]
        ms_vocab_tokens = tokenize_vocab(ms_vocab_words, bpe_tokenizer)
        eng_vocab_tokens = tokenize_vocab(eng_vocab_words, bpe_tokenizer)
        chi_vocab_tokens = tokenize_vocab(chinese_words, bpe_tokenizer)
        # Convert ms_vocab_tokens to a set
        ms_vocab_set = set(token for sublist in ms_vocab_tokens for token in sublist)

        # Convert eng_vocab_tokens to a set
        eng_vocab_set = set(token for sublist in eng_vocab_tokens for token in sublist)

        # Convert chi_vocab_tokens to a set
        chi_vocab_set = set(token for sublist in chi_vocab_tokens for token in sublist)
        
        mixedxlm = MixedXLM(
            vocab_size, 
            ms_vocab_set, 
            eng_vocab_set, 
            chi_vocab_set,
            d_model=args.d_model,
            n_layers=args.n_layers,
            heads=args.heads,
            dropout=args.dropout,
            debug=False)
        
        model = MixedXLM_LM(mixedxlm, vocab_size)
    model.to(device)


    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    # Prepare variables for training
    
    MASKED_TOKEN_ID = args.mask_token_id  
    epochs = args.epochs
    log_step = args.log_step
    validation_interval = args.validation_interval
    checkpoint_dir = args.result_dir +'/checkpoints'
    log_file_dir = args.result_dir + 'training_log.json'
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_loss = float('inf')
    no_improvement_count = 0
    patience = args.patience
    improvement_count = 0
    best_model_interval = args.best_model_interval

    current_step = 0
    total_steps = len(train_dataloader) * epochs

    log_entries = []


    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        log_loss = 0
        
        for batch in tqdm(train_dataloader, total=len(train_dataloader), desc="Training Progress", initial=current_step):
            current_step += 1
            input_ids_batch = batch['input_ids'].to(device)
            attention_mask_batch = batch['attention_mask'].to(device)
            
            masked_input_ids, labels = mask_tokens(input_ids_batch, MASKED_TOKEN_ID)
            masked_input_ids, labels = masked_input_ids.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(masked_input_ids, attention_mask=attention_mask_batch)
                logits = output.view(-1, vocab_size)
                labels = labels.view(-1)
                loss = F.cross_entropy(logits, labels, ignore_index=-100)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            log_loss += loss.item()

            if current_step % log_step == 0:
                avg_log_loss = log_loss / log_step
                tqdm.write(f"Step {current_step}, Training Loss: {avg_log_loss:.4f}")

                log_entry = {
                    "epoch": (epoch + current_step / len(train_dataloader)) / epochs,
                    "step": current_step,
                    "loss": avg_log_loss,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "gpu_memory": get_gpu_memory(),
                    "gradient_norm": None
                }
                log_entries.append(log_entry)
                with open(log_file_dir, 'w') as f:
                    json.dump(log_entries, f, indent=4)
                log_loss = 0

            if (current_step + 1) % validation_interval == 0:
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for val_step, val_batch in tqdm(enumerate(validation_dataloader), total=len(validation_dataloader), desc=f"Validation at Step {current_step + 1}"):
                        input_ids_val = val_batch['input_ids'].to(device)
                        attention_mask_val = val_batch['attention_mask'].to(device)

                        masked_input_ids_val, labels_val = mask_tokens(input_ids_val, MASKED_TOKEN_ID)
                        masked_input_ids_val, labels_val = masked_input_ids_val.to(device), labels_val.to(device)

                        output_val = model(masked_input_ids_val, attention_mask=attention_mask_val)
                        logits_val = output_val.view(-1, vocab_size)
                        labels_val = labels_val.view(-1)
                        val_loss = F.cross_entropy(logits_val, labels_val, ignore_index=-100)
                        total_val_loss += val_loss.item()

                avg_val_loss = total_val_loss / len(validation_dataloader)
                print(f"Step {current_step + 1}, Average Validation Loss: {avg_val_loss:.4f}, Best Validation Loss: {best_val_loss:.4f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    no_improvement_count = 0

                    checkpoint_filename = f'{checkpoint_dir}/best_model_step_{current_step}.pth'
                    torch.save({
                        'epoch': epoch,
                        'step': current_step,
                        'model_state_dict': model.state_dict(),
                        'validation_loss': best_val_loss
                    }, checkpoint_filename)
                    print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")
                else:
                    no_improvement_count += 1
                    print(f"No improvement. Early stopping counter: {no_improvement_count}/{patience}")

                print(f"Improvement count: {improvement_count}")
                improvement_count += 1
                if improvement_count >= best_model_interval:
                    improvement_count = 0
                    checkpoint_filename = f'{checkpoint_dir}/best_model_interval_{current_step}.pth'
                    torch.save({
                        'epoch': epoch,
                        'step': current_step,
                        'model_state_dict': model.state_dict(),
                        'validation_loss': best_val_loss
                    }, checkpoint_filename)
                    print(f"Best model saved after {best_model_interval} improvements: {checkpoint_filename}")

                    manage_checkpoints(checkpoint_dir, current_step)

                checkpoint_filename = f'{checkpoint_dir}/checkpoint_step_{current_step}.pth'
                torch.save({
                    'epoch': epoch,
                    'step': current_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                }, checkpoint_filename)
                print(f"Checkpoint saved as {checkpoint_filename}")

                manage_checkpoints(checkpoint_dir, current_step)

                model.train()

                if no_improvement_count >= patience:
                    print("Early stopping triggered! No improvement in validation loss.")
                    break

        if no_improvement_count >= patience:
            break

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Training Loss: {avg_train_loss:.4f}")

    print("Training complete!")

    final_model_save_path = f'{checkpoint_dir}/final_model_step_{current_step}.bin'
    torch.save(model.state_dict(), final_model_save_path)
    print(f"Final model saved as {final_model_save_path}")
    
    with open(log_file_dir, 'w') as f:
        json.dump(log_entries, f, indent=4)

    print("Training complete!")



if __name__ == "__main__":
    train()
