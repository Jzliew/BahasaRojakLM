import argparse
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import os
from sklearn.model_selection import KFold
from model import *
from tokenizers import Tokenizer
import pandas as pd
import json
from tqdm import tqdm

def load_sentibahasarojak_dataset(dataset_type):
    """
    Load the SentiBahasaRojak dataset by type: 'movie', 'product', or 'stock'.

    Returns:
        train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels
    """
    base_dir = '/home/s6321012100/fyp/data/BahasaRojak Datasets/SentiBahasaRojak'

    if dataset_type == 'movie':
        dir = os.path.join(base_dir, 'SentiBahasaRojak-Movie')
        train_texts = open(os.path.join(dir, 'mix.train')).readlines()
        train_labels = [0 if l.strip() == 'negative' else 1 for l in open(os.path.join(dir, 'mix.train.label')).readlines()]

        valid_texts = open(os.path.join(dir, 'mix.valid')).readlines()
        valid_labels = [0 if l.strip() == 'negative' else 1 for l in open(os.path.join(dir, 'mix.valid.label')).readlines()]

        test_texts = open(os.path.join(dir, 'mix.test')).readlines()
        test_labels = [0 if l.strip() == 'negative' else 1 for l in open(os.path.join(dir, 'mix.test.label')).readlines()]

    elif dataset_type == 'product':
        dir = os.path.join(base_dir, 'SentiBahasaRojak-Product')
        train_texts = open(os.path.join(dir, 'mix.train')).readlines()
        train_labels = [0 if l.strip() == 'negative' else 1 for l in open(os.path.join(dir, 'mix.train.label')).readlines()]

        valid_texts = open(os.path.join(dir, 'mix.valid')).readlines()
        valid_labels = [0 if l.strip() == 'negative' else 1 for l in open(os.path.join(dir, 'mix.valid.label')).readlines()]

        test_texts = open(os.path.join(dir, 'mix.test')).readlines()
        test_labels = [0 if l.strip() == 'negative' else 1 for l in open(os.path.join(dir, 'mix.test.label')).readlines()]

    elif dataset_type == 'stock':
        dir = os.path.join(base_dir, 'SentiBahasaRojak-Stock')

        def load_tsv(file_path):
            df = pd.read_csv(file_path, sep='\t', header=None, names=['text', 'label'])
            df['label'] = df['label'].map({-1: 0, 1: 1})
            return df['text'].tolist(), df['label'].tolist()

        train_texts, train_labels = load_tsv(os.path.join(dir, 'train_labeled.tsv'))
        valid_texts, valid_labels = load_tsv(os.path.join(dir, 'valid_labeled.tsv'))
        test_texts, test_labels = load_tsv(os.path.join(dir, 'test_labeled.tsv'))

    else:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'. Must be one of: 'movie', 'product', 'stock'.")

    return train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels



# ------------------- Argument Parser -------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['BERT', 'MixedXLM', 'TinyXLMR'],  default='MixedXLM')

    parser.add_argument("--d_model", type=int, default=512 , help='hidden state size of model')
    parser.add_argument("--n_layers", type=int, default=6 , help='number of transformer layers in model')
    parser.add_argument("--heads", type=int, default=8, help='number of attention heads in MAH')
    parser.add_argument("--dropout", type=float, default=0.1, help='droupout ratio')

    parser.add_argument("--fit_size", type=int, default=768, help='student model fit size')


    parser.add_argument('--tokenizer_path', type=str, default='./models/mixed_xlm/bpe_tokenizer.json',
                        help='Path to the tokenizer JSON file')
    parser.add_argument('--model_ckpt', type=str,  default='./result/bert_test/checkpoints/final_model_step_17344.bin')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')

    parser.add_argument('--dataset', type=str, choices=['movie', 'product', 'stock'],  default='movie')
    args = parser.parse_args()
    return args

# Tokenization function for bpe
def tokenize_function_bpe(examples, tokenizer):
    
    return {
        "input_ids": [
            tokenizer.encode(text).ids[:256] + [0] * max(0, 256 - len(tokenizer.encode(text).ids))
            for text in examples["text"]
        ],
        "attention_mask": [
            [1] * min(len(tokenizer.encode(text).ids), 256) + [0] * max(0, 256 - len(tokenizer.encode(text).ids))
            for text in examples["text"]
        ],
    }

def tokenize_function_xlmr(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True,max_length=256)

def main():
    args = parse_args()


    # ------------------- Load Tokenizer and Data -------------------
    if args.model == 'TinyXLMR':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        
    else:
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
    

    train_texts, train_labels, valid_texts, valid_labels, test_texts, test_labels = load_sentibahasarojak_dataset(args.dataset)

    


    if args.model == 'MixedXLM':
        # Load the vocab sets from the JSON file
        input_file = "./models/mixed_xlm/vocab_sets_v3.json"
        with open(input_file, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)

        # Extract vocab sets from the loaded dictionary
        ms_vocab_words = vocab_dict["ms_vocab_words"]
        eng_vocab_words = vocab_dict["eng_vocab_words"]
        chinese_words = vocab_dict["chinese_words"]

        device = args.device
        PAD_TOKEN_ID = 3
        MASKED_TOKEN_ID = 4
        
        # Function to tokenize vocab set
        def tokenize_vocab(vocab, tokenizer):
            return [tokenizer.encode(word).ids for word in tqdm(vocab)]
            
        # Tokenize the vocab sets
        ms_vocab_tokens = tokenize_vocab(ms_vocab_words, tokenizer)
        eng_vocab_tokens = tokenize_vocab(eng_vocab_words, tokenizer)
        chi_vocab_tokens = tokenize_vocab(chinese_words, tokenizer)
        # Convert ms_vocab_tokens to a set
        ms_vocab_set = set(token for sublist in ms_vocab_tokens for token in sublist)

        # Convert eng_vocab_tokens to a set
        eng_vocab_set = set(token for sublist in eng_vocab_tokens for token in sublist)

        # Convert chi_vocab_tokens to a set
        chi_vocab_set = set(token for sublist in chi_vocab_tokens for token in sublist)
        print(len(ms_vocab_set))
        print(len(eng_vocab_set))
        print(len(chi_vocab_set))



    # Combine all data
    all_texts = train_texts + valid_texts + test_texts
    all_labels = train_labels + valid_labels + test_labels

    # Create a single dataset
    dataset = Dataset.from_dict({'text': all_texts, 'label': all_labels})
    if args.model=='TinyXLMR':
        dataset = dataset.map(lambda examples: tokenize_function_xlmr(examples, tokenizer), batched=True)
    else:
        dataset = dataset.map(lambda examples: tokenize_function_bpe(examples, tokenizer), batched=True)

    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    # ------------------- K-Fold Setup -------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    fold_accuracies, fold_f1 = [], []

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold+1}/10 ---")
        
        train_subset = dataset.select(train_idx.tolist())
        valid_subset = dataset.select(valid_idx.tolist())

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(valid_subset, batch_size=args.batch_size)

        # ------------------- Model Initialization -------------------
        if args.model == 'BERT':
            # Initialize model with parameters from args
            vocab_size = 50000
            bert_model = BERT(
                vocab_size=vocab_size,
                d_model=args.d_model,
                n_layers=args.n_layers,
                heads=args.heads,
                dropout=args.dropout
            )
            model = BERTCLassifier(bert_model, vocab_size,num_labels=2)
        elif args.model == 'MixedXLM':
            # Load the vocab sets from the JSON file
            
            vocab_size = 50000
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
            
            model = MixedXLM_CLassifier(mixedxlm, vocab_size,num_labels=2)
        elif args.model == 'TinyXLMR':
            # Load the vocab sets from the JSON file
            vocab_size = tokenizer.vocab_size
            student_config = XLMRobertaConfig(
                num_hidden_layers=args.n_layers,  # M=4 layers
                hidden_size=args.d_model,      # d0=312
                intermediate_size=1200,  # d0i=1200
                num_attention_heads=args.heads,  # h=12 heads
                vocab_size=vocab_size,  # Assuming this is the size of the vocab
                max_position_embeddings=512,  # Or use the size you need
                output_attentions=True, 
                output_hidden_states=True ,
            )
            model = TinyXLMRobertaForSequenceClassification(student_config, fit_size=768,num_labels=2)
        #total_params = sum(p.numel() for p in model.parameters())
        #print(f"Total parameters: {total_params:,}")
        #print(model)
        if os.path.exists(args.model_ckpt):
            state_dict = torch.load(args.model_ckpt, map_location=device)
            # **Ensure compatibility by removing "output_layer" from state_dict if present**
            '''if "output_layer.weight" in state_dict:
                del state_dict["output_layer.weight"]
                del state_dict["output_layer.bias"]
            '''
            # Load the modified state_dict
            model.load_state_dict(state_dict['model_state_dict'], strict=False)  # strict=False allows missing keys
            print(f"✅ Loaded checkpoint: {args.model_ckpt}")

        model.to(device)

        

        # ------------------- Training -------------------
        # Optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        #print(device)
        # Training loop
        num_epochs = args.epochs
        best_accuracy=0
        best_f1=0
        if args.model == 'TinyXLMR':
           
            criterion = torch.nn.CrossEntropyLoss()
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Forward pass
                    outputs = model(input_ids, attention_mask=attention_mask)
                    
                    logits = outputs.logits  # or just outputs if your model directly returns logits
                
                    # Compute loss manually
                    loss = criterion(logits, labels)
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()

                # Print the training loss after each epoch
                print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}")

                
                model.eval()
                test_predictions, test_true_labels = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['label'].to(device)
                        
                        outputs = model(input_ids, attention_mask=attention_mask)
                    
                        logits = outputs.logits  # or just outputs if your model directly returns logits
                    
                        # Compute loss manually
                        #loss = criterion(logits, labels)
                        test_predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                        test_true_labels.extend(labels.cpu().numpy())
                test_accuracy = accuracy_score(test_true_labels, test_predictions)
                test_f1 = f1_score(test_true_labels, test_predictions, average='macro')
                if test_accuracy>best_accuracy: 
                    best_f1=test_f1
                    best_accuracy=test_accuracy
                print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


        else:  #bert and mixedXLM      
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    # Forward pass
                    outputs = model(input_ids, attention_mask=attention_mask,labels=labels)
                    loss = outputs['loss']
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()

                # Print the training loss after each epoch
                print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_loader)}")

                
                model.eval()
                test_predictions, test_true_labels = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['label'].to(device)
                        
                        outputs = model(input_ids, attention_mask=attention_mask)
                        logits = outputs['logits']
                        test_predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                        test_true_labels.extend(labels.cpu().numpy())

                test_accuracy = accuracy_score(test_true_labels, test_predictions)
                test_f1 = f1_score(test_true_labels, test_predictions, average='macro')
                if test_accuracy>best_accuracy: 
                    best_f1=test_f1
                    best_accuracy=test_accuracy
                print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
                
                scheduler.step()  # Update learning rate scheduler

        fold_accuracies.append(best_accuracy)
        fold_f1.append(best_f1)
        print(f"Best Test Accuracy: {best_accuracy * 100:.2f}%")

    
    # ------------------- Final Results -------------------
    average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    average_f1 = sum(fold_f1) / len(fold_f1)    
    print("\n==================")
    print(f"Average Accuracy over 10 folds: {average_accuracy * 100:.2f}%")
    print(f"Average F1-score over 10 folds: {average_f1 * 100:.2f}%")
    print("==================")


if __name__ == "__main__":
    main()
