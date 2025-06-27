import argparse
from datasets import Dataset, load_dataset
from tokenizers import Tokenizer as BPETokenizer
from transformers import AutoTokenizer
from tqdm import tqdm
from utils import merge_subwords


def tokenize_with_bpe(tokenizer_path, dataset_path, save_path, max_seq_length):
    print("=== Using BPE Tokenizer ===")
    tokenizer = BPETokenizer.from_file(tokenizer_path)
    PAD_TOKEN_ID = tokenizer.token_to_id("<pad>")
    MASKED_TOKEN_ID = tokenizer.token_to_id("<mask>")
    print("<pad>: ", PAD_TOKEN_ID)
    print("<mask>: ", MASKED_TOKEN_ID)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    input_ids = []
    current_tokens = []

    for line in tqdm(lines, desc="Tokenizing with BPE"):
        encoding = tokenizer.encode(line)
        line_tokens = encoding.ids

        if len(line_tokens) > max_seq_length:
            line_tokens = line_tokens[:max_seq_length]

        if len(current_tokens) + len(line_tokens) <= max_seq_length:
            current_tokens.extend(line_tokens)
        else:
            if len(current_tokens) < max_seq_length:
                current_tokens += [PAD_TOKEN_ID] * (max_seq_length - len(current_tokens))
            input_ids.append(current_tokens)
            current_tokens = line_tokens

    if current_tokens:
        if len(current_tokens) < max_seq_length:
            current_tokens += [PAD_TOKEN_ID] * (max_seq_length - len(current_tokens))
        input_ids.append(current_tokens)

    attention_mask = [
        [1 if token != PAD_TOKEN_ID else 0 for token in seq]
        for seq in input_ids
    ]

    dataset = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask})
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    dataset.save_to_disk(save_path)
    print(f"✅ BPE tokenized dataset saved to {save_path}")


def tokenize_with_xlmr(tokenizer_path, dataset_path, save_path, max_seq_length):
    print("=== Using XLM-R Tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    raw_dataset = load_dataset('text', data_files={'train': dataset_path})
    train_ds = raw_dataset['train']

    def preprocess_and_tokenize(batch):
        merged_lines = []
        current_line = ""

        for line in batch['text']:
            tentative_line = current_line + " " + line if current_line else line
            tokenized_tentative = tokenizer(
                tentative_line,
                truncation=True,
                max_length=max_seq_length,
                add_special_tokens=True,
            )

            if len(tokenized_tentative['input_ids']) <= max_seq_length - 2:
                current_line = tentative_line
            else:
                if current_line:
                    merged_lines.append(current_line.strip())
                current_line = line

        if current_line:
            merged_lines.append(current_line.strip())

        tokenized = tokenizer(
            merged_lines,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
        )

        return tokenized

    processed_dataset = train_ds.map(
        preprocess_and_tokenize,
        batched=True,
        remove_columns=["text"],
        num_proc=4
    )

    processed_dataset.save_to_disk(save_path)
    print(f"✅ XLM-R tokenized dataset saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess and tokenize dataset for pretraining")
    parser.add_argument("--tokenizer_type", type=str, choices=["bpe", "xlmr"], required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=256)
    args = parser.parse_args()

    if args.tokenizer_type == "bpe":
        tokenize_with_bpe(args.tokenizer_path, args.dataset_path, args.save_path, args.max_seq_length)
    elif args.tokenizer_type == "xlmr":
        tokenize_with_xlmr(args.tokenizer_path, args.dataset_path, args.save_path, args.max_seq_length)


if __name__ == "__main__":
    main()
