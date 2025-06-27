import os
import json
import argparse

from transformers import XLMRobertaConfig, XLMRobertaModel , AutoTokenizer, PreTrainedModel
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW , get_scheduler

from datasets import load_dataset, Dataset, load_from_disk, DatasetDict
from tqdm import tqdm
from model import * 
from loss import * 
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT MLM model with XLM tokenizer")


    parser.add_argument('--dataset_path', type=str, default="./data/xlmr_tokenized", help='Path to the dataset folder')
    parser.add_argument('--tokenizer_path', type=str, default='./models/xlmr', help='Path to the tokenizer folder')
    parser.add_argument('--result_dir', type=str, default='./result/xlmr_test/', help='Directory to save checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=40, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--patience', type=int, default=20000, help='Early stopping patience')
    parser.add_argument('--validation_interval', type=int, default=5000, help='Validation interval (steps)')
    parser.add_argument('--log_step', type=int, default=500, help='Logging step interval')
    parser.add_argument('--best_model_interval', type=int, default=50, help='Interval for saving best model checkpoints')

    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')

    parser.add_argument("--d_model", type=int, default=516, help='Hidden state size of model')
    parser.add_argument("--intermediate_size", type=int, default=1200, help='FFN size of transformer')
    parser.add_argument("--n_layers", type=int, default=6, help='Number of transformer layers in model')
    parser.add_argument("--heads", type=int, default=12, help='Number of attention heads in MAH')
    parser.add_argument("--dropout", type=float, default=0.1, help='Dropout ratio')

    parser.add_argument('--teacher_model_path', type=str, default='./models/xlmr', help='Path to the teacher model')
    

    args = parser.parse_args()
    return args

args = parse_args()

# Device setup
device = args.device
# Load dataset
full_dataset = load_from_disk(args.dataset_path)
split_dataset = full_dataset.train_test_split(test_size=0.01)
tokenized_dataset = DatasetDict({
    "train": split_dataset["train"],
    "validation": split_dataset["test"]
})
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
batch_size = args.batch_size
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=batch_size, shuffle=True)
validation_dataloader = DataLoader(tokenized_dataset["validation"], batch_size=batch_size)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

# Load teacher model
teacher_model_path = args.teacher_model_path
teacher_config = XLMRobertaConfig.from_pretrained(teacher_model_path)
teacher_model = XLMRobertaModel.from_pretrained(teacher_model_path, config=teacher_config, ignore_mismatched_sizes=True)

# Define student model config with the smaller dimensions
student_config = XLMRobertaConfig(
    num_hidden_layers=args.n_layers,
    hidden_size=args.d_model,
    intermediate_size=args.intermediate_size,
    num_attention_heads=args.heads,
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=args.max_seq_length,
    output_attentions=True,
    output_hidden_states=True,
)

student_model = TinyXLMRobertaForPreTraining(student_config, fit_size=768)


# Load pre-trained teacher model
# Move models to device
teacher_model.to(device)
student_model.to(device)
#roberta.to(device)
print(student_model)
total_params = sum(p.numel() for p in student_model.roberta.parameters())
print(f"Total student_model parameters: {total_params:,}")


# Set models to training mode
teacher_model.eval()  # Teacher is not being trained, so keep it in eval mode
student_model.train()


# Optimizer and Scheduler
optimizer = AdamW(student_model.parameters(), lr=args.learning_rate)
num_epochs = args.epochs
num_training_steps = num_epochs * len(train_dataloader)

# For FP16 training
scaler = torch.cuda.amp.GradScaler()

# Directory for saving results
result_dir = args.result_dir
checkpoint_dir = os.path.join(result_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

log_step = args.log_step  # Log every 50 steps
validation_interval = args.validation_interval  # Validate every 500 steps
log_file = result_dir+"training_log.json"
log_entries = []
log_loss = 0
attention_loss_accumulated = 0.0  # Accumulated attention loss
hidden_state_loss_accumulated = 0.0  # Accumulated hidden state loss
current_step = 0
best_val_loss = float('inf')
no_improvement_count = 0
improvement_count = 0
patience = args.patience  # For early stopping
best_model_interval = args.best_model_interval




# Optimizer and Scheduler
optimizer = AdamW(student_model.parameters(), lr=args.learning_rate)
num_epochs = args.epochs
num_training_steps = num_epochs * len(train_dataloader)

# For FP16 training
scaler = torch.cuda.amp.GradScaler()






for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss = 0.0
    log_loss = 0.0
    # Training Loop
    student_model.train()
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
 
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                output_attentions=True, 
                output_hidden_states=True
            )

        with torch.cuda.amp.autocast():  # FP16 Training
            student_outputs = student_model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
            )
           
            
            attention_loss, hidden_state_loss = distillation_loss(student_outputs, teacher_outputs ,device)
            loss = attention_loss + hidden_state_loss
            
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()



        # Accumulate the attention and hidden state loss
        attention_loss_accumulated += attention_loss.item()
        hidden_state_loss_accumulated += hidden_state_loss.item()

        train_loss += loss.item()
        log_loss += loss.item()  # Accumulate loss for logging

        current_step += 1

        # Log at regular intervals
        if current_step % log_step == 0:
            # Calculate the average losses for attention and hidden state
            avg_attention_loss = attention_loss_accumulated / log_step
            avg_hidden_state_loss = hidden_state_loss_accumulated / log_step
            avg_log_loss = log_loss / log_step

            tqdm.write(f"Step {current_step}, Attention Loss: {avg_attention_loss:.4f}, "
                       f"Hidden State Loss: {avg_hidden_state_loss:.4f}, Total Loss: {avg_log_loss:.4f}")

            # Logging the required data
            log_entry = {
                "epoch": (epoch + current_step / len(train_dataloader)) / num_epochs,
                "step": current_step,
                "attention_loss": avg_attention_loss,
                "hidden_state_loss": avg_hidden_state_loss,
                "total_loss": avg_log_loss,
                "learning_rate": optimizer.param_groups[0]['lr'],
                "gradient_norm": None  # You can compute gradient norm here if needed
            }
            log_entries.append(log_entry)

            # Save the log to JSON file after each log step
            with open(log_file, 'w') as f:
                json.dump(log_entries, f, indent=4)

            # Reset accumulated losses after logging
            attention_loss_accumulated = 0.0
            hidden_state_loss_accumulated = 0.0
            log_loss = 0  # Reset log_loss after logging

        # Perform validation at set intervals
        if (current_step + 1) % validation_interval == 0:
            student_model.eval()  # Set model to evaluation mode
            total_val_loss = 0  # Track total validation loss

            with torch.no_grad():  # No gradient computation during validation
                for val_step, val_batch in tqdm(enumerate(validation_dataloader), total=len(validation_dataloader), desc=f"Validation at Step {current_step + 1}"):
                    input_ids_val = val_batch['input_ids'].to(device)
                    attention_mask_val = val_batch['attention_mask'].to(device)

                    teacher_outputs = teacher_model(
                        input_ids=input_ids_val, 
                        attention_mask=attention_mask_val, 
                        output_attentions=True, 
                        output_hidden_states=True
                    )
                    student_outputs = student_model(
                        input_ids=input_ids_val, 
                        attention_mask=attention_mask_val, 
                    )

                    attention_loss, hidden_state_loss = distillation_loss(student_outputs, teacher_outputs,device)
                    val_loss = attention_loss + hidden_state_loss
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(validation_dataloader)
            print(f"Step {current_step + 1}, Average Validation Loss: {avg_val_loss:.4f}")


            checkpoint_filename = f'{checkpoint_dir}/checkpoint_step_{current_step}.pth'
            torch.save({
                'epoch': epoch,
                'step': current_step,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # For AMP training
                #'scheduler_state_dict': scheduler.state_dict(),  # Save the LR scheduler state
            }, checkpoint_filename)
            print(f"Checkpoint saved as {checkpoint_filename}")

            print(f"Improvement count: {improvement_count}")
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement_count = 0
                
                # Save the best model and optimizer state
                checkpoint_filename = f'{checkpoint_dir}/best_model_step_{current_step}.pth'
                torch.save({
                    'epoch': epoch,
                    'step': current_step,
                    'model_state_dict': student_model.state_dict(),
                    'validation_loss': best_val_loss
                }, checkpoint_filename)
                print(f"New best model saved with Validation Loss: {best_val_loss:.4f}")
            else:
                no_improvement_count += 1
                print(f"No improvement. Best val loss: {best_val_loss} ,Early stopping counter: {no_improvement_count}/{patience}")
            
            # Model saved at regular intervals
            if improvement_count >= best_model_interval:
                improvement_count = 0
                checkpoint_filename = f'{checkpoint_dir}/best_model_interval_{current_step}.pth'
                torch.save({
                    'epoch': epoch,
                    'step': current_step,
                    'model_state_dict': student_model.state_dict(),
                    'validation_loss': best_val_loss
                }, checkpoint_filename)
                print(f"Best model saved after {best_model_interval} improvements: {checkpoint_filename}")
            
            manage_checkpoints(checkpoint_dir, current_step)
            improvement_count += 1






