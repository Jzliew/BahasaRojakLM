import os

def merge_subwords(tokens, continuing_subword_prefix="_<w>", end_of_word_suffix="</w>"):
    """Merge subwords back into words by removing the prefixes and suffixes."""
    merged_tokens = []
    temp_token = ""
    tokens = tokens.split(' ')
    for token in tokens:
        # Remove continuing subword prefix
        
        token = token.lstrip(continuing_subword_prefix)
        # Remove end of word suffix
        #token = token.rstrip(end_of_word_suffix)
        #print(token)
        if temp_token:
            temp_token += token  # Continue appending the subword to the word
        else:
            temp_token = token  # Start a new word
        
        # If we reach the last token (or it's a full word), add it to the list
        if temp_token and (token.endswith(end_of_word_suffix) or token == tokens[-1]):
            merged_tokens.append(temp_token.rstrip(end_of_word_suffix))
            temp_token = ""  # Reset for the next word
    
    return " ".join(merged_tokens)

def manage_checkpoints(checkpoint_dir, current_step):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    regular_checkpoints = [f for f in checkpoint_files if f.startswith('checkpoint_step')]
    best_checkpoints = [f for f in checkpoint_files if f.startswith('best_model_step')]
    regular_checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)
    best_checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]), reverse=True)

    if len(regular_checkpoints) > 2:
        for old_checkpoint in regular_checkpoints[2:]:
            os.remove(os.path.join(checkpoint_dir, old_checkpoint))
            print(f"Deleted old checkpoint: {old_checkpoint}")

    if len(best_checkpoints) > 2:
        for old_checkpoint in best_checkpoints[2:]:
            os.remove(os.path.join(checkpoint_dir, old_checkpoint))
            print(f"Deleted old checkpoint: {old_checkpoint}")

    print(f"Retained checkpoints: {regular_checkpoints[:2]} + {best_checkpoints}")
