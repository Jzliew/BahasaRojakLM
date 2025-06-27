
import torch.nn.functional as F
import torch
from transformers import AdamW


def distillation_loss(student_outputs, teacher_outputs, device,
                      temperature=1.0, attention_loss_weight=1.0, hidden_state_loss_weight=1.0):
    """
    Compute the total distillation loss using MSE between:
      - Student and Teacher attentions
      - Student and Teacher hidden states (already transformed in the student model)
    """
    # Attention Distillation Loss
    attention_loss = 0.0
    num_student_layers = len(student_outputs.attentions)
    num_teacher_layers = len(teacher_outputs.attentions)
    teacher_layer_indices = torch.linspace(0, num_teacher_layers - 1, steps=num_student_layers).long()
    #print(teacher_layer_indices)
    
    num_heads = student_outputs.attentions[0].size(1)
    teacher_attentions = [att.detach() for att in teacher_outputs.attentions]
    for student_layer_idx, teacher_layer_idx in enumerate(teacher_layer_indices):
        student_attention = student_outputs.attentions[student_layer_idx]
        teacher_attention = teacher_attentions[teacher_layer_idx]
        
        if teacher_layer_idx >= num_teacher_layers:
            break
        
        
        teacher_attention = [teacher_att.detach() for teacher_att in teacher_outputs.attentions]
        teacher_attention = teacher_attention[teacher_layer_idx] #/ temperature
        #print(teacher_attention.sum(dim=-1))
        layer_attention_loss = 0.0
        for head in range(num_heads):
            student_head_attention = student_attention[:, head, :, :]
            teacher_head_attention = teacher_attention[:, head, :, :]
            
            student_head_attention = torch.where(student_head_attention <= -1e2, torch.zeros_like(student_head_attention).to(device),
                                              student_head_attention)
            teacher_head_attention = torch.where(teacher_head_attention <= -1e2, torch.zeros_like(teacher_head_attention).to(device),
                                              teacher_head_attention)
            

            head_loss = F.mse_loss(student_head_attention, teacher_head_attention, reduction='mean')
            
            
            layer_attention_loss += head_loss*1000
        
        attention_loss += layer_attention_loss / num_heads  # Average over heads

    # Hidden State Distillation Loss
    hidden_state_loss = 0.0
    
    #print(len(teacher_outputs.hidden_states))
    #print(teacher_outputs.hidden_states[1].shape)
    
    
    num_student_layers = len(student_outputs.hidden_states)
    num_teacher_layers = len(teacher_outputs.hidden_states)

    teacher_hidden_states = [h.detach() for h in teacher_outputs.hidden_states]
    teacher_layer_indices = torch.linspace(0, num_teacher_layers - 1, steps=num_student_layers).long()
    #print(teacher_layer_indices)
    for student_layer_idx, teacher_layer_idx in enumerate(teacher_layer_indices):
        student_hidden = student_outputs.hidden_states[student_layer_idx]
        teacher_hidden = teacher_hidden_states[teacher_layer_idx]
        #print(student_layer_idx , teacher_layer_idx)
        if teacher_layer_idx >= num_teacher_layers:
            break
        
        
        
        layer_hidden_loss = F.mse_loss(student_hidden, teacher_hidden, reduction='mean')
        hidden_state_loss += layer_hidden_loss

    # Combine attention loss and hidden state loss
    #total_loss = (attention_loss_weight * attention_loss) + (hidden_state_loss_weight * hidden_state_loss)
    
    return attention_loss, hidden_state_loss


