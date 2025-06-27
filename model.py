
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, XLMRobertaConfig, XLMRobertaModel , AutoTokenizer, PreTrainedModel
from datasets import load_dataset
import math


### embedding
class PositionalEmbedding(torch.nn.Module):

    def __init__(self, d_model, max_len=128):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        for pos in range(max_len):
            # for each dimension of the each position
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        # include the batch size
        self.pe = pe.unsqueeze(0)
        # self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe

class BERTEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, seq_len=256, dropout=0.1,device='cuda'):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0).to(device)
        self.segment = torch.nn.Embedding(3, embed_size, padding_idx=0).to(device)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len).to(device)
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.device = device


    def forward(self, sequence): #, segment_label
        #print(self.token.device, self.token.device)
        x = self.token(sequence).to(self.device) + self.position(sequence).to(self.device) #+ self.segment(segment_label)
        return self.dropout(x)
    



### attention layers
class MultiHeadedAttention(torch.nn.Module):

    def __init__(self, heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads
        self.dropout = torch.nn.Dropout(dropout)

        self.query = torch.nn.Linear(d_model, d_model)
        self.key = torch.nn.Linear(d_model, d_model)
        self.value = torch.nn.Linear(d_model, d_model)
        self.output_linear = torch.nn.Linear(d_model, d_model)

        #self.pad_token_id = 0

    def forward(self, query, key, value, mask):
        """
        query, key, value of shape: (batch_size, max_len, d_model)
        mask of shape: (batch_size, 1, 1, max_words)
        """
        # (batch_size, max_len, d_model)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)

        # (batch_size, max_len, d_model) --> (batch_size, max_len, h, d_k) --> (batch_size, h, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        # (batch_size, h, max_len, d_k) matmul (batch_size, h, d_k, max_len) --> (batch_size, h, max_len, max_len)
        scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / math.sqrt(query.size(-1))

        # fill 0 mask with super small number so it wont affect the softmax weight
        # (batch_size, h, max_len, max_len)
        #print(scores.shape, mask.shape)
        scores = scores.masked_fill(mask == 0 , float('-inf'))

        # (batch_size, h, max_len, max_len)
        # softmax to put attention weight for all non-pad tokens
        # max_len X max_len matrix of attention
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # (batch_size, h, max_len, max_len) matmul (batch_size, h, max_len, d_k) --> (batch_size, h, max_len, d_k)
        context = torch.matmul(weights, value)

        # (batch_size, h, max_len, d_k) --> (batch_size, max_len, h, d_k) --> (batch_size, max_len, d_model)
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)

        # (batch_size, max_len, d_model)
        return self.output_linear(context)

class FeedForward(torch.nn.Module):
    "Implements FFN equation"

    def __init__(self, d_model, middle_dim=2048, dropout=0.1):
        super(FeedForward, self).__init__()

        self.fc1 = torch.nn.Linear(d_model, middle_dim)
        self.fc2 = torch.nn.Linear(middle_dim, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out

class EncoderLayer(torch.nn.Module):
    def __init__(
        self,
        d_model=512,
        heads=8,
        feed_forward_hidden=512 * 4,
        dropout=0.1
        ):
        super(EncoderLayer, self).__init__()
        self.layernorm = torch.nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadedAttention(heads, d_model)
        self.feed_forward = FeedForward(d_model, middle_dim=feed_forward_hidden)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, embeddings, mask):
        # embeddings: (batch_size, max_len, d_model)
        # encoder mask: (batch_size, 1, 1, max_len)
        # result: (batch_size, max_len, d_model)
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        # residual layer
        interacted = self.layernorm(interacted + embeddings)
        # bottleneck
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        return encoded


class BERT(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, d_model=512, n_layers=6, heads=8, dropout=0.1,device='cuda'):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.pad_token_id = 0
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_model * 4
        self.device = device
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=d_model,device=device)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model, heads, d_model * 4, dropout) for _ in range(n_layers)])

    def forward(self, x ,attention_mask=None):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)]
        x = x.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        else:
            mask = (x != self.pad_token_id).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        if attention_mask is not None:
            x *= (attention_mask).unsqueeze(-1).to(x.dtype)  # Ensure masked positions remain unchanged
        
        return x

class NextSentencePrediction(torch.nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 2)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # use only the first token which is the [CLS]
        return self.softmax(self.linear(x[:, 0]))

class MaskedLanguageModel(torch.nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class BERTLM(torch.nn.Module):
    """
    BERT Language Model
    Masked Language Model
    """

    def __init__(self, bert: BERT, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.d_model, vocab_size)

    def forward(self, x, attention_mask=None):
        x = self.bert(x,attention_mask)
        return self.mask_lm(x)

class BERTCLassifier(torch.nn.Module):
    """
    BERT Language Model
    Classification Model 
    """

    def __init__(self, bert: BERT, vocab_size,num_labels=2):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert.d_model, num_labels)
        

    def forward(self, x, attention_mask=None,labels=None):
        outputs = self.bert(x,attention_mask)
        pooled_output = outputs[:, 0, :]  # First token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {'loss': loss, 'logits': logits}
        





class MixedXLMEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, ms_vocab, eng_vocab, chi_vocab, 
                 embed_size, seq_len=256, dropout=0.1,device='cuda',
                 use_language_emb=True,
                 debug=False,
                 ):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """

        super().__init__()
        self.embed_size = embed_size
        # (m, seq_len) --> (m, seq_len, embed_size)
        # padding_idx is not updated during training, remains as fixed pad (0)
        self.token = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0).to(device)
        self.language = torch.nn.Embedding(4, embed_size, padding_idx=0).to(device)
        self.position = PositionalEmbedding(d_model=embed_size, max_len=seq_len).to(device)
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.device = device
        self.use_language_emb = use_language_emb
        self.debug=debug
        self.ms_vocab = torch.tensor(list(ms_vocab), device=self.device)
        self.eng_vocab = torch.tensor(list(eng_vocab), device=self.device)
        self.chi_vocab = torch.tensor(list(chi_vocab), device=self.device)

    def generate_language_embeddings(self, tokens):
            """
            Function to generate language embeddings based on the token's vocabulary.
            :param tokens: Tensor of token IDs (batch_size, seq_length)
            :return: Tensor of language IDs (batch_size, seq_length)
            """
            batch_size, seq_length = tokens.size()
            
            # Initialize language_ids tensor with the default value for "Other" (3)
            language_ids = torch.full(tokens.shape, 0, dtype=torch.long, device=tokens.device)

            # Create boolean masks for each language
            ms_mask = torch.isin(tokens, self.ms_vocab)
            eng_mask = torch.isin(tokens, self.eng_vocab)
            chi_mask = torch.isin(tokens, self.chi_vocab)

            # Assign language IDs based on masks
            
            
            language_ids[chi_mask] = 1 # Chinese
            language_ids[eng_mask] = 2  # English
            language_ids[ms_mask] = 3  # Malay
            return language_ids
    
    def forward(self, sequence): 
        
        if self.use_language_emb:
            language_ids = self.generate_language_embeddings(sequence)
        else:
            language_ids = torch.full(sequence.shape, 0, dtype=torch.long, device=sequence.device)
        lang_embeds = self.language(language_ids).to(self.device)
        position_embeds = self.position(sequence).to(self.device)
        token_embeds = self.token(sequence).to(self.device)
        x = token_embeds + position_embeds + lang_embeds
        if getattr(self, 'debug', False):
            print(f"\n{'='*20} DEBUG INFO {'='*20}")
            print(f"Input IDs:\n{sequence}")
            
            print(f"Word Embeddings Shape: {token_embeds.shape}")
            ms_count = (language_ids == 3).sum().item()
            eng_count = (language_ids == 2).sum().item()
            chi_count = (language_ids == 1).sum().item()
            other_count = (language_ids == 0).sum().item()
            print(f"MS Tokens: {ms_count}, ENG Tokens: {eng_count}, CHI Tokens: {chi_count}, other Tokens: {other_count}")

            print(f"Position Embeddings Shape: {position_embeds.shape}")
            
            # Reverse map tokens if tokenizer is available
            #print(self.tokenizer)
            if getattr(self, 'tokenizer', False):
                for i in range(batch_size):
                    token_ids = padded_input_ids[i].tolist()
                    #tokens = self.tokenizer.decode(token_ids) 
                    tokens = [self.tokenizer.id_to_token(tid) for tid in token_ids]
                    #tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

                    print(f"Batch {i+1} Tokens: {merge_subwords(' '.join(tokens))}")
            print(f"{'='*50}\n")
        return self.dropout(x)
    




class MixedXLM(torch.nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, ms_vocab, eng_vocab, chi_vocab,
                 d_model=512, n_layers=6, heads=8, dropout=0.1,device='cuda',
                 debug=False):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.heads = heads
        self.pad_token_id = 0
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_model * 4
        self.device = device
        self.debug = debug
        # embedding for BERT, sum of positional, segment, token embeddings
        self.embedding = MixedXLMEmbedding(vocab_size, 
                                      ms_vocab, eng_vocab, chi_vocab,
                                      embed_size=d_model,device=device,
                                      use_language_emb=True,debug=self.debug,)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = torch.nn.ModuleList(
            [EncoderLayer(d_model, heads, d_model * 4, dropout) for _ in range(n_layers)])

    def forward(self, x ,attention_mask=None):
        # attention masking for padded token
        # (batch_size, 1, seq_len, seq_len)]
        x = x.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        else:
            mask = (x != self.pad_token_id).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x)

        # running over multiple transformer blocks
        for encoder in self.encoder_blocks:
            x = encoder.forward(x, mask)
        return x




class MixedXLM_LM(torch.nn.Module):
    """
    BERT Language Model
    Masked Language Model
    """

    def __init__(self, MixedXLM: MixedXLM, vocab_size):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.MixedXLM = MixedXLM
        self.mask_lm = MaskedLanguageModel(self.MixedXLM.d_model, vocab_size)

    def forward(self, x, attention_mask=None):
        x = self.MixedXLM(x,attention_mask)
        return self.mask_lm(x)

class MixedXLM_CLassifier(torch.nn.Module):
    """
    BERT Language Model
    Classification Model 
    """

    def __init__(self, MixedXLM: MixedXLM, vocab_size,num_labels=2):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.MixedXLM = MixedXLM
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(MixedXLM.d_model, num_labels)
        

    def forward(self, x, attention_mask=None,labels=None):
        outputs = self.MixedXLM(x,attention_mask)
        pooled_output = outputs[:, 0, :]  # First token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {'loss': loss, 'logits': logits}
        

class TinyXLMRobertaForPreTraining(PreTrainedModel):
    def __init__(self, config, fit_size=768):
        super(TinyXLMRobertaForPreTraining, self).__init__(config)
        self.roberta = XLMRobertaModel(config)  # Enable attention output
        #self.cls = XLMRobertaModel(config)
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)
        #self.apply(self.init_weights)

    def forward(self, input_ids, attention_mask=None, output_attentions=True, output_hidden_states=True):
        # Forward pass with attention and hidden states output
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        #print(outputs)
        # Extract the hidden states and attentions
        hidden_states = outputs.hidden_states  # Tuple of (layer_0, layer_1, ..., layer_N)
        attentions = outputs.attentions        # Tuple of attention matrices from each layer
        
        # Apply the fit_dense transformation to all hidden states
        
        transformed_hidden_states = [self.fit_dense(layer) for layer in hidden_states]

        # Return as an object with attributes for compatibility
        output = type('ModelOutput', (object,), {})()
        output.attentions = attentions
        output.hidden_states = transformed_hidden_states
        
        return output
    


class TinyXLMRobertaForSequenceClassification(PreTrainedModel):
    def __init__(self, config, fit_size=312, num_labels=2):
        super(TinyXLMRobertaForSequenceClassification, self).__init__(config)
        
        # Initialize XLM-Roberta model with attention outputs
        self.roberta = XLMRobertaModel(config)  # Enable attention output
        self.cls = nn.Linear(config.hidden_size, num_labels)  # Classification head
        self.fit_dense = nn.Linear(config.hidden_size, fit_size)  # Optional hidden size reduction
        #self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # Forward pass with attention outputs
        outputs = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        
        # Extract the last hidden states and attention weights
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        attention_output = outputs[1]  # Attention weights
        
        # Apply the dense transformation to each hidden layer for fit_size
        tmp = []
        for s_id, sequence_layer in enumerate(sequence_output):
            tmp.append(self.fit_dense(sequence_layer))
        hidden_output = tmp
        
        # Use the final hidden state of the [CLS] token for classification (first token)
        #print(sequence_output.shape)
        pooled_output = sequence_output[:, 0]  # The representation for the first token [CLS]

        # Pass through the classification head
        logits = self.cls(pooled_output)
        output = type('ModelOutput', (object,), {})()
        output.attention_output = attention_output
        output.hidden_output = hidden_output
        output.logits = logits

        return output
    

