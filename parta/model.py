import torch
import torch.nn as nn
from typing import Any, Dict, List

# Monkey-patch torch.load to default weights_only=False for PyTorch 2.6+ compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

def _get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        self.config = config
        self.load_config()
        super().__init__()
        #will initialize the weights with hardcoded names in the set_weights func via register buffers -> need to change this if i want to be able to train the model
        self.word_embeddings = None
        self.L = None
        
        #n_transformer_blocks
        self.x_l = None # will be initialized in forward when L is known
        self.z_l_1 = None
        self.z_l_2 = None
        self.logits = None
        self.probs = None
        
        
        
    def load_config(self):
        self.d_model = self.config["d_model"]
        self.n_heads = self.config["n_heads"]
        self.d_head = self.config["d_head"]
        self.n_layers = self.config["n_layers"]
        self.vocab_size = self.config["vocab_size"]
        self.mode = self.config["mode"]
        self.tau = self.config.get("tau", None)

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        # Embeddings
        self.register_buffer('W_vocab', weights["W_vocab"].T)      # (vocab_size, d_model)
        self.register_buffer('W_devocab', weights["W_devocab"])    # (d_model, vocab_size)
        
        # Per-layer weights
        for l in range(1, self.n_layers + 1):
            # Attention: Q, K, V per head + output projection
            for k in range(1, self.n_heads + 1):
                self.register_buffer(f'W_{l}_Q_{k}', weights[f"W_{l}_Q_{k}"])
                self.register_buffer(f'W_{l}_K_{k}', weights[f"W_{l}_K_{k}"])
                self.register_buffer(f'W_{l}_V_{k}', weights[f"W_{l}_V_{k}"])
            self.register_buffer(f'W_{l}_O', weights[f"W_{l}_O"])
            
            # Feedforward
            self.register_buffer(f'W_{l}_up', weights[f"W_{l}_up"])
            self.register_buffer(f'b_{l}_up', weights[f"b_{l}_up"])
            self.register_buffer(f'W_{l}_down', weights[f"W_{l}_down"])
            self.register_buffer(f'b_{l}_down', weights[f"b_{l}_down"])
            
            # Layer norms
            self.register_buffer(f'beta_{l}_1', weights[f"beta_{l}_1"])
            self.register_buffer(f'gamma_{l}_1', weights[f"gamma_{l}_1"])
            self.register_buffer(f'beta_{l}_2', weights[f"beta_{l}_2"])
            self.register_buffer(f'gamma_{l}_2', weights[f"gamma_{l}_2"])
        
        self.register_buffer('beta_final', weights["beta_final"])
        self.register_buffer('gamma_final', weights["gamma_final"])
    
    def set_weights_randomly(self):
        #do now usign nn.parameters so tthat we can train
        self.W_vocab = nn.Parameter(torch.empty(self.vocab_size, self.d_model))
        self.W_devocab = nn.Parameter(torch.empty(self.d_model, self.vocab_size))
        
        for l in range(1, self.n_layers + 1):
            for k in range(1, self.n_heads + 1):
                setattr(self, f'W_{l}_Q_{k}', nn.Parameter(torch.empty(self.d_head, self.d_model)))
                setattr(self, f'W_{l}_K_{k}', nn.Parameter(torch.empty(self.d_head, self.d_model)))
                setattr(self, f'W_{l}_V_{k}', nn.Parameter(torch.empty(self.d_head, self.d_model)))
            setattr(self, f'W_{l}_O', nn.Parameter(torch.empty(self.d_model, self.d_model)))
            setattr(self, f'W_{l}_up', nn.Parameter(torch.empty(self.d_model, self.d_model * 4)))
            setattr(self, f'b_{l}_up', nn.Parameter(torch.empty(self.d_model * 4)))
            setattr(self, f'W_{l}_down', nn.Parameter(torch.empty(self.d_model * 4, self.d_model)))
            setattr(self, f'b_{l}_down', nn.Parameter(torch.empty(self.d_model)))
            setattr(self, f'beta_{l}_1', nn.Parameter(torch.empty(self.d_model)))
            setattr(self, f'gamma_{l}_1', nn.Parameter(torch.empty(self.d_model)))
            setattr(self, f'beta_{l}_2', nn.Parameter(torch.empty(self.d_model)))
            setattr(self, f'gamma_{l}_2', nn.Parameter(torch.empty(self.d_model)))
            setattr(self, f'beta_{l}_3', nn.Parameter(torch.empty(self.d_model)))
            setattr(self, f'gamma_{l}_3', nn.Parameter(torch.empty(self.d_model)))
            setattr(self, f'beta_{l}_4', nn.Parameter(torch.empty(self.d_model)))
            setattr(self, f'gamma_{l}_4', nn.Parameter(torch.empty(self.d_model)))

            
        self.beta_final = nn.Parameter(torch.empty(self.d_model))
        self.gamma_final = nn.Parameter(torch.empty(self.d_model))
        # Initialize parameters (e.g., with Xavier initialization)
        
        # self.word_embeddings = nn.Parameter(torch.empty(self.vocab_size, self.d_model))
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the model. The output should be a tensor of shape (T, |Vocab|).

        Parameters:
            - input_ids: A tensor of shape (batch_size, sequence_len) containing token IDs.
            - attention_mask: A tensor of shape (batch_size, sequence_len) containing 1s for valid tokens and 0s for padding.

        Returns:
            - A tensor of shape (batch_size, sequence_len, vocab_size) containing the logits for each token in the vocabulary.
            Logits are the raw, unnormalized scores output by the model, which can be converted to probabilities using a softmax function.
        """
        self.encode(input_ids, attention_mask)   # this function finds the word embeddings and the positional embeddings
        self.n_transformer_blocks()     # this function runs a for loop for n transformer blocks, and update the hidden states in each block
        self.final_norm()       # this function applies the final layer norm
        self.devocab()          # converts the final hidden states to logits over the vocabulary
        self.find_prob()        # applies softmax to obtain the probabilities
        return self.logits
        # raise NotImplementedError("Implement forward as described in assignment document")
        
    def encode(self, input_ids, attention_mask):
        self.apply_word_embeddings(input_ids, attention_mask)
        self.apply_positional_embeddings(attention_mask)
        self.x_l = self.word_embeddings
        self.attention_mask = attention_mask
        pass
    
    def n_transformer_blocks(self):
        for i in range(1, self.n_layers+1):
            self.apply_transformer_block(i) # this function applies the i-th transformer block and updates the hidden states accordingly
        pass
    
    def final_norm(self):
        self.x_l = self.layer_norm(self.x_l, self.beta_final, self.gamma_final)
    
    def devocab(self):
        # self.x_l is of shape (batch_size, sequence_len, d_model)
        # self.W_devocab is of shape (d_model, vocab_size)
        self.logits = torch.matmul(self.x_l, self.W_devocab) #projected on to vocab space
        pass
    
    def find_prob(self):
        self.probs = torch.softmax(self.logits, dim=-1)
        pass
    
    def apply_word_embeddings(self, input_ids, attention_mask):
        # input_ids is of shape (batch_size, sequence_len)
        # attention_mask is of shape (batch_size, sequence_len)
        self.word_embeddings = self.W_vocab[input_ids] #dim will be (batch_size, sequence_len, d_model)
        self.L = input_ids.shape[1]
        pass
    
    def apply_positional_embeddings(self, attention_mask):
        # self.word_embeddings is of shape (batch_size, sequence_len, d_model)
        # attention_mask is of shape (batch_size, sequence_len)
        pos_range = torch.arange(self.L, device=self.word_embeddings.device).view(1, -1, 1)
        i_range = torch.arange(self.d_model, device=self.word_embeddings.device).view(1, 1, -1)
        pos_emb = self.comp_pos_emb(pos_range, i_range)
        self.word_embeddings = self.word_embeddings + pos_emb
        # attn_mask_exp = attention_mask.unsqueeze(-1)
        self.word_embeddings = self.word_embeddings 
        pass
    def comp_pos_emb(self, position, i):
        denom = torch.pow(10000.0, ((i//2)*2) / self.d_model)
        sin = torch.sin(position / denom)
        cos = torch.cos(position / denom)
        return torch.where(i % 2 == 0, sin, cos)
    
    
    def apply_transformer_block(self, l):
        self.apply_layer_norm(l, 1)
        self.apply_attention(l)
        self.apply_layer_norm(l, 3)
        self.add(1)
        self.apply_layer_norm(l, 2)
        self.apply_layer_norm(l, 4)
        self.apply_up_proj(l)
        self.apply_swiglu()
        self.apply_down_proj(l)
        self.add(2)
        
    def apply_layer_norm(self, l, part):
        if part == 1:
            beta = getattr(self, f'beta_{l}_1')
            gamma = getattr(self, f'gamma_{l}_1')
            self.z_l_1 = self.layer_norm(self.x_l, beta, gamma)
        elif part == 2:
            beta = getattr(self, f'beta_{l}_2')
            gamma = getattr(self, f'gamma_{l}_2')
            self.z_l_2 = self.layer_norm(self.x_l, beta, gamma)
        elif part == 3:
            beta = getattr(self, f'beta_{l}_3')
            gamma = getattr(self, f'gamma_{l}_3')
            self.z_l_1 = self.layer_norm(self.z_l_1, beta, gamma)
        elif part == 4:
            beta = getattr(self, f'beta_{l}_4')
            gamma = getattr(self, f'gamma_{l}_4')
            self.z_l_2 = self.layer_norm(self.z_l_2, beta, gamma)
                 
        pass
    
    
    
    def add(self, part):
        if part == 1:
            self.x_l = self.x_l + self.z_l_1
        else:
            self.x_l = self.x_l + self.z_l_2
        pass
    
    def apply_up_proj(self, l):
        self.z_l_2 = torch.matmul(self.z_l_2, getattr(self, f'W_{l}_up')) + getattr(self, f'b_{l}_up')
        pass
    
    def apply_swiglu(self):
        self.z_l_2 = torch.nn.functional.swiglu(self.z_l_2)
        pass
    
    def apply_down_proj(self, l):
        self.z_l_2 = torch.matmul(self.z_l_2, getattr(self, f'W_{l}_down')) + getattr(self, f'b_{l}_down')
        pass
    
    def layer_norm(self, x, beta, gamma):
        #normalize x
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (torch.sqrt(x.var(dim=-1, keepdim=True, correction=0) + 1e-5))
        #scale and shift
        x = gamma * x + beta
        return x
    
    def apply_attention(self, l):
        # self.n_heads_splitted = self.split_heads(l)
        # for i in range(1, self.n_heads+1):
        #     self.compute_qkv(l, i)
        #     self.compute_unnormalized_attention(l,i)
        #     if self.mode == "tanh-clipped":
        #         self.clip_attention_scores(l, i)
        #     self.compute_attention_weights(l, i)
        #     self.compute_attended_values(l, i)
        # self.concatenate_heads(l)
        # self.project_attention_output(l) #should convert the output to self.z_l_1
        
        W_Q_l = torch.cat([getattr(self, f'W_{l}_Q_{k}') for k in range(1, self.n_heads+1)], dim=0)
        W_K_l = torch.cat([getattr(self, f'W_{l}_K_{k}') for k in range(1, self.n_heads+1)], dim=0)
        W_V_l = torch.cat([getattr(self, f'W_{l}_V_{k}') for k in range(1, self.n_heads+1)], dim=0)
        Q = torch.matmul(self.z_l_1, W_Q_l)
        K = torch.matmul(self.z_l_1, W_K_l)
        V = torch.matmul(self.z_l_1, W_V_l)
        # print(f"Shape of Q: {Q.shape}, Shape of K: {K.shape}, Shape of V: {V.shape}")
        
        #changing hte shape
        batch_size, sequence_len, d_model = self.z_l_1.shape
        Q = Q.view(batch_size, sequence_len, self.n_heads, self.d_head)
        K = K.view(batch_size, sequence_len, self.n_heads, self.d_head)
        V = V.view(batch_size, sequence_len, self.n_heads, self.d_head)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        #attention
        S = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)
        if self.mode == "tanh-clipped":
            S = self.tau * torch.tanh(S)
        # Causal mask (lower-triangular keep): mask strictly upper triangle (future tokens).
        # causal_mask = torch.triu(torch.ones(sequence_len, sequence_len, device=S.device), diagonal=1).bool()
        # S = S.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        causal_mask = torch.triu(torch.full((sequence_len, sequence_len), float('-inf'), device=S.device), diagonal=1)
        S = S + causal_mask.unsqueeze(0).unsqueeze(0)
        padding_mask = (self.attention_mask == 0).unsqueeze(1).unsqueeze(2)  # shape (batch_size, 1, 1, sequence_len)
        S = S.masked_fill(padding_mask, float('-inf'))
        attn_wts = torch.softmax(S, dim=-1)
        attn_wts = torch.nan_to_num(attn_wts, nan=0.0)
        attended_values = torch.matmul(attn_wts, V)
        #concat heads
        attended_values = attended_values.transpose(1, 2)
        attended_values = attended_values.contiguous()
        attended_values = attended_values.view(batch_size, sequence_len, self.n_heads * self.d_head)
        
        #final projection
        self.z_l_1 = torch.matmul(attended_values, getattr(self, f'W_{l}_O'))
        # print(f"Shape of attention output (z_l_1): {self.z_l_1.shape}, Shape of W_O_l: {getattr(self, f'W_{l}_O').shape}")
        
        

    # def split_heads(self, l):
    #     #need to split self.z_l_1 into n different parts
    #     #dim of self.z_l_1 is batch_size, sequence_len, d_model
    #     pass
    
    # def compute_qkv(self, l, i):
    #     pass
    
    # def compute_unnormalized_attention(self, l, i):
    #     pass
    
    # def clip_attention_scores(self, l, i):
    #     pass
    
    # def compute_attention_weights(self, l, i):
    #     pass
    
    # def compute_attended_values(self, l, i):
    #     pass
    
    # def concatenate_heads(self, l):
    #     pass
    
    # def project_attention_output(self, l):
    #     pass
    
        
        



def load_model(config: Dict[str, Any], weights: Dict[str, Any]):
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function inputs config and weights and outputs a nn.Module derived object.
    """

    model = LanguageModel(config)
    model.set_weights(weights)

    return model


def collate_fn(batch: Dict[str, List[torch.tensor]]) -> Dict[str, torch.Tensor]:
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function takes in a batch of data and outputs a dictionary of tensors ready to be fed into the model.
    """
    PAD_ID = 0  # Assume 0 is the padding token ID
    # max_len = 0
    input_ids = batch['input_ids']
    # for ids in input_ids:
    #     max_len = max(max_len, len(ids))     #vectorize?
    padded_batch = {}       #padd
    # padded_batch['input_ids'] = []
    # padded_batch['attention_mask'] = []
    # for ids in input_ids:
    #     padded_ids = ids.tolist() + [PAD_ID] * (max_len - len(ids))
    #     attention_mask = [1] * len(ids) + [0] * (max_len - len(ids))
    #     padded_batch['input_ids'].append(torch.tensor(padded_ids))
    #     padded_batch['attention_mask'].append(torch.tensor(attention_mask))
    # padded_batch['input_ids'] = torch.stack(padded_batch['input_ids'])
    # padded_batch['attention_mask'] = torch.stack(padded_batch['attention_mask'])
    padded_batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD_ID)
    padded_batch['attention_mask'] = (padded_batch['input_ids'] != PAD_ID).long()
    return padded_batch
    # raise NotImplementedError("Implement collate_fn as described in assignment document")



