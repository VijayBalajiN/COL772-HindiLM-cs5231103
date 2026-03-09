import torch
import torch.nn as nn
from typing import Any, Dict, List


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
        # self.d_model = None
        # self.n_heads = None
        # self.d_head = None
        # self.n_layers = None
        # self.vocab_size = None
        # self.mode = None
        # self.tau = None
        self.load_config()
        super().__init__()
        #initialize the weights with None
        #emdeddings
        self.W_vocab = [None for _ in range(self.vocab_size)]
        self.W_devocab = None
        #attention
        self.W_Q_l_k = [[None for _ in range(self.n_heads+1)] for _ in range(self.n_layers+1)]
        self.W_K_l_k = [[None for _ in range(self.n_heads+1)] for _ in range(self.n_layers+1)]
        self.W_V_l_k = [[None for _ in range(self.n_heads+1)] for _ in range(self.n_layers+1)]
        self.W_O_l = [None for _ in range(self.n_layers+1)]
        
        # feedforward
        self.W_l_up = [None for _ in range(self.n_layers+1)]
        self.b_l_up = [None for _ in range(self.n_layers+1)]
        self.W_l_down = [None for _ in range(self.n_layers+1)]
        self.b_l_down = [None for _ in range(self.n_layers+1)]
        
        # layer norms
        self.beta_l_1 = [None for _ in range(self.n_layers+1)]
        self.gamma_l_1 = [None for _ in range(self.n_layers+1)]
        self.beta_l_2 = [None for _ in range(self.n_layers+1)]
        self.gamma_l_2 = [None for _ in range(self.n_layers+1)]
        self.final_beta = None
        self.final_gamma = None
        
        
        #initializing the hidden states needed in different function in forward pass to none
        #encode
        self.word_embeddings = None #initially will contain the word embeddigs alone then will add positional to it
        self.L = None
        
        #n_transformer_blocks
        self.x_l = torch.zeros(self.L, self.d_model) # empty tensor of shape (self.L, self.d_model)
        self.z_l_1 = torch.zeros(self.L, self.d_model)
        self.z_l_2 = torch.zeros(self.L, self.d_model)
                
        #final_norm
        #nothing is needed i think
        
        #devocab
        self.logits = None
        
        #find_prob
        self.probs = None
        
        #attention
        self.n_heads_splitted = [None for _ in range(self.n_layers+1)] # this will be a list of length n_layers+1
        self.Q_l_n = [[None for _ in range(self.n_heads+1)] for _ in range(self.n_layers+1)]
        self.K_l_n = [[None for _ in range(self.n_heads+1)] for _ in range(self.n_layers+1)]
        self.V_l_n = [[None for _ in range(self.n_heads+1)] for _ in range(self.n_layers+1)]
        self.S_l_n = [[None for _ in range(self.n_heads+1)] for _ in range(self.n_layers+1)]
        self.attended_values_l_n = [[None for _ in range(self.n_heads+1)] for _ in range(self.n_layers+1)]
        #do i need to store all the l and n things?
        self.Q = None
        self.K = None
        self.V = None
        self.S = None
        self.attended_values = None
        self.concatenated_attended_values = None
        
        
    def load_config(self):
        self.d_model = self.config["d_model"]
        self.n_heads = self.config["n_heads"]
        self.d_head = self.config["d_head"]
        self.n_layers = self.config["n_layers"]
        self.vocab_size = self.config["vocab_size"]
        self.mode = self.config["mode"]
        self.tau = self.config["tau"]

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        # embeddigns
        self.W_vocab = weights["W_vocab"]
        self.W_devocab = weights["W_devocab"]
        
        #attention
        for l in range(1, self.n_layers+1):
            for k in range(1, self.n_heads+1):
                self.W_Q_l_k[l][k] = weights[f"W_{l}_Q_{k}"]
                self.W_K_l_k[l][k] = weights[f"W_{l}_K_{k}"]
                self.W_V_l_k[l][k] = weights[f"W_{l}_V_{k}"]
            self.W_O_l[l] = weights[f"W_{l}_O"]
            
        #feedfrward
        for l in range(1, self.n_layers+1):
            self.W_l_up[l] = weights[f"W_{l}_up"]
            self.b_l_up[l] = weights[f"b_{l}_up"]
            self.W_l_down[l] = weights[f"W_{l}_down"]
            self.b_l_down[l] = weights[f"b_{l}_down"]
            
        # layer norms
        for l in range(1, self.n_layers+1):
            self.beta_l_1[l] = weights[f"beta_{l}_1"]
            self.gamma_l_1[l] = weights[f"gamma_{l}_1"]
            self.beta_l_2[l] = weights[f"beta_{l}_2"]
            self.gamma_l_2[l] = weights[f"gamma_{l}_2"]
        self.final_beta = weights["final_beta"]
        self.final_gamma = weights["final_gamma"]
        
        # raise NotImplementedError("Implement set_weights as described in assignment document")

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
        pass
    
    def n_transformer_blocks(self):
        for i in range(1, self.n_layers+1):
            self.apply_transformer_block(i) # this function applies the i-th transformer block and updates the hidden states accordingly
        pass
    
    def final_norm(self):
        beta = self.final_beta
        gamma = self.final_gamma
        self.x_l = self.layer_norm(self.x_l, beta, gamma)
        pass
    
    def devocab(self):
        # self.x_l is of shape (batch_size, sequence_len, d_model)
        # self.W_devocab is of shape (d_model, vocab_size)
        self.logits = torch.matmul(self.x_l, self.W_devocab) #projected on to vocab space
        pass
    
    def find_prob(self):
        self.probs = torch.softmax(self.logits, axis=-1)
        pass
    
    def apply_word_embeddings(self, input_ids, attention_mask):
        # input_ids is of shape (batch_size, sequence_len)
        # attention_mask is of shape (batch_size, sequence_len)
        self.word_embeddings = self.W_vocab[input_ids] #dim will be (batch_size, sequence_len, d_model)
        self.L = input_ids.shape[1]
        # attn_mask_exp = attention_mask.unsqueeze(-1)
        # self.word_embeddings = self.word_embeddings * attn_mask_exp
        # self.L = self.word_embeddings.shape[1] # sequence length
        pass
    
    def apply_positional_embeddings(self, attention_mask):
        # self.word_embeddings is of shape (batch_size, sequence_len, d_model)
        # attention_mask is of shape (batch_size, sequence_len)
        # batch_size, sequence_len, d_model = self.word_embeddings.shape
        # pos_range = torch.arange(sequence_len).unsqueeze(0).expand(batch_size, -1).unsqueeze(-1) # shape (batch_size, sequence_len, 1)
        # i_range = torch.arange(d_model).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1) # shape (batch_size, 1, d_model)
        # # entire_range = pos_range + i_range
        # position_embeddings = self.comp_pos_emb(pos_range, i_range) # shape (batch_size, sequence_len, d_model)
        # position_embeddings = torch.zeros_like(self.word_embeddings) # shape (batch_size, sequence_len, d_model)
        # for i in range(d_model):
        #     position_embeddings[:,:,i] = self.comp_pos_emb(position_ids, i)
        pos_range = torch.arange(self.L).view(1, -1, 1)
        i_range = torch.arange(self.d_model).view(1, 1, -1)
        pos_emb = self.comp_pos_emb(pos_range, i_range)
        self.word_embeddings = self.word_embeddings + pos_emb
        attn_mask_exp = attention_mask.unsqueeze(-1)
        self.word_embeddings = self.word_embeddings * attn_mask_exp
        pass
    def comp_pos_emb(self, position, i):
        # if i % 2 == 0:
        #     return torch.sin(position / (10000 ** (i / self.d_model)))
        # else:
        #     return torch.cos(position / (10000 ** ((i-1) / self.d_model)))
        denom = torch.pow(10000, ((i//2)*2) / self.d_model)
        sin = torch.sin(position / denom)
        cos = torch.cos(position / denom)
        return torch.where(i % 2 == 0, sin, cos)
    
    
    def apply_transformer_block(self, l):
        self.apply_layer_norm(l, 1)
        self.apply_attention(l)
        self.add(1)
        self.apply_layer_norm(l, 2)
        self.apply_up_proj(l)
        self.apply_gelu()
        self.apply_down_proj(l)
        self.add(2)
        
    def apply_layer_norm(self, l, part):
        # beta = self.f"beta_l_{part}"[l]
        if part == 1:
            beta = self.beta_l_1[l]
            gamma = self.gamma_l_1[l]
            self.z_l_1 = self.layer_norm(self.x_l, beta, gamma)
        else:
            beta = self.beta_l_2[l]
            gamma = self.gamma_l_2[l]
            self.z_l_2 = self.layer_norm(self.x_l, beta, gamma)
                 
        pass
    
    # def apply_attention(self, l):
    #     pass
    
    def add(self, part):
        if part == 1:
            self.x_l = self.x_l + self.z_l_1
        else:
            self.x_l = self.x_l + self.z_l_2
        pass
    
    def apply_up_proj(self, l):
        self.z_l_2 = torch.matmul(self.z_l_2, self.W_l_up[l]) + self.b_l_up[l]
        pass
    
    def apply_gelu(self):
        self.z_l_2 = torch.nn.functional.gelu(self.z_l_2)
        pass
    
    def apply_down_proj(self, l):
        self.z_l_2 = torch.matmul(self.z_l_2, self.W_l_down[l]) + self.b_l_down[l]
        pass
    
    def layer_norm(self, x, beta, gamma):
        #normalize x
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-5)
        #scale and shift
        x = gamma * x + beta
        return x
    
    def apply_attention(self, l):
        self.n_heads_splitted = self.split_heads(l)
        for i in range(1, self.n_heads+1):
            self.compute_qkv(l, i)
            self.compute_unnormalized_attention(l,i)
            if self.mode == "tanh-clipped":
                self.clip_attention_scores(l, i)
            self.compute_attention_weights(l, i)
            self.compute_attended_values(l, i)
        self.concatenate_heads(l)
        self.project_attention_output(l) #should convert the output to self.z_l_1
        
        
    def split_heads(self, l):
        #need to split self.z_l_1 into n different parts
        #dim of self.z_l_1 is batch_size, sequence_len, d_model
        pass
    
    def compute_qkv(self, l, i):
        pass
    
    def compute_unnormalized_attention(self, l, i):
        pass
    
    def clip_attention_scores(self, l, i):
        pass
    
    def compute_attention_weights(self, l, i):
        pass
    
    def compute_attended_values(self, l, i):
        pass
    
    def concatenate_heads(self, l):
        pass
    
    def project_attention_output(self, l):
        pass
    
        
        



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
    raise NotImplementedError("Implement collate_fn as described in assignment document")
