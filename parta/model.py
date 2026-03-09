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
        self.W_vocab = None
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
        self.encode()   # this function finds the word embeddings and the positional embeddings
        self.n_transformer_blocks()     # this function runs a for loop for n transformer blocks, and update the hidden states in each block
        self.final_norm()       # this function applies the final layer norm
        self.devocab()          # converts the final hidden states to logits over the vocabulary
        self.find_prob()        # applies softmax to obtain the probabilities
        return self.logits
        # raise NotImplementedError("Implement forward as described in assignment document")
        
    def encode(self):
        self.apply_word_embeddings()
        self.apply_positional_embeddings()
        pass
    
    def n_transformer_blocks(self):
        for i in range(1, self.n_layers+1):
            self.apply_transformer_block(i) # this function applies the i-th transformer block and updates the hidden states accordingly
        pass
    
    def final_norm(self):
        pass
    
    def devocab(self):
        pass
    
    def find_prob(self):
        pass
    
    def apply_word_embeddings(self):
        pass
    
    def apply_positional_embeddings(self):
        pass
    
    def apply_transformer_block(self, l):
        self.apply_layer_norm(l, 1)
        self.apply_attention(l)
        self.add()
        self.apply_layer_norm(l, 2)
        self.apply_up_proj(l)
        self.apply_gelu()
        self.apply_down_proj(l)
        self.add()
        
    def apply_layer_norm(self, l, part):
        pass
    
    # def apply_attention(self, l):
    #     pass
    
    def add(self):
        pass
    
    def apply_up_proj(self, l):
        pass
    
    def apply_gelu(self):
        pass
    
    def apply_down_proj(self, l):
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
