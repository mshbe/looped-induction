from dataclasses import dataclass

import einops
import torch as t
from jaxtyping import Float, Int
from torch import Tensor, nn


@dataclass
class Config:
    d_model: int = 256
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 256
    d_head: int = 32
    n_heads: int = 8 #n_heads x d_head= d_model
    n_layers: int = 1 #number of distinct layers in stack
    n_loops: int = 2 #number of times model loops through the stack
    tokenizer_name: str = "EleutherAI/gpt-neox-20b"

# modules needed for the model

#token embedding
class Embed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_E = nn.Parameter(t.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        return self.W_E[tokens]

#positional embedding
class PosEmbed(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_pos = nn.Parameter(t.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_model"]:
        batch, seq_length=tokens.shape
        return einops.repeat(self.W_pos[:seq_length], "seq d_model -> b seq d_model", b=batch)



# Attention with no biases
# W_Q, W_K, W_V have shape [n_heads, d_model, d_head], W_O has shape [n_heads,d_head, d_model]

class Attention(nn.Module):
    IGNORE: Float[Tensor, ""]

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_K = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_V = nn.Parameter(t.empty((cfg.n_heads, cfg.d_model, cfg.d_head)))
        self.W_O = nn.Parameter(t.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        #self.b_Q = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        #self.b_K = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        #self.b_V = nn.Parameter(t.zeros((cfg.n_heads, cfg.d_head)))
        #self.b_O = nn.Parameter(t.zeros((cfg.d_model)))
        nn.init.normal_(self.W_Q, std=self.cfg.init_range)
        nn.init.normal_(self.W_K, std=self.cfg.init_range)
        nn.init.normal_(self.W_V, std=self.cfg.init_range)
        nn.init.normal_(self.W_O, std=self.cfg.init_range)
        self.register_buffer("IGNORE", t.tensor(float("-inf"), dtype=t.float32))

    def forward(self, resid_pre: Float[Tensor, "batch posn d_model"], pos_emb: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_model"]:

        #defining keys, queries, values
        x=resid_pre
        y=pos_emb

        #pos_emb added to input of keys and queries, but not values. No biases.
        keys=einops.einsum(x+y, self.W_K,
                           "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") #+ self.b_K

        queries=einops.einsum(x+y, self.W_Q,
                           "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head") #+ self.b_Q

        values=einops.einsum(x, self.W_V,
                           "batch posn d_model, n_heads d_model d_head -> batch posn n_heads d_head")# + self.b_V

        #attn scores and probabilities
        attn_scores=einops.einsum(
            keys,
            queries,
            "batch posn_K n_heads d_head, batch posn_Q n_heads d_head -> batch n_heads posn_Q posn_K")

        #Attn scores scaled by 1/(d_head)^0.5, causally masked
        attn_scores_masked=self.apply_causal_mask(attn_scores/(self.cfg.d_head**0.5))
        attn_prob=attn_scores_masked.softmax(-1)

        #applying attn prob to values
        weighted_val=einops.einsum(
            attn_prob,
            values,
            "batch n_heads posn_Q posn_K, batch posn_K n_heads d_head ->batch posn_Q n_heads d_head")

        #final linear transformation and summing over heads
        result=einops.einsum(
            weighted_val,
            self.W_O,
            "batch posn_Q n_heads d_head, n_heads d_head d_model -> batch posn_Q d_model"
        ) #+ self.b_O

        return result


    def apply_causal_mask(
        self, attn_scores: Float[Tensor, "batch n_heads query_pos key_pos"]
    ) -> Float[Tensor, "batch n_heads query_pos key_pos"]:
        """
        Applies a causal mask to attention scores, and returns masked scores.
        """
        query_pos=attn_scores.shape[2]
        key_pos=attn_scores.shape[3]
        trueMtx=t.full((query_pos, key_pos), True, device=attn_scores.device)
        mask=t.triu(trueMtx, diagonal=1)
        return t.masked_fill(attn_scores, mask, self.IGNORE)


#transformer block
class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = Attention(cfg)

    def forward(self, resid_pre, pos_emb):
        return resid_pre + self.attn(resid_pre, pos_emb)


#unembedding

class Unembed(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_U = nn.Parameter(t.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)
        #self.b_U = nn.Parameter(t.zeros((cfg.d_vocab)), requires_grad=False)

    def forward(
        self, resid_final: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        return (einops.einsum(resid_final,
                              self.W_U,
                              "batch posn d_model, d_model d_vocab -> batch posn d_vocab"))

#model
class LoopedAttnOnly(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.unembed = Unembed(cfg)

    def forward(self, tokens: Int[Tensor, "batch position"]) -> Float[Tensor, "batch position d_vocab"]:
        residual=self.embed(tokens)
        pos=self.pos_embed(tokens)
        for _ in range(self.cfg.n_loops):
            for block in self.blocks:
                residual=block(residual, pos) 
        return self.unembed(residual)