import torch.nn as nn
import torch.nn.functional as F
import torch


class BigramModel(nn.Module):
    def __init__(self, vocab_size, pad_token_idx):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size, padding_idx=pad_token_idx)

    def forward(self, x, targets=None):
        logits = self.embeddings(x)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, x, max_new_tokens, end_token_idx):
        for i in range(max_new_tokens):
            logits, _ = self(x)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=1)
            x_next = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, x_next), dim=1)
            if x_next == end_token_idx:
                return x
        return x


class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, head_dim, block_size, dropout):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim

        self.key = nn.Linear(d_model, head_dim, bias=False)
        self.query = nn.Linear(d_model, head_dim, bias=False)
        self.value = nn.Linear(d_model, head_dim, bias=False)
        self.register_buffer("trill", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1) / self.d_model ** 0.5
        wei = wei.masked_fill(self.trill[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head_dim, block_size, n_heads, dropout):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(d_model, head_dim//n_heads, block_size, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(head_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FFN(nn.Module):
    def __init__(self, ffn_dim,  d_model, dropout):
        super().__init__()
        self.ffn1 = nn.Linear(in_features=d_model, out_features=ffn_dim)
        self.relu = nn.ReLU()
        self.ffn2 = nn.Linear(in_features=ffn_dim, out_features=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ffn1(x)
        x = self.relu(x)
        x = self.dropout(self.ffn2(x))
        return x


class Block(nn.Module):
    def __init__(self, d_model, head_dim, block_size, n_heads, ffn_dim, vocab_size, dropout):
        super().__init__()
        self.ffn_dim = ffn_dim
        self.vocab_size = vocab_size
        self.sa = MultiHeadAttention(d_model, head_dim, block_size, n_heads, dropout)
        self.ffn = FFN(ffn_dim, d_model, dropout)
        self.relu = nn.ReLU()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        out = x + self.ffn(self.ln2(x))
        return out



class DecoderModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, head_dim, block_size, n_heads, ffn_dim, n_layers,
                 dropout, pad_token_idx):
        super().__init__()
        self.block_size = block_size
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_token_idx)
        self.positional_embeddings = nn.Embedding(num_embeddings=block_size, embedding_dim=embedding_dim)
        self.blocks = nn.Sequential(*[Block(d_model=embedding_dim, head_dim=head_dim,
                                    block_size=block_size, n_heads=n_heads,
                                    ffn_dim=ffn_dim, vocab_size=vocab_size, dropout=dropout) for _ in range(n_layers)])
        self.lm_head = nn.Linear(in_features=embedding_dim, out_features=vocab_size)
        self.final_ln = nn.LayerNorm(embedding_dim)

    def forward(self, x, targets=None):
        B, T= x.shape
        x = self.embeddings(x)
        x += self.positional_embeddings(torch.arange(T, device=x.device))

        x = self.blocks(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, x, max_new_tokens, end_token_idx):
        for i in range(max_new_tokens):
            idx_cond = x[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=1)
            x_next = torch.multinomial(probs, num_samples=1)

            x = torch.cat((x, x_next), dim=1)
            if x_next == end_token_idx:
                return x
        return x