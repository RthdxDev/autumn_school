import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        args = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(position * args)
        pos_enc[:, 1::2] = torch.cos(position * args)
        pos_enc = pos_enc.unsqueeze(0)
        self.register_buffer('pe', pos_enc)

    def forward(self, x):
        len_ = x.size(1)
        return self.pe[:, :len_, :].to(x.device)


def subsequent_mask(size):
    return torch.tril(torch.ones((1, size, size), device=device, dtype=torch.bool))


def make_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_head, dropout=0.3):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.dropout = nn.Dropout(dropout)
        self.inner_dim = n_head * d_head
        self.w_q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.w_k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.w_v = nn.Linear(d_model, self.inner_dim, bias=False)
        self.w_out = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        B = q.size(0)
        Q = self.w_q(q).view(B, -1, self.n_head, self.d_head).transpose(1, 2)
        K = self.w_k(k).view(B, -1, self.n_head, self.d_head).transpose(1, 2)
        V = self.w_v(v).view(B, -1, self.n_head, self.d_head).transpose(1, 2)

        attention_score = Q @ K.transpose(-2, -1) / self.d_head ** 0.5

        if mask is not None:
            attention_score = attention_score.masked_fill(~mask, float('-inf'))

        attention_score = F.softmax(attention_score, dim=-1)
        attention_score = self.dropout(attention_score)

        out = attention_score @ V
        out = out.transpose(1, 2).reshape(B, -1, self.inner_dim)
        out = self.w_out(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderOnlyLayer(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, d_head, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_res = self.self_attention(x, x, x, mask=mask)
        x = x + self.dropout(attention_res)
        x = self.norm1(x)

        ff_res = self.ff(x)
        x = x + self.dropout(ff_res)
        x = self.norm2(x)

        return x


class DecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, d_head, d_ff, max_len, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = SinusoidalPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderOnlyLayer(d_model, n_head, d_head, d_ff, dropout) for _ in range(n_layer)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, mask=None):
        x = self.token_embed(seq)
        x = x + self.pos_embed(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)
        return x


class TransformerDecoderOnly(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layer=6, n_head=8, d_head=64, d_ff=2048,
                 max_len=512, dropout=0.3, pad_idx=0):
        super().__init__()
        self.decoder = DecoderOnly(vocab_size, d_model, n_layer, n_head, d_head, d_ff, max_len, dropout)
        self.out = nn.Linear(d_model, vocab_size)
        self.pad_idx = pad_idx

    def forward(self, seq):
        pad_mask = make_pad_mask(seq, self.pad_idx)
        seq_len = seq.size(1)
        causal_mask = subsequent_mask(seq_len).unsqueeze(0)

        mask = pad_mask & causal_mask

        decoder_output = self.decoder(seq, mask=mask)
        logits = self.out(decoder_output)
        return logits
