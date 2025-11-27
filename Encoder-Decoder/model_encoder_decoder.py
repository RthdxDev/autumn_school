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
        attention_score = self.dropout(attention_score) # можно убрать

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


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_ff, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head, d_head, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, seq_mask=None):
        attention_res = self.attention(x, x, x, mask=seq_mask)
        x = x + self.dropout(attention_res)
        x = self.norm1(x)

        ff = self.ff(x)
        x = x + self.dropout(ff)
        x = self.norm2(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_ff, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, d_head, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_head, d_head, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, seq_mask=None, target_mask=None):
        self_attention_res = self.self_attention(x, x, x, mask=target_mask)
        x = x + self.dropout(self_attention_res)
        x = self.norm1(x)

        cross_attention_res = self.cross_attention(x, enc_output, enc_output, mask=seq_mask)
        x = x + self.dropout(cross_attention_res)
        x = self.norm2(x)

        ff = self.ff(x)
        x = x + self.dropout(ff)
        x = self.norm3(x)

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, d_head, d_ff, max_len, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = SinusoidalPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_head, d_head, d_ff, dropout) for _ in range(n_layer)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq, seq_mask=None):
        x = self.token_embed(seq)
        x = x + self.pos_embed(x)

        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, seq_mask=seq_mask)
        x = self.norm(x)

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layer, n_head, d_head, d_ff, max_len, dropout):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = SinusoidalPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_head, d_head, d_ff, dropout) for _ in range(n_layer)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, target, encoder_output, seq_mask=None, target_mask=None):
        x = self.token_embed(target)
        x = x + self.pos_embed(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, seq_mask=seq_mask, target_mask=target_mask)
        x = self.norm(x)

        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size_seq, vocab_size_target, d_model=512, n_layer=6, n_head=8, d_head=64, d_ff=2048,
                 max_len=512, dropout=0.3, pad_idx=0):
        super().__init__()
        self.encoder = Encoder(vocab_size_seq, d_model, n_layer, n_head, d_head, d_ff, max_len, dropout)
        self.decoder = Decoder(vocab_size_target, d_model, n_layer, n_head, d_head, d_ff, max_len, dropout)
        self.out = nn.Linear(d_model, vocab_size_target)
        self.pad_idx = pad_idx

    def forward(self, seq, target):
        seq_mask = make_pad_mask(seq, self.pad_idx)
        target_pad_mask = make_pad_mask(target, self.pad_idx)

        target_len = target.size(1)
        causal_mask = subsequent_mask(target_len).unsqueeze(0)
        target_mask = target_pad_mask & causal_mask

        encoder_output = self.encoder(seq, seq_mask=seq_mask)
        decoder_output = self.decoder(target, encoder_output, seq_mask=seq_mask, target_mask=target_mask)
        logits = self.out(decoder_output)

        return logits
