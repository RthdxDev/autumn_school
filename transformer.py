import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
	"""
	Multiple heads of self-attention in parallel
	"""
	def __init__(self, emb_size, num_heads, head_size, dropout=0.0):
		super().__init__()
		self.head_size = head_size
		self.query = nn.Linear(emb_size, num_heads * head_size, bias=False)
		self.key = nn.Linear(emb_size, num_heads * head_size, bias=False)
		self.value = nn.Linear(emb_size, num_heads * head_size, bias=False)
		self.dropout = nn.Dropout(dropout)
		self.softmax = nn.Softmax(dim=-1)
		self.linear = nn.Linear(num_heads * head_size, emb_size)
		
	def forward(self, x):
		batch_size, seq_len, emb_size = x.shape
		Q = self.query(x).view(batch_size, seq_len, -1, self.head_size).transpose(1, 2)
		K = self.key(x).view(batch_size, seq_len, -1, self.head_size).transpose(1, 2)
		V = self.value(x).view(batch_size, seq_len, -1, self.head_size).transpose(1, 2)

		attention_score = Q @ K.transpose(-2, -1) / (self.head_size ** 0.5)

		mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
		attention_score = attention_score.masked_fill(mask == 0, float('-inf'))

		out = self.softmax(attention_score) @ V
		out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
		out = self.linear(out)
		out = self.dropout(out)

		return out


class FeedForwardNetwork(nn.Module):
	"""
	A simple linear layer followed by a non-linearity
	"""
	def __init__(self, emb_size, hidden_size, dropout=0.0):	
		super().__init__()
		self.linear1 = nn.Linear(emb_size, hidden_size)
		self.linear2 = nn.Linear(hidden_size, emb_size)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		x = self.linear1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.linear2(x)
		return x


class TransformerBlock(nn.Module):
	"""
	Transformer block: communication followed by computation
	"""
	def __init__(self, emb_size, num_heads, head_size, hidden_size, post_ln=True, dropout=0.0):
		super().__init__()
		self.post_ln = post_ln
		self.attention = SelfAttention(emb_size, num_heads, head_size, dropout)
		self.ffn = FeedForwardNetwork(emb_size, hidden_size, dropout)
		self.norm1 = nn.LayerNorm(emb_size)
		self.norm2 = nn.LayerNorm(emb_size)
		
	def forward(self, x):
		if self.post_ln:
			x = self.norm1(x + self.attention(x))
			x = self.norm2(x + self.ffn(x))
		return x


class GPT(nn.Module):
	def __init__(self, dict_size, emb_size, seq_len, num_heads=8, head_size=64, hidden_size=2048, dropout=0.3):
		super().__init__()
		self.max_len = seq_len
		self.token_embedding_table = nn.Embedding(num_embeddings=dict_size, embedding_dim=emb_size)
		self.blocks = nn.Sequential(
			*[
				TransformerBlock(
				emb_size=emb_size,
		   		num_heads=num_heads,
				head_size=head_size,
				hidden_size=hidden_size,
				dropout=dropout
				) for _ in range(6)
			]
		)
		self.flatten = nn.Flatten()
		self.linear = nn.Linear(emb_size, dict_size)
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, nn.Linear):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
			if module.bias is not None:
				torch.nn.init.zeros_(module.bias)
		elif isinstance(module, nn.Embedding):
			torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, idx):
		emb = self.token_embedding_table(idx)
		out = self.blocks(emb)
		logits = self.linear(out)
		return logits

	@torch.no_grad()
	def generate(self, idx, num_tokens=1):
		for _ in range(num_tokens):
			idx_crop = idx if idx.shape[1] <= self.max_len else idx[:, -self.max_len:]

			logits = self.forward(idx_crop)[:, -1, :]
			probs = F.softmax(logits, dim=-1)

			id_next = torch.multinomial(probs, num_samples=1)
			idx = torch.cat((idx, id_next), dim=1)

		return idx
