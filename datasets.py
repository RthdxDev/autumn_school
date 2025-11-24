import torch
from torch.utils.data import Dataset
from torch import Tensor


class NamesDataset(Dataset):
	def __init__(self, names: list[str], seq_len: int=3):
		self._seq_len = seq_len
		self.__stoi = {
			"<pad>": 0,
			"<start>": 1,
			"<end>": 2,	
		}

		for i, ch in enumerate(sorted(set(''.join(names)))):
			self.__stoi[ch] = i + 3
		self.__itos = {i: ch for ch, i in self.__stoi.items()}

		self.names: list[tuple[str, str]] = []
		self.names_encoded: list[tuple[Tensor, Tensor]] = []

		for name in names:
			tokens = [self.__stoi['<start>']] + [self.__stoi[ch] for ch in name] + [self.__stoi['<end>']]
			strs = ['<start>'] + list(name) + ['<end>']

			if len(tokens) < seq_len:
				tokens = tokens + [self.__stoi['<pad>']] * (seq_len - len(tokens))
				strs = strs + ['<pad>'] * (seq_len - len(strs))
			elif len(tokens) > seq_len:
				tokens = tokens[:seq_len]
				strs = strs[:seq_len]

			input_seq = torch.tensor(tokens[:-1])
			target_seq = torch.tensor(tokens[1:])

			self.names.append((''.join(strs[:-1]), ''.join(strs[1:])))
			self.names_encoded.append((input_seq, target_seq))

	def __len__(self):
		return len(self.names_encoded)
	
	@property
	def seq_len(self):
		return self._seq_len
	
	@property
	def stoi(self):
		return self.__stoi
	
	@property	
	def itos(self):
		return self.__itos
	
	def get_sample(self, idx):
		return self.names[idx]
	
	def __getitem__(self, idx):
		return self.names_encoded[idx]
