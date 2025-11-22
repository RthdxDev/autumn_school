import torch
from torch.utils.data import Dataset
from torch import Tensor


class NamesDataset(Dataset):
	def __init__(self, path, names: list[str], seq_len: int=3):
		with open(path, 'r') as f:
			names = f.read().splitlines()
			self._seq_len = seq_len
			self.__stoi = {
				"<start>": 0,
				"<end>": 1,
			}
			for i, ch in enumerate(sorted(set(''.join(names)))):
				self.__stoi[ch] = i + 2
			self.__itos = {i: ch for ch, i in self.__stoi.items()}
			self.names: list[tuple[list[str], str]] = []
			self.names_encoded: list[tuple[Tensor, Tensor]] = []
			for name in names:
				name_lst = list(name) + ['<end>']
				for i in range(len(name_lst)):
					target: str = name_lst[i]
					target_encoded = torch.tensor([self.__stoi[target]])
					part_name = name_lst[max(0, i - self.seq_len):i]
					entry = ["<start>"] * (self.seq_len - len(part_name)) + part_name
					entry_encoded = torch.tensor([self.__stoi["<start>"]] * (self.seq_len - len(part_name)) + [self.__stoi[ch] for ch in part_name])
					self.names.append((entry, target))
					self.names_encoded.append((entry_encoded, target_encoded))

	def __len__(self):
		return len(self.names)
	
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
