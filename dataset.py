import torch
from torch.utils.data import DataLoader, Dataset
from config import parse_option



opt = parse_option()

WINDOW_SIZE = opt.window_size
HEAD_SIZE = opt.head_size
N_EMBED = opt.n_embed

class BitcoinDataset(Dataset):
    def __init__(self, data, window_size=WINDOW_SIZE):
        self.data = data
        self.window_size = window_size

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        try:
            x = self.data[idx:idx+self.window_size]
            y = self.data['close'][idx+1:idx+self.window_size+1]
        except IndexError:
            idx = torch.randint(0, len(self.data) - self.window_size - 1, (1,))
            x = self.data[idx:idx+self.window_size]
            y = self.data['close'][idx+1:idx+self.window_size+1]
        x = list(x.values())
        x = torch.tensor(x, dtype=torch.float32)   
        y = torch.tensor(y, dtype=torch.float32)
        return x, y
