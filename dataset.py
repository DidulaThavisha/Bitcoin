import torch
from torch.utils.data import DataLoader, Dataset
from config import parse_option



opt = parse_option()

WINDOW_SIZE = opt.window_size
HEAD_SIZE = opt.head_size
N_EMBED = opt.n_embed
BATCH_SIZE = opt.batch_size

class BitcoinDataset(Dataset):
    def __init__(self, data, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE):
        self.data = data
        self.window_size = window_size
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
     
        print(idx)
        print(idx+self.window_size)
        try:
            x = self.data[idx:idx+self.window_size]
            y = self.data[idx+1:idx+self.window_size+1]['close']
        except IndexError:
            idx = torch.randint(0, len(self.data) - self.window_size - 1, (1,))
            x = self.data[idx:idx+self.window_size]
            y = self.data[idx+1:idx+self.window_size+1]['close']
        x = list(x.values())
        x = torch.tensor(x, dtype=torch.float32)   
        y = torch.tensor(y, dtype=torch.float32)
        return x, y
