import torch
import tqdm
from datasets import load_dataset
from dataset import BitcoinDataset
from torch.utils.data import DataLoader
from config import parse_option
from model import Bitcoin



class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, epochs, device):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device

    def train(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, (x, y) in progress_bar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                y_hat = self.model(x)
                y_hat = y_hat.squeeze(-1)
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_description(f"Epoch: {epoch}, Loss: {total_loss/(i+1)}")
            
        
    @torch.no_grad()    
    def valid(self):
        self.model.eval()
        total_loss = 0
        for i, (x, y) in enumerate(tqdm.tqdm(self.train_loader)):
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            loss = self.criterion(y_hat, y)
            total_loss += loss.item()
        print(f"Validation Loss: {total_loss/len(self.train_loader)}")

    def fit(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.valid()



if __name__ == "__main__":
    opt = parse_option()

    DEVICE = opt.device
    BATCH_SIZE = opt.batch_size
    EPOCHS = opt.epochs
    LEARNING_RATE = opt.lr  


    model = Bitcoin()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()


    train_dataset = load_dataset("Onegai/BitcoinPrice", split="train[:90%]")
    valid_dataset = load_dataset("Onegai/BitcoinPrice", split="train[90%:]")
    columns_to_drop = ["timestamp", "target","__index_level_0__"]

    # Drop the specified columns
    train_dataset = train_dataset.remove_columns(columns_to_drop)
    valid_dataset = valid_dataset.remove_columns(columns_to_drop)
    train_dataset = BitcoinDataset(train_dataset)
    valid_dataset = BitcoinDataset(valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    trainer = Trainer(model, optimizer, criterion, train_loader, EPOCHS, DEVICE)

    trainer.fit()









