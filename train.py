import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from datasets import load_dataset
from dataset import BitcoinDataset
from torch.utils.data import DataLoader
from config import parse_option
from model import Bitcoin


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        train_loader,
        valid_loader,
        epochs,
        batch_size,
        window_size,
        device,
    ):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.batch_size = batch_size
        self.window_size = window_size
        self.device = device

    def train(self, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, (x, y) in progress_bar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                y_hat = self.model(x)
                y_hat = y_hat.squeeze(-1)
                y = y.unsqueeze(0)
                y_hat = y_hat.unsqueeze(0) 
                Y = torch.cat([y_hat, y], dim=0)*1000
                Y = F.normalize(Y, dim=-1)
                y = Y[0]
                y_hat = Y[1]
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()
            total_loss += loss.item()
            progress_bar.set_description(f"Epoch: {epoch+1}, Loss: {loss.item()}")
        progress_bar.set_description(f"Epoch: {epoch+1}, Loss: {total_loss/len(self.train_loader)}")

    @torch.no_grad()
    def valid(self):
        self.model.eval()
        total_loss = 0
        for x, y in self.valid_loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            y_hat = y_hat.squeeze(-1)
            y = y.unsqueeze(0)
            y_hat = y_hat.unsqueeze(0) 
            Y = torch.cat([y_hat, y], dim=0)*1000
            Y = F.normalize(Y, dim=-1)
            y = Y[0]
            y_hat = Y[1]
            loss = self.criterion(y_hat, y)
            total_loss += loss.item()
        print(f"Validation Loss: {total_loss/len(self.valid_loader)}")

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
    WINDOW_SIZE = opt.window_size
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    model = Bitcoin()
    model.to(DEVICE)
    #model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss(reduction="mean")    

    train_dataset = load_dataset("Onegai/BitcoinPrice", split="train[:90%]")
    valid_dataset = load_dataset("Onegai/BitcoinPrice", split="train[90%:]")
    columns_to_drop = ["timestamp", "target", "__index_level_0__"]

    # Drop the specified columns
    train_dataset = train_dataset.remove_columns(columns_to_drop)
    valid_dataset = valid_dataset.remove_columns(columns_to_drop)
    train_dataset = BitcoinDataset(train_dataset)
    valid_dataset = BitcoinDataset(valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    trainer = Trainer(model, optimizer, criterion, train_loader, valid_loader, EPOCHS, BATCH_SIZE, WINDOW_SIZE, DEVICE)
    print("Training the model on", DEVICE)

    trainer.fit()
