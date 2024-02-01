import torch
import torch.nn as nn

class LstmModelSimple(nn.Module):
    def __init__(self, num_classes, num_layers=3, hidden_size=64, embed_size=64):
        super(LstmModelSimple, self).__init__()
        self.embed = torch.nn.Linear(192, embed_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = torch.nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.8,
        )
        self.fc = torch.nn.Linear(hidden_size, num_classes)
        self.name = "LstmModelSimple"
        self.dropout1 = nn.Dropout(0.5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initHidden(self, batch_size):
        return (
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
            torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device),
        )

    def forward(self, x, h):
        y = self.embed(x)
        y = self.dropout1(y)
        y, (h, c) = self.lstm(y, h)
        y = self.fc(y)
        return y, (h, c)
