import torch
import torch.nn.functional as F
from torch import Tensor, nn

from core.config import Config


class SpeechEmbedder(nn.Module):
    def __init__(self, hp: Config):
        super(SpeechEmbedder, self).__init__()
        self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden, num_layers=hp.model.num_layer, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hp.model.hidden, hp.model.proj)

    def forward(self, x):
        x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        # print("Shape is: ", x.shape)
        return x

    def get_embedding(self, x: Tensor) -> Tensor:
        return self(x)


class SpeechEmbedder_Softmax(nn.Module):
    def __init__(self, num_classes, hp):
        super(SpeechEmbedder_Softmax, self).__init__()
        self.LSTM_stack = nn.LSTM(hp.data.nmels, hp.model.hidden, num_layers=hp.model.num_layer, batch_first=True)
        for name, param in self.LSTM_stack.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        self.projection = nn.Linear(hp.model.hidden, hp.model.proj)

        # Classification purpose
        # self.relu1 = nn.ReLU()
        self.projection2 = nn.Linear(hp.model.proj, num_classes)
        self.bn1 = nn.BatchNorm1d(num_classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        # x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        # # only use last frame
        # x = x[:, x.size(1) - 1]
        # x = self.projection(x.float())
        # x = x / torch.norm(x, dim=1).unsqueeze(1)
        x = self.get_embedding(x)

        x = self.projection2(x.float())
        x = self.bn1(x)
        x = self.softmax(x)
        # x = self.get_confidence(x)

        return x

    def get_embedding(self, x: Tensor) -> Tensor:
        x, _ = self.LSTM_stack(x.float())  # (batch, frames, n_mels)
        # only use last frame
        x = x[:, x.size(1) - 1]
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=1).unsqueeze(1)
        return x

    def get_confidence(self, x: Tensor) -> Tensor:
        x = self.projection2(x.float())
        x = self.bn1(x)
        x = F.softmax(x)
        # x = self.softmax(x)
        return x
