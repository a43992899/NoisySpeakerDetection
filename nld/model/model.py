from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SpeechEmbedder(nn.Module):
    lstm: nn.LSTM
    projection: nn.Linear
    should_softmax: bool
    projection2: Optional[nn.Linear]
    bn1: Optional[nn.BatchNorm1d]
    softmax: Optional[nn.LogSoftmax]

    def __init__(
        self, lstm_input_size: int, lstm_hidden_size: int, lstm_num_layers: int, projection_size: int,
        should_softmax: bool = False, num_classes: Optional[int] = None
    ):
        super().__init__()
        self.lstm = nn.LSTM(lstm_input_size, lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.projection = nn.Linear(lstm_hidden_size, projection_size)

        self.should_softmax = should_softmax
        if should_softmax:
            if num_classes is None:
                raise ValueError()
            self.projection2 = nn.Linear(projection_size, num_classes)
            self.bn1 = nn.BatchNorm1d(num_classes)
            self.softmax = nn.LogSoftmax(dim=-1)
        else:
            self.projection2 = None
            self.bn1 = None
            self.softmax = None

    def forward(self, x: Tensor):
        x = self.get_embedding(x)
        if self.should_softmax:
            x = self.projection2(x.float())
            x = self.bn1(x)
            x = self.softmax(x)
        return x

    def get_embedding(self, x: Tensor) -> Tensor:
        x, _ = self.lstm(x.float())  # (batch, frames, n_mels)
        x = x[:, -1, :]  # only use last frame
        x = self.projection(x.float())
        x = x / torch.norm(x, dim=-1).unsqueeze(1)
        return x

    # def get_confidence(self, x: Tensor) -> Tensor:
    #     if not self.should_softmax:
    #         raise ValueError()
    #     assert self.projection2 is not None
    #     assert self.bn1 is not None
    #     assert self.softmax is not None

    #     x = self.projection2(x.float())
    #     x = self.bn1(x)
    #     # x = F.softmax(x)  # TODO: logsoftmax or softmax?
    #     x = self.softmax(x)
    #     return x
