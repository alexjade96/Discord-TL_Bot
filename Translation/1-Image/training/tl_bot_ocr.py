"""EasyOCR custom model definition — None-VGG-BiLSTM-CTC.

Matches EasyOCR's default recognition architecture. This file must live in
~/.EasyOCR/user_network/ alongside tl_bot_ocr.yaml to be loadable via:

    easyocr.Reader(['en'], recog_network='tl_bot_ocr')
"""

import torch
import torch.nn as nn


class VGGFeatureExtractor(nn.Module):
    def __init__(self, input_channel: int, output_channel: int):
        super().__init__()
        self.output_channel = [
            output_channel // 8,
            output_channel // 4,
            output_channel // 2,
            output_channel,
        ]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True),
        )

    def forward(self, x):
        return self.ConvNet(x)


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.linear(out)


class Model(nn.Module):
    """None-VGG-BiLSTM-CTC recognition network."""

    def __init__(self, input_channel: int, output_channel: int, hidden_size: int, num_class: int):
        super().__init__()
        self.feature_extraction = VGGFeatureExtractor(input_channel, output_channel)
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.sequence_modeling = nn.Sequential(
            BidirectionalLSTM(output_channel, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )
        self.prediction = nn.Linear(hidden_size, num_class)

    def forward(self, x, text=None, is_train=True):
        features = self.feature_extraction(x)
        features = self.adaptive_avg_pool(features.permute(0, 3, 1, 2))
        features = features.squeeze(3)
        seq = self.sequence_modeling(features)
        return self.prediction(seq)
