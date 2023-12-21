import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRNN(nn.Module):
    def __init__(
        self, num_chars: int, dropout: float = 0.2, bidirectional: bool = True
    ):
        super(ConvRNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Convert (bs, C, H, W) to (bs, W, H*C) and project to (bs, W, hidden_dim)
        # We want W to be present as sequence length, as the width of the image
        # has the information of the text.
        proj_out_features = 256
        self.projection = self.projection_layer(out_features=proj_out_features)
        self.drop1 = nn.Dropout(dropout)

        # RNN
        rnn_hidden_dim = 128
        self.rnn = nn.GRU(
            input_size=proj_out_features,
            hidden_size=rnn_hidden_dim,
            num_layers=2,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )

        # out_features = num_chars, we already added black char " " with code `0` in dataset
        bidir_factor = 2 if bidirectional else 1
        self.output_layer = nn.Linear(
            in_features=rnn_hidden_dim * bidir_factor, out_features=num_chars
        )

    def projection_layer(self, out_features: int):
        def forward(x):
            bs, c, h, w = x.shape
            # Bring W of image as seq_length
            x = x.permute(0, 3, 1, 2)  # bs,c,h,w -> bs,w,c,h
            x = x.view(bs, w, -1)  # bs,w,c,h -> bs,w,c*h
            proj = nn.Linear(in_features=x.shape[-1], out_features=out_features).to(
                x.device
            )
            return proj(x)

        return forward

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)

        # if self.projection is None:
        #     bs, c, h, w = x.shape
        #     # Bring W of image as seq_length
        #     x = x.permute(0, 3, 1, 2)  # bs,c,h,w -> bs,w,c,h
        #     x = x.view(bs, w, -1)  # bs,w,c,h -> bs,w,c*h
        #     self.projection = nn.Linear(in_features=x.shape[-1], out_features=128)

        x = self.drop1(self.projection(x))

        x, _ = self.rnn(x)

        x = self.output_layer(x)

        return F.log_softmax(x, dim=-1)



if __name__ == "__main__":
    model = ConvRNN(num_chars=20)
    rand_input = torch.rand((2, 3, 75, 300))
    output = model(rand_input)
    print(output.shape)