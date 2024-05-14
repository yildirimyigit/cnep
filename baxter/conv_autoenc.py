import torch
import torch.nn as nn
import torch.nn.functional as F


'''
Convolutional Autoencoder for processing images on Baxter
'''
class ConvAE(nn.Module):
    def __init__(self, filter_sizes=[2048,1024,512,256]) -> None:
        super(ConvAE, self).__init__()

        nof_layers = len(filter_sizes)

        # # Encoder
        # layers = []
        # layers.append(nn.Conv2d(3, filter_sizes[0], kernel_size=3, stride=1, padding=1))
        # layers.append(nn.ReLU())
        # layers.append(nn.MaxPool2d(2))
        # for i in range(1, nof_layers):
        #     layers.append(nn.Conv2d(filter_sizes[i-1], filter_sizes[i], kernel_size=3, stride=1, padding=1))
        #     layers.append(nn.ReLU())
        #     layers.append(nn.MaxPool2d(2))

        # self.encoder = nn.Sequential(*layers)
        
        # # Decoder
        # layers = []
        # for i in range(nof_layers, 1):
        #     layers.append(nn.ConvTranspose2d(filter_sizes[i], filter_sizes[i-1], kernel_size=3, stride=2, padding=1, output_padding=1))
        #     layers.append(nn.ReLU())
        # layers.append(nn.ConvTranspose2d(filter_sizes[-1], 3, kernel_size=3, stride=2, padding=1, output_padding=1))
        # # layers.append(nn.Sigmoid())  # Sigmoid so output is between 0 and 1

        # self.decoder = nn.Sequential(*layers)

        # Encoder
        layers = []
        layers.append(nn.Conv2d(3, filter_sizes[0], kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))  # Downsample by a factor of 2
        for i in range(1, nof_layers):
            layers.append(nn.Conv2d(filter_sizes[i-1], filter_sizes[i], kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))  # Downsample by a factor of 2

        self.encoder = nn.Sequential(*layers)
        
        # Decoder
        layers = []
        for i in range(nof_layers - 1, 0, -1):
            layers.append(nn.ConvTranspose2d(filter_sizes[i], filter_sizes[i-1], kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.ConvTranspose2d(filter_sizes[0], 3, kernel_size=3, stride=2, padding=1, output_padding=1))
        # Uncomment the following line if your output needs to be between 0 and 1
        # layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):  # x: (batch_size, nof_channels, H, H)
        latent = self.encoder(x)    # latent: (batch_size, filter_sizes[-1])
        decoded_frames = self.decoder(latent)  # decoded_frames: (batch_size, H, H)
        return decoded_frames

    def loss(self, prediction, target):
        return F.mse_loss(prediction, target)