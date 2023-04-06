import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class SPAE_With_Auto_Encoder(nn.Module):
    def __init__(self, layer_dims) -> None:
        super(SPAE_With_Auto_Encoder, self).__init__()
        self.layer_dims = layer_dims
        self.spae = nn.ModuleList()

        for i in range(len(layer_dims)-1):
            self.spae.append(AutoEncoder(layer_dims[i], layer_dims[i+1]))
    
    def forward(self, x):
        for i in range(len(self.spae)):
            x = self.autoencoders[i].encoder(x)

        return x


class SPAE_No_AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(SPAE_No_AutoEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Define encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims)):
            if i == 0:
                encoder_layer = nn.Linear(input_dim, hidden_dims[i])
            else:
                encoder_layer = nn.Linear(hidden_dims[i-1], hidden_dims[i])
            self.encoder_layers.append(encoder_layer)
        
        # Define decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1, -1, -1):
            if i == len(hidden_dims) - 1:
                decoder_layer = nn.Linear(hidden_dims[i], input_dim)
            else:
                decoder_layer = nn.Linear(hidden_dims[i+1], hidden_dims[i])
            self.decoder_layers.append(decoder_layer)
    
    def forward(self, x):
        # Encode
        for layer in self.encoder_layers:
            x = torch.relu(layer(x))
        
        # Decode
        for layer in self.decoder_layers:
            x = torch.relu(layer(x))
        
        return x