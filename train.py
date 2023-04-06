import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import GaitDataset
from stacked_auto_encoders import SPAE_With_Auto_Encoder as SPAE


def train_spae(img_folder, batch_size, layer_dims, epochs):
    # Load the dataset and the dataloader
    gait_dataset = GaitDataset(img_folder)
    train_loader = DataLoader(gait_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model, optimizer, and loss function
    model = SPAE(layer_dims)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    # Train the model
    for e in epochs:
        epoch_loss = 0.0

        for cycle in train_loader:
            x = cycle.view(cycle.size(0), -1)
            for i in range(len(model.spae)):
                encoded, decoded = model.spae[i](x)
                x = encoded
                loss = loss_fn(decoded, x)
                epoch_loss += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if e % 10 == 0:
            print("Epoch {}, loss {}".format(e+1, epoch_loss))
    
    # Save the model
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, 'train_spae.pth')


