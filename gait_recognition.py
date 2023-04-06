import torch 
from sklearn.model_selection import train_test_split

from train import train_spae
from dataset import GaitDataset
from stacked_auto_encoders import SPAE_With_Auto_Encoder as SPAE 


if __name__ == "__main__":
    # Please define your own dataset folder address
    img_folder = "" 
    labels = ""
    batch_size = 64
    layer_dims = [64*64, 512, 256, 128]

    # Initialize the dataset
    data = GaitDataset(img_folder)

    train_model = False
    if train_model:
        train_spae(img_folder, batch_size, layer_dims)
    
    # Load the trained model
    checkpoint = torch.load('spae.pth')
    spae = SPAE(layer_dims)
    spae.load_state_dict(checkpoint['state_dict'])
    spae.eval()

 