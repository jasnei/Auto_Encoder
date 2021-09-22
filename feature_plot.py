import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D
from model_design import AutoEncoder
from data import get_dataset


if __name__ == '__main__':

    train_loader, train_data_y, test_loader, test_data_y = get_dataset(test_batch_size=500)
    auto_encoder_model = AutoEncoder()

    checkpoint = torch.load("./checkpoint/auto_encoder.pth")
    auto_encoder_model.load_state_dict(checkpoint["state_dict"])
    auto_encoder_model.eval()
    print("Load model done")

    for i, test_data in enumerate(test_loader):
        if i > 0:
            break
    test_encoder, _ = auto_encoder_model(test_data)
    print(f"test_encoder.shape: {test_encoder.shape}")

    test_encoder_arr = test_encoder.data.numpy()
    
    fig = plt.figure(figsize=(12, 8))
    ax_1 = Axes3D(fig)
    X = test_encoder_arr[:, 0]
    Y = test_encoder_arr[:, 1]
    Z = test_encoder_arr[:, 2]
    ax_1.set_xlim(min(X), max(X))
    ax_1.set_ylim(min(Y), max(Y))
    ax_1.set_zlim(min(Z), max(Z))
    for i in range(test_encoder.shape[0]):
        text = test_data_y.data.numpy()[i]
        ax_1.text(X[i], Y[i], Z[i], str(text), fontsize=8, bbox=dict(boxstyle="round", facecolor=plt.cm.Set1(text), alpha=0.7))
    plt.show()