import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from model_design import AutoEncoder
from data import get_dataset


if __name__ == '__main__':

    train_loader, train_data_y, test_loader, test_data_y = get_dataset()
    auto_encoder_model = AutoEncoder()

    checkpoint = torch.load("./checkpoint/auto_encoder.pth")
    auto_encoder_model.load_state_dict(checkpoint["state_dict"])
    auto_encoder_model.eval()
    print("Load model done")

    for i, test_data in enumerate(test_loader):
        if i > 0:
            break
    _, test_decoder = auto_encoder_model(test_data)

    plt.figure(figsize=(10, 10))
    for i in range(test_decoder.shape[0]):
        plt.subplot(10, 10, i + 1)
        img = test_data[i, :]
        img = img.data.numpy().reshape(28, 28)
        plt.imshow(img, 'gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(test_decoder.shape[0]):
        plt.subplot(10, 10, i + 1)
        img = test_decoder[i, :]
        img = img.data.numpy().reshape(28, 28)
        plt.imshow(img, 'gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


