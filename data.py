import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

def get_dataset(root="../data/", batch_size=64, num_workers=4):

    train_data = MNIST(
        root=root,
        train=True,
        transform=transforms.ToTensor(),
        download=False,
    )

    test_data = MNIST(
        root=root,
        train=False,
        download=False,
    )

    train_data_x = train_data.data.type(torch.FloatTensor) / 255.0
    train_data_x = train_data_x.reshape(train_data_x.shape[0], -1)
    train_data_y = train_data.targets

    train_loader = Data.DataLoader(
        dataset=train_data_x,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
    test_data_x = test_data_x.reshape(test_data_x.shape[0], -1)
    test_data_y = test_data.targets

    test_loader = Data.DataLoader(
        dataset=test_data_x,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    print(f"train_data_x.shape: {train_data_x.shape}")
    print(f"test_data_x.shape: {test_data_x.shape}")

    return train_loader, train_data_y, test_loader, test_data_y

if __name__ == '__main__':

    train_loader, train_data_y, test_loader, test_data_y = get_dataset()
    
    for step, image in enumerate(train_loader):
        if step > 0:
            break

    img = make_grid(image.reshape(-1, 1, 28, 28))
    img = img.data.numpy().transpose(1, 2, 0)
    plt.figure()
    plt.imshow(img, 'gray')
    plt.show()