
from mpl_toolkits.mplot3d import Axes3D
import hiddenlayer as hl
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, accuracy_score
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from model_design import AutoEncoder
from data import get_dataset


if __name__ == '__main__':

    train_loader, train_data_y, test_loader, test_data_y = get_dataset(train_batch_size=128)

    lr = 3e-3
    auto_encoder_model = AutoEncoder().cuda()
    optimizer = optim.Adam(auto_encoder_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = hl.History()
    canvas = hl.Canvas()
    
    print("Training...")
    for epoch in range(10):
        train_loss_epoch = 0
        train_num = 0
        for step, images in enumerate(train_loader):
            images = images.cuda()
            _, output = auto_encoder_model(images)
            loss = criterion(output, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
            train_num = train_num + images.size(0)
        train_loss = train_loss_epoch / train_num
        history.log(epoch, train_loss=train_loss)
        with canvas:
            canvas.draw_plot(history['train_loss'])
            
    torch.save({"state_dict": auto_encoder_model.state_dict(),}, "./checkpoint/auto_encoder.pth")
    print('Save model done!')