from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from model_design import AutoEncoder
from data import get_dataset
import torch

if __name__ == '__main__':
    train_loader, train_data_y, test_loader, test_data_y = get_dataset(train_batch_size=60000, test_batch_size=10000)
    auto_encoder_model = AutoEncoder()

    checkpoint = torch.load("./checkpoint/auto_encoder.pth")
    auto_encoder_model.load_state_dict(checkpoint["state_dict"])
    auto_encoder_model.eval()
    print("Load model done")

    for i, train_data_x in enumerate(train_loader):
        if i > 0:
            break
    train_x_encoder, _ = auto_encoder_model(train_data_x)
    train_x_encoder = train_x_encoder.data.numpy()
    train_y = train_data_y.data.numpy()

    for i, test_data_x in enumerate(test_loader):
        if i > 0:
            break
    test_x_encoder, _ = auto_encoder_model(test_data_x)
    test_x_encoder = test_x_encoder.data.numpy()
    test_y = test_data_y.data.numpy()

    pca_model = PCA(n_components=3, random_state=100)
    pca_train_x = pca_model.fit_transform(train_data_x.data.numpy())
    pca_test_x = pca_model.fit_transform(test_data_x.data.numpy())
    print(f"pca_train_x.shape: {pca_train_x.shape}")
    
    encoder_svc = SVC(kernel='rbf', random_state=123)
    encoder_svc.fit(train_x_encoder, train_y)
    pred_svc = encoder_svc.predict(test_x_encoder)
    print(classification_report(test_y, pred_svc))