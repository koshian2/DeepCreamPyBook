import torchvision
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from mask import MaskGenerator

def load_svhn():
    # データをダウンロードするためにTorchVisionを使う
    torchvision.datasets.SVHN("./data", split="extra", download=True)
    extra = np.array(loadmat("./data/extra_32x32.mat")["X"], np.uint8)
    extra = np.transpose(extra, [3,0,1,2]) # sample, height, width, channelに
    # シャッフル
    np.random.seed(114514)
    np.random.shuffle(extra)
    X_train, X_test = extra[10240:], extra[:10240]
    return X_train, X_test

def plot_svhn():
    train, _ = load_svhn()
    plt.subplots_adjust(hspace=0.02, wspace=0.02, top=0.98, bottom=0.02, left=0.02, right=0.98)
    for i in range(100):
        ax = plt.subplot(10, 10, i+1)
        ax.imshow(train[i])
        ax.axis("off")
    plt.show()

def plot_svhn_with_masks():
    _, X_test = load_svhn()
    maskgen = MaskGenerator(64, 64, rand_seed=114514)
    plt.subplots_adjust(hspace=0.02, wspace=0.02, top=0.98, bottom=0.02, left=0.02, right=0.98)
    for i in range(3):
        img = X_test[i].repeat(2, axis=0).repeat(2, axis=1)
        ax = plt.subplot(3,3,3*i+1)
        ax.imshow(img)
        ax.axis("off")
        mask = maskgen.sample()
        ax = plt.subplot(3,3,3*i+2)
        ax.imshow(mask*255)
        ax.axis("off")
        joined = mask * img
        ax = plt.subplot(3,3,3*i+3)
        ax.imshow(joined)
        ax.axis("off")
    plt.show()
