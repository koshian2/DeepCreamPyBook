import numpy as np

def random_erasing(image, prob=0.5, sl=0.2, sh=0.8, r1=0.2, r2=0.8):
    # パラメーター
    # - image = 入力画像
    # - prob = random erasingをする確率
    # - sl, sh = random erasingする面積の比率[sl, sh]
    # - r1, r2 = random erasingのアスペクト比[r1, r2]
    assert image.ndim == 3
    assert image.dtype == np.uint8
    if np.random.rand() >= prob:
        return image
    else:
        H, W, C = image.shape # 縦横チャンネル
        S = H * W # 面積
        while True:
            S_eps = np.random.uniform(sl, sh) * S
            r_eps = np.random.uniform(r1, r2)
            H_eps, W_eps = np.sqrt(S_eps*r_eps), np.sqrt(S_eps/r_eps)
            x_eps, y_eps = np.random.uniform(0, W), np.random.uniform(0, H)
            if x_eps + W_eps <= W and y_eps + H_eps <= H:
                out_image = image.copy()
                out_image[int(y_eps):int(y_eps+H_eps), int(x_eps):int(x_eps+W_eps), 
                          :] = int(np.random.uniform(0, 255))
                return out_image

from keras.datasets import cifar10
import matplotlib.pyplot as plt

def random_erasing_demo():
    image = cifar10.load_data()[0][0][0] # X_trainの1枚目の画像だけ取得
    plt.subplots_adjust(hspace=0.02, wspace=0.02, left=0.02, top=0.9, bottom=0.02, right=0.98)
    for i in range(100):
        ax = plt.subplot(10, 10, i+1) # 10x10の分割表示
        ax.imshow(random_erasing(image)) # Random Erasingした画像の表示
        ax.axis("off") # 軸をオフにする
    plt.suptitle("Random Erasing : $p=0.5, s_l=0.2, s_h=0.8, r_1=0.2. r_2=0.8$")
    plt.show()

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import LearningRateScheduler, History
from tensorflow.contrib.tpu.python.tpu import keras_support

from keras.utils import to_categorical
import pickle, os

# VGGライクなブロック
def create_block(inputs, ch, rep):
    x = inputs
    for i in range(rep):
        x = layers.Conv2D(ch, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x

# モデルの作成
def create_model():
    input = layers.Input((32, 32, 3))
    x = create_block(input, 64, 3) # 32x32x64を3層
    x = layers.AveragePooling2D(2)(x)
    x = create_block(x, 128, 3) # 16x16x128を3層
    x = layers.AveragePooling2D(2)(x)
    x = create_block(x, 256, 3) # 8x8x256を3層
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(10, activation="softmax")(x)
    return Model(input, x)

# データのジェネレーター
def data_generator(X, y, batch_size, is_train, use_random_erasing):
    X_cache, y_cache = [], []
    while True:
        indices = np.arange(X.shape[0])
        if is_train:
            np.random.shuffle(indices) # 訓練データならシャッフルする
        for i in indices:
            if use_random_erasing:
                X_cache.append(random_erasing(X[i])) # Random Erasingする場合
            else:
                X_cache.append(X[i]) # しない場合
            y_cache.append(y[i])

            if(len(X_cache) == batch_size):
                X_batch = np.asarray(X_cache, dtype=np.float32) / 255.0
                y_batch = np.asarray(y_cache, dtype=np.float32)
                X_cache, y_cache = [], []
                yield X_batch, y_batch

def lr_scheduler(epoch):
    x = 0.4
    if epoch >= 50: x /= 5.0
    if epoch >= 100: x /= 5.0
    if epoch >= 125: x /= 5.0
    return x

def train(use_random_erasing):
    tf.logging.set_verbosity(tf.logging.FATAL)

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = create_model()
    model.compile(SGD(0.4, momentum=0.9), "categorical_crossentropy", ["acc"]) # 学習率0.4のモメンタム

    # TPUモデルに変換
    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    batch_size = 512
    cb = LearningRateScheduler(lr_scheduler) # 学習率減衰
    hist = History()
    model.fit_generator(data_generator(X_train, y_train, batch_size, True, use_random_erasing),
                        steps_per_epoch=X_train.shape[0]//batch_size,
                        validation_data=data_generator(X_test, y_test, batch_size, False, False),
                        validation_steps=X_test.shape[0]//batch_size,
                        callbacks=[cb, hist], epochs=150)

    history = hist.history
    with open(f"random_erasing_{use_random_erasing}.dat", "wb") as fp:
        pickle.dump(history, fp)

if __name__ == "__main__":
    for flag in [False, True]:
        K.clear_session()
        train(flag)
