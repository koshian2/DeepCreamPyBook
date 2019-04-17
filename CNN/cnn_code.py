## CIFAR-10の呼び出し
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

## ロジスティック回帰＋CIFAR-10
from keras.datasets import cifar10
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def logistic_regression():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # (50000, 32, 32, 3) (50000, 1) (10000, 32, 32, 3) (10000, 1)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # ロジスティック回帰にするため画像をベクトル化する
    X_train = X_train.astype(np.float32).reshape(X_train.shape[0], -1) / 255.0
    X_test = X_test.astype(np.float32).reshape(X_test.shape[0], -1) / 255.0
    print(X_train.shape, X_test.shape) # (50000, 3072) (10000, 3072)
    # ロジスティック回帰
    classifer = LogisticRegression().fit(X_train, y_train)
    y_pred = classifer.predict(X_test)
    # テスト精度 2時間弱かかる
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)
    
## MLP
from keras.datasets import cifar10
from keras import layers
from keras.models import Model
from keras.utils import to_categorical

def dense_bn_relu(input, ch):
    x = layers.Dense(ch)(input)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def cifar_mlp():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # ラベルをOnehotベクトルにする
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    # ネットワークを作る
    input = layers.Input((32,32,3))
    x = layers.Flatten()(input) # 多層パーセプトロンにするためベクトル化する
    x = dense_bn_relu(x, 1024)
    x = dense_bn_relu(x, 256)
    x = dense_bn_relu(x, 128)
    x = layers.Dense(10, activation="softmax")(x)
    model = Model(input, x)
    # 訓練
    model.compile("adam", "categorical_crossentropy", ["acc"])
    model.fit(X_train/255.0, y_train, batch_size=128, epochs=100, validation_data=
    (X_test/255.0, y_test))
    
## AlexNetもどき
from keras.datasets import cifar10
from keras import layers
from keras.models import Model
from keras.utils import to_categorical

def conv_bn_relu(input, ch):
    x = layers.Conv2D(ch, kernel_size=3, padding="same")(input)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def cnn():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    # AlexNetもどきを作る
    input = layers.Input((32,32,3))
    x = conv_bn_relu(input, 96)
    x = layers.AveragePooling2D(pool_size=2)(x)
    x = conv_bn_relu(x, 256)
    x = layers.AveragePooling2D(pool_size=2)(x)
    x = conv_bn_relu(x, 384)
    x = conv_bn_relu(x, 384)
    x = conv_bn_relu(x, 256)
    x = layers.Flatten()(x)
    x = layers.Dense(10, activation="softmax")(x)
    model = Model(input, x)
    # 訓練
    model.compile("adam", "categorical_crossentropy", ["acc"])
    model.fit(X_train/255.0, y_train, batch_size=128, epochs=100, validation_data=
    (X_test/255.0, y_test))
