## MNISTの可視化
from keras.datasets import mnist
import matplotlib.pyplot as plt

def plot_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    plt.subplots_adjust(hspace=0.02, wspace=0.02, top=0.98, bottom=0.02, left=0.02, right=0.98)
    for i in range(100):
        ax = plt.subplot(10, 10, i+1)
        ax.imshow(X_train[i], cmap="gray")
        ax.axis("off")
    plt.show()

plot_mnist()

## NumpyによるMNISTの可視化
import numpy as np
def view_numpy():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train.shape)
    print(X_train[0])

## シグモイド関数の可視化
import numpy as np
import matplotlib.pyplot as plt
def plot_sigmoid():
    x = np.arange(-5,5,0.01) # -5から5まで0.01刻みにXを作りなさいということ
    y = 1.0/(1+np.exp(-x))
    plt.plot(x, y)
    plt.show()
    
##　ロジスティック回帰による分類
from keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
def logistic_regression():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28*28) / 255.0
    clf = LogisticRegression(multi_class="ovr").fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print("精度")
    print(accuracy_score(y_train, y_pred))
    print("混同⾏列")
    print(confusion_matrix(y_train, y_pred))
    
## Kereasによるロジスティック回帰の分類
from keras.datasets import mnist
from keras import layers
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np

def logistic_keras():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28*28) / 255.0
    y_train = to_categorical(y_train)
    # モデルの定義
    input = layers.Input((784,))
    x = layers.Dense(10, activation="softmax")(input)
    model = Model(input, x)
    # 訓練
    model.compile("adam", "categorical_crossentropy", ["acc"])
    model.fit(X_train, y_train, epochs=10)
    # 評価
    y_pred = model.predict(X_train) # 確率の推定であることに注意
    y_true_label, y_pred_label = np.argmax(y_train, axis=-1), np.argmax(y_pred, axis=-1)
    print("精度")
    print(accuracy_score(y_true_label, y_pred_label))
    print("混同⾏列")
    print(confusion_matrix(y_true_label, y_pred_label))
    
## 隠れ層の追加
from keras.datasets import mnist
from keras import layers
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
def dnn():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 28*28) / 255.0
    y_train = to_categorical(y_train)
    # モデルの定義
    input = layers.Input((784,))
    x = layers.Dense(128, activation="relu")(input) #
    x = layers.Dense(128, activation="relu")(x) # この2⾏を追加しただけ
    x = layers.Dense(10, activation="softmax")(x)
    model = Model(input, x)
    # 訓練
    model.compile("adam", "categorical_crossentropy", ["acc"])
    model.fit(X_train, y_train, epochs=10)
    # 評価
    y_pred = model.predict(X_train)
    y_true_label, y_pred_label = np.argmax(y_train, axis=-1), np.argmax(y_pred, axis=-1)
    print("精度")
    print(accuracy_score(y_true_label, y_pred_label))
    print("混同⾏列")
    print(confusion_matrix(y_true_label, y_pred_label))
