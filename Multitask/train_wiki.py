import tensorflow as tf
from tensorflow.keras.applications import mobilenet
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, History
from tensorflow.contrib.tpu.python.tpu import keras_support

import numpy as np
import os, pickle
from PIL import Image

def create_model():
    model = mobilenet.MobileNet(include_top=False, input_shape=(160,160,3),
                                weights="imagenet", pooling="avg")
    x = layers.Dense(2)(model.layers[-1].output) # 活性化関数は入れない
    x = layers.Lambda(multitask_activation)(x)
    return Model(model.layers[0].input, x)

# マルチタスクな活性化関数
def multitask_activation(inputs):
    # 0はSigmoid、1はReLUをかける
    sigmoid = K.sigmoid(inputs[:,0])
    relu = K.relu(inputs[:,1])
    result = K.stack([sigmoid, relu], axis=-1)
    return result

# マルチタスクな損失関数
def multitask_loss(y_true, y_pred):
    cross_entropy = K.binary_crossentropy(y_true[:,0], y_pred[:,0])
    mse = (y_true[:,1] - y_pred[:,1]) ** 2
    return cross_entropy + 0.1 * mse # MSEを1/10にして重み付け

# メモリー対策にジェネレーターを作る
def data_generator(image, gender, age, select_index, batch_size, shuffle):
    indices = select_index.copy()
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for i in range(len(indices)//batch_size):
            current_indices = indices[i*batch_size:(i+1)*batch_size]
            X_batch = mobilenet.preprocess_input(image[current_indices])
            y_batch = np.c_[gender[current_indices], age[current_indices]]
            yield X_batch, y_batch

# モデルのチェックポイント
class Checkpoint(Callback):
    def __init__(self, model):
        self.model = model
        self.min_loss = np.inf

    def on_epoch_end(self, epoch, logs):
        if logs["val_loss"] < self.min_loss:
            print(f"Val loss improved {self.min_loss:.04} to {logs['val_loss']:.04}")
            self.min_loss = logs["val_loss"]
            self.model.save_weights("wiki_model.hdf5")

def train():
    # データの読み込み
    data = np.load("wiki_crop/wiki_all.npz")
    image, gender, age = data["image"], data["gender"], data["age"]
    # TrainとTestのSplit（インデックスを指定するだけ）
    np.random.seed(45)
    indices = np.random.permutation(image.shape[0])
    n_test = 8192 # 8192枚をテストとする
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # モデルの作成
    model = create_model()
    # 損失関数を自作してコンパイル
    model.compile(tf.train.RMSPropOptimizer(3e-3), multitask_loss)
    # TPUモデルに変換
    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    # 訓練
    batch_size = 512
    history = History()
    checkpoint = Checkpoint(model)
    model.fit_generator(data_generator(image, gender, age, train_indices, batch_size, True), 
                        steps_per_epoch=len(train_indices)//batch_size,
                        validation_data=data_generator(image, gender, age, test_indices, batch_size, False),
                        validation_steps=len(test_indices)//batch_size,
                        max_queue_size=1, callbacks=[history, checkpoint], epochs=50)

    # 結果保存
    hist = history.history
    with open("history.dat", "wb") as fp:
        pickle.dump(hist, fp)

def predict(image_path):
    # モデルの読み込み
    model = create_model()
    model.load_weights("wiki_model.hdf5")
    # 対象画像
    image = np.asarray(Image.open(image_path).convert("RGB").resize((160,160)), np.uint8)
    x = mobilenet.preprocess_input(np.expand_dims(image, axis=0))
    # 推論
    pred = model.predict(x)
    print(image_path, pred)


if __name__ == "__main__":
    K.clear_session()
    train()
