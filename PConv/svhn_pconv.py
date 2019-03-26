import tensorflow as tf
from tensorflow.keras import layers
from libs.pconv_layer import PConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
import tensorflow.keras.backend as K
from tensorflow.contrib.tpu.python.tpu import keras_support

import numpy as np
from libs.svhn import load_svhn
from libs.mask import MaskGenerator
from libs.loss_layer import LossLayer
from libs.vgg16 import extract_vgg_features
from libs.utils import tile_images
from keras.applications.vgg16 import preprocess_input
import os

# tpuだとUpsampling2Dでエラーになるので独自に定義する
# 参考：https://blog.shikoan.com/tpu-upsampling/
def upsampling2d_tpu(inputs, scale=2):
    x = K.repeat_elements(inputs, scale, axis=1)
    x = K.repeat_elements(x, scale, axis=2)
    return x

def conv_bn_relu(image_in, mask_in, filters, kernel_size, downsampling=1, upsampling=1, act="relu"):
    # Upsamplingする場合
    if upsampling > 1:
        conv = layers.Lambda(upsampling2d_tpu, arguments={"scale":upsampling})(image_in)
        mask = layers.Lambda(upsampling2d_tpu, arguments={"scale":upsampling})(mask_in)
    else:
        conv, mask = image_in, mask_in
    # strideでダウンサンプリング
    conv, mask = PConv2D(filters=filters, kernel_size=kernel_size, 
                         padding="same", strides=downsampling)([conv, mask])
    # Image側だけBN->ReLUを入れる
    conv = layers.BatchNormalization()(conv)
    if act == "relu":
        conv = layers.Activation("relu")(conv)
    elif act == "output":
        conv = layers.Lambda(output_activation, name="unmasked")(conv)
    return conv, mask

def output_activation(input):
    # 画像の出力層の活性化関数（前処理に合わせる）
    # tanhで-1～1のスケールに
    x = K.tanh(input)
    # [-1,1] -> [0,255]のスケール
    x = x * 127.5 + 127.5
    # カラーチャンネルごとのシフト
    mean = np.array([103.939, 116.779, 123.68]).reshape(1,1,1,-1)
    x = x - K.variable(mean)
    return x

def create_model():
    input_image = layers.Input((64,64,3))
    input_mask = layers.Input((64,64,3))
    input_grandtruth = layers.Input((64,64,3))
    ## メインのCNN
    # Encoder
    conv, mask = conv_bn_relu(input_image, input_mask, 128, 3, downsampling=2) # 32x32
    conv, mask = conv_bn_relu(conv, mask, 256, 3, downsampling=2) # 16x16
    conv, mask = conv_bn_relu(conv, mask, 512, 3, downsampling=2) # 8x8
    # Decoder
    conv, mask = conv_bn_relu(conv, mask, 256, 3, upsampling=2) # 16x16
    conv, mask = conv_bn_relu(conv, mask, 128, 3, upsampling=2) # 32x32
    conv, mask = conv_bn_relu(conv, mask, 3, 3, upsampling=2, act="output") # 64x64

    ## 損失関数（VGGの扱いが大変なのでモデル内で損失値を計算する）
    # マスクしていない部分の真の画像＋マスク部分の予測画像
    y_comp = layers.Lambda(lambda inputs: inputs[0]*inputs[1] + (1-inputs[0])*inputs[2])(
        [input_mask, input_grandtruth, conv])
    # vggの特徴量
    vgg_pred_1, vgg_pred_2, vgg_pred_3 = extract_vgg_features(conv, (64,64,3), 0)
    vgg_true_1, vgg_true_2, vgg_true_3 = extract_vgg_features(input_grandtruth, (64,64,3), 1)
    vgg_comp_1, vgg_comp_2, vgg_comp_3 = extract_vgg_features(y_comp, (64,64,3), 2)
    # 画像＋損失
    join = LossLayer()([input_mask, 
                        conv, input_grandtruth, y_comp,
                        vgg_pred_1, vgg_pred_2, vgg_pred_3,
                        vgg_true_1, vgg_true_2, vgg_true_3,
                        vgg_comp_1, vgg_comp_2, vgg_comp_3])
    # lossやmetricsの表示がうまくいかないので出力は1つにする
    return Model([input_image, input_mask, input_grandtruth], join)

# 損失関数側だけ取る
def identity_loss(y_true, y_pred):
    return K.mean(y_pred[:,:,:,3], axis=(1,2))

def PSNR(y_true, y_pred):
    # 参考：https://ja.wikipedia.org/wiki/%E3%83%94%E3%83%BC%E3%82%AF%E4%BF%A1%E5%8F%B7%E5%AF%BE%E9%9B%91%E9%9F%B3%E6%AF%94
    # 前処理は引き算しているだけなのでMaxは255
    pic_gt = y_true[:,:,:,:3]
    pic_pred = y_pred[:,:,:,:3]
    return 20 * K.log(255.0) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(pic_gt-pic_pred), axis=(1,2,3))) / K.log(10.0) 


def load_data():
    # SVHNのロード (extraのうち10240をテスト、残りを訓練）
    X_train, X_test = load_svhn()
    # X_train = X_train[:102400] # 訓練データ多すぎて遅い場合はこのコメントアウトを消す
    # テスト用のマスク画像を作る（テスト用は決定的であったほうが良い）
    # SVHNは32x32だが2倍に拡大して使うので、64x64でマスクを作る
    mask_test = np.zeros((X_test.shape[0], 64, 64, 3), dtype=np.uint8)
    mask_gen = MaskGenerator(64, 64, rand_seed=10)
    for i in range(mask_test.shape[0]):
        mask_test[i] = mask_gen.sample()
    return X_train, X_test, mask_test

def data_generator(image_data, mask_data, batch_size, shuffle):
    image_cache, mask_cache = [], []
    maskgen = MaskGenerator(64, 64)
    while True:
        indices = np.arange(image_data.shape[0])
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            image_cache.append(image_data[i])
            if mask_data is None:
                mask_cache.append(maskgen.sample())
            else:
                mask_cache.append(mask_data[i])
            if len(image_cache) == batch_size:
                batch_gt = np.asarray(image_cache, np.float32)
                batch_gt = batch_gt.repeat(2, axis=1).repeat(2, axis=2) # 2倍に拡大
                batch_mask = np.asarray(mask_cache, np.float32)
                batch_masked_image = batch_gt * batch_mask
                image_cache, mask_cache = [], []
                # 前処理
                batch_gt = preprocess_input(batch_gt)
                batch_masked_image = preprocess_input(batch_masked_image)
                # yはgt, dummy
                batch_y = np.zeros((batch_gt.shape[0], batch_gt.shape[1], batch_gt.shape[2], 4), np.float32)
                batch_y[:,:,:,:3] = batch_gt
                yield [batch_masked_image, batch_mask, batch_gt], batch_y

class SamplingCallback(Callback):
    def __init__(self, model, X_test, mask):
        self.model = model
        self.X_test = X_test
        self.mask = mask
        self.min_val_loss = np.inf

    def on_epoch_end(self, epoch, logs):
        # エポックごとにマスク修復の訓練の進みを可視化して保存
        gen = data_generator(self.X_test, self.mask, 16, False)
        [masked, mask, gt], _ = next(gen)
        unmasked = self.model.predict([masked, mask, gt])[:,:,:,:3]
        tile_images(masked, unmasked, gt, f"sampling/epoch_{epoch:03}.png")
        # モデルの保存
        if self.min_val_loss > logs["val_loss"]:
            print(f"Val loss improved {self.min_val_loss:.04} to {logs['val_loss']:.04}")
            self.min_val_loss = logs["val_loss"]
            self.model.save_weights("svhn_pconv_train.hdf5" ,save_format="h5")

def lr_schduler(epoch):
    x = 5e-4
    if epoch >= 10: x /= 10.0
    return x

def train():
    X_train, X_test, mask_test = load_data()

    model = create_model()
    model.compile("rmsprop", identity_loss, [PSNR])
    model.summary()

    # TPUモデルに変換
    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)
 
    batch_size=768
    cb = SamplingCallback(model, X_test, mask_test)
    scheduler = LearningRateScheduler(lr_schduler)

    model.fit_generator(data_generator(X_train, None, batch_size, True), 
                        steps_per_epoch=X_train.shape[0]//batch_size,
                        validation_data=data_generator(X_test, mask_test, batch_size, False),
                        validation_steps=X_test.shape[0]//batch_size,
                        callbacks=[cb, scheduler], epochs=20, max_queue_size=1)

def convert_to_pred_model():
    # 訓練モデルを推論モデルに変換
    train_model = create_model()
    train_model.load_weights("svhn_pconv_train.hdf5")
    image = train_model.inputs[0]
    mask = train_model.inputs[1]
    output = train_model.get_layer("unmasked").output
    model = Model([image, mask], output)
    # 推論モデルを保存
    model.save_weights("svhn_pconv_pred.hdf5")
    model.summary()

    # テストする
    _, X_test, mask_test = load_data()
    [batch_img, batch_mask, batch_gt], _ = next(data_generator(X_test[100:], mask_test[100:], 16, False))
    batch_pred = model.predict([batch_img, batch_mask], verbose=1)
    tile_images(batch_img, batch_pred, batch_gt, "./pred_sample.png")
    print("推論モデルの結果をpred_sample.pngに保存しました")


if __name__ == "__main__":
    K.clear_session()
    #train()
    convert_to_pred_model()
