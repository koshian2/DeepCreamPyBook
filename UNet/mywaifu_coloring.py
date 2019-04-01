import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.contrib.tpu.python.tpu import keras_support

import numpy as np
import os
from PIL import Image
from libs.utils import *
from libs.vgg16 import convert_to_caffe_colorscale, extract_vgg_features
from libs.loss_layer import LossLayer

def conv_bn_relu(input, ch, kernel_size):
    x = layers.Conv2D(ch, kernel_size, padding="same")(input)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)

def upsampling2d_tpu(inputs, scale=2):
    x = K.repeat_elements(inputs, scale, axis=1)
    x = K.repeat_elements(x, scale, axis=2)
    return x

# main側のUpsampling＋SkipConnectionのConcat
def upsampling_concat(main_tensor, skip_connection, upsampling_scale):
    x = layers.Lambda(upsampling2d_tpu, 
                      arguments={"scale":upsampling_scale})(main_tensor)
    return layers.Concatenate()([x, skip_connection])

# U-Netライクなアーキテクチャを作る
def create_train_unet():
    input_gray = layers.Input((512, 512, 1)) # 入力はモノクロ（ch=1）
    input_color = layers.Input((512, 512, 3))
    # 入力も出力もカラースケールは[-1,1]とする（tfのカラースケール）
    # Encoder
    conv1 = conv_bn_relu(input_gray, 32, 5) # 512x512x32
    conv2 = conv_bn_relu(layers.AveragePooling2D(4)(conv1), 64, 5) # 128x128x64
    conv3 = conv_bn_relu(layers.AveragePooling2D(4)(conv2), 128, 3) # 32x32x128
    conv4 = conv_bn_relu(layers.AveragePooling2D(2)(conv3), 256, 3) # 16x16x256
    conv5 = conv_bn_relu(layers.AveragePooling2D(2)(conv4), 512, 3) # 8x8x512
    conv6 = conv_bn_relu(layers.AveragePooling2D(2)(conv5), 1024, 3) # 4x4x1024
    # Decoder
    x = conv_bn_relu(upsampling_concat(conv6, conv5, 2), 512, 3) # 8x8x512
    x = conv_bn_relu(upsampling_concat(x, conv4, 2), 256, 3) # 16x16x256
    x = conv_bn_relu(upsampling_concat(x, conv3, 2), 128, 3) # 32x32x128
    x = conv_bn_relu(upsampling_concat(x, conv2, 4), 64, 5) # 128x128x64
    x = conv_bn_relu(upsampling_concat(x, conv1, 4), 32, 5) # 512x512x32
    pred = layers.Conv2D(3, 1, activation="tanh", padding="same")(x) # 出力はカラー（ch=3）

    # StyleLossを使いたいので、P-Convと同様に、画像＋損失の「損失関数のレイヤー」を作る
    # VGG用にカラースケールの変換(tf->Caffe)
    gt_caffe = layers.Lambda(convert_to_caffe_colorscale)(input_color)
    pred_caffe = layers.Lambda(convert_to_caffe_colorscale)(pred)
    # VGGの特徴量の取得
    vgg_true_1, vgg_true_2, vgg_true_3 = extract_vgg_features(gt_caffe, (512,512,3), 0)
    vgg_pred_1, vgg_pred_2, vgg_pred_3 = extract_vgg_features(pred_caffe, (512,512,3), 1)
    # 損失関数
    join = LossLayer()([pred, input_color, # この2つはTFのカラースケールで与える 
                        vgg_pred_1, vgg_pred_2, vgg_pred_3,
                        vgg_true_1, vgg_true_2, vgg_true_3])

    model = Model([input_gray, input_color], join) 
    return model

# 損失関数（形式上）
def identity_loss(y_true, y_pred):
    return K.mean(y_pred[:,:,:,3], axis=(1,2))

def PSNR(y_true, y_pred):
    # tfのカラースケールなのでMaxは2.0
    pic_gt = y_true[:,:,:,:3]
    pic_pred = y_pred[:,:,:,:3]
    return 20 * K.log(2.0) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(pic_gt-pic_pred), axis=(1,2,3))) / K.log(10.0) 

def image_generaor(paths, batch_size, shuffle):
    # グレーイメージとカラーイメージの配列を用意
    gray_img, color_img = [], []
    while True:
        indices = np.arange(len(paths))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            with Image.open(paths[i]) as img:
                resize = img.convert("RGB").resize((512, 512), Image.BILINEAR)
                color = np.asarray(resize, np.uint8)
                gray = np.expand_dims(np.asarray(resize.convert("L"), np.uint8), axis=-1)
            gray_img.append(gray)
            color_img.append(color)
            if len(color_img) == batch_size:
                color_batch = np.asarray(color_img, np.float32)
                gray_batch = np.asarray(gray_img, np.float32)
                gray_img, color_img = [], []
                # 前処理
                color_batch = preprocess_inputs(color_batch)
                gray_batch = preprocess_inputs(gray_batch)
                # y側はC+1にする
                y_batch = np.zeros((*color_batch.shape[:3], color_batch.shape[3]+1), np.float32)
                y_batch[:, :, :, :3] = color_batch
                yield [gray_batch, color_batch], y_batch

class UNetCallback(Callback):
    def __init__(self, model, sampling_paths):
        assert len(sampling_paths) == 32
        self.model = model
        self.gen = image_generaor(sampling_paths, 32, False)
        self.min_val_loss = np.inf

    def on_epoch_end(self, epoch, logs):
        # サンプリング
        [gray, y_true], _ = next(self.gen)
        y_pred = self.model.predict([gray, y_true])[:,:,:,:3]
        y_true, y_pred = deprocess_inputs(y_true), deprocess_inputs(y_pred)
        tile_images(f"sampling/{epoch:03}.png", y_true, y_pred, 
                    f"Ground Truth - Pred / epoch = {epoch:03}")
        # モデルの保存
        if self.min_val_loss > logs["val_loss"]:
            self.model.save_weights("waifu_unet.hdf5", save_format="h5")
            print(f"Val loss improved from {self.min_val_loss:.04} to {logs['val_loss']:.04}")
            self.min_val_loss = logs["val_loss"]

def lr_decay(epoch):
    x = 2e-5
    if epoch >= 10: x /= 5.0
    if epoch >= 15: x /= 5.0
    return x

def train(image_directory):
    train, test = load_wifus_data(image_directory)

    tf.logging.set_verbosity(tf.logging.FATAL)
    model = create_train_unet()
    model.summary()
    model.compile("adam", identity_loss, [PSNR])

    # TPUモデルに変換
    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    batch_size = 8
    cb = UNetCallback(model, test[:32])
    scheduler = LearningRateScheduler(lr_decay)
    model.fit_generator(image_generaor(train, batch_size, True), 
                        steps_per_epoch=len(train)//batch_size,
                        validation_data=image_generaor(test, batch_size, False),
                        validation_steps=len(test)//batch_size,
                        callbacks=[cb, scheduler], max_queue_size=10, epochs=20)


def generator_test(image_directory):
    ## 書き直す

    train, test = load_wifus_data(image_directory) # train, testの画像パスを取得
    G, C = next(image_generaor(train, 32, True)) # 1バッチ分読み出し
    G, C = deprocess_inputs(G), deprocess_inputs(C) # generatorでは前処理がされているので逆変換
    tile_images("./generator_sample.png", C, G, "Color-Gray") # プロット

if __name__ == "__main__":
    K.clear_session()
    train("images")
