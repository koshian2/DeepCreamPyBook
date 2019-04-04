import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.contrib.tpu.python.tpu import keras_support

import numpy as np
import os
from PIL import Image
from libs.utils import *

## RGB <- -> YCbCrの変換行列（BT.601）
# https://ja.wikipedia.org/wiki/YUV
# R,G,B,Y=[0,1], Cb,Cr=[-0.5,0.5]
RGB2YCbCr = np.array([[0.299, 0.587, 0.114],
                      [-0.168736, -0.331264, 0.5],
                      [0.5, -0.418688, -0.081312]], np.float32).T
YCbCr2RGB = np.array([[1.0, 0, 1.402],
                      [1.0, -0.344136,  -0.714136],
                      [1.0, 1.772, 0]], np.float32).T

def conv_bn_relu(input, ch, kernel_size, reps):
    x = input
    for i in range(reps):
        x = layers.Conv2D(ch, kernel_size, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x

def upsampling2d_tpu(inputs, scale=2):
    x = K.repeat_elements(inputs, scale, axis=1)
    x = K.repeat_elements(x, scale, axis=2)
    return x

# main側のUpsampling＋SkipConnectionのConcat
def upsampling_concat(main_tensor, skip_connection, upsampling_scale):
    x = layers.Lambda(upsampling2d_tpu, 
                      arguments={"scale":upsampling_scale})(main_tensor)
    return layers.Concatenate()([x, skip_connection])

# YCbCrのテンソルをRGBに変換
def convert_rgb(ycbcr_tensor):
    rgb = K.dot(ycbcr_tensor, K.variable(YCbCr2RGB))
    return K.clip(rgb, 0, 1)

# 大域特徴量を取る
def squeeze_global_features(conv_tensor, ch, size):
    x = layers.GlobalAveragePooling2D()(conv_tensor)
    x = layers.Dense(ch)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Reshape((1,1,ch))(x)
    x = layers.Lambda(upsampling2d_tpu, 
                      arguments={"scale":size})(x)
    return x

def create_unet():
    # 輝度（Y）を入力とし、色差を予測する（CbCr）
    input_y = layers.Input((512, 512, 1)) # 入力は輝度（ch=1）
    # Encoder
    conv1 = conv_bn_relu(input_y, 32, 5, 1) # 512x512x32
    conv2 = conv_bn_relu(layers.AveragePooling2D(4)(conv1), 64, 3, 2) # 128x128x64
    conv3 = conv_bn_relu(layers.AveragePooling2D(2)(conv2), 128, 3, 2) # 64x64x128
    conv4 = conv_bn_relu(layers.AveragePooling2D(2)(conv3), 256, 3, 2) # 32x32x256
    conv5 = conv_bn_relu(layers.AveragePooling2D(2)(conv4), 512, 3, 2) # 16x16x512
    middle = conv_bn_relu(layers.AveragePooling2D(2)(conv4), 512, 3, 2) # 8x8x512
    large = conv_bn_relu(layers.AveragePooling2D(2)(middle), 512, 3, 2) # 4x4x512
    # 大域特徴量を入れる
    middle_features = squeeze_global_features(middle, 512, 16)
    large_features = squeeze_global_features(large, 512, 16)
    x = layers.Concatenate()([conv5, middle_features, large_features]) # 16x16x1536
    x = conv_bn_relu(x, 1024, 1, 1)
    # Decoder
    x = conv_bn_relu(upsampling_concat(x, conv4, 2), 256, 3, 2) # 32x32x256
    x = conv_bn_relu(upsampling_concat(x, conv3, 2), 128, 3, 2) # 64x64x128
    x = conv_bn_relu(upsampling_concat(x, conv2, 2), 64, 3, 2) # 128x128x64
    x = conv_bn_relu(upsampling_concat(x, conv1, 4), 32, 5, 1) # 512x512x32
    x = layers.Conv2D(2, 1, activation="tanh", padding="same")(x) # 色差
    cbcr = layers.Lambda(lambda input: 0.5*input)(x) # 出力の色差は[-0.5,0.5]のスケール
    # YCbCr -> RGB
    pred = layers.Concatenate()([input_y, cbcr]) # 輝度（Y）は入力の値をそのまま使う
    pred = layers.Lambda(convert_rgb)(pred) # RGB[0,1]の色空間に変更

    model = Model(input_y, pred) 
    return model

def jointed_loss(y_true, y_pred):
    # 合成損失関数＝色差（CrCb）の損失関数＋RGBの損失関数
    # 色差の損失関数　→　YCrCbのCrCbのL1距離
    ycbcr_convert_matrix = K.variable(RGB2YCbCr)
    ycbcr_true = K.dot(y_true, ycbcr_convert_matrix)
    ycbcr_pred = K.dot(y_pred, ycbcr_convert_matrix)
    color_difference_loss = K.mean(K.abs(ycbcr_true[:,:,:,1:]-ycbcr_pred[:,:,:,1:]), axis=(1,2,3))
    # RGBのL1距離
    rgb_loss = K.mean(K.abs(y_true-y_pred), axis=(1,2,3))
    return 2*color_difference_loss+rgb_loss

def PSNR(y_true, y_pred):
    # RGB[0,1]なのでMax_Iは1.0
    return 20 * K.log(1.0) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(y_true-y_pred), axis=(1,2,3))) / K.log(10.0) 

# uint8のRGB→floatのRGBとYに変換
def preprocess(rgb_inputs):
    assert rgb_inputs.ndim == 4
    assert rgb_inputs.dtype == np.uint8
    # floatのRGBに
    float_rgb = (rgb_inputs / 255.0).astype(np.float32)
    # YCbCrのYだけ取る
    yvalue = np.dot(float_rgb, RGB2YCbCr[:,:1])
    return yvalue, float_rgb

# floatのRGB→uintのRGBに変換
def deprocess(rgb_inputs):
    x = np.clip(rgb_inputs*255.0, 0.0, 255.0)
    return x.astype(np.uint8)

def image_generaor(paths, batch_size, shuffle):
    # 輝度（Y）の画像と、RGBの画像を用意
    imgs = []
    while True:
        indices = np.arange(len(paths))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            with Image.open(paths[i]) as img:
                resize = img.convert("RGB").resize((512, 512), Image.BILINEAR)
                img_array = np.asarray(resize, np.uint8)
                imgs.append(img_array)

            if len(imgs) == batch_size:
                img_batch = np.asarray(imgs, np.uint8)
                y_batch, rgb_batch = preprocess(img_batch)
                imgs = []
                yield y_batch, rgb_batch

class UNetCallback(Callback):
    def __init__(self, model, sampling_paths):
        assert len(sampling_paths) == 32
        self.model = model
        self.gen = image_generaor(sampling_paths, 32, False)
        self.min_val_loss = np.inf

    def on_epoch_end(self, epoch, logs):
        # サンプリング
        gray, y_true = next(self.gen)
        y_pred = self.model.predict(gray)
        y_true, y_pred = deprocess(y_true), deprocess(y_pred)
        tile_images(f"sampling/{epoch:03}.png", y_true, y_pred, 
                    f"Ground Truth - Pred / epoch = {epoch:03}")
        # モデルの保存
        if self.min_val_loss > logs["val_loss"]:
            self.model.save_weights("waifu_unet.hdf5", save_format="h5")
            print(f"Val loss improved from {self.min_val_loss:.04} to {logs['val_loss']:.04}")
            self.min_val_loss = logs["val_loss"]

def train(image_directory):
    train, test = load_wifus_data(image_directory)

    tf.logging.set_verbosity(tf.logging.FATAL)
    model = create_unet()
    model.summary()
    model.compile(tf.train.AdamOptimizer(2e-5), jointed_loss, [PSNR])

    # TPUモデルに変換
    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    batch_size = 16
    cb = UNetCallback(model, test[:32])
    model.fit_generator(image_generaor(train, batch_size, True), 
                        steps_per_epoch=len(train)//batch_size,
                        validation_data=image_generaor(test, batch_size, False),
                        validation_steps=len(test)//batch_size,
                        callbacks=[cb], max_queue_size=5, epochs=45)

if __name__ == "__main__":
    K.clear_session()
    train("images")
