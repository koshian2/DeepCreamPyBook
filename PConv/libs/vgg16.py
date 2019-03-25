"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556)
"""

# レイヤーの名前衝突を避けるためにVGG16を自分で作る（KerasのVGG16はinput_tensorがうまく作動しないバグもある）

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras import layers
from tensorflow.keras import utils as keras_utils
from tensorflow.keras import models
import os
import numpy as np



WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


def VGG16(input_shape, cnt,
          **kwargs):
    prefix = str(cnt)+"_"

    # input_tensorをInput以外にするとモデルが作れない
    img_input = layers.Input(input_shape)
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=prefix+'block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=prefix+'block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=prefix+'block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=prefix+'block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name=prefix+'block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=prefix+'block5_pool')(x)

    # Create model.
    model = models.Model(img_input, x, name='vgg16')

    # すべてのレイヤーの係数を訓練しない
    for layer in model.layers:
        layer.trainable = False

    # Load weights.
    weights_path = keras_utils.get_file(
        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        file_hash='6d6bbae143d832006294945121d1f1fc')
    model.load_weights(weights_path, by_name=True)

    return model

def extract_vgg_features(input_tensor, input_shape, cnt):
    # VGGモデルの作成（レイヤーの名前衝突を避けるためにカウンターを入れる）
    model = VGG16(input_shape, cnt)
    # グラフを組み直す
    x = input_tensor
    result = []
    for i, l in enumerate(model.layers):
        if i == 0: continue
        l.trainable = False
        x = l(x)
        if i in [3, 6, 10]:
            result.append(x)
    return result

# ニューラルネットワーク→元の画像に戻すための処理（前処理の逆変換）
# VGG19がCaffeモードの前処理なのでちょっと複雑
def deprocess_image(x):
    img = x.copy()
    # 平均を0にしているのを戻す
    img[:, :, :, 0] += 103.939
    img[:, :, :, 1] += 116.779
    img[:, :, :, 2] += 123.68
    # CaffeモードではBGRで定義しているので、BGR->RGBへの変換をする
    img = img[:, :, :, ::-1]
    return np.clip(img, 0, 255).astype('uint8')

