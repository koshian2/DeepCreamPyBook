import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, History
from tensorflow.contrib.tpu.python.tpu import keras_support

from libs.pconv_layer import PConv2D
from libs.loss_layer import LossLayer
from libs.vgg16 import extract_vgg_features
import libs.utils as utils
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from tqdm import tqdm
from PIL import Image

CENSOR_MODE = "mosaic" # hikariなら謎の光

# 画像のカラースケールはNan対策でtf[-1,1]とする。VGGは別に処理する。
def conv_bn_relu(image_in, mask_in, filters, kernel_size, 
                 downsampling=1, upsampling=1, act="relu",
                 concat_img=None, concat_mask=None, reps=1):
    assert not (concat_img is None)^(concat_mask is None) # XORは常にFalse
    # Upsamplingする場合
    if upsampling > 1:
        conv = layers.Lambda(upsampling2d_tpu, arguments={"scale":upsampling})(image_in)
        mask = layers.Lambda(upsampling2d_tpu, arguments={"scale":upsampling})(mask_in)
    else:
        conv, mask = image_in, mask_in
    if concat_img is not None and concat_mask is not None:
        conv = layers.Concatenate()([conv, concat_img])
        mask = layers.Concatenate()([mask, concat_mask])

    for i in range(reps):
        stride = downsampling if i == 0 else 1
        # strideでダウンサンプリング
        conv, mask = PConv2D(filters=filters, kernel_size=kernel_size, 
                             padding="same", strides=stride)([conv, mask])
        # Image側だけBN->ReLUを入れる
        conv = layers.BatchNormalization()(conv)
        if act == "relu":
            conv = layers.Activation("relu")(conv)
        if act == "tanh":
            conv = layers.Activation("tanh", name="unmasked")(conv)
    return conv, mask

def convert_caffe_color_space(tf_color_input):
    # 画像の出力層の活性化関数（前処理に合わせる）
    # [-1,1] -> [0,255]のスケール
    x = tf_color_input * 127.5 + 127.5
    # RGB -> BGR
    x = x[:,:,:,::-1]
    # カラーチャンネルごとのシフト
    mean = np.array([103.939, 116.779, 123.68]).reshape(1,1,1,-1)
    x = x - K.variable(mean)
    return x

def upsampling2d_tpu(inputs, scale=2):
    x = K.repeat_elements(inputs, scale, axis=1)
    x = K.repeat_elements(x, scale, axis=2)
    return x

def create_train_pconv_unet():
    input_image = layers.Input((256,256,3))
    input_mask = layers.Input((256,256,3))
    input_groundtruth = layers.Input((256,256,3))

    # マスクを0-1にして広げる
    expanded_mask = layers.Lambda(lambda x: 1.0-K.sign(1.0-x))(input_mask)

    ## U-Net
    # Encoder
    conv1, mask1 = conv_bn_relu(input_image, expanded_mask, 
                                filters=32, kernel_size=3, downsampling=1, reps=2) # 256x256
    conv2, mask2 = conv_bn_relu(conv1, mask1,
                                filters=64, kernel_size=5, downsampling=4, reps=2) # 64x64
    conv3, mask3 = conv_bn_relu(conv2, mask2,
                                filters=128, kernel_size=3, downsampling=2, reps=2) # 32x32
    conv4, mask4 = conv_bn_relu(conv3, mask3,
                                filters=256, kernel_size=3, downsampling=2, reps=2) # 16x16
    conv5, mask5 = conv_bn_relu(conv4, mask4,
                                filters=512, kernel_size=3, downsampling=2, reps=2) # 8x8
    ## Decoder
    img, mask = conv_bn_relu(conv5, conv5,
                             filters=256, kernel_size=3, upsampling=2, reps=2,
                             concat_img=conv4, concat_mask=mask4) # 16x16
    img, mask = conv_bn_relu(img, mask,
                             filters=128, kernel_size=3, upsampling=2, reps=2,
                             concat_img=conv3, concat_mask=mask3) # 32x32
    img, mask = conv_bn_relu(img, mask,
                             filters=64, kernel_size=3, upsampling=2, reps=2,
                             concat_img=conv2, concat_mask=mask2) # 64x64
    img, mask = conv_bn_relu(img, mask,
                             filters=32, kernel_size=5, upsampling=4, reps=2,
                             concat_img=conv1, concat_mask=mask1) # 256x256
    img, mask = conv_bn_relu(img, mask,
                             filters=3, kernel_size=3, upsampling=1, reps=1, act="tanh")

    ## 損失関数
    # マスクしていない部分の真の画像＋マスク部分の予測画像
    y_comp = layers.Lambda(lambda inputs: inputs[0]*inputs[1] + (1-inputs[0])*inputs[2])(
        [expanded_mask, input_groundtruth, img])
    # Caffeカラースケールに変換
    vgg_in_pred = layers.Lambda(convert_caffe_color_space)(img)
    vgg_in_groundtruth = layers.Lambda(convert_caffe_color_space)(input_groundtruth)
    vgg_in_comp = layers.Lambda(convert_caffe_color_space)(y_comp)
    # vggの特徴量
    vgg_pred_1, vgg_pred_2, vgg_pred_3 = extract_vgg_features(vgg_in_pred, (256,256,3), 0)
    vgg_true_1, vgg_true_2, vgg_true_3 = extract_vgg_features(vgg_in_groundtruth, (256,256,3), 1)
    vgg_comp_1, vgg_comp_2, vgg_comp_3 = extract_vgg_features(vgg_in_comp, (256,256,3), 2)
    # 画像＋損失
    join = LossLayer()([expanded_mask,
                        img, input_groundtruth, y_comp,
                        vgg_pred_1, vgg_pred_2, vgg_pred_3,
                        vgg_true_1, vgg_true_2, vgg_true_3,
                        vgg_comp_1, vgg_comp_2, vgg_comp_3])
    # lossやmetricsの表示がうまくいかないので出力は1つにする
    model = Model([input_image, input_mask, input_groundtruth], join) # このモデルは>100MBだが、推論用モデルは93MB

    return model

# 損失関数側だけ取る
def identity_loss(y_true, y_pred):
    return K.mean(y_pred[:,:,:,3], axis=(1,2))

def PSNR(y_true, y_pred):
    # 参考：https://ja.wikipedia.org/wiki/%E3%83%94%E3%83%BC%E3%82%AF%E4%BF%A1%E5%8F%B7%E5%AF%BE%E9%9B%91%E9%9F%B3%E6%AF%94
    pic_gt = y_true[:,:,:,:3]
    pic_pred = y_pred[:,:,:,:3]
    return 20 * K.log(2.0) / K.log(10.0) - 10.0 * K.log(K.mean(K.square(pic_gt-pic_pred), axis=(1,2,3))) / K.log(10.0) 

def data_generator(data, batch_size, shuffle):
    masked_image_cache, mask_cache, gt_cache = [], [], []
    while True:
        indices = np.arange(len(data["masked_image"]))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            masked_image_cache.append(data["masked_image"][i])
            mask_cache.append(data["mask"][i])
            gt_cache.append(data["ground_truth"][i])

            if len(masked_image_cache) == batch_size:
                batch_masked_image = np.asarray(masked_image_cache, np.uint8)
                batch_gt = np.asarray(gt_cache, np.uint8)
                batch_mask = np.asarray(mask_cache, np.float32) / 255.0
                masked_image_cache, mask_cache, gt_cache = [], [], []
                # 前処理
                batch_gt = utils.preprocess_image(batch_gt)
                batch_masked_image = utils.preprocess_image(batch_masked_image)

                # yはgt+dummy
                batch_y = np.zeros((batch_gt.shape[0], batch_gt.shape[1], batch_gt.shape[2], 4), np.float32)
                batch_y[:,:,:,:3] = batch_gt

                yield [batch_masked_image, batch_mask, batch_gt], batch_y
                
def load_data():
    if not os.path.exists("oppai_dataset.job.gz"):
        utils.create_dataset(mode=CENSOR_MODE)
    data = joblib.load("oppai_dataset.job.gz")
    return data

class SamplingCallback(Callback):
    def __init__(self, model, data, batch_size):
        self.model = model
        self.sampling_data = data["sampling"]
        self.batch_size = batch_size
        self.min_val_loss = np.inf

    def on_epoch_end(self, epoch, logs):
        # 数エポックごとにマスク修復の訓練の進みを可視化して保存
        if epoch % 3 == 0:
            gen = data_generator(self.sampling_data, self.batch_size, False)
            pred_crops = self.model.predict_generator(gen, 
                                        steps=len(self.sampling_data["masked_image"])//self.batch_size)[:,:,:,:3]
            utils.save_tiled_images(self.sampling_data, pred_crops, epoch, "Censored / Pred / Ground Truth",
                                    mode=CENSOR_MODE)

        # モデルの保存
        if self.min_val_loss > logs["loss"]:
            print(f"Val loss improved {self.min_val_loss:.04} to {logs['loss']:.04}")
            self.min_val_loss = logs["loss"]
            self.model.save_weights("oppai_train.hdf5" ,save_format="h5")

def train():
    data = load_data()

    n_train, n_test = len(data["train"]["masked_image"]), len(data["sampling"]["masked_image"])
    print(n_train, n_test)

    model = create_train_pconv_unet()
    model.summary()
    model.compile(tf.train.RMSPropOptimizer(1e-4), identity_loss, [PSNR])

    # TPUモデルに変換
    tpu_grpc_url = "grpc://"+os.environ["COLAB_TPU_ADDR"]
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)
    strategy = keras_support.TPUDistributionStrategy(tpu_cluster_resolver)
    model = tf.contrib.tpu.keras_to_tpu_model(model, strategy=strategy)

    batch_size=32
    cb = SamplingCallback(model, data, batch_size)
    hist = History()

    model.fit_generator(data_generator(data["train"], batch_size, True),
                        steps_per_epoch=n_train//batch_size,
                        callbacks=[cb, hist], epochs=150, max_queue_size=5)
    history = hist.history
    joblib.dump(history, "history.job")

## 推論用
def create_pred_model():
    train_model = create_train_pconv_unet()
    train_model.load_weights("oppai_train.hdf5")
    pred_model = Model([train_model.inputs[0], train_model.inputs[1]],
                       train_model.get_layer("unmasked").output)
    pred_model.summary()
    return pred_model

def plot_results():
    data = load_data()
    # URL別リストを作る
    urls = ["" for i in range(len(data["train"]["masked_image"]))]
    for mapper in data["train"]["mapper"]:
        for ind in mapper["index"]:
            urls[ind] = mapper["json"]["original_url"]

    model = create_pred_model()
    gen = data_generator(data["train"], 8, False)
    cnt = 0

    if not os.path.exists("plot"):
        os.mkdir("plot")

    def blending(base_img, sub_img, mask_img):
        with Image.fromarray(mask_img) as mask:
            mask = mask.convert("L")
            with Image.fromarray(base_img) as base_pil:
                with Image.fromarray(sub_img) as sub_pil:
                    merge = Image.composite(base_pil, sub_pil, mask)
                    merge = Image.blend(merge, base_pil, 0.5)
                    return np.asarray(merge, np.uint8)


    for i in tqdm(range(len(data["train"]["masked_image"])//8 )):
        [batch_img, batch_mask, batch_gt], _ = next(gen)
        batch_pred = model.predict([batch_img, batch_mask], batch_size=8)
        batch_img = utils.deprocess_image(batch_img)
        batch_gt = utils.deprocess_image(batch_gt)
        batch_pred = utils.deprocess_image(batch_pred)
        for j in range(batch_pred.shape[0]):
            fig = plt.figure(figsize=(8,8))
            fig.subplots_adjust(hspace=0.02, wspace=0.02,
                                left=0.02, top=0.90, bottom=0.02, right=0.98)
            ax = fig.add_subplot(2, 2, 1)
            ax.imshow(batch_img[j])
            ax.axis("off")
            ax = fig.add_subplot(2, 2, 2)
            ax.imshow(batch_pred[j])
            ax.axis("off")
            blend = blending(batch_img[j], batch_pred[j], 
                    data["train"]["mask"][i*8+j])
            ax = fig.add_subplot(2, 2, 3)
            ax.imshow(blend)
            ax.axis("off")
            ax = fig.add_subplot(2, 2, 4)
            ax.imshow(batch_gt[j])
            ax.axis("off")
            fig.suptitle("Mosaic - Decensored - Blend - Ground Truth \n ("+ urls[cnt]+")")
            fig.savefig(f"plot/compare_{cnt:04}.png")
            plt.close()
            cnt += 1

if __name__ == "__main__":
    K.clear_session()
    #train()
    plot_results()
