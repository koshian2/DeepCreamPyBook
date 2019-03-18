from keras.applications import vgg19
import keras.backend as K
from scipy.optimize import fmin_l_bfgs_b

from PIL import Image
import numpy as np
import time, os

# 出力サイズ
IMG_WIDTH, IMG_HEIGHT = 256, 256

# 入力→ニューラルネットワークの前処理（KerasのVGG19はCaffeモードでの前処理）
def preprocess_image(image_path):
    img = np.asarray(Image.open(image_path).convert("RGB").resize((IMG_HEIGHT, IMG_WIDTH)))
    img = np.expand_dims(img, axis=0)
    return vgg19.preprocess_input(img)

# ニューラルネットワーク→元の画像に戻すための処理（前処理の逆変換）
# VGG19がCaffeモードの前処理なのでちょっと複雑
def decompress_image(x):
    img = x.reshape((IMG_HEIGHT, IMG_WIDTH, 3))
    # 平均を0にしているのを戻す
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # CaffeモードではBGRで定義しているので、BGR->RGBへの変換をする
    img = img[:, :, ::-1]
    return np.clip(img, 0, 255).astype('uint8')

# 画像の保存
def save_image(dest_dir, image_array, iter):
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    with Image.fromarray(image_array) as img:
        img.save(f"{dest_dir}/{iter:03}.png")

# モデルの作成＋損失関数に使う中間層の出力
def create_model(base_image_path, style_reference_image_path):
    # 入力テンソルを作る
    base_image = K.variable(preprocess_image(base_image_path))
    style_reference_image = K.variable(preprocess_image(style_reference_image_path))
    combination_image = K.placeholder((1, IMG_HEIGHT, IMG_WIDTH, 3))
    # 元の画像、スタイル画像、組み合わせた画像の3枚を入力（バッチ）として渡す
    input_tensor = K.concatenate([base_image, style_reference_image, combination_image], axis=0)

    # VGG19
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=input_tensor)

    # 損失関数に使う中間層の出力（Neural Style Transferの論文より）
    # content loss : conv4_2
    # style loss : conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
    content_loss_features, style_loss_features = [], []
    for layer in model.layers:
        if layer.name == "block4_conv2":
            content_loss_features.append(layer.output)
        if layer.name in ["block1_conv1", "block2_conv1", "block3_conv1",
                          "block4_conv1", "block5_conv1"]:
            style_loss_features.append(layer.output)
    return model, content_loss_features, style_loss_features, combination_image

# グラム行列
def gram_matrix(x):
    assert K.ndim(x) == 3 # ランク3なのに注意（1枚単位のグラム行列）: (H, W, C)
    height, width, channel = K.int_shape(x)
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1))) # (C, H, W) -> (C, HW)
    gram = K.dot(features, K.transpose(features)) # (C, C)
    return gram

# Style Loss
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    height, width, channel = K.int_shape(style)
    return K.sum((S - C)**2) / 4.0 / channel ** 2 / (height * width) ** 2

# Content Loss
def content_loss(base, combination):
    #return K.sum((combination - base) ** 2) / 2.0 # ただのL2 loss
    height, width, channel = K.int_shape(base)
    return K.sum((combination - base) ** 2) / 2.0 / height / width / channel 

# L_content + L_style
def total_loss(content_loss_features, style_loss_features, content_weight):
    # ハイパーパラメータ：Content_lossの係数α(content_weight)
    # Style lossの係数βは1で固定

    # 損失を0で初期化
    loss = K.variable(0.0)
    # content loss
    # layer_features の1つ目の軸は [元画像, スタイル画像, 組み合わせた画像]の順
    for layer_features in content_loss_features:
        loss = loss + content_weight * content_loss(layer_features[0, :, :, :],
                                                    layer_features[2, :, :, :]) # base, combination

    # style lossの係数w_l＝1÷(style lossを計算するレイヤー数) 論文より
    style_weighting_factor = 1.0 / len(style_loss_features)
    # style loss
    for layer_features in style_loss_features:
        loss = loss + style_weighting_factor * style_loss(layer_features[1, :, :, :],
                                                          layer_features[2, :, :, :]) # style, combination
    return loss

# 損失＋勾配を同時に返す関数を作る
def create_loss_grads_func(loss, combination_image):
    # 勾配の取得
    grads = K.gradients(loss, combination_image)
    # 損失＋勾配の出力
    outputs = [loss]
    if isinstance(grads, (list, tuple)):
        outputs += grads
    else:
        outputs.append(grads)

    # 合成画像を入力とし、損失と勾配を返す関数
    f_outputs = K.function([combination_image], outputs)
    return f_outputs

# 損失＋勾配の計算（L-BFGSの更新に勾配を明示的に与えてあげる必要がある）
def eval_loss_and_grads(f_outputs, inputs):
    x = inputs.reshape((1, IMG_HEIGHT, IMG_WIDTH, 3))
    # create_loss_grads_funcで作った関数の実行
    outs = f_outputs([x])

    # 損失の値
    loss_value = outs[0]
    # 勾配の値
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# L-BFGS用に損失と勾配を1回の計算で評価できる用のEvaluatorを作る
# 損失と勾配を別々に計算するのは効率が悪い
class Evaluator(object):
    def __init__(self, f_outputs):
        self.loss_value = None
        self.grads_values = None
        self.f_outputs = f_outputs

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(self.f_outputs, x)
        self.loss_value = loss_value
        self.grad_values = grad_values # 損失評価時に勾配の値をキャッシュしておく
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values) # 勾配評価時はキャッシュから取り出す
        self.loss_value = None
        self.grad_values = None
        return grad_values

# Neural style transferのメイン部分
def main(base_image_path, style_reference_image_path, content_weight, output_dir="generated"):
    # パラメーター：
    # base_image_path : 元の画像のパス
    # style_reference_image_path : スタイル画像のパス
    # content_weight : 元の画像とスタイル画像の混合比率の調整パラメーター。大きいほど元の画像に比重が置かれる。

    # モデルの作成
    model, content_loss_features, style_loss_features, generated_image = create_model(base_image_path, style_reference_image_path)
    # 損失関数のグラフの作成
    loss_graph = total_loss(content_loss_features, style_loss_features, content_weight)
    # 損失と勾配を同時に返す関数
    func_loss_grad = create_loss_grads_func(loss_graph, generated_image)
    # 最適化用のEvaluator
    evaluator = Evaluator(func_loss_grad)

    x = preprocess_image(base_image_path)

    for i in range(300):
        if i % 10 == 0:
            print('Start of iteration', i)
            start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                         fprime=evaluator.grads, maxfun=20)
        if not i % 10 == 0:
            continue
        print('Current loss value:', min_val)
        # 生成された画像の保存
        img = decompress_image(x.copy()) # ここでコピーするのが大事
        save_image(dest_dir=output_dir, image_array=img, iter=i)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))


if __name__ == "__main__":
    main("dancing.jpg", "picasso.jpg", 1e-4)
