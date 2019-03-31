from PIL import Image
import numpy as np
import glob
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# カラーチャンネルあたりの相関行列を計算する
def color_ch_corr(image):
    assert image.ndim == 3
    assert image.shape[2] == 3
    flatten = image.reshape(-1, 3) # (H, W, C) -> (HW, C)
    return np.corrcoef(flatten, rowvar=False) # 相関行列

def load_color_images(image_directory, use_cache_file=True):
    cache_file_path = "waifus_cache.npy"
    if use_cache_file and os.path.exists(cache_file_path):
        return np.load(cache_file_path)

    images = glob.glob(image_directory+"/*.png")
    images += glob.glob(image_directory+"/*.jpeg")
    images = sorted(images) # 環境によってglobの順番が維持されないので順番保証
    result = []
    # カラー画像のみ抽出
    print("カラー画像のみを抽出…")
    for path in tqdm(images):
        try:
            with Image.open(path) as img:
                array = np.asarray(img.convert("RGB"), np.uint8)
            x = color_ch_corr(array)
        except:
            continue
        # 相関行列の平均が0.995未満ならカラー画像とみなす
        if np.mean(x) < 0.995:
            result.append(path)

    # キャッシュに保存
    np.save(cache_file_path, np.array(result))

    return result

# train test splitを行う
def load_wifus_data(image_directory):
    img_paths = load_color_images(image_directory)
    train, test = train_test_split(img_paths, test_size=0.2, random_state=114514)
    return train, test

# 前処理
def preprocess_inputs(input):
    assert input.ndim == 4
    x = input / 127.5 - 1.0
    x = np.clip(x, -1.0, 1.0)
    return x

# 前処理の逆
def deprocess_inputs(input):
    assert input.ndim == 4
    assert input.dtype == np.float32
    x = (input + 1.0) * 127.5
    x = np.clip(x, 0, 255.0).astype(np.uint8)
    return x

# 画像を並べてプロットして保存
def tile_images(save_path, x1, x2, title):
    assert x1.shape[0] == 32 and x2.shape[0] == 32
    assert x1.dtype == np.uint8 and x2.dtype == np.uint8
    assert x1.ndim == 4 and x2.ndim == 4
    plt.clf()
    plt.figure(figsize=(8,8))
    plt.subplots_adjust(hspace=0.02, wspace=0.02, 
                        top=0.92, bottom=0.02, left=0.02, right=0.98)
    def plot_ax(ax, img):
        if img.shape[-1] == 3:
            ax.imshow(img)
        if img.shape[-1] == 1:
            ax.imshow(img[:, :, 0], cmap="gray")
        ax.axis("off")

    for i in range(x1.shape[0]):
        ax = plt.subplot(8, 8, 2*i+1)
        plot_ax(ax, x1[i])
        ax = plt.subplot(8, 8, 2*i+2)
        plot_ax(ax, x2[i])
    plt.suptitle(title)

    if not os.path.exists(os.path.dirname(save_path)):
        os.mkdir(os.path.dirname(save_path))
    plt.savefig(save_path)
