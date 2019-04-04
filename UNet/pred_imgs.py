from mywaifu_coloring import create_unet, preprocess
from PIL import Image
import numpy as np
import shutil, os
from tqdm import tqdm
import matplotlib.pyplot as plt
from libs.utils import load_wifus_data
import tensorflow.keras.backend as K

K.clear_session()
model = create_unet()
model.load_weights("waifu_unet.hdf5")

def copy_all_test():
    train, test = load_wifus_data("./images")
    if not os.path.exists("test"):
        os.mkdir("test")
    for path in tqdm(test):
        shutil.copy(path, path.replace("images", "test"))

def pred_images(img_path):
    with Image.open(img_path) as pillow_img:
        img = np.expand_dims(np.asarray(pillow_img.convert("RGB").resize((512,512), Image.LANCZOS), 
                                        np.uint8), 0)
        original_size = pillow_img.size

    gray, color = preprocess(img)

    pred = model.predict(gray)

    plt.clf()
    plt.figure(figsize=(12,4))
    plt.subplots_adjust(hspace=0.02, wspace=0.02, 
                        top=0.98, bottom=0.02, left=0.02, right=0.98)
    def plot_ax(ax, img):
        if img.ndim == 3:
            ax.imshow(img)
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        ax.axis("off")

    def convert_to_original_size(img):
        convert = (img*255.0).astype(np.uint8)
        if convert.shape[-1] == 1:
            x = convert[:,:,0]
        else:
            x = convert
        with Image.fromarray(x) as pillow_img:
            pillow_img = pillow_img.resize(original_size, Image.LANCZOS)
            return np.asarray(pillow_img, np.uint8)

    ax = plt.subplot(1, 3, 1)
    plot_ax(ax, convert_to_original_size(color[0]))
    ax = plt.subplot(1, 3, 2)
    plot_ax(ax, convert_to_original_size(gray[0]))
    ax = plt.subplot(1, 3, 3)
    plot_ax(ax, convert_to_original_size(pred[0]))

    plt.savefig("pred_high_res.png")
