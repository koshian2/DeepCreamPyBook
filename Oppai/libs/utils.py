from PIL import Image, ImageDraw, ImageFilter
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import joblib
import os

# https://github.com/deeppomf/DeepCreamPy/blob/master/libs/utils.py　より改変
# risk of box being bigger than the image
def expand_bounding(img_size_dict, bbox_dict, expand_factor=1.5, min_size = 256):
    #expand bounding box to capture more context
    min_x, min_y = bbox_dict["left"], bbox_dict["top"]
    max_x, max_y = bbox_dict["width"]+min_x, bbox_dict["height"]+min_y
    width, height = img_size_dict["width"], img_size_dict["height"]
    width_center = width//2
    height_center = height//2
    bb_width = max_x - min_x
    bb_height = max_y - min_y
    x_center = (min_x + max_x)//2
    y_center = (min_y + max_y)//2
    current_size = max(bb_width, bb_height)
    current_size  = int(current_size * expand_factor) #長辺を1.5倍
    max_size = min(width, height)
    if current_size > max_size:
        current_size = max_size
    elif current_size < min_size:
        current_size = min_size
    x1 = x_center - current_size//2
    x2 = x_center + current_size//2
    y1 = y_center - current_size//2
    y2 = y_center + current_size//2
    x1_square = x1
    y1_square = y1
    x2_square = x2
    y2_square = y2
    #move bounding boxes that are partially outside of the image inside the image
    if (y1_square < 0 or y2_square > (height - 1)) and (x1_square < 0 or x2_square > (width - 1)):
        #conservative square region
        if x1_square < 0 and y1_square < 0:
            x1_square = 0
            y1_square = 0
            x2_square = current_size
            y2_square = current_size
        elif x2_square > (width - 1) and y1_square < 0:
            x1_square = width - current_size - 1
            y1_square = 0
            x2_square = width - 1
            y2_square = current_size
        elif x1_square < 0 and y2_square > (height - 1):
            x1_square = 0
            y1_square = height - current_size - 1
            x2_square = current_size
            y2_square = height - 1
        elif x2_square > (width - 1) and y2_square > (height - 1):
            x1_square = width - current_size - 1
            y1_square = height - current_size - 1
            x2_square = width - 1
            y2_square = height - 1
        else:
            x1_square = x1
            y1_square = y1
            x2_square = x2
            y2_square = y2
    else:
        if x1_square < 0:
            difference = x1_square
            x1_square -= difference
            x2_square -= difference
        if x2_square > (width - 1):
            difference = x2_square - width + 1
            x1_square -= difference
            x2_square -= difference
        if y1_square < 0:
            difference = y1_square
            y1_square -= difference
            y2_square -= difference
        if y2_square > (height - 1):
            difference = y2_square - height + 1
            y1_square -= difference
            y2_square -= difference
    # if y1_square < 0 or y2_square > (height - 1):

    #if bounding box goes outside of the image for some reason, set bounds to original, unexpanded values
    #print(width, height)
    if x2_square > width or y2_square > height:
        print("bounding box out of bounds!")
        print(x1_square, y1_square, x2_square, y2_square)
        x1_square, y1_square, x2_square, y2_square = min_x, min_y, max_x, max_y

    # 元のBounding Boxの拡大したBBoxの中での相対座標
    original_rel_position = [min_x - x1_square,
                             min_y - y1_square,
                             max_x - x1_square,
                             max_y - y1_square]
    return [x1_square, y1_square, x2_square, y2_square], original_rel_position

# テスト用：Bouding Boxの拡大のチェック
def view_expanded_bbox():
    with open("oppai_dataset/oppai_meta.json", "r") as fp:
       metadata = json.load(fp)
    for pic in metadata:
        with Image.open(pic["filepath"]) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            for bbox in pic["bounding_boxes"]:
                # オリジナルのBoudingBox
                draw.rectangle((bbox["left"], bbox["top"], 
                                bbox["width"]+bbox["left"], bbox["height"]+bbox["top"]),
                               outline=(0,255,0), width=5)
                expanded_bbox, original_rel = expand_bounding(pic["size"], bbox)
                print(original_rel)
                # 拡大したBoudingBox
                draw.rectangle(expanded_bbox, outline=(0,0,255), width=5)
            img_array = np.asarray(img, np.uint8)
            plt.imshow(img_array)
            plt.show()

def color_ch_corr(image):
    assert image.ndim == 3
    assert image.shape[2] == 3
    flatten = image.reshape(-1, 3) # (H, W, C) -> (HW, C)
    return np.corrcoef(flatten, rowvar=False) # 相関行列

def crop_by_expanded_bbox(img_json, output_size=(256, 256), excluded_grayscale_imgs=True):
    """
    返り値：クロップしたNumpy配列のリスト、元のBBOXの拡大BBOX内の相対座標（リサイズ後）、
    元のBBOXの画像内の絶対座標
    """
    cropped_images = []
    original_bboxes_rel = []
    original_bboxes_abs = []

    try:
        with Image.open(img_json["filepath"]) as img:
            img = img.convert("RGB")
            
            # 白黒画像の除外（訓練ノイズになる）
            if excluded_grayscale_imgs:
                img_array = np.asarray(img, np.float32)
                corrcoef = np.mean(color_ch_corr(img_array))
                # 相関行列の平均0.995以上で白黒画像とみなす
                if corrcoef >= 0.995:
                    return [], [], []

            # Bouding Boxの拡大
            for bbox in img_json["bounding_boxes"]:
                expand, original_rel = expand_bounding(img_json["size"],bbox)
                # リサイズ係数
                expand_size = (expand[2]-expand[0], expand[3]-expand[1])
                scale_x = output_size[0] / expand_size[0]
                scale_y = output_size[1] / expand_size[1]
                # リサイズ
                crop = img.crop(expand).resize(output_size, Image.BICUBIC)
                # 相対座標もリサイズ対応させる
                resized_original_rel = [original_rel[0]*scale_x,
                                        original_rel[1]*scale_y,
                                        original_rel[2]*scale_x,
                                        original_rel[3]*scale_y]
                # 元のBBOXの絶対座標
                original_abs = [bbox["left"], bbox["top"],
                                bbox["left"]+bbox["width"], 
                                bbox["top"]+bbox["height"]]
                # リストに追加
                cropped_images.append(np.asarray(crop, np.uint8))
                original_bboxes_rel.append(resized_original_rel)
                original_bboxes_abs.append(original_abs)
    except Exception as e:
        print(e)
        return [], [], []
    else:
        return cropped_images, original_bboxes_rel, original_bboxes_abs

# 謎の光を作る
def nazo_no_hikari(cropped_img_array, rel_bbox, blur_size=10):
    """
    出力：謎の光を入れた画像、謎の光の白黒マスク（マスク部分は黒に変更）
    """
    assert cropped_img_array.ndim == 3 and cropped_img_array.dtype == np.uint8
    assert len(rel_bbox) == 4
    # マスク画像を作る    
    with Image.new("RGB", (cropped_img_array.shape[1], cropped_img_array.shape[0])) as mask:
        draw = ImageDraw.Draw(mask)
        mask_width = rel_bbox[2] - rel_bbox[0]
        mask_height = rel_bbox[3] - rel_bbox[1]

        draw.rectangle((rel_bbox[0]+mask_width*0.1, rel_bbox[1]+mask_width*0.1, 
                        rel_bbox[0]+mask_width*0.9, rel_bbox[1]+mask_height*0.9),
                       fill=(255,255,255))
        mask = mask.filter(ImageFilter.GaussianBlur(blur_size))
        mask_array = np.asarray(mask, np.int16)
    # 合成する
    merge = cropped_img_array.astype(np.int16) + mask_array
    merge = np.clip(merge, 0, 255).astype(np.uint8)
    # 前処理でハマるので、マスク部分を黒に変える
    reversed_mask = (255 - mask_array).astype(np.uint8)

    return merge, reversed_mask
    
def add_mosaic(cropped_img_array, rel_bbox, blur_size=10):
    """
    出力：モザイクを画像、ブレンディングの白黒マスク（マスク部分は黒に変更）
    """
    assert cropped_img_array.ndim == 3 and cropped_img_array.dtype == np.uint8
    assert len(rel_bbox) == 4
    # マスク画像を作る    
    with Image.new("L", (cropped_img_array.shape[1], cropped_img_array.shape[0])) as mask:
        draw = ImageDraw.Draw(mask)
        mask_width = rel_bbox[2] - rel_bbox[0]
        mask_height = rel_bbox[3] - rel_bbox[1]

        draw.rectangle((rel_bbox[0]+mask_width*0.1, rel_bbox[1]+mask_width*0.1, 
                        rel_bbox[0]+mask_width*0.9, rel_bbox[1]+mask_height*0.9),
                       fill=(255))
        mask = mask.filter(ImageFilter.GaussianBlur(blur_size))
        mask_array = np.expand_dims(np.asarray(mask, np.uint8), -1) * np.ones((1,1,3), np.uint8)

        # 元画像、モザイク画像
        with Image.fromarray(cropped_img_array) as original:
            mosaic = original.filter(ImageFilter.GaussianBlur(5))
            mosaic = mosaic.resize([x // 8 for x in mosaic.size]).resize(mosaic.size)
            merge = Image.composite(mosaic, original, mask)
            merge_array = np.asarray(merge, np.uint8)

    # 前処理でハマるので、マスク部分を黒に変える
    reversed_mask = (255 - mask_array).astype(np.uint8)

    return merge_array, reversed_mask

# 有効な画像数をカウント
def mosaic_test():
    with open("oppai_dataset/oppai_meta.json", "r") as fp:
       metadata = json.load(fp)
    for pic in tqdm(metadata):
        imgs, rels, abss = crop_by_expanded_bbox(pic)
        if len(imgs) == 0: continue
        nazos = []
        for img, rel in zip(imgs, rels):
            hikari, mask = add_mosaic(img, rel)
            nazos.append(hikari)
        merge = merge_to_original(pic, nazos, rels, abss)
        plt.title(pic["original_url"].replace("?", "?\n"))
        plt.imshow(merge)
        plt.show()

# クロップした画像を1つに束ねる
def merge_to_original(img_json, cropped_img_arrays, orig_bbox_relative, orig_bbox_absolute):
    """
    img_json:画像のJSON、cropped_img_arrays:クロップしたNumpy配列のリスト
    orig_bbox_relative:オリジナルBBOXの相対座標、orig_bbox_absolute:オリジナルBBOXの絶対座標

    出力：合成された画像のNumpy配列
    """

    with Image.open(img_json["filepath"]) as original:
        for crop, rel, abs in zip(cropped_img_arrays, orig_bbox_relative, orig_bbox_absolute):
            with Image.fromarray(crop) as bbox:
                # 拡大されたBBOX内の相対座標でクロップ
                bbox_crop = bbox.crop(rel)
                # 元のBBOXとサイズが一致するようにリサイズ
                bbox_crop = bbox_crop.resize((int(abs[2]-abs[0]),
                                             int(abs[3]-abs[1])), Image.BICUBIC)
                # 元の画像にペースト
                dest = [int(abs[0]),
                        int(abs[1]),
                        int(abs[0] + bbox_crop.size[0]),
                        int(abs[1] + bbox_crop.size[1])]
                orinal = original.paste(bbox_crop, dest)
        img_array = np.asarray(original, np.uint8)
    return img_array

def save_tiled_images(data, pred_crops, epoch, base_title, directory="sampling", mode="hikari"):
    assert pred_crops.dtype == np.float32 and pred_crops.ndim == 4
    assert "mapper" in data.keys()
    # カラースケールを元に戻す
    preds = deprocess_image(pred_crops)

    if not os.path.exists(directory):
        os.mkdir(directory)

    def plot_subplot(index, image):
        ax = plt.subplot(2, 2, index)
        ax.imshow(image)
        ax.axis("off")

    for i in range(50):
        img_json = data["mapper"][i]["json"]
        # オリジナル画像
        with Image.open(img_json["filepath"]) as img:
            original = np.asarray(img, np.uint8)
        # 謎の光orモザイク画像
        imgs, rels, abss = crop_by_expanded_bbox(img_json, 
                                                 excluded_grayscale_imgs=False) # 高速化のためカラーチェックは切る
        censored_images = []
        for im, re in zip(imgs, rels):
            if mode == "hikari":
                censor, _ = nazo_no_hikari(im, re)
            elif mode == "mosaic":
                censor, _ = add_mosaic(im, re)
            censored_images.append(censor)
        masked_image = merge_to_original(img_json, censored_images, rels, abss) # 謎の光orモザイク

        # 復元画像
        used_preds = [preds[k] for k in data["mapper"][i]["index"]]
        reconstruct_image = merge_to_original(img_json, used_preds,
                                              data["mapper"][i]["bbox_rel"],
                                              data["mapper"][i]["bbox_abs"])
        plt.clf()
        plt.subplots_adjust(hspace=0.02, wspace=0.02, top=0.95, bottom=0.02, left=0.02, right=0.98)
        plt.figure(figsize=(10, 10))

        plot_subplot(1, masked_image)
        plot_subplot(2, reconstruct_image)
        plot_subplot(3, original)

        plt.suptitle(base_title+f" (epoch={epoch:03} {i+1}/50)")
        plt.savefig(directory+f"/epoch_{epoch:03}_{i:02}.png")

# 有効な画像数をカウント
def count_valid_images():
    with open("oppai_dataset/oppai_meta.json", "r") as fp:
       metadata = json.load(fp)

    valid = []
    for pic in tqdm(metadata):
        imgs, rels, abss = crop_by_expanded_bbox(pic)
        if len(imgs) == 0: continue
        valid.append(pic)
    return valid

def create_dataset_by_json(json_paths, mode="hikari"):
    """
    出力：マスク済み画像[masked_image]、マスク[mask]、真の画像[ground_trth]、マッパー[mapper]
    マッパー＝[index]:全体リストのインデックス、[json]:元のJSON
            [bbox_rel]:元のBBoxの相対座標、[bbox_abs]：元のBBoxの絶対座標
    """
    masked_images, masks, ground_truths = [], [], []
    mappers = []

    cnt = 0
    for pic in tqdm(json_paths):
        # マッパー情報
        mapper_item = {"json": pic}
        mapper_indices = []

        # クロップ
        img, rel, abs = crop_by_expanded_bbox(pic)
        for i, r in zip(img, rel):
            if mode == "hikari":
                masked_item, mask_item = nazo_no_hikari(i, r)
            elif mode == "mosaic":
                masked_item, mask_item = add_mosaic(i, r)
            # 画像側に追加
            masked_images.append(masked_item)
            masks.append(mask_item)
            ground_truths.append(i)

            # マッパーにインデックスの記録
            mapper_indices.append(cnt)
            cnt += 1

        # マッパーに記録
        mapper_item["index"] = mapper_indices
        mapper_item["bbox_rel"] = rel
        mapper_item["bbox_abs"] = abs
        mappers.append(mapper_item)

    result = {
        "masked_image": masked_images, "mask": masks,
        "ground_truth": ground_truths, "mapper": mappers
    }

    return result


# データセットの作成
def create_dataset(mode="hikari"):
    # 記録するもの

    # 訓練テストの分割：ファイル単位
    # 訓練データ、テストデータ
    # 　拡張してクロップした：マスク済み、マスクフラグ、真の画像
    #
    # ファイルとクロップのマッパー（合成時に必要）
    #　　JSON＋インデックス＋相対座標＋絶対座標

    # 有効画像数のカウント
    valid_images = count_valid_images()
    print("有効画像数", len(valid_images)) # 開発環境では3973件

    # 訓練データの作成
    print("mode =", mode, "でデータを作成")
    print("訓練データの作成中")
    train_data = create_dataset_by_json(valid_images, mode=mode)

    # ToDO:もっとデータを増やす
    # データが少なすぎるので過学習がひどい
    # そのため、train_test_splitで分けるのではなく、
    # 訓練データの一部をサンプリング用に分割する（汎化性能は今後の課題）
    indices = np.random.seed(114514)
    sampling_indices = np.random.permutation(len(valid_images))[:200]

    # サンプリング用を作成
    sampling_data = create_dataset_by_json([valid_images[x] for x in sampling_indices],
                                           mode=mode)
    # 全体の保存
    result = {"train": train_data, "sampling": sampling_data}
    joblib.dump(result, "oppai_dataset.job.gz", compress=3)

def preprocess_image(x):
    assert x.dtype == np.uint8
    return x.astype(np.float32) / 127.5 - 1.0

def deprocess_image(x):
    assert x.dtype == np.float32
    img = (x + 1.0) * 127.5
    return np.clip(img, 0.0, 255.0).astype(np.uint8)

if __name__ == "__main__":
    create_dataset("mosaic")
