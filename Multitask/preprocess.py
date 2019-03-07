from scipy.io import loadmat
import pandas as pd
import datetime
from tqdm import tqdm
import numpy as np
from PIL import Image

# 異常なデータを取り除く
def remove_abnormal_data():
    data = loadmat("wiki_crop/wiki.mat") # Matlab形式のファイルの読み込み
    # 誕生年、Matlabのシリアル。Mathlabが西暦0年起点なのに対し、Pythonが西暦1年起点なので補正する
    birthyear = [datetime.datetime.fromordinal(x - 366).year for x in data["wiki"][0][0][0][0]]
    # メタデータをPandasのDataFrameにする
    df = pd.DataFrame({"birthyear":birthyear,
                       "photo_taken":data["wiki"][0][0][1][0],
                       "full_path":[x[0] for x in data["wiki"][0][0][2][0]],
                       "gender":data["wiki"][0][0][3][0],
                       "face_score":data["wiki"][0][0][6][0]})
    df["age"]=df["photo_taken"]-df["birthyear"]
    print(df.shape) # (62328, 6)

    # 異常なデータのフィルタリング [62328 -> 42089]
    # 年齢：5歳以上100歳以下、データミスがあり、負の年齢や1歳の大人の画像、2000歳近い人物もいる
    # 性別：性別不明は除外
    # 顔スコア：クロップされた顔画像がどれだけ確からしいか。顔画像以外も含まれるのでそれを除外
    query = df.query("(age >= 5) & (age <= 100) & (gender in [0,1]) & (face_score >= 1.0)")
    print(query.shape) # (42089, 6)
    # csvに保存
    query.reset_index(drop=True).to_csv("wiki_crop/valid_face.csv", index=False)

# Numpy配列で画像の保存
def pack_data():
    df = pd.read_csv("wiki_crop/valid_face.csv")
    # 160x160の画像にリサイズ
    image = np.zeros((df.shape[0], 160, 160, 3), dtype=np.uint8) #メモリが溢れないようにuint8で定義すること
    for i in tqdm(range(image.shape[0])):
        path = "wiki_crop/"+df.loc[i, "full_path"]
        image[i] = np.asarray(Image.open(path).convert("RGB").resize((160,160)), np.uint8)
    # Numpy配列で保存
    np.savez_compressed("wiki_crop/wiki_all", image=image,
                        gender=df["gender"].values.astype(np.float32),
                        age=df["age"].values.astype(np.float32))

# 試しに読んでみる
def test_loading():
    data = np.load("wiki_crop/wiki_all.npz")
    image, gender, age = data["image"], data["gender"], data["age"]
    print(image.shape) # (42089, 160, 160, 3)
    print(gender.shape) # (42089,)
    print(age.shape) # (42089,)

if __name__ == "__main__":
    remove_abnormal_data()
    pack_data()
    test_loading()
