# oppai_meta.jsonから必要なファイルをダウンロードする

# 【要】pixivpy3
# https://github.com/upbit/pixivpy
# !pip install pixivpy --upgrade

# 【注意】pixiv.jsonにはPixivのidとパスワードが記載されるので不特定多数に公開しないように注意すること

import json
import os
import time
from tqdm import tqdm
from pixivpy3 import *

def set_login_attribute(pixiv_id, password):
    """
    PixivにログインするユーザーのID、パスワードを設定
    pixiv_id : PixivID（プロフィール画面→ユーザー設定→pixiv ID [user_…]の値）
    password : Pixivのログインパスワード
    """

    settings = {
        "pixiv_id": pixiv_id,
        "password": password
    }
    with open("pixiv.json", "w") as fp:
        json.dump(settings, fp)

def login():
    """
    Pixivにログインし、セッションを獲得する
    出力：AppPixivAPIによるセッション
    """

    if not os.path.exists("pixiv.json"):
        raise FileNotFoundError("pixiv.jsonが存在しません。「set_login_attribute」を先に呼んでください")
    
    _aapi = AppPixivAPI()
    with open("pixiv.json", "r") as fp:
        settings = json.load(fp)
    keys = settings.keys()
    if not "pixiv_id" in keys or not "password" in keys:
        raise AttributeError("pixiv.jsonのキーが不正です")

    _aapi.login(*settings.values())
    return _aapi

def download_all(aapi_session, meta_data_path):
    if not os.path.exists(meta_data_path):
        raise FileNotFoundError("meta_data_pathで指定されたoppai_meta.jsonが存在しません : "+meta_data_path)

    base_dir = os.getcwd().replace("\\", "/").split("/")
    if len(base_dir) >= 2 and base_dir[-1] == "images" and base_dir[-2] == "oppai_dataset":
        base_dir = base_dir[:-1]
    if len(base_dir) >= 1 and base_dir[-1] == "oppai_dataset":
        base_dir = base_dir[:-1]

    dest_dir = "/".join(base_dir + ["oppai_dataset", "images"])
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    with open(meta_data_path) as fp:
        meta = json.load(fp)

    trial_cnt = 0
    success_cnt = 0

    for item in tqdm(meta):
        try:
            aapi_session.download(item["image_url"], path=dest_dir)
        except:
            pass
        else:
            success_cnt += 1
        finally:
            trial_cnt += 1

        if trial_cnt >= 30:
            time.sleep(1)
            trial_cnt = 0

    print(success_cnt, "/", len(meta), "件のダウンロードが完了しました")

if __name__ == "__main__":
    set_login_attribute("your pixiv id", "your pixiv password")
    session = login()
    download_all(session, "oppai_dataset/oppai_meta.json")
