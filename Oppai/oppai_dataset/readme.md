# おっぱいデータセット（Oppai Dataset）
## これは何？
Pixivから4200枚超の「おっぱい」イラストをダウンロードし、「おっぱい」の位置をアノテーション付けしたもの。アノテーション作業は私が1人で手動で行った。

![](images/oppai_dataset.jpg)

元イラスト：[https://www.pixiv.net/member_illust.php?mode=medium&illust_id=72656796](https://www.pixiv.net/member_illust.php?mode=medium&illust_id=72656796)　（新条アカネ by 富岡二郎）

## 含まれるもの・含まれないもの
### 含まれるもの
* 元のイラストのURL
* 画像サイズ
* おっぱいの位置のBouding Box（座標）

### 含まれないもの
* 元のイラストのjpegファイル（各自ダウンロードすること）

## クイックスタート（ダウンロード）
画像のファイルのダウンロードは以下の通り。**要：Pixivアカウント**。ただし、**パスワードを直に記録することになるので取扱に注意**すること。

### Colabの場合
**Subversion**をインストールする

```python
!apt install subversion
```

「おっぱいデータセット」のリポジトリをコピーする

```python
!svn export https://github.com/koshian2/DeepCreamPyBook/trunk/Oppai/oppai_dataset --force
```

**[pixivpy](https://github.com/upbit/pixivpy)**をインストールする

```python
!pip install pixivpy --upgrade
```

画像をダウンロードする

```python
from oppai_dataset.download_images import *

set_login_attribute("your pixiv user id", "your pixiv password") # ここにID名、パスワードを入力する
session = login()
download_all(session, "oppai_dataset/oppai_meta.json")
```

画像は、「**oppai_dataset/images**」に保存される。

* PixivIDの確認方法は、「Pixivのユーザー画面→基本設定→pixiv ID」から。
* login()関数を何度もたたくとログイン履歴がたまり、その分登録メールアドレスにログインメールが大量に送られるので注意すること。
* セッションは1時間程度で有効期限が切れるので（2019年4月現在）、失効したら再ログインすること

## 想定される使い方
* 物体検出。Object Detectionならぬ**Oppai Detection**。
* [Partial Convolution](https://arxiv.org/abs/1804.07723)による**謎の光**の復元

## 注意点
* 全て全年齢対象のイラストからなる。R-18イラストは含まれない。
* 2019年4月時点でPixivを「おっぱい」でタグ検索した4264件のイラストからなる。
* 元のイラストのデータは配布していない。URLのみ記載している。利用する場合は、oppai_meta.json内のURLから各自ダウンロードすること。
* Pixivの元のイラストの著作権は作者に帰属するので、利用者はイラストの作者をリスペクトしながら使わなければならない。また、著作権法上の例外を超える利用をする場合は、必ずイラストの作者の許諾を得ること。
* 境界箱のデータ（oppai_meta.json）は「[AGPL-v3.0ライセンス](https://www.gnu.org/licenses/agpl-3.0.html)」の下で利用できる。
