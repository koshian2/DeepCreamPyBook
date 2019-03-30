## 始め方
Google Colabの**TPU**インスタンスを立ち上げる。

subversionをインストールし、libs以下をコピーする

```
!apt install subversion > /dev/null
!svn export https://github.com/koshian2/DeepCreamPyBook/trunk/PConv/libs
```

「svhn_pconv.py」の内容を全てコピペし、実行する。

## Reference
以下のリポジトリのコードの一部を利用している。

* MathiasGruber, PConv-Keras, [https://github.com/MathiasGruber/PConv-Keras](https://github.com/MathiasGruber/PConv-Keras), MIT License
