from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class LossLayer(Layer):
    """
    損失関数の値を計算するレイヤー
    Input：復元カラー画像（TF）、真のカラー画像（TF）、VGGの特徴量（Caffe）
    Output：予測画像＋損失値の合成
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        y_pred, y_true, vgg_pred_1, vgg_pred_2, vgg_pred_3, vgg_true_1, vgg_true_2, vgg_true_3 = inputs

        # 特徴量（ネストしたテンソルを渡せないのでこうする）
        vgg_pred = [vgg_pred_1, vgg_pred_2, vgg_pred_3]
        vgg_true = [vgg_true_1, vgg_true_2, vgg_true_3]
 
        # 各損失関数
        content = content_loss(y_true, y_pred)
        style = style_loss(vgg_true, vgg_pred)
        tv = total_variation_loss(y_true, y_pred)

        # 全体の損失関数
        total_loss = 5*content + style + 0.1*tv

        # (batch,H,W,1)のテンソルを作る
        ones = K.sign(K.abs(y_pred) + 1) # (batch,H,W,3)のすべて1のテンソル
        ones = K.expand_dims(K.mean(ones, axis=-1), axis=-1) # (batch,H,W,1)
        # 画像と結合
        total_loss = K.expand_dims(K.expand_dims(K.expand_dims(total_loss))) # (batch,1,1,1)
        join = K.concatenate([y_pred, ones*total_loss], axis=-1)
        return join

    def get_config(self):
        return super().get_config()

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],input_shape[0][1],input_shape[0][2],input_shape[0][3]+1)

def content_loss(y_true, y_pred):
    # content loss
    return l1(y_true, y_pred)

def style_loss(vgg_true, vgg_pred):
    loss = 0
    for p, t in zip(vgg_pred, vgg_true):
        loss += l1(gram_matrix(p), gram_matrix(t))
    return loss

def total_variation_loss(y_true, y_pred):
    loss = 0
    loss += l1(y_pred[:,:,1:,:], y_pred[:,:,:-1,:])
    loss += l1(y_pred[:,1:,:,:], y_pred[:,:-1,:,:])
    return loss
    
def l1(y_true, y_pred):
    """Calculate the L1 loss used in all loss calculations"""
    if K.ndim(y_true) == 4:
        return K.mean(K.abs(y_pred - y_true), axis=[1,2,3])
    elif K.ndim(y_true) == 3:
        return K.mean(K.abs(y_pred - y_true), axis=[1,2])
    else:
        raise NotImplementedError("Calculating L1 loss on 1D tensors? should not occur for this network")

def gram_matrix(x, norm_by_channels=False):
    """Calculate gram matrix used in style loss"""
        
    # Assertions on input
    assert K.ndim(x) == 4, 'Input tensor should be a 4d (B, H, W, C) tensor'
    assert K.image_data_format() == 'channels_last', "Please use channels-last format"        
        
    # Permute channels and get resulting shape
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    shape = K.shape(x)
    B, C, H, W = shape[0], shape[1], shape[2], shape[3]
        
    # Reshape x and do batch dot product
    features = K.reshape(x, K.stack([B, C, H*W]))
    gram = K.batch_dot(features, features, axes=2)
        
    # Normalize with channels, height and width
    gram = gram /  K.cast(C * H * W, x.dtype)
        
    return gram