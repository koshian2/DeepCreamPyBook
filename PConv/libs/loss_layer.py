from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

class LossLayer(Layer):
    """
    損失関数の値を計算するレイヤー
    Input：入力マスク画像、入力真の画像、予測画像、VGGの特徴量
    Output：予測画像＋損失値の合成
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        mask, y_pred, y_true, y_comp, vgg_pred_1, vgg_pred_2, vgg_pred_3, vgg_true_1, vgg_true_2, vgg_true_3, vgg_comp_1, vgg_comp_2, vgg_comp_3 = inputs

        # 特徴量（ネストしたテンソルを渡せないのでこうする）
        vgg_out = [vgg_pred_1, vgg_pred_2, vgg_pred_3]
        vgg_gt = [vgg_true_1, vgg_true_2, vgg_true_3]
        vgg_comp = [vgg_comp_1, vgg_comp_2, vgg_comp_3]

        # 各損失関数
        l1 = loss_valid(mask, y_true, y_pred)
        l2 = loss_hole(mask, y_true, y_pred)
        l3 = loss_perceptual(vgg_out, vgg_gt, vgg_comp)
        l4 = loss_style(vgg_out, vgg_gt)
        l5 = loss_style(vgg_comp, vgg_gt)
        l6 = loss_tv(mask, y_comp)

        # 全体の損失関数
        total_loss = l1 + 6*l2 + 0.05*l3 + 120*(l4+l5) + 0.1*l6

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

def loss_hole(mask, y_true, y_pred):
    """Pixel L1 loss within the hole / mask"""
    return l1((1-mask) * y_true, (1-mask) * y_pred)
    
def loss_valid(mask, y_true, y_pred):
    """Pixel L1 loss outside the hole / mask"""
    return l1(mask * y_true, mask * y_pred)
    
def loss_perceptual(vgg_out, vgg_gt, vgg_comp): 
    """Perceptual loss based on VGG16, see. eq. 3 in paper"""       
    loss = 0
    for o, c, g in zip(vgg_out, vgg_comp, vgg_gt):
        loss += l1(o, g) + l1(c, g)
    return loss
        
def loss_style(output, vgg_gt):
    """Style loss based on output/computation, used for both eq. 4 & 5 in paper"""
    loss = 0
    for o, g in zip(output, vgg_gt):
        loss += l1(gram_matrix(o), gram_matrix(g))
    return loss
    
def loss_tv(mask, y_comp):
    """Total variation loss, used for smoothing the hole region, see. eq. 6"""

    # Create dilated hole region using a 3x3 kernel of all 1s.
    kernel = K.ones(shape=(3, 3, mask.shape[3], mask.shape[3]))
    dilated_mask = K.conv2d(1-mask, kernel, data_format='channels_last', padding='same')

    # Cast values to be [0., 1.], and compute dilated hole region of y_comp
    dilated_mask = K.cast(K.greater(dilated_mask, 0), 'float32')
    P = dilated_mask * y_comp

    # Calculate total variation loss
    a = l1(P[:,1:,:,:], P[:,:-1,:,:])
    b = l1(P[:,:,1:,:], P[:,:,:-1,:])        
    return a+b

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
