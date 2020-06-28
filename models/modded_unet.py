from keras.models import Model, Sequential
from keras.utils import Sequence
from keras.layers import Concatenate, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D,Lambda,LeakyReLU
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras import optimizers
from keras.optimizers import SGD
import keras

import tensorflow as tf

'''
    Modified U-Net. Less depth. linear final activation.
'''

def model_name():
    return "unet"

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2]).value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1]).value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)

def conv_layer(name,prev_layer,kernels,kernel_size,batch_norm=True,dropout_rate=0):
    conv = Conv2D(kernels, kernel_size, padding="same", name=name, data_format="channels_last")(prev_layer)
    #conv = PReLU(shared_axes=[1, 2])(conv)
    conv = LeakyReLU()(conv)
    
    if dropout_rate:
        conv = Dropout(rate=dropout_rate)(conv)
    if batch_norm:
        conv = BatchNormalization()(conv)

    return conv

def up_cat(prev_layer,skip_layer,kernels,kernel_size):
    up_samp = UpSampling2D(size=(2, 2), data_format="channels_last")(prev_layer)
    up_conv = Conv2D(kernels, kernel_size, padding="same", data_format="channels_last")(up_samp)
    #ch, cw = get_crop_shape(skip_layer,up_conv)
    #crop_conv = Cropping2D(cropping=(ch,cw), data_format="channels_last")(skip_layer)
    up   = keras.layers.merge.concatenate([up_conv,skip_layer])
    
    return up

def get_model(n_ch,patch_height,patch_width,kernel_size = 64):
    concat_axis = 3
    dropout_rate = 0.1
    b_norm = True

    inputs = Input((patch_height, patch_width, n_ch))
    
    #flat = Flatten()(inputs)
    #dense = Dense(1)(inputs)
    
    conv1 = conv_layer("conv1_1",inputs,kernel_size,7,batch_norm=b_norm)
    conv1 = conv_layer("conv1_2",conv1,kernel_size,3,batch_norm=b_norm,dropout_rate=dropout_rate)
    
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)
    conv2 = conv_layer("conv2_1",pool1,kernel_size*2,3,batch_norm=b_norm,dropout_rate=dropout_rate)
    conv2 = conv_layer("conv2_2",conv2,kernel_size*2,3,batch_norm=b_norm,dropout_rate=dropout_rate)
    
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)
    conv3 = conv_layer("conv3_1",pool2,kernel_size*4,3,batch_norm=b_norm,dropout_rate=dropout_rate)
    conv3 = conv_layer("conv3_2",conv3,kernel_size*4,3,batch_norm=b_norm,dropout_rate=dropout_rate)
    
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)
    conv4 = conv_layer("conv4_1",pool3,kernel_size*8,3,batch_norm=b_norm)
    conv4 = conv_layer("conv4_2",conv4,kernel_size*8,3,batch_norm=b_norm)
    
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)
    conv5 = conv_layer("conv5_1",pool4,kernel_size*16,3,batch_norm=b_norm)
    conv5 = conv_layer("conv5_2",conv5,kernel_size*16,3,batch_norm=b_norm)
    
    pool5 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv5)
    conv6 = conv_layer("conv6_1",pool5,kernel_size*32,3,batch_norm=b_norm)
    conv6 = conv_layer("conv6_2",conv6,kernel_size*32,3,batch_norm=b_norm)

    up7 = up_cat(conv6,conv5,kernel_size*16,3)
    conv7 = conv_layer("conv7_1",up7,kernel_size*16,3,batch_norm=b_norm)
    conv7 = conv_layer("conv7_2",conv7,kernel_size*16,3,batch_norm=b_norm)

    up8 = up_cat(conv7,conv4,kernel_size*8,3)
    conv8 = conv_layer("conv8_1",up8,kernel_size*8,3,batch_norm=b_norm)
    conv8 = conv_layer("conv8_2",conv8,kernel_size*8,3,batch_norm=b_norm)
    
    up9 = up_cat(conv8,conv3,kernel_size*4,3)
    conv9 = conv_layer("conv9_1",up9,kernel_size*4,3,batch_norm=b_norm,dropout_rate=dropout_rate)
    conv9 = conv_layer("conv9_2",conv9,kernel_size*4,3,batch_norm=b_norm,dropout_rate=dropout_rate)
    
    up10 = up_cat(conv9,conv2,kernel_size*2,3)
    conv10 = conv_layer("conv10_1",up10,kernel_size*2,3,batch_norm=b_norm,dropout_rate=dropout_rate)
    conv10 = conv_layer("conv10_2",conv10,kernel_size*2,3,batch_norm=b_norm,dropout_rate=dropout_rate)
    
    up11 = up_cat(conv10,conv1,kernel_size,3)
    conv11 = conv_layer("conv11_1",up11,kernel_size,3,batch_norm=b_norm,dropout_rate=dropout_rate)
    conv11 = conv_layer("conv11_2",conv11,kernel_size,3,batch_norm=b_norm,dropout_rate=dropout_rate)
    
    #ch, cw = get_crop_shape(inputs, conv9)
    #conv9  = ZeroPadding2D(padding=(ch[0],cw[0]), data_format="channels_last")(conv9)
    #conv10 = Conv2D(1, (1, 1), data_format="channels_last", activation="sigmoid")(conv9)
    
    #output = Lambda(lambda v: np.angle(tf.spectral.ifft2d(v[...,:]+1j*v[...,:])))(conv9)
    
    #flat = Flatten()(conv9)
    #decoded  = Dense(1)(flat)
    
    decoded = Conv2D(1, (1,1), padding="same",  name="conv_final", activation='linear', data_format="channels_last")(conv11)
    #printed = print_layer(decoded,"final_layer=")
    #final = keras.layers.merge.concatenate([decoded,tf.zeros(decoded.shape)],axis=-1)

    model = Model(inputs=inputs, outputs=decoded)
    
    return model