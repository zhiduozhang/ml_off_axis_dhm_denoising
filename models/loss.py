# https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras

from keras import backend as K
from keras import losses
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops as _ops
from tensorflow.python.ops import manip_ops

def fftshift(x, axes=None, name=None):
    """Shift the zero-frequency component to the center of the spectrum.
    This function swaps half-spaces for all axes listed (defaults to all).
    Note that ``y[0]`` is the Nyquist component only if ``len(x)`` is even.
    @compatibility(numpy)
    Equivalent to numpy.fft.fftshift.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftshift.html
    @end_compatibility
    For example:
    ```python
    x = tf.signal.fftshift([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
    x.numpy() # array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    ```
    Args:
    x: `Tensor`, input tensor.
    axes: `int` or shape `tuple`, optional Axes over which to shift.  Default is
      None, which shifts all axes.
    name: An optional name for the operation.
    Returns:
    A `Tensor`, The shifted tensor.
    """
    with _ops.name_scope(name, "fftshift") as name:
        x = _ops.convert_to_tensor(x)
    if axes is None:
        axes = tuple(range(x.shape.ndims))
        shift = [int(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = int(x.shape[axes] // 2)
    else:
        shift = [int((x.shape[ax]) // 2) for ax in axes]

    return manip_ops.roll(x, shift, axes)

def ifftshift(x, axes=None, name=None):
    """The inverse of fftshift.
    Although identical for even-length x,
    the functions differ by one sample for odd-length x.
    @compatibility(numpy)
    Equivalent to numpy.fft.ifftshift.
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.ifftshift.html
    @end_compatibility
    For example:
    ```python
    x = tf.signal.ifftshift([[ 0.,  1.,  2.],[ 3.,  4., -4.],[-3., -2., -1.]])
    x.numpy() # array([[ 4., -4.,  3.],[-2., -1., -3.],[ 1.,  2.,  0.]])
    ```
    Args:
    x: `Tensor`, input tensor.
    axes: `int` or shape `tuple` Axes over which to calculate. Defaults to None,
      which shifts all axes.
    name: An optional name for the operation.
    Returns:
    A `Tensor`, The shifted tensor.
    """
    with _ops.name_scope(name, "ifftshift") as name:
        x = _ops.convert_to_tensor(x)
    if axes is None:
        axes = tuple(range(x.shape.ndims))
        shift = [-int(dim // 2) for dim in x.shape]
    elif isinstance(axes, int):
        shift = -int(x.shape[axes] // 2)
    else:
        shift = [-int(x.shape[ax] // 2) for ax in axes]

    return manip_ops.roll(x, shift, axes)

def covar(a,b):
    ma = K.mean(a)
    mb = K.mean(b)
    am, bm = a-ma, b-mb
    s = K.sum(tf.multiply(am,bm))
    
    #return s/tf.cast((tf.size(a)[-1]-1),tf.float32)
    return s

def npcc(y_true, y_pred):
    print(y_true.shape)
    r_num = covar(y_true,y_pred)
    r_den = K.sqrt(tf.multiply(covar(y_true,y_true),covar(y_pred,y_pred)))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return -r

def npcc_fft(y_true, y_pred):
    pred_fft = tf.signal.fft2d(tf.cast(tf.reshape(y_pred,[-1,y_pred.shape[1],y_pred.shape[2]]),dtype=tf.complex64))
    #pred_fft = tf.signal.rfft2d(tf.reshape(y_pred,[-1,y_pred.shape[1],y_pred.shape[2]]))
    
    return npcc(y_true,y_pred) + 0.1*K.mean(K.abs(pred_fft))

def npcc_npcc_fft(y_true,y_pred):
    pred_fft_abs = tf.math.log(K.abs(tf.signal.fft2d(tf.cast(tf.reshape(y_pred,[-1,y_pred.shape[1],y_pred.shape[2]]),dtype=tf.complex64)))+1)
    true_fft_abs = tf.math.log(K.abs(tf.signal.fft2d(tf.cast(tf.reshape(y_true,[-1,y_pred.shape[1],y_pred.shape[2]]),dtype=tf.complex64)))+1)
    
    pred_fft_phase = tf.math.angle(tf.signal.fft2d(tf.cast(tf.reshape(y_pred,[-1,y_pred.shape[1],y_pred.shape[2]]),dtype=tf.complex64)))
    true_fft_phase = tf.math.angle(tf.signal.fft2d(tf.cast(tf.reshape(y_true,[-1,y_pred.shape[1],y_pred.shape[2]]),dtype=tf.complex64)))
    
    return 0.5*npcc(y_true,y_pred) + 0.5*npcc(pred_fft_abs,true_fft_abs) # + 0.25 * npcc(tf.math.multiply_no_nan(pred_fft_phase,pred_fft_abs),tf.math.multiply_no_nan(true_fft_phase,true_fft_abs))
    

def npcc_mse(y_true, y_pred):
    ratio = 0.8
    return (ratio * npcc(y_true,y_pred)) + ((1-ratio) * losses.mean_squared_error(y_true,y_pred))

def volume(y_true, y_pred):
    return tf.div(K.sum(K.abs(y_true-y_pred)),K.sum(y_true))

def npcc_volume(y_true,y_pred):
    return 0.9 * npcc(y_true,y_pred) + 0.1*volume(y_true, y_pred)

def ms_ssim(y_true,y_pred):
    return 1-tf.image.ssim_multiscale(y_true,y_pred,1)

class phase_loss:    
    def __init__(self,x,y,bb):
        self.bb = bb
        #self.x = x
        #self.y = y
        self.np_mask = np.zeros((x,y))
        self.np_mask[bb[0]:bb[0]+bb[2],bb[1]:bb[1]+bb[3]] = 1
        
        #self.mask = tf.ragged.boolean_mask(np_mask.reshape((x,y,2)))
    
    def phase(self,y_true_comb,y_pred_comb):
        y_true, y_mask = tf.split(y_true_comb,[1,1],axis=-1)
        y_pred, y_blah = tf.split(y_pred_comb,[1,1],axis=-1)
        
        print('true comb',y_true_comb.shape)
        print('pred comb',y_pred_comb.shape)
        
        #print(tf.reshape(y_pred,[-1,y_pred.shape[1],y_pred.shape[2]]).shape)
        pred_fft = tf.signal.fft2d(tf.cast(tf.reshape(y_pred,[-1,y_pred.shape[1],y_pred.shape[2]]),dtype=tf.complex64))
        pred_fft_shifted = self.fftshift(pred_fft,axes=(1,2))
        true_fft = tf.signal.fft2d(tf.cast(tf.reshape(y_true,[-1,y_pred.shape[1],y_pred.shape[2]]),dtype=tf.complex64))
        true_fft_shifted = self.fftshift(true_fft,axes=(1,2))
        
        #print(pred_fft.dtype)
        #print(tf.convert_to_tensor(self.np_mask,dtype=tf.complex64).dtype)
        #mask = tf.cast(tf.convert_to_tensor(self.np_mask),dtype=tf.complex64)
        #mask_b = tf.broadcast_to(mask,tf.shape(true_fft))
        
        #masked_true = tf.ragged.boolean_mask(data=true_fft,mask=self.np_mask.reshape((x,y,2)))
        #masked_pred = tf.ragged.boolean_mask(data=pred_fft,mask=self.np_mask.reshape((x,y,2)))
        y_mask_comp = tf.cast(tf.reshape(y_mask,[-1,y_pred.shape[1],y_pred.shape[2]]),dtype=tf.complex64)
        #y_mask_comp = tf.cast(y_mask,dtype=tf.complex64)
        masked_true = tf.multiply(y_mask_comp,true_fft_shifted)
        masked_pred = tf.multiply(y_mask_comp,pred_fft_shifted)
        
        pred_ifft = tf.signal.ifft2d(self.ifftshift(masked_pred,axes=(1,2)))
        true_ifft = tf.signal.ifft2d(self.ifftshift(masked_true,axes=(1,2)))
        
        pred_phase = tf.atan2(tf.imag(pred_ifft),tf.real(pred_ifft))
        true_phase = tf.atan2(tf.imag(true_ifft),tf.real(true_ifft))
        #print(y_pred.shape)
        return pred_phase,true_phase
    
    def mse(self,y_true,y_pred):
        print('mse', y_true.shape)
        pred_phase,true_phase = self.phase(y_true,y_pred)
        
        #loss = losses.mean_squared_error(pred_ifft,true_ifft)
        return losses.mean_squared_error(pred_phase,true_phase)
    
    def npcc(self,y_true,y_pred):
        pred_phase,true_phase = self.phase(y_true,y_pred)
        
        return npcc(true_phase,pred_phase)
    
    def holo_npcc(self,y_true, y_pred):
        r_num = covar(y_true,y_pred)
        r_den = K.sqrt(tf.multiply(covar(y_true,y_true),covar(y_pred,y_pred)))
        r = r_num / r_den

        r = K.maximum(K.minimum(r, 1.0), -1.0)
        return -r
        
    def mixed(self,y_true,y_pred):
        print('mixed', y_true.shape)
        return 0.1*self.mse(y_true,y_pred) + 0.9*self.holo_npcc(y_true,y_pred)
        
        