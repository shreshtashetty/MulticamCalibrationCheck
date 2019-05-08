import matplotlib
import matplotlib.cm
import tensorflow as tf
import numpy as np


def print_shape(t):
    print(t.name, t.get_shape().as_list())

def colorize(value, normalize=True, vmin=None, vmax=None, cmap=None, vals=255):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.

    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.

    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
      - vals: the number of values in the cmap minus one

    Example usage:

    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """
    value = tf.squeeze(value, axis=3)

    if normalize:
        vmin = tf.reduce_min(value) if vmin is None else vmin
        vmax = tf.reduce_max(value) if vmax is None else vmax
        value = (value - vmin) / (vmax - vmin) # vmin..vmax

        # dma = tf.reduce_max(value)
        # dma = tf.Print(dma, [dma], 'dma', summarize=16)
        # tf.summary.histogram('dma', dma) # just so tf.Print works
        
        # quantize
        indices = tf.to_int32(tf.round(value * float(vals)))
    else:
        # quantize
        indices = tf.to_int32(value)

    # 00 Unknown 0 0 0
    # 01 Terrain 210 0 200
    # 02 Sky 90 200 255
    # 03 Tree 0 199 0
    # 04 Vegetation 90 240 0
    # 05 Building 140 140 140
    # 06 Road 100 60 100
    # 07 GuardRail 255 100 255
    # 08 TrafficSign 255 255 0
    # 09 TrafficLight 200 200 0
    # 10 Pole 255 130 0
    # 11 Misc 80 80 80
    # 12 Truck 160 60 60
    # 13 Car:0 200 200 200
  
    if cmap=='vkitti':
        colors = np.array([0, 0, 0,
                           210, 0, 200,
                           90, 200, 255,
                           0, 199, 0,
                           90, 240, 0,
                           140, 140, 140,
                           100, 60, 100,
                           255, 100, 255,
                           255, 255, 0,
                           200, 200, 0,
                           255, 130, 0,
                           80, 80, 80,
                           160, 60, 60,
                           200, 200, 200,
                           230, 208, 202]);
        colors = np.reshape(colors, [15, 3]).astype(np.float32)/255.0
        colors = tf.constant(colors)
    else:
        # gather
        cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
        if cmap=='RdBu' or cmap=='RdYlGn':
            colors = cm(np.arange(256))[:, :3]
        else:
            colors = cm.colors
        colors = np.array(colors).astype(np.float32)
        colors = np.reshape(colors, [-1, 3])
        colors = tf.constant(colors, dtype=tf.float32)
    
    value = tf.gather(colors, indices)
    # value is float32, in [0,1]
    return value
