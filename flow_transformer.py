import tensorflow as tf

def transformer(im, flow, out_size, name='SpatialTransformer', **kwargs):

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolateHW(im, x, y, out_size):
        with tf.variable_scope('_interpolateHW'):
            # constants
            shape = im.get_shape()
            bs = int(shape[0])
            h = int(shape[1])
            w = int(shape[2])
            c = int(shape[3])

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            h_f = tf.cast(h, 'float32')
            w_f = tf.cast(w, 'float32')
            out_h = out_size[0]
            out_w = out_size[1]

            # clip coordinates to [0, dim-1]
            x = tf.clip_by_value(x, 0, w_f-1)
            y = tf.clip_by_value(y, 0, h_f-1)
            
            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1
            y1_f = y0_f + 1
            x0 = tf.cast(x0_f, 'int32')
            y0 = tf.cast(y0_f, 'int32')
            x1 = tf.cast(tf.minimum(x1_f, w_f-1), 'int32')
            y1 = tf.cast(tf.minimum(y1_f, h_f-1), 'int32')
            dim2 = w
            dim1 = w*h
            base = _repeat(tf.range(bs)*dim1, out_h*out_w)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore c dim
            im_flat = tf.reshape(im, tf.stack([-1, c]))
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # grab the occluded pixels this flow uncovers
            d_a = tf.sparse_to_dense(tf.cast(idx_a,'int32'), [bs*h*w], 1, default_value=0, validate_indices=False, name="d_a")
            d_a = tf.reshape(tf.tile(tf.reshape(tf.cast(d_a,'float32'),[bs,h,w,1]),[1,1,1,c]),[-1,c])

            d_b = tf.sparse_to_dense(tf.cast(idx_b,'int32'), [bs*h*w], 1, default_value=0, validate_indices=False, name="d_b")
            d_b = tf.reshape(tf.tile(tf.reshape(tf.cast(d_b,'float32'),[bs,h,w,1]),[1,1,1,c]),[-1,c])

            d_c = tf.sparse_to_dense(tf.cast(idx_c,'int32'), [bs*h*w], 1, default_value=0, validate_indices=False, name="d_c")
            d_c = tf.reshape(tf.tile(tf.reshape(tf.cast(d_c,'float32'),[bs,h,w,1]),[1,1,1,c]),[-1,c])

            d_d = tf.sparse_to_dense(tf.cast(idx_d,'int32'), [bs*h*w], 1, default_value=0, validate_indices=False, name="d_d")
            d_d = tf.reshape(tf.tile(tf.reshape(tf.cast(d_d,'float32'),[bs,h,w,1]),[1,1,1,c]),[-1,c])
            
            # and finally calculate interpolated values
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            warp = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            occ = tf.add_n([wa*d_a, wb*d_b, wc*d_c, wd*d_d])
            return warp, occ

    def _meshgridHW(h, w):
        with tf.variable_scope('_meshgridHW'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(0, w-1, w),
            #                         np.linspace(0, h-1, h))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([h, 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(0.0, w-1, w), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0, h-1, h), 1),
                            tf.ones(shape=tf.stack([1, w])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(flow, im, out_size):
        with tf.variable_scope('_transform'):
            shape = im.get_shape()
            bs = int(shape[0])
            h = int(shape[1])
            w = int(shape[2])
            c = int(shape[3])
            
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            h_f = tf.cast(h, 'float32')
            w_f = tf.cast(w, 'float32')
            out_h = out_size[0]
            out_w = out_size[1]
            grid = _meshgridHW(out_h, out_w)
            
            [x_f, y_f] = tf.unstack(flow, axis=3)
            x_f_flat = tf.expand_dims(tf.reshape(x_f, (bs, -1)),1)
            y_f_flat = tf.expand_dims(tf.reshape(y_f, (bs, -1)),1)
            zeros = tf.zeros_like(x_f_flat)
            flowgrid = tf.concat(axis=1, values=[x_f_flat,y_f_flat,zeros])

            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([bs]))
            grid = tf.reshape(grid, tf.stack([bs, 3, -1]),name="grid")

            # using flow from 1 to 2, warp image 2 to image 1
            grid=grid+flowgrid
                
            x_s = tf.slice(grid, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(grid, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            w, o = _interpolateHW(
                im, x_s_flat, y_s_flat,
                out_size)
            warp = tf.reshape(w,tf.stack([bs,
                                         out_h,
                                         out_w,
                                         c]),
                              name="warp")
            occ = tf.reshape(o,tf.stack([bs,
                                        out_h,
                                        out_w,
                                        c]),
                             name="occ")
            return warp, occ

    with tf.variable_scope(name):
        warp, occ = _transform(flow, im, out_size)
        return warp, occ

