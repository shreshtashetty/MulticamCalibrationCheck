from PIL import Image
import scipy.misc
import scipy.io
import os
import argparse
import numpy as np
import tensorflow as tf
import sys
from os import listdir
from os.path import isfile, join
import glob, os
from utils import *
import imageio

out_dir = '.'
# for idx in [0, 10, 20]:
for idx in [100]:
    idx2 = idx + 10
    # set up intrinsics
    H = 128
    W = 384
    FOV = 90.0
    focal = float(float(W)/(2.0*np.tan(FOV*np.pi/360.0)))
    print 'focal = %.2f' % focal
    fx = focal
    fy = focal
    x0 = float(W)/2.0
    y0 = float(H)/2.0

    # set data paths
    root = 'city_02_vehicles_140_episode_0061_2019-05-07'
    cam1 = 0
    cam2 = 20
    depth_path1 = '%s/CameraDepth%d/%06d.npy' % (root, cam1, idx)
    depth_path2 = '%s/CameraDepth%d/%06d.npy' % (root, cam2, idx2)
    rgb_path1 = '%s/CameraRGB%d/%06d.npy' % (root, cam1, idx)
    rgb_path2 = '%s/CameraRGB%d/%06d.npy' % (root, cam2, idx2)
    p1_path = '%s/CameraRGB%d/%06d_c2w.npy' % (root, cam1, idx)
    p2_path = '%s/CameraRGB%d/%06d_c2w.npy' % (root, cam2, idx2)

    # load the image data
    i1 = np.load(rgb_path1)
    i2 = np.load(rgb_path2)
    i1 = tf.expand_dims(tf.constant(i1.astype(np.uint8)), axis=0)
    i2 = tf.expand_dims(tf.constant(i2.astype(np.uint8)), axis=0)

    # load the camera poses
    origin_T_carla1 = np.load(p1_path)
    origin_T_carla2 = np.load(p2_path)

    # convert camera poses into standard format (origin_T_cam)
    carla_T_cam = np.eye(4, dtype=np.float32)
    carla_T_cam[0,0] = -1.0
    carla_T_cam[1,1] = -1.0
    origin_T_cam1 = np.matmul(origin_T_carla1, carla_T_cam)
    origin_T_cam2 = np.matmul(origin_T_carla2, carla_T_cam)

    # load the depth data
    d1 = np.load(depth_path1)
    d1 = d1.astype(np.float64)
    d1 = d1*1000.0
    d1 = tf.constant(d1)
    d1 = tf.expand_dims(tf.expand_dims(d1, axis=0), axis=3)

    # compute the relative transformation between cameras
    cam2_T_cam1 = np.matmul(np.linalg.inv(origin_T_cam2), origin_T_cam1)


    ## tensorflow
    
    # add a batch dim
    cam2_T_cam1 = tf.expand_dims(tf.constant(cam2_T_cam1.astype(np.float32)), 0)
    x0_ = tf.expand_dims(x0,0)
    y0_ = tf.expand_dims(y0,0)
    fx = tf.expand_dims(fx,0)
    fy = tf.expand_dims(fy,0)
    cam2_T_cam1 = tf.cast(cam2_T_cam1, tf.float32)
    d1 = tf.cast(d1, tf.float32)
    
    # turn the flow and relative transformatino into a flow field
    flow, XYZ2 = zrt2flow(d1, cam2_T_cam1, fy, fx, y0_, x0_)
    # visualize this
    flow_z = flow2color(flow, adaptive_max=False, maxFlow=100)

    # turn the images into floats (in range [-0.5, 0.5])
    i1_g = preprocess_color(i1)
    i2_g = preprocess_color(i2)
    
    # warp image2 into image1
    i1_m, _ = warper(i2_g, flow)
    
    # convert things to uint8 images for visualization
    d1_z = depth2inferno(d1)
    i1_m = back2color(i1_m)
    i1_g = back2color(i1_g)
    i2_g = back2color(i2_g)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()

        # evaluate the tensors
        cam2_T_cam1_, flow_z_, d1_z_, i1_g_, i2_g_, i1_m_ = sess.run([cam2_T_cam1, flow_z, d1_z, i1_g, i2_g, i1_m])

        # print/save what we got
        
        print 'cam2_T_cam1:', cam2_T_cam1_
        scipy.misc.imsave("%s/%d_zrt_flow.png" % (out_dir, idx), flow_z_[0])
        scipy.misc.imsave("%s/%d_d1.png" % (out_dir, idx), d1_z_[0])
        scipy.misc.imsave("%s/%d_i1_original.png" % (out_dir, idx), i1_g_[0])
        scipy.misc.imsave("%s/%d_i2_original.png" % (out_dir, idx), i2_g_[0])
        scipy.misc.imsave("%s/%d_i2_zrt_warp.png" % (out_dir, idx), i1_m_[0])
        imageio.mimsave("%s/%d_i1i2_zrt_warp.gif" % (out_dir, idx), [i1_g_[0],i1_m_[0]], 'GIF')
