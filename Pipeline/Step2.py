import sys
import os
from collections import namedtuple
import operator
import glob
import csv 

import numpy as np

import math
from math import sqrt
from math import ceil

import PIL
from PIL import Image, ImageDraw, ImageFilter, ImageFile

import skimage
import skimage.io
import skimage.measure

import scipy

import shapely
import shapely.geometry
from shapely.geometry import Polygon

import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

# List of modifiable variables. Will become object properties once encapsulated

# Image and training set processing

subimage_width = None                   # Width of each subimage after slicing out of the original
subimage_height = None                  # Height of each subimage after slicing out of the original
height_spacing = None                   # How far apart vertically each subimage is sliced
width_spacing = None                    # How far apart horizontally each subimage is sliced

# Neural net parameters


# Step1: Functions for splitting images apart into subimages

def split_image(image, sub_width = 200, sub_height = 300, height_spacing = 50, width_spacing = 50):

    '''
    Given an input image and the space between subimage splits, splits and stores all subimages 
    for training as well as the relative coordinates.
    Currently lacks a good solution to the far edges.
    '''

    # Defaults

    if sub_width == None:
        sub_width = 200

    if sub_height == None:
        sub_height= 300

    if height_spacing == None:
        height_spacing = 50

    if width_spacing == None:
        width_spacing = 50

    # Body starts here

    if type(image) != np.ndarray:

        return None

    total_width = image.shape[1]
    total_height = image.shape[0]

    width_splits = total_width/width_spacing
    height_splits = total_height//height_spacing

    subimages = []

    for width_scan in range(int(width_splits - ceil(sub_width/width_spacing))):

        for height_scan in range(int(height_splits - ceil(sub_height//height_spacing))):

            current_section = image[height_scan*height_spacing:height_scan*height_spacing + sub_height,
                                    width_scan*width_spacing:width_scan*width_spacing + sub_width]

            subimages.append(current_section)

    return subimages


def coords(subimage, dot_subimage, padding = 0.2):

        """ Extract coordinates of dotted sealions and return list of SeaLionCoord objects)
            Code is adapted from:
            __description__ = 'Sea Lion Prognostication Engine'
            __version__ = '0.1.0'
            __license__ = 'MIT'
            __author__ = 'Gavin Crooks (@threeplusone)'
            __status__ = "Prototype"
            __copyright__ = "Copyright 2017"
        """         

        src_img = subimage[padding-1 : subimage.shape[1]-padding , padding-1 : subimage.shape[0]-padding]
        dot_img = dot_subimage[padding-1 : dot_subimage.shape[1]-padding , padding-1 : dot_subimage.shape[0]-padding]
        
        normsubsave = scipy.misc.imsave('normal_subimage.jpeg', src_img)
        dotsave = scipy.misc.imsave('dot_subimage.jpeg', dot_img)
        
        # Potential dot colors

        cls_colors = (
            (243,8,5),          # red
            (244,8,242),        # magenta
            (87,46,10),         # brown 
            (25,56,176),        # blue
            (38,174,21),        # green
            )

        # Empirical constants
        MIN_DIFFERENCE = 16
        MIN_AREA = 9
        MAX_AREA = 100
        MAX_AVG_DIFF = 50
        MAX_COLOR_DIFF = 32
       
        #src_img = np.asarray(self.load_train_image(train_id, mask=True)

        img_diff = np.abs(src_img-dot_img)

        diffsave= scipy.misc.imsave('diff_save.jpeg', img_diff)
        
        # Detect bad data. If train and dotted images are very different then somethings wrong.
        avg_diff = img_diff.sum() / (img_diff.shape[0] * img_diff.shape[1])

        if avg_diff > MAX_AVG_DIFF:
             return None
        
        img_diff = np.max(img_diff, axis=-1)   
           
        img_diff[img_diff<MIN_DIFFERENCE] = 0
        img_diff[img_diff>=MIN_DIFFERENCE] = 255

        diffsaveBW= scipy.misc.imsave('diff_saveBW.jpeg', img_diff)

        sealions = []
        
        for cls, color in enumerate(cls_colors):
            # color search backported from @bitsofbits.
            color_array = np.array(color)[None, None, :]
            has_color = np.sqrt(np.sum(np.square(dot_img * (img_diff > 0)[:,:,None] - color_array), axis=-1)) < MAX_COLOR_DIFF 
            contours = skimage.measure.find_contours(has_color.astype(float), 0.5)
            for cnt in contours :

                if len(cnt)<3:                  #Handles:"A LinearRing must have at least 3 coordinate tuples" error
                    continue

                p = Polygon(shell=cnt)
                area = p.area 
                if(area > MIN_AREA and area < MAX_AREA) :
                    y, x= p.centroid.coords[0] # DANGER : skimage and cv2 coordinates transposed?
                    x = int(round(x))
                    y = int(round(y))
                    sealions.append([cls, x, y])


        label = [0,0,0,0,0]
        left_x = padding*subimage.shape[1]
        right_x = (1-padding)*subimage.shape[1]
        left_y = padding*subimage.shape[0]
        right_y = (1-padding)*subimage.shape[0]
        for dot in sealions:
            if left_x < dot[1] < right_x:
                if left_y < dot[2] < right_y:

                    label[dot[0]] += 1

        return label     

def create_training_single(normal_path, dot_path)

    '''
    Given the paths a single "Full" image's normal and dotted versions, function will:
        1.) Split the images apart into subimages
        2.) Locate the dots in each subimage
        3.) Output lists of valid subimages and corresponding labels
    This output represents a portion of the overall training data for the neural net
    '''

    # Read in images
    normal_image = np.asarray(Image.open(normal_path), dtype = 'float')
    dot_image = np.asarray(Image.open(dot_path), dtype = 'float')

    # Save images
    # normalsave = scipy.misc.imsave('normalsave.jpeg', normal_image)
    # dotsave = scipy.misc.imsave('dotsave.jpeg', dot_image)


    # Split images into subimages
    normal_splits = split_image(normal_image)
    dot_splits = split_image(dot_image)

    # Find location of dots, and create a label for the input image
    splits_labels = []
    valid_index = []
    for subimage_number in range(len(normal_splits)):
        sub_image_label = coords(normal_splits[subimage_number],dot_splits[subimage_number])
        if sub_image_label != None:
            valid_index.append(subimage_number)
        splits_labels.append(sub_image_label)

    valid_labels = [splits_labels[i] for i in valid_index]
    valid_subimages = [normal_splits[i] for i in valid_index]

    print(valid_labels, len(valid_subimages),len(valid_labels))

    return valid_subimages, valid_labels


'''

# Step3

dataset_path      = "/path/to/out/dataset/mnist/"
test_labels_file  = "test-labels.csv"
train_labels_file = "train-labels.csv"

inputdata = "bla"

input_width = 200
input_height = 200
channels = 3
input_full_length = input_height*input_width*channels

n_nodes_hl1 = 10000
n_nodes_hl2 = 1000
n_nodes_hl3 = 200

n_classes = 5
batch_size = 1 #Will vary based on choice

x_placeholder = tf.placeholder('float',[None, input_fulllength])
y_placeholder = tf.placeholder('float')



def neural_network_model(data):

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([input_full_length, n_nodes_hl1])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
    'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
    'biases':tf.Variable(tf.random_normal([n_classes]))}

    # The actual model starts here

    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3) 
    
    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output
    
def train_neural_network(x_placeholder):

    prediction = neural_network_model(x_placeholder)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_placeholder))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    num_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(image_count/batch_size)):
                batch_x = training_images[(batch_size * _): (batch_size * _) + batch_size]
                batch_y = training_labels[(batch_size * _): (batch_size * _) + batch_size]
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(epoch_y,1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:test_images, y:test_labels})) 

#train_neural_network(x_placeholder)
'''
    