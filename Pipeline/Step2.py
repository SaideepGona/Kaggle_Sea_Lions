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

import random
import pickle

import shapely
import shapely.geometry
from shapely.geometry import Polygon

import tensorflow as tf 
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
# import prettytensor as pt

#***************************************************************************************************************

# -------------------------- List of GLOBALS. Can modify input parameters here. --------------------------------

# Toggles
train_toggle = 1

# Image and training set processing
base_data_path = '/home/saideep/Github_Repos/Saideep/Kaggle_Sea_Lions/Sample_Data'
# base_data_path = '/home/saideep/Github_Repos/Saideep/Kaggle_Sea_Lions/Pipeline/Step2&3trainingstorage/Total_Training_Set'
normal_path = '/Train/'
dot_path = '/TrainDotted/'
subimage_width = 100                   # Width of each subimage after slicing out of the original
subimage_height = 100               # Height of each subimage after slicing out of the original
height_spacing = 6             # How far apart vertically each subimage is sliced
width_spacing = 6            # How far apart horizontally each subimage is sliced
channels = 3
subimage_stretched = subimage_height * subimage_width * channels             # The length of image vectors after flattening
padding_percentage = 20               # Perventage value for how far from the edge a dot must be to "count" towards the label
padding = padding_percentage/100  
total_images = 10
full_image_count = 10              # Number of full training images that will be used to train   
test_image_count = 5    
negated_images = 0
# Neural net parameters

# Convolutional layers
filter_size1 = 5
num_filters1 = 16
filter_size2 = 5
num_filters2 = 36
# Fully connected layers
hidden_layer_1_nodes = 80
hidden_layer_2_nodes = 10
hidden_layer_3_nodes = 200
# Labels and output
max_seal_count = 2
color_index = 0
label_set = []
for label in range(max_seal_count):
    current_label = [0] * (max_seal_count)
    current_label[label] += 1
    label_set.append(current_label)
output_classes = max_seal_count
# binary_label_length = output_classes * (max_seal_count+1)
label_weighting = 1000
batch_size = 1 
# Placeholders and other
images_placeholder = tf.placeholder(dtype='float',shape=[batch_size, subimage_stretched])         # Placeholder for subimage neural net input
labels_placeholder = tf.placeholder(dtype='float',shape=[batch_size, output_classes])             # Placeholder for subimage label neural net input
num_epochs = 20                                                         # Number of training epochs

graph_save_path = '/home/saideep/Github_Repos/Saideep/Kaggle_Sea_Lions/Pipeline/Saved_Models/save.cpkt'
#maxlabel(for testing)

MAXLABEL = 0

# Step1: Functions for splitting images apart into subimages

def split_image(image):

    '''
    Given an input image and the space between subimage splits, splits and stores all subimages 
    for training as well as the relative coordinates.
    Currently lacks a good solution to the far edges.
    '''

    # Body starts here

    if type(image) != np.ndarray:

        return None

    total_width = image.shape[1]
    total_height = image.shape[0]

    width_splits = total_width/width_spacing
    height_splits = total_height//height_spacing

    subimages = []
    flattened_subimages = []

    for width_scan in range(int(width_splits - ceil(subimage_width/width_spacing))):

        for height_scan in range(int(height_splits - ceil(subimage_height//height_spacing))):

            current_section = image[height_scan*height_spacing:height_scan*height_spacing + subimage_height,
                                    width_scan*width_spacing:width_scan*width_spacing + subimage_width]

            subimages.append(current_section)

    return subimages


def coords(subimage, dot_subimage):

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

        # binary_label = [0] * binary_label_length
        # # Constructs a "binary_label" version of the label

        # for color_num in range(len(label)):
        #     color_count = label[color_num]
        #     binary_label[color_num*(max_seal_count+1) + color_count] += 1
        # weighted_label = [val*2 for val in label]

        return label   

def create_training_single(normal_path, dot_path):
    global MAXLABEL
    global negated_images
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
        print(subimage_number)
        full_sub_image_label = coords(normal_splits[subimage_number],dot_splits[subimage_number])
        sub_image_label = [0] * (max_seal_count)
        if full_sub_image_label != None:
            valid_index.append(subimage_number)
            cur_maxlabel = max(full_sub_image_label)
            if cur_maxlabel > MAXLABEL:
                MAXLABEL = cur_maxlabel
            if full_sub_image_label[color_index] >= max_seal_count:
                full_sub_image_label[0] = max_seal_count-1
            sub_image_label[full_sub_image_label[0]] += 1
        splits_labels.append(sub_image_label)

    valid_labels = [splits_labels[i] for i in valid_index]
    if len(valid_labels) == 0:
        negated_images += 1
        return 0, 0
    
    valid_subimages = [normal_splits[i] for i in valid_index]

    # Normalization step to create a balanced test set

    valid_label_counts = []
    for label in label_set:
        valid_label_single_count = valid_labels.count(label)
        if valid_label_single_count == 0:
            negated_images += 1
            return 0,0
        valid_label_counts.append(valid_label_single_count)
    valid_label_count_min = min(valid_label_counts)

    print(valid_label_counts, "<---------------------valid label counts")

    base_lists = []

    for label_num in range(max_seal_count):
        current_base_list = [0] * (valid_label_counts[label_num]-valid_label_count_min)
        current_base_list = current_base_list + ([1]*valid_label_count_min)
        random.shuffle(current_base_list)
        base_lists.append(current_base_list)

    super_valid_subimages = []
    super_valid_labels = []

    for valid_label_num in range(len(valid_labels)):   #Loop through all the valid labels

        label_count = 0
        for label in label_set:                         #Loop through the possible label types

            # if base_lists[label_count] == None:
            #     label_count += 1
            #     continue   
            # else:                    
            if label == valid_labels[valid_label_num]:                                      #Verify that the valid label matches the label type
                if base_lists[label_count][0] == 1:                                         #Verify the normalized contribution of that label
                    super_valid_labels.append(valid_labels[valid_label_num])
                    super_valid_subimages.append(valid_subimages[valid_label_num])
                del base_lists[label_count][0]                                   #Delete the "used up" label type
                break
            else:
                label_count += 1

    print(super_valid_labels)

    return super_valid_subimages, super_valid_labels

# Step3
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights

def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features

def neural_network_model(data):

    '''
    Specifies the TensorFlow neural net model to be used. Does not actively run anything.
    '''

    # The actual model starts here

    # Convolution layers
    reshaped_placeholder = tf.reshape(data, [-1, subimage_width, subimage_height, channels])

    layer_conv1, weights_conv1 = new_conv_layer(input=reshaped_placeholder,
                   num_input_channels=channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)

    layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

    layer_conv3, weights_conv3 = new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

    layer_conv4, weights_conv4 = new_conv_layer(input=layer_conv3,
                   num_input_channels=num_filters2,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)

    flat_conv_out, len_conv_out = flatten_layer(layer_conv4)
    
    # Fully connected layers
    # Reference dictionaries for each fully connected layer

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len_conv_out, hidden_layer_1_nodes])),
    'biases':tf.Variable(tf.random_normal([hidden_layer_1_nodes]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([hidden_layer_1_nodes, hidden_layer_2_nodes])),
    'biases':tf.Variable(tf.random_normal([hidden_layer_2_nodes]))}

    # hidden_3_layer = {'weights':tf.Variable(tf.random_normal([hidden_layer_2_nodes, hidden_layer_3_nodes])),
    # 'biases':tf.Variable(tf.random_normal([hidden_layer_3_nodes]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([hidden_layer_2_nodes, output_classes])),
    'biases':tf.Variable(tf.random_normal([output_classes]))}

    l1 = tf.add(tf.matmul(flat_conv_out,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    # l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    # l3 = tf.nn.relu(l3) 
    
    output = tf.matmul(l2,output_layer['weights']) + output_layer['biases']
    print(type(output),"neural net model function")
    return output

def create_training_set():

    '''
    Runs through images on disc, processes, and "pickles" a balanced training set for later use.
    '''

    full_training_images = []
    full_training_labels = []

    for full_image_num in range(full_image_count):

        print('--------------------------------------FULL-IMAGE#  %d' % (full_image_num))
        
        normal_full_path = base_data_path + normal_path + str(full_image_num) + '.jpg'
        dots_full_path = base_data_path + dot_path + str(full_image_num) + '.jpg'

        whole_batch_x, whole_batch_y = create_training_single(normal_full_path, dots_full_path)
        if whole_batch_x == 0:
            continue

        for image in whole_batch_x:
            full_training_images.append(image)
        
        for label in whole_batch_y:
            full_training_labels.append(label)

    print(len(full_training_images))
    pickle.dump(full_training_images, open( "train_images.p", "wb" ))
    pickle.dump(full_training_labels, open( "train_labels.p", "wb" ))

        

def train_neural_network_preloaded():

    '''
    Instantiates the neural network model and runs the main training session. Takes as input the full training set rather
    than generating it within the training loops
    '''

    prediction = neural_network_model(images_placeholder)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_placeholder))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
 
    saver = tf.train.Saver()
   
       
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('sess')
        epoch_accuracies = []
        for epoch in range(num_epochs):
            epoch_runs = 0
            summed_predictions = 0
               
            whole_batch_x = pickle.load(open( "train_images.p", "rb" ))
            whole_batch_y = pickle.load(open( "train_labels.p", "rb" ))

            whole_batches = list(zip(whole_batch_x,whole_batch_y))
            random.shuffle(whole_batches)
            whole_batch_x, whole_batch_y = zip(*whole_batches)

            # whole_batch_x = training_set_subimages
            # whole_batch_y = training_set_labels

            if whole_batch_x == None:
                return None
            batch_count = len(whole_batch_x)//batch_size

            training_cutoff = math.ceil(0.8*len(batch_count))

            for batch_num in range(training_cutoff):
                print(batch_num, "<--------------------NEW BATCH")
                current_batch_x = np.zeros((batch_size, subimage_stretched))

                for batch in range(batch_size):
                    current_batch_x[batch,:] = whole_batch_x[batch_num*batch_size + batch].flatten()
                    
                current_batch_y = whole_batch_y[batch_num*batch_size: (batch_num+1)*batch_size]
                feed_dict = {images_placeholder: current_batch_x, labels_placeholder: current_batch_y}
                trained_model = sess.run([optimizer, cost], 
                    feed_dict = {images_placeholder: current_batch_x.reshape(batch_size, subimage_stretched), 
                        labels_placeholder: current_batch_y})
                
            # Test the accuracy of the neural net on the training set after each epoch of training

            for batch_num in range(training_cutoff, batch_count):

                current_batch_x = np.zeros((batch_size, subimage_stretched))
                for batch in range(batch_size):
                    current_batch_x[batch,:] = whole_batch_x[batch_num*batch_size + batch].flatten()
                current_batch_y = whole_batch_y[batch_num*batch_size: (batch_num+1)*batch_size]
    
                trained_out = sess.run(prediction, 
                    feed_dict = {images_placeholder: current_batch_x.reshape(batch_size, subimage_stretched)})

                if trained_out[0][0] > trained_out[0][1]:
                    out_lab = [1,0]
                else:
                    out_lab = [0,1]

                print(out_lab,current_batch_y, '<------------Comparison')
                
                if out_lab == current_batch_y[0]:
                    summed_predictions += 1
                    print('YES')

                epoch_runs += 1

            saver.save(sess, 'seals-model')
                # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(batch_y,1))
                # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))


            epoch_accuracy = summed_predictions/epoch_runs
            epoch_accuracies.append(epoch_accuracy)
            print(epoch_runs, summed_predictions, epoch_accuracy)
            print(epoch_accuracies)

        # sess.close()
        saver.save(sess, graph_save_path)
        return trained_model

# def train_neural_network():

#     '''
#     Instantiates the neural network model and runs the main training session.
#     '''

#     prediction = neural_network_model(images_placeholder)

#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels_placeholder))
#     print(type(cost),cost,"---------------COST") 
#     optimizer = tf.train.AdamOptimizer().minimize(cost)
#     print(type(optimizer),optimizer,"---------------Optimizer")     
#     saver = tf.train.Saver()
#     print(type(saver),saver,"---------------Saver")     
       
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         print('sess')
#         epoch_accuracies = []
#         for epoch in range(num_epochs):
#             epoch_runs = 0
#             summed_predictions = 0
#             for full_image_num in range(full_image_count):

#                 print('--------------------------------------FULL-IMAGE#  %d' % (full_image_num))
                
#                 normal_full_path = base_data_path + normal_path + str(full_image_num) + '.jpg'
#                 dots_full_path = base_data_path + dot_path + str(full_image_num) + '.jpg'

#                 whole_batch_x, whole_batch_y = create_training_single(normal_full_path, dots_full_path)
#                 if whole_batch_x == 0:
#                     continue
#                 batch_count = len(whole_batch_x)//batch_size
#                 true_count = 0

#                 if full_image_num % 2 == 0:
#                     saver.save(sess, 'seals-model')

#                 for batch_num in range(batch_count):
#                     print(batch_num, "<--------------------NEW BATCH")
#                     current_batch_x = np.zeros((batch_size, subimage_stretched))

#                     for batch in range(batch_size):
#                         current_batch_x[batch,:] = whole_batch_x[batch_num*batch_size + batch].flatten()
                        
#                     current_batch_y = whole_batch_y[batch_num*batch_size: (batch_num+1)*batch_size]
#                     feed_dict = {images_placeholder: current_batch_x, labels_placeholder: current_batch_y}
#                     trained_model = sess.run([optimizer, cost], 
#                         feed_dict = {images_placeholder: current_batch_x.reshape(batch_size, subimage_stretched), 
#                             labels_placeholder: current_batch_y})
                    
#                     trained_out = sess.run(prediction, 
#                         feed_dict = {images_placeholder: current_batch_x.reshape(batch_size, subimage_stretched)})

#                     if trained_out[0][0] > trained_out[0][1]:
#                         out_lab = [1,0]
#                     else:
#                         out_lab = [0,1]

#                     print(out_lab,current_batch_y, '<------------Comparison')

#                     if out_lab == current_batch_y[0]:
#                         summed_predictions += 1
#                         print('YES')

#                     epoch_runs += 1

#                     print(type(trained_out),trained_out, '<-----------Trained Out')
#                     print(current_batch_y, "<-------------------------correct")
#                 if full_image_num % 1 == 0:
#                     saver.save(sess, 'seals-model')
#                 # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(batch_y,1))
#                 # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
#                 print(type(trained_model), trained_model)

#             epoch_accuracy = summed_predictions/epoch_runs
#             epoch_accuracies.append(epoch_accuracy)
#             print(epoch_runs, summed_predictions, epoch_accuracy)
#             print(epoch_accuracies)

#         # sess.close()
#         saver.save(sess, graph_save_path,)
#         return trained_model


if train_toggle == 1:
    train_neural_network_preloaded()

else:
    create_training_set()

print(MAXLABEL, "maxlabel")


    