import cv2
import numpy as np
import tensorflow as tf

sample_dots = []

for x in range(10):

    current_num = str(x)
    img_path = 'Sample_Data/TrainDotted/'+current_num+'.jpg'
    current_image = cv2.imread(img_path)
    sample_dots.append(current_image)
    print(current_image[20,20])



first_dot = cv2.imread('Sample_Data/TrainDotted/0.jpg')
second_dot = cv2.imread('Sample_Data/TrainDotted/1.jpg')

print(first_dot.shape)
print(first_dot.dtype)

#TensorFlow

x1 = tf.constant([5])
x2 = tf.constant([6])

result = tf.mul(x1,x2)
print(result)

