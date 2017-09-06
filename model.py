import csv
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sklearn
from scipy.misc import *
import math

from PIL import Image

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Cropping2D
from keras.preprocessing.image import *
from keras.optimizers import Adam

lines = []

# set random seed 
seed = 11
np.random.seed(seed)

# Checkout the data using using Pandas 
columns = ['center', 'left', 'right', 'steering_angle', 'throttle', 'brake', 'speed']
#my_dir = '../udacity_data/data/'
my_dir = '../mydata/'
data = pd.read_csv(my_dir+'driving_log.csv', names=columns)

print("Dataset Columns:", columns, "\n")
print("Shape of the dataset:", data.shape, "\n")
print(data.describe(), "\n")

print("Data loaded...")

# open csv data and append  
with open( my_dir+'driving_log.csv' ) as csv_file:
  reader = csv.reader(csv_file)  
  for line in reader:
    lines.append(line)  

images = []
measurements = []

WIDTH_SHIFT_RANGE = 100
HEIGHT_SHIFT_RANGE = 40
# shift height/width of the image by a small fraction
def height_width_shift(img, steering_angle):
    rows, cols, channels = img.shape
    # Translation
    # Image will be shifted by left or right (up or down) by
    #+- WIDTH_SHIFT_RANGE/2 (HEIGHT_SHIFT_RANGE/2)  respectively

    # Idea from Vivek Yadav (https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)
    tx = WIDTH_SHIFT_RANGE * np.random.uniform() - WIDTH_SHIFT_RANGE / 2
    ty = HEIGHT_SHIFT_RANGE * np.random.uniform() - HEIGHT_SHIFT_RANGE / 2
    steering_angle = steering_angle + tx / WIDTH_SHIFT_RANGE * 2 * .2


    transform_matrix = np.float32([[1, 0, tx],
                                   [0, 1, ty]])
    
    translated_image = cv2.warpAffine(img, transform_matrix, (cols, rows))
    return translated_image, steering_angle


# Apply random brightness shift  
def brightness_shift(img, bright_value=None):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    if bright_value:
        img[:,:,2] += bright_value
    else:
        random_bright = .25 + np.random.uniform()
        img[:,:,2] = img[:,:,2] * random_bright
    
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img

  
def get_image(source_path):
  ''' function which given a path return an image '''
  filename = source_path.split('/')[-1]
  current_path = my_dir+'IMG/' + filename
  #image = cv2.imread(current_path)
  image = mpimg.imread(current_path)
  return image

def horizontal_flip(img, steering_angle):
    flipped_image = cv2.flip(img, 1)
    steering_angle = -1 * steering_angle
    return flipped_image, steering_angle

def apply_transformation(img, steering_angle):
    
    transformed_image, steering_angle = height_width_shift(img, steering_angle)
    transformed_image = brightness_shift(transformed_image)
   
    if np.random.random() < 0.5:
        transformed_image, steering_angle = horizontal_flip(transformed_image, steering_angle)
            
    #transformed_image = crop_resize_image(transformed_image)   
    return transformed_image, steering_angle

def get_image(source_path):
  ''' function which given a path return an image '''
  filename = source_path.split('/')[-1]
  current_path = my_dir+'IMG/' + filename
  #image = cv2.imread(current_path)
  image = mpimg.imread(current_path)
  return image

print("Entering image augmentation")

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        np.random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:      
            
              image_ori = get_image(batch_sample[0])            
              angle_ori = float(batch_sample[3])
              images.append(image_ori) 
              angles.append(angle_ori)

              image,angle = apply_transformation(image_ori,angle_ori)
              images.append(image)
              
              angles.append(angle)
              correction = 0.2
              # Load left frames and modity steering 
              image_left_0 = get_image(line[1])          
              angle_left_0 = angle + correction
              images.append(image_left_0)
              angles.append(angle_left_0)

              image_left,angle_left = apply_transformation(image_left_0,angle_left_0)
              images.append(image_left)
              angles.append(angle_left)
  
              # Load right frames and modity steering 
              image_right_0 = get_image(line[2])          
              angle_right_0 = angle - correction

              images.append(image_right_0)
              angles.append(angle_right_0)

              image_right,angle_right = apply_transformation(image_right_0,angle_right_0)
              images.append(image_right)
              angles.append(angle_right)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
batch_size = 32
train_generator = generator(train_samples, batch_size)
validation_generator = generator(validation_samples, batch_size)

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(160,320,3)))

model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample = (2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample = (2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample = (2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3,activation='relu'))
#model.add(Dropout(0.75))
model.add(Flatten()) 
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

opt = Adam(lr=0.0002)
model.compile(loss='mse', optimizer=opt)

model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*6, validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)

model.summary()

model.save('model.h5')

