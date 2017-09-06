# **Behavioral Cloning** 

## Writeup

Pre-stuff

To set-up the environment in the AWS instance do: source activate carnd-term1 and ALSO conda install opencv
---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/angles_before_augment_mydata.png "Udacity Data Before Augmentation"
[image3]: ./examples/angles_after_augment_mydata.png "Udacity Data After Augmentation"
[image4]: ./examples/original_image.png "Original Test Image"	  
[image5]: ./examples/augmented_images.png "Augmented Images"

## Rubric Points
### Here I consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a neural network inspired by the NVIDIA team (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with 5 convulutional networks (CovNets): the first 3 with 5x5 filter size, stride of 2 and depths 24, 36 and 48 respectively. The last two CovNets consist of 3x3 filter size and depths 64 and 64. (model.py lines 172-187) 

The model includes RELU activation layers after each CovNet to introduce nonlinearity (code line 177-181), and the data is normalized in the model using a Keras lambda layer (code line 174). 

#### 2. Attempts to reduce overfitting in the model

No overfitting was observed as the mse in the training and validation sets were very similar (around 0.06).

#### 3. Model parameter tuning

The model used an adam optimizer, with a learning rate of 0.002. This seems to yield the better driving results than staying with the default
0.001.

#### 4. Appropriate training data

First I recorded data as suggested in the lessons and then added some more: 2 laps each way of the track and 2 recovery
laps also each way. I then had trouble in specific parts of the track, for example leaving the bridge and also in the dirt
right after. I proceeded to collect some more data for this particular parts. After this the car didn't have any more problems
driving in those specific parts. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start very simple as the lessons suggest. First building a model
with one flat layer, then implementing LeNet and finally moving on the NVIDIA model. One of the most effort-demanding and probably
critical task is making sure one does not get a biased data set as seen in the first figure where only steering angles of
zero are obtained since the car is mostly driving straight.

![alt text][image2]

In order to alleviate this, one can add the left and right images which introduce more data at non-zero angles. Additionally,
a random height and width shift is added which also introduces corresponding changes in the steering angle. This transformation in
addition to a random brightness and a horizontal random shift are applied to ALL images(center, left and right). The steering angle
distribution of the center, left and right images plus all the augmented images is shown in the following figure. 

![alt text][image3]

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
The final step was to run the simulator to see how well the car was driving around track one. 
At the end of the process, the vehicle is able to drive autonomously (and smootly when going straight)
around the track without leaving the road.

#### 2. Final Model Architecture

My model consists of a neural network inspired NVIDIA team (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with 5 convulutional networks: the first 3 with 5x5 filter size, stride of 2 and depths 24, 36 and 48 respectively. The last two CovNets consist of 3x3 filter size and depths 64 and 64 respectively. (model.py lines 162-184) 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To augment the dataset, I transformed the images randomly applying random shifts in: horizontal orientation, brightness and random height and weight shifts. With the height and weight shifts the steering angle was modified accordingly. I think that having dataset that resebles actual driving is crucial. That is why these random shifts in steering angles are important. The change of the steering angles after the width change is about 0.004 steering units per pixel by plotting the offset from right and left sides to match the center according to https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9. Examples of the shifts created by these random transformations are observed in the following figures which show an original images with 4 corresponding random transformation. Printed is also shown the new steering angle obtained. The cropping is not show since it is done internally by Keras. This was chosen to be removing 70px from the top and 25px from the bottom.   

![alt text][image4]

![alt text][image5]

After the collection process 12655 frames were collected. For the validation set, 20% is taken leaving us, with 10124 frames or lines. The images and angles added to the training sample were: the original, left and images and the augmented original, left and right images. This leaves us with a todal of 60744 data points for training. 