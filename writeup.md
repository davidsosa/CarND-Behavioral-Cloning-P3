# **Behavioral Cloning** 

## Writeup

Pre-stuff

To set-up the environment in the AWS instance do: source activate carnd-term1 and ALSO
conda install opencv
---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/angles_before_augment_udacitydata.png "Udacity Data Before Augmentation"
[image4]: ./examples/angles_after_augment_udacitydata.png "Udacity Data After Augmentation"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

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

My model consists of a neural network inspired NVIDIA team (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with 5 convulutional networks: the first 3 with 5x5 filter size, stride of 2 and depths 24, 36 and 48 respectively. The last two CovNets consist of 3x3 filter size and depths 64 and 64. (model.py lines 162-184) 

The model includes RELU layers after each CovNet to introduce nonlinearity (code line 167-171 ), and the data is normalized in the model using a Keras lambda layer (code line 164). 

#### 2. Attempts to reduce overfitting in the model

At the moment I see no overfitting as the mse in the training and validation sets are very similar.

#### 3. Model parameter tuning

The model used an adam optimizer, with a learning rate of 0.002. This seems to yield the better driving results.

#### 4. Appropriate training data

Since I wanted to make sure I had a good model going on I used the Udacity data, since I have read online and in the forums
people have gotten good results with this.

I did however tried to record my data with the runs as suggested in the lessons. However it often happened that updated
the data and my model would end up doing worse. That is why I decided to stick with the Udacity data.   

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start very simple as the lessons suggest. First building a model
with one flat layer, then implementing LeNet and finally moving on the NVIDIA model. One of the most effort-demanding and probably
critica task is making sure one does not get an biased data set as seen in the first figure where one onle get steering angles of
zero.

![alt text][image2]

In order to alleviate this, one can add the left and right images which introduce more data at non-zero angles. Additionally,
a random height and width shift which also introduces corresponding changes in the steering angle. This transformation in
addition to a random brightness and a horizontal random shift are applied to ALL images. The steering angle distribution of
all the augmented images is shown in the following figure. 

![alt text][image3]

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. 
The final step was to run the simulator to see how well the car was driving around track one. 
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My model consists of a neural network inspired NVIDIA team (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with 5 convulutional networks: the first 3 with 5x5 filter size, stride of 2 and depths 24, 36 and 48 respectively. The last two CovNets consist of 3x3 filter size and depths 64 and 64. (model.py lines 162-184) 

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To augment the dataset, I transformed the images randomly applying random shifts in : horizontal orientation, brightness and (small) random height and weight shifts. With the height and weight shifts the steering angle was modified accordingly. I think that having dataset that resebles actual driving is crucial. That is why this random shifts in in steering angles are important 

After the collection process, I around 30000 images counting left and right images. This data was preprocessed with the augmentation just mentioned, the normalization and the cropping. I finally randomly shuffled the data set and put 20% of the data into a validation set. The model did not show signs of overfitting since the training erros was ... and the validation erros was ... . 
