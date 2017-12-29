# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./vitual_net_structure.png "Model Visualization"
[image2]: ./flip_image3.png "Flipped Image"
[image3]: ./center.jpg "center image"
[image4]: ./left.jpg "left Image"
[image5]: ./right.jpg "right Image"
[image6]: ./flip_image.png "Flipped Image"
[image7]: ./flip_image2.png "Flipped Image"


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 108-127) 

The model includes RELU layers to introduce nonlinearity (code line 108), and the data is normalized in the model using a Keras lambda layer (code line 99). 
My final model consisted of the following layers(reference: nvidia self-drive model).


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 108).

Besides, cropping2d is used to avoid overfit and underfit. because overfit come from noise datas which include nonsense obejects for instance trees and other scenes, and too small feature datas will result underfit. thus, appropriate image train datas will help us.


The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 70-75). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.



####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 134).


####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road in high speed to avoid too similar image data.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to training the steer angel and image datas to get a classification for real-time judge which time to change car steer.

My first step was to use a convolution neural network model similar to the nvidia model. I thought this model might be appropriate because nvidia have many experiment in self-drive car.

To combat the overfitting which  is low mean squared error in valid and is high mean squared error in training set, I added local normal step to change image pixel value in range (0,1).

Then I extract image to region of interests for reduing noise pixel. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track,as follow:
1)turn across

2)run in high speed

3)change road type

to improve the driving behavior in these cases, I turn down the speed , make more turn acorss and change road type datas.  

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes :

|Layer (type)      |           Output Shape    |          Param # |  
|:----------------:|:-------------------------:|:----------------:|
|lambda_1 (Lambda)  |          (None, 160, 320, 3)  |     0      |   
|cropping2d_1 (Cropping2D)  |  (None, 76, 320, 3) |       0       |  
|conv2d_1 (Conv2D)      |      (None, 36, 158, 24)   |    1824    |  
|conv2d_2 (Conv2D)      |      (None, 16, 77, 36)    |    21636 |    
|conv2d_3 (Conv2D)      |      (None, 6, 37, 48)    |     43248 |    
|conv2d_4 (Conv2D)     |       (None, 4, 35, 64)   |      27712 |    
|conv2d_5 (Conv2D)     |       (None, 2, 33, 64)    |     36928 |    
|flatten_1 (Flatten)   |       (None, 4224)       |       0    |     
|dense_1 (Dense)      |        (None, 100)        |       422500    
|dense_2 (Dense)       |       (None, 50)         |       5050   |   
|dense_3 (Dense)     |         (None, 1)           |      51     |   


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to change steer in specific situation. These images show what a recovery looks like starting from center camera,left camera and right camera :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would help model avoid noise image. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]



After the collection process, I had X number of data points. I then preprocessed this data by normalizing image_data = image_data/255.0 -0.5 for reducing overfit.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by stable training loss and stable valid accuration . I used an adam optimizer so that manually training the learning rate wasn't necessary.
