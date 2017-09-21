# Vehicle Detection Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains my implementation of the Vehicle Detection Project (Term 1 - Project 5), which is part of the Udacity Self Driving Car NanoDegree.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. 
Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself.   You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

This data has to be unzipped in the `data/vehicles` and `data/non-vehicles` subfolders. 

### Files

* `vehicle_detection.ipynb`:  Jupyter notebook contains the project code. 
* `writeup_report.md`: The project writeup.

Folders:
* `data`: Contains the Labeled training data
* `test_images`: Images for testing your pipeline on single frames.  
* `test_images_output`: Contains intermediate images from my processing pipeling, using input images from the `test_images` folder.
* `test_videos`: Input videos for the project.
* `test_videos_output`: Output videos using input videos from the `test_videos` folder.

The folder `test_images_output` contains the following:
* Input image (`01_image_in_test#.jpg`)
* Image with all positively predicted boxes (`02_image_boxes_test#.jpg`)
* Heatmap of the combined boxes (`03_heatmap_test#.jpg`)
* Output image (`04_out_img_test#.jpg`)

### The Project

The goals / steps of this project were the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

