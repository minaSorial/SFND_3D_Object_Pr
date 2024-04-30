# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
  * Install Git LFS before cloning this Repo.
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

# Writeup
##  TASK FP.1 : Match 3D Objects

In this task, we implemented the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property)“. Matches is done with the ones with the highest number of keypoint correspondences.


##  TASK FP.2 : Compute Lidar-based TTC

In this task, we implemented the method "computeTTCLidar" that computes the  Time-to-collision for all matched 3D objects based on Lidar measurements alone.
to make the TTC robust against outliers which might be way too close and thus lead to faulty estimates. we use the average distance of all the 3d lidar points instead of using the closest point only. 

##  TASK FP.3 : Associate Keypoint Correspondences with Bounding Boxes

In this task, we implemented the method "clusterKptMatchesWithROI" that finds all keypoint matches that belong to each 3D object.
To remove outliers among your matches. we compute the mean of all the euclidean distances between keypoint matches and then remove those that are too far away from the mean.
to compute the distance threshold we can use 
 * Constant multiplication of the mean for example  " 1.5 * mean ".
 *  standard deviation * mean.
 
both of the methods above works but for some (detector type/descriptor type) combination work best one of them produce better result than the other.


## TASK FP.4 : Compute Camera-based TTC
In this task, we implemented the method "computeTTCCamera" based on the distance ratio between matched keypoints pairs from matched bounding box pairs from frame to frame.  


## TASK FP.5 : Performance Evaluation 1 (TTC Lidar Evaluation)
The computation of TTC with lidar point cloud is based on estimation of the distance between between the lidar car and the car in front of it from frame to frame.

<img src="images/lidarttc.png" width="779" height="414" />

The accurate estimation/calculation of this distance across time is essitnial for a roubost and account TTC estimation.

Many factors can intudice faulty measurment of distance thus faulty estimation of TTC.
  * If our calculation of the distance is based on finding the nearest point from the lidar point cloud, any Lidar noisy measurement or other reflection points due to dust or weather conditions appear between the lidar position and the actual points that represent the car will be used as the base to calculate the TTC. This problem can be avoided by not using the nearest point but instead using the average distance of all the point clouds that represent the rear of the car

  * Another source of error is where there are outlier points due to the lidar scanning other parts of the vehicle.
  In our calculation of the TTC, we are only interested in lidar points that are on the rear surface of the car. 
  so to mitigate this problem, the outliers points can be filtered out by simply increasing the shrinking factor when clustering lidar points with ROI



Althogh after taking this sulution into consideratin the TTC estimation using lidar points don,t usually produce stabel/rouboust estimamtion.
For example the 3 photos below show the TTC lidar estimation from 3 consctive frames it not stable.
we can notice also the lidar point clould not always concetrated in the rear of the vehcile and also the depth span of the points in the X direction is not small that also cause some erros when caluclating the avrage distance.    

<img src="images/AKAZE | BRIEF0002.jpg" width="900" height="300" />
<img src="images/AKAZE | BRIEF0003.jpg" width="900" height="300" />
<img src="images/AKAZE | BRIEF0004.jpg" width="900" height="300" />

Depth span of the points in the X direction.

<img src="images/TTClidar4.jpg" width="300" height="300" />



## TASK FP.6 : Performance Evaluation 2 (TTC Camera Evaluation)
To Evaluate the TTC camera estimation we used diffrent combinations from (detector/descriptor)

 * Certain combinations of ORB and Harris detectors produce very poor and unreliable TTC estimation.

 * The top 3 detectors/descriptors recommended from the previous project produced good results especially FAST/ORB but still all of them produce some unreliable TTC estimation in some frames. 
   * FAST/BRIEF  
   * FAST/ORB
   * FAST/BRISK    

 * Other detectors SIFT- SHITOMASI- AKAZE  also produces good results especially SIFT and AZAKAN which produce better results than FAST detector. 
 * BRISK  Detector as we saw from the previous project is the one that produced the largest number of Keypoints matches but also during the TTC estimation the performance wasn't good. One idea to fix the problem is to filter out keypoint matches to select a good quality keypoints.

 * Given that the process of estimation of the TTC is a very fast-paced process and needed to be done many times in one second I will recommend using FAST/ORB or  SHITOMASI/BRIEF but if the computation power is not a problem, I will recommend AKAZE/AKAZE , AKAZE/FREAK,  SIFT/SIFT  as they produce better and more stable results.

 Here is an example of using AKAZE/FREAK in 3 consecutive frames.

 <img src="images/AKAZE | FREAK0012.jpg" width="900" height="300" />
 <img src="images/AKAZE | FREAK0013.jpg" width="900" height="300" />
 <img src="images/AKAZE | FREAK0014.jpg" width="900" height="300" />





































