# SFND 3D Object Tracking


# Rubric

## FP.0 Final Report
this file is providing all information

## FP.1 Match 3D Objects
 commit 484e7695d9a35c26

## FP.2 Compute Lidar-based TTC
 commit 4353321aa3b11

## FP.3 Associate Keypoint Correspondences with Bounding Boxes 
 commit d080e54e2856

## FP.4 Compute Camera-based TTC 
 commit ebebdce06549

## FP.5 Performance Evaluation 1 
At several frames from the Lidar cloud points there are outlier points that are appearing and making calculations to jump.
I observe places where after filtering the outliers points the TTC result again jumps too much. 

I believe possible reasons for this are:

 1.The Lidar sensor frame rate is not constant and TTC calculations relying on constant frame rate gives estimation error
 
 2.The Lidar frame rate is too low and cannot detect reliably the quicker dynamic changes in the acceleration/deceleration between the ego car and the car in front of us. We see jumping in the results
 
 3.The surface of the preceding vehicle is making noisy reflections and the resulting point cloud is with inconsistent points density

Until frame 6 we have ttc=~12s then it jumps in frame[7]=7s;  frame[8]=44s frame[9]=24s then it again is returned to ttc= ~12s. Because we use results from one frame in two calculations the error is spread in several calculations. 
At frame[44]=16s the TTC is jumping a lot compared to the previous and next frames that have ttc=~5s
After frame[48] results are jumping a lot but actually the cars are not moving.

 

## FP.6 Performance Evaluation 2 
Shared [table](https://docs.google.com/spreadsheets/d/1hf6fANiiCgElG7U1ZphyN-TaG75hdg9SFj6NIW7dZl8/edit?usp=sharing) with all calculations.
 
The detectors returning more keypoints calculates better the TTC. The reason is in the stable search for match points. When more points are used the TTC is calculated more robust.

`AKAZE` detector is most stable method no matter which descriptor is used after that.
`SHI-THOMASI` detector is giving good stable results but only when used with descriptor `BRISK` and `FREAK`. With other descriptors there are some fluctuations in several frames (frame 8-9).

Some of the detectors are giving not stable estimations and are not good to be used like `ORB`, `BRISK` and `HARRIS`.

I have organized several tables with all results. 
In the sheet “raw data” I have all results from the calculations. Then split them in the following sheets to have more detailed charts.
In sheet “lidar” are the TTC calculated from LiDAR.
In sheet “all TTC cam” are all combinations descriptor/detectors but it is difficult to analyze the resulting chart because too many fluctuations.
Then in the following sheets are charts with only one detector type to observe and compare stability.
