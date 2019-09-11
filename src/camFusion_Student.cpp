
#include <iostream>
#include <algorithm>
#include <numeric>
#include <map>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 2);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // Zero movement detection threshold
    const double zeroMovementDistance = 0.000001;
    // Time between two measurements in seconds
    double dT = 1/frameRate;

    double minXPrev = 1e9, minXCurr = 1e9;
    std::vector<double> minXPrevVector(10, minXPrev);
    std::vector<double> minXCurrVector(10, minXCurr);

    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); it++)
    {
        if (minXPrev > it->x)
        {
            minXPrev = it->x;
        }

        for (auto i = minXPrevVector.begin(); i != minXPrevVector.end(); i++)
        {
            if (*i > it->x)
            {
                minXPrevVector.insert(i, minXPrev);
                minXPrevVector.resize(10);
                break;
            }
        }
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); it++)
    {
        if (minXCurr > it->x)
        {
            minXCurr = it->x;
//            minXCurrVector.push_back(minXCurr);
        }
    }

    // Make some filtering of outliers
    for (auto i = minXPrevVector.begin(); i != minXPrevVector.end(); i++)
    {
        std::cout<<"Values min="<<*i<<std::endl;
    }




    double distanceChange = minXPrev - minXCurr;
    if (distanceChange < zeroMovementDistance)
        distanceChange = zeroMovementDistance;
    TTC = minXCurr * dT / distanceChange;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    map<pair<int, int>, int> matchesCounter;
    // Assign only the best matches pairs according this threshold
    const int thresholdKeypointsMatchesCount = 20;

    // Loop all matchpoints
    // Find to which bounding box pair (ids of both image bounding boxes) belongs
    for (auto it = matches.begin(); it != matches.end(); ++it)
    {
        int kptIdxPrevImg = (*it).queryIdx;
        int kptIdxCurrImg = (*it).trainIdx;
        cv::Point2f prevImgKpt;
        cv::Point2f currImgKpt;
        try
        {
            prevImgKpt = prevFrame.keypoints.at(kptIdxPrevImg).pt;
            currImgKpt = currFrame.keypoints.at(kptIdxCurrImg).pt;
        }
        catch (const out_of_range &ex) // We should never land hare as the indices are taken from the keypoints vectors
        {
            cout << "out_of_range Exception Caught :: " << ex.what() << endl;
            continue;
        }

        int bbIdxPrev;

        // if the keypoint does not belong to any bounding box in previous image skip
        if (!pointFromBoundingBox(prevFrame.boundingBoxes, prevImgKpt, &bbIdxPrev))
        {
            continue;
        }

        int bbIdxCurr;
        // if the keypoint does not belong to any bounding box in current image skip
        if (!pointFromBoundingBox(currFrame.boundingBoxes, currImgKpt, &bbIdxCurr))
        {
            continue;
        }

        map<pair<int, int>, int>::iterator mapIt;
        pair<int, int> bbIdxPair = make_pair(bbIdxPrev, bbIdxCurr);

        mapIt = matchesCounter.find(bbIdxPair);
        if (mapIt == matchesCounter.end()) // key not present in map
        {
            // Add new matching bounding boxes pair
            matchesCounter.insert(make_pair(bbIdxPair, 1));
        }
        else // key present in map
        {
            // Count matching bounding boxes pair hits
            mapIt->second = mapIt->second + 1;
        }
    }


    multimap<int, pair<int, int>> matchesCountKeys;
    map<pair<int, int>, int>::iterator mapIt = matchesCounter.begin();

    // Swap key-value from the original map in order to sort by value
    // We can have same value(our counter) and this will lead to repeating key in the new map
    // thats why we need multimap
    while (mapIt != matchesCounter.end())
    {
        matchesCountKeys.insert(make_pair(mapIt->second, mapIt->first));
        mapIt++;
    }

    multimap<int, pair<int, int>>::iterator multimapIt = matchesCountKeys.begin();
    while (multimapIt != matchesCountKeys.end())
    {
        // Assign only the best matches according a threshold
        if (multimapIt->first > thresholdKeypointsMatchesCount)
        {
            bbBestMatches.insert(multimapIt->second);
        }
        multimapIt++;
    }
}










bool pointFromBoundingBox(std::vector<BoundingBox> &boundingBoxes, cv::Point2f kpt, int *bbIdx)
{
    float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI

    *bbIdx = -1;
    vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current match point
    for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
    {
        // shrink current bounding box slightly to avoid having too many outlier points around the edges
        cv::Rect smallerBox;
        smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
        smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
        smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
        smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

        // check wether point is within current bounding box
        if (smallerBox.contains(kpt))
        {
            enclosingBoxes.push_back(it2);
        }

    } // eof loop over all bounding boxes

    // check wether point has been enclosed by one or by multiple boxes
    if (enclosingBoxes.size() == 1)
    {
        *bbIdx = enclosingBoxes[0]->boxID;
        return true;
    }
    else
    {
        return false;
    }


}

