
#include <iostream>
#include <algorithm>
#include <numeric>
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
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0);
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

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

/*
 * The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size.
 * However, you can make this function work for other sizes too.
 * For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
 */
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f((left - 250) / 4, (bottom + 50) / 4), cv::FONT_ITALIC, 0.5, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f((left - 250) / 4, (bottom + 125) / 4), cv::FONT_ITALIC, 0.5, currColor);
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
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    //string name =  "TTClidar.jpg";  

    //cv::imwrite("../result/"+name, topviewImg);

    if (bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
 
    vector<cv::DMatch> matchesInBX;
    vector<double> Distance;

    for (auto matche : kptMatches)
    {

        // get the keypoints  pair  ( query form prefious frame , train from current frame)

        cv::KeyPoint kPrev = kptsPrev[matche.queryIdx];
        cv::KeyPoint kCurr = kptsCurr[matche.trainIdx];

        if (boundingBox.roi.contains(kPrev.pt) && boundingBox.roi.contains(kCurr.pt))
        {

            matchesInBX.push_back(matche);
            Distance.push_back(cv::norm(kCurr.pt - kPrev.pt));
        }
    }
    double meanDist = std::accumulate(Distance.begin(), Distance.end(), 0.0) / Distance.size();



    double sq_sum = std::inner_product(
        Distance.begin(), Distance.end(), Distance.begin(), 0.0,
        [](double const &x, double const &y)
        { return x + y; },
        [meanDist](double const &x, double const &y)
        { return (x - meanDist) * (y - meanDist); });

    double standard_deviation = std::sqrt(sq_sum / Distance.size());

    //cout << "meanDist = " << meanDist << " ,standard_deviation = " << standard_deviation << endl;

    double distRatio = 1.5;
    for (int i = 0; i < matchesInBX.size(); i++)
    {

         if  (Distance.at(i) < (distRatio * meanDist) )
        //if ((Distance.at(i) < (meanDist + 1* standard_deviation)))
        {

            boundingBox.kptMatches.push_back(matchesInBX.at(i));

            boundingBox.keypoints.push_back(kptsCurr[matchesInBX.at(i).trainIdx]);
        }
    }
    // cout << "boundingBox.kptMatches = " << boundingBox.kptMatches.size() << endl;
    // cout << "boundingBox.keypoints = " << boundingBox.keypoints.size() << endl;
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    // cout << "distRatios.size() = " << distRatios.size() << endl;
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1. / frameRate;
    TTC = -dT / (1 - medDistRatio);
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{

    double laneWidth = 4.0; // assumed width of the ego lane

    // Avg distance distance to Lidar points within ego lane
    double distXPrev = 1e9, distXCurr = 1e9;

    double avgPrev = 0;
    int pointCountPrev = 0;
    double avgCurr = 0;
    int pointCountCurr = 0;
    
    // 3D point within ego lane laneWidth = 4 from previous frame
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {
        if (abs(it->y) <= laneWidth / 2.0)
        { // 3D point within ego lane?

            
            avgPrev += it->x;
            pointCountPrev += 1;
            
        }
    }
    // 3D point within ego lane laneWidth = 4 from current frame
    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {

        if (abs(it->y) <= laneWidth / 2.0)

        {
            avgCurr += it->x;
            pointCountCurr += 1;
           
        }
    }

    // calulate distance using avrage distance to reduce effect of outlier

    distXPrev = avgPrev / pointCountPrev;
    distXCurr = avgCurr / pointCountCurr;

    //cout<< "distXPrev = " << distXPrev<< " , distXCurr = "  << distXCurr << " , distdiffr = " <<distXPrev - distXCurr <<endl;

    TTC = (distXCurr * (1 / frameRate)) / (distXPrev - distXCurr);
}

// takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property)“.
// Matches must be the ones with the highest number of keypoint correspondences.
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    //
    int prevBoxNumbers = prevFrame.boundingBoxes.size();
    int currBoxNumbers = currFrame.boundingBoxes.size();

    // creat an array that stores keypoint counts in each pair of bounding boxs
    int matches_counts[prevBoxNumbers][currBoxNumbers] = {};

    for (auto matche : matches)
    {

        // get the keypoints  pair  ( query form prefious frame , train from current frame)

        cv::Point query = prevFrame.keypoints[matche.queryIdx].pt;
        cv::Point train = currFrame.keypoints[matche.trainIdx].pt;

        bool query_found = false;
        bool train_found = false;
        int query_box_id = 0;
        int train_box_id = 0;

        // to discard any keypoint that is contained in more than one bounding box
        int number_boxs_query_in = 0;
        int number_boxs_train_in = 0;

        for (int i = 0; i < prevBoxNumbers; i++)
        {
            if (prevFrame.boundingBoxes[i].roi.contains(query))
            {
                query_found = true;
                query_box_id = i;
                number_boxs_query_in += 1;
                if (number_boxs_query_in > 1)
                {
                    // discard point if it contained in more than one bounding box
                    break;
                }
            }
        }

        for (int i = 0; i < currBoxNumbers; i++)
        {
            if (currFrame.boundingBoxes[i].roi.contains(train))
            {
                train_found = true;
                train_box_id = i;
                number_boxs_train_in += 1;

                if (number_boxs_query_in > 1)
                { // discard point if it contained in more than one bounding box
                    break;
                }
            }
        }

         // increase the count of matched keypoints in a specific pair of bounding boxes
        if ((query_found && train_found) && (number_boxs_query_in == 1) && (number_boxs_train_in == 1))
        {
        
            matches_counts[query_box_id][train_box_id] += 1;
        }
    }

    // for each bounding box in the previous frame find the corresponding matched bounding box in the current frame that contains the biggest number of matched keypoints
    for (int i = 0; i < prevBoxNumbers; i++)
    {

        int max_count = 0;
        int max_id = -1;
        for (int j = 0; j < currBoxNumbers; j++)
        {
            if (matches_counts[i][j] > max_count)
            {
                max_count = matches_counts[i][j];
                max_id = j;
            }
        }

        // std::cout << "matched pair of bounding boxs = (" << i << ", "<<max_id << ") with point counts = " << max_count<<std::endl;

        // store pair (previous bounding box ID , Current matched Bounding box ID )
        bbBestMatches[i] = max_id;
    }
}
