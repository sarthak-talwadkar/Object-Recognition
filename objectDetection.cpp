/*
    Sarthak Uday Talwadkar
    Date - February 13, 2025
    Real time Object Detection
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <functions.h>

#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{

    VideoCapture *capture;

    // const string videoStreamAddress = "http://192.168.1.111:2333";
    capture = new VideoCapture(0);
    if (!capture->isOpened())
    {
        cerr << "Unable to open video device !!" << endl;
        return -1;
    }

    /*
        Path to DB file
    */
    string filename = "training_data.csv";
    loadTrainingData(filename);

    Mat frame, processedFrame, eroded, dilated;

    while (true)
    {
        *capture >> frame;
        if (frame.empty())
        {
            cerr << "Error opening frame !!" << endl;
            break;
        }

        imshow("Frame ", frame);

        /*
            Converting th RGB image to grayscale and cleaning the noise
        */
        preProcessImage(frame, processedFrame);

        /*
            Converting the grayscale image to binary image
        */
        applyThreshold(processedFrame, processedFrame);

        imshow("After Thresholding", processedFrame);

        /*
            Defining the filter for morphological operations
        */
        int filterSize = 3;
        Mat filter = Mat::ones(filterSize, filterSize, CV_8UC1);
        Mat crossFilter = getStructuringElement(MORPH_CROSS, Size(5, 5));

        // erode(processedFrame, eroded, filter, Point(-1, -1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());
        // dilate(eroded, dilated, filter, Point(-1, -1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());
        // morphologyEx(processedFrame, dilated, MORPH_OPEN, filter, Point(-1, -1), 1, BORDER_CONSTANT, morphologyDefaultBorderValue());
        // morphologyEx(processedFrame, eroded, MORPH_TOPHAT, filter);
        // morphologyEx(processedFrame, eroded, MORPH_GRADIENT, crossFilter);

        /*
            Applying Erosion and Dilation to the Binary Image
        */
        erosion(processedFrame, eroded, filter);
        dilation(eroded, dilated, filter);

        imshow("After Opening ", dilated);

        /*
            Defing the Mat for features
        */
        Mat labels, stats, centroids;

        /*
            Using OpenCV connectedComponentsWithStats for features
        */
        int numLabels = connectedComponentsWithStats(dilated, labels, stats, centroids);

        /*
            Ignoring the Regions with area less than 1000 pixels and the regions which are touching the boundaries
            Taking only 15 regions to compute features and classify to reduce computational cost
        */
        int minArea = 1000;
        int maxRegions = 15;

        vector<int> validLabels;
        for (int i = 1; i < numLabels; i++)
        {
            int area = stats.at<int>(i, CC_STAT_AREA);
            int x = stats.at<int>(i, CC_STAT_LEFT);
            int y = stats.at<int>(i, CC_STAT_TOP);
            int width = stats.at<int>(i, CC_STAT_WIDTH);
            int height = stats.at<int>(i, CC_STAT_HEIGHT);

            bool touchesBoundary = (x <= 0 || y <= 0 || (x + width) >= dilated.cols - 1 || (y + height) > dilated.rows - 1);
            if (area >= minArea && !touchesBoundary)
            {
                validLabels.push_back(i);
            }
        }

        /*
            Sorting the labels from highest to lowest as per the area
        */
        sort(validLabels.begin(), validLabels.end(), [&](int a, int b)
             { return stats.at<int>(a, CC_STAT_AREA) > stats.at<int>(b, CC_STAT_AREA); });

        if (validLabels.size() > maxRegions)
            validLabels.resize(maxRegions);

        /*
            Giving color to identify each identified regions
        */
        RNG rng(getTickCount());
        map<int, Vec3b> colorMap;

        for (int label : validLabels)
        {
            colorMap[label] = Vec3b(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        }

        Mat regionMap = Mat::zeros(dilated.size(), CV_8UC3);
        for (int i = 1; i < labels.rows; i++)
        {
            for (int j = 0; j < labels.cols; j++)
            {
                int label = labels.at<int>(i, j);
                if (colorMap.count(label))
                    regionMap.at<Vec3b>(i, j) = colorMap[label];
            }
        }
        imshow("Region Map", regionMap);

        /*
            Computing the region features for all valid regions - Percentage of bounding box occupied,
            Aspect Ratio for scale invariant
            and eccentricity, Orientation and centroid
        */
        for (int id : validLabels)
        {
            /*
                Compute the features for regions in current frame
            */
            regionFeatures f = computeFeatures(labels, id);
            /*
                Draws the bounding box for each region as per ther orientation
            */
            drawRegionFeatures(frame, f);

            vector<float> features = {
                f.percentage,
                f.aspectRatio,
                f.eccentricity};

            string label;
            /*
                If kmean is passed as an argument then it will classify the object using kmean
            */
            if (argv[1] == "kmean")
                label = knnClassify(features, 3);
            else
            {
                /*
                Uses NN with Euclidean metrics to clssify and identify the regions
                */
                string label = classify(features);
            }

            putText(frame, label, Point(f.centroid.x, f.centroid.y), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        }

        imshow("Final Frame with label", frame);

        char key = waitKey(10);
        if (key == 'q' || key == 'Q')
            break;
        else if (key == 'n' || key == 'N')
        {
            /*
                Training mode
                It will prompt the user to give a label to the largest object in the image
                It will store percentage, aspect ratio and eccentricity in training_data.csv file
                If there are no valid region or features then it will give an error "No valid regions to save"
            */
            if (!validLabels.empty())
            {
                int regionId = validLabels[0];

                regionFeatures f = computeFeatures(labels, regionId);

                cout << "Enter Object Label : ";
                string label;
                cin >> label;

                ofstream file("training_data.csv", ios::app);
                if (file)
                {
                    if (file.tellp() == 0)
                    {
                        file << "Label,Percentage,AspectRatio,Eccentricity" << endl;
                    }

                    file << label << "," << f.percentage << "," << f.aspectRatio << "," << f.eccentricity << endl;
                    cout << "Saved Features for : " << label << endl;
                }
                else
                {
                    cerr << "Error to open the file !!" << endl;
                }
            }
            else
            {
                cout << "No valid regions to save" << endl;
            }
        }
    }

    /*
        Relieving the Memory
    */
    delete capture;
    destroyAllWindows();
    return 0;
}
