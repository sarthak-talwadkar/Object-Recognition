/*
    Sarthak Uday Talwadkar
    Date - February 13, 2025
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <fstream>
#include <cmath>
#include <sstream>

using namespace cv;
using namespace std;

/*
    Gobal struct to store features of valid regions
*/
struct regionFeatures
{
    float percentage;
    float aspectRatio;
    float eccentricity;
    Point2f centroid;
    RotatedRect orientedBox;
    float orientation;
};

/*
    Global stuct to test the features of TrainingDB to the currently identified object
*/
struct trainingData
{
    string label;
    vector<float> features;
};

/*
    Global vectors to to store the training database called at start and updated in real time
*/
vector<trainingData> trainingDb;
vector<float> featureMeans;
vector<float> stdDeviation;

/*
    Convert the RGB image to Grayscale and applyies Blur to remove the noise
*/
int preProcessImage(const Mat &src, Mat &dst)
{

    // Grayscale
    Mat gray;
    dst.create(src.size(), CV_8UC1);

    cvtColor(src, gray, COLOR_BGR2GRAY);

    // Blurring
    Mat blur;
    GaussianBlur(gray, dst, Size(5, 5), 0);

    return 0;
}

/*
    Function calculates an optimal threshold using Otsu's method
    Input is Grayscale Image and output is int threshold
    It calculates the threshold by analyzing the histogram of grayscale image for
    separating the foreground and background pixel based on inter-class variance
*/
int calculateThreshold(const Mat &src)
{
    int threshold;

    // Creating a grayscale Histogram
    int hist[256] = {0};
    int sum;
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            hist[src.at<uchar>(y, x)]++;
        }
    }
    int totalPixel = src.rows * src.cols;
    for (int i = 0; i < 256; i++)
    {
        sum += hist[i];
    }

    // Finding the Inter-class variance by dividing the pixel into foreground and background
    int pB, pF;
    double sumB;
    double maxVariance = 0;

    for (int i = 0; i < 256; i++)
    {
        pB += hist[i];
        if (pB == 0)
            continue;

        pF = totalPixel - pB;
        if (pF == 0)
            break;

        sumB += hist[i];

        double mB = sumB / sum;
        double mF = (sum - sumB) / pF;

        double variance = pB * pF * (mB - mF) * (mB - mF);

        if (variance > maxVariance)
        {
            maxVariance = variance;
            threshold = i;
        }
    }

    return threshold;
}

/*
    Function to compute an optimal threshold using K-Means clustering
    Input is Grayscale Image and output is double threshold value
    It segments pixel intensities into two clusters(foreground and background) using an iterative approach
    returns the mean of the two cluster centers as the threshold.
*/
double kMeanThreshold(const Mat &src, double sampleFraction = 0.1)
{
    vector<uchar> pixel;
    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            pixel.push_back(src.at<uchar>(y, x));
        }
    }
    //  randomly sampling the pixel from the image
    random_device rd;
    mt19937 mt{rd()};
    shuffle(pixel.begin(), pixel.end(), mt);

    int sampleSize = pixel.size() * sampleFraction;
    vector<uchar> samplePixels(pixel.begin(), pixel.begin() + sampleSize);

    uchar center1 = samplePixels[0];
    uchar center2 = samplePixels[1];

    vector<uchar> cluster1, cluster2;
    for (int i = 0; i < 10; i++)
    {
        cluster1.clear();
        cluster2.clear();

        for (uchar pixel : samplePixels)
        {
            if (abs(pixel - center1) < abs(pixel - center2))
                cluster1.push_back(pixel);
            else
                cluster2.push_back(pixel);
        }

        if (!cluster1.empty())
            center1 = accumulate(cluster1.begin(), cluster1.end(), 0) / cluster1.size();

        if (!cluster2.empty())
            center2 = accumulate(cluster2.begin(), cluster2.end(), 0) / cluster2.size();
    }

    return (center1 + center2) / 2.0;
}
/*
    Function to apply a threshold to an image
    Inputs is the preprocess image and output is Binary image of same size
    2 methods of thresholding are used (Otsu's and K-means)
*/
int applyThreshold(const Mat &src, Mat &dst)
{
    dst.create(src.rows, src.cols, CV_8UC1);

    // threshold = calculateThreshold(src);
    double threshold = kMeanThreshold(src);

    for (int y = 0; y < src.rows; y++)
    {
        for (int x = 0; x < src.cols; x++)
        {
            if (src.at<uchar>(y, x) < threshold)
            {
                dst.at<uchar>(y, x) = 255;
            }
            else
            {
                dst.at<uchar>(y, x) = 0;
            }
        }
    }
    return 0;
}

/*
    Function to perform morphological erosion on an image
    Input is binary image and filter, Ouput is eroded image
    It removes Noise and shrinks the white region by applying the filter
*/
int erosion(const Mat &src, Mat &dst, Mat &filter)
{

    dst.create(src.size(), CV_8UC1);

    int radius = filter.rows / 2;

    // Exclude the border of the image
    for (int y = radius; y < src.rows - radius; y++)
    {
        for (int x = radius; x < src.cols - radius; x++)
        {
            bool allWhite = true;
            for (int i = -radius; i <= radius; i++)
            {
                for (int j = -radius; j <= radius; j++)
                {
                    if (filter.at<uchar>(i + radius, j + radius) &&
                        src.at<uchar>(y + i, x + j) == 0)
                    {
                        allWhite = false;
                        break;
                    }
                }
                if (!allWhite)
                    break;
            }
            dst.at<uchar>(y, x) = allWhite ? 255 : 0;
        }
    }

    return 0;
}
/*
    Function to perform morphological dilation on an image
    Input is eroded binary image and filter,  Ouput is dilated image
    It bring back the object/element to its size while filling holes by applying the filter
*/
int dilation(const Mat &src, Mat &dst, Mat &filter)
{
    dst.create(src.size(), CV_8UC1);

    int radius = filter.rows / 2;

    // Exclude the borders of the image
    for (int y = radius; y < src.rows - radius; y++)
    {
        for (int x = radius; x < src.cols - radius; x++)
        {
            bool anyWhite = false;
            for (int i = -radius; i < radius; i++)
            {
                for (int j = -radius; j < radius; j++)
                {
                    if (filter.at<uchar>(i + radius, j + radius) && src.at<uchar>(y + i, x + j) == 255)
                    {
                        anyWhite = true;
                        break;
                    }
                }
                if (anyWhite == true)
                    break;
            }
            dst.at<uchar>(y, x) = anyWhite ? 255 : 0;
        }
    }
    return 0;
}

/*
    Function to compute various features of a specific region in an image
    Input is the Image matrix for each regions and valid regions Id
    It calculates centroid, orientation, eccentricity, bounding box,
    percentage area, and aspect ratio for a given region ID in the region map.
    It return struct of regionFeatures
*/
regionFeatures computeFeatures(const Mat &regionMap, int regionID)
{
    regionFeatures features;
    Moments m = moments(regionMap == regionID, true);
    if (m.m00 <= 0.001)
        return features; // Avoid division by zero

    Mat mask;
    compare(regionMap, regionID, mask, CMP_EQ);
    if (countNonZero(mask) == 0)
    {
        cerr << "Empty region: " << regionID << endl;
        return features; // Return default features
    }

    /*
        Compute centroid of the region
    */
    features.centroid = Point2f(m.m10 / m.m00, m.m01 / m.m00);

    /*
        Compute second-order central moments
    */
    float mu20 = m.mu20 / m.m00;
    float mu02 = m.mu02 / m.m00;
    float mu11 = m.mu11 / m.m00;

    /*
        Compute orientation of the region in degrees
    */
    features.orientation = 0.5 * atan2(2 * mu11, mu20 - mu02) * 180 / CV_PI;

    /*
        Compute eigenvalues (lambda1 and lambda2) to determine eccentricity
    */
    float lambda1 = (mu20 + mu02) / 2 + sqrt(4 * mu11 * mu11 + (mu20 - mu02) * (mu20 - mu02)) / 2;
    float lambda2 = (mu20 + mu02) / 2 - sqrt(4 * mu11 * mu11 + (mu20 - mu02) * (mu20 - mu02)) / 2;
    features.eccentricity = sqrt(1 - lambda2 / lambda1);

    /*
        Compute the minimum area bounding box for the region
    */
    vector<Point> points;
    findNonZero(mask, points);
    features.orientedBox = minAreaRect(points);

    /*
        Compute percentage of region area within the bounding box
    */
    float regionArea = m.m00;
    float boxArea = features.orientedBox.size.area();
    features.percentage = regionArea / boxArea;

    /*
        Compute aspect ratio of the bounding box
    */
    Size2f size = features.orientedBox.size;
    features.aspectRatio = size.height > size.width ? size.height / size.width : size.width / size.height;

    return features;
}

/*
    Function to visualize the computed region features on an image
    Input are the Output Image and Region features and provide output is image
    It draws a bounding box around the detected region and an arrow indicating orientation.
*/
void drawRegionFeatures(Mat &dst, const regionFeatures &features)
{
    if (features.orientedBox.size.area() <= 10)
        return;

    Point2f vertices[4];
    features.orientedBox.points(vertices);

    for (int i = 0; i < 4; i++)
    {
        line(dst, vertices[i], vertices[(i + 1) % 4], Scalar(0, 255, 0), 2);
    }
    float lenght = 50;
    Point2f endPoint(
        features.centroid.x + lenght * cos(features.orientation * CV_PI / 180),
        features.centroid.y + lenght * sin(features.orientation * CV_PI / 180));

    arrowedLine(dst, features.centroid, endPoint, Scalar(0, 255, 0), 2);
}

/*
    Function reads feature vectors and labels from a CSV file,
    stores them in  vector `trainingDb`, and computes the mean and standard deviation
    for feature normalization.
    Input is the file path to db file
*/
void loadTrainingData(string &filename)
{
    trainingDb.clear();
    ifstream file(filename);
    string line;

    getline(file, line);

    vector<vector<float>> allFeatures;
    while (getline(file, line))
    {
        stringstream ss(line);
        trainingData data;
        string token;

        getline(ss, token, ',');
        data.label = token;

        while (getline(ss, token, ','))
        {
            data.features.push_back(stof(token));
        }

        trainingDb.push_back(data);
        allFeatures.push_back(data.features);
    }

    if (!allFeatures.empty())
    {
        int numFeatures = allFeatures[0].size();
        featureMeans.resize(numFeatures, 0.0f);
        stdDeviation.resize(numFeatures, 0.0f);

        for (const auto &vec : allFeatures)
        {
            for (int i = 0; i < numFeatures; i++)
            {
                featureMeans[i] += vec[i];
            }
        }
        for (auto &mean : featureMeans)
        {
            mean /= trainingDb.size();
        }

        for (const auto &vec : allFeatures)
        {
            for (int i = 0; i < numFeatures; i++)
            {
                stdDeviation[i] += (vec[i] - featureMeans[i]) * (vec[i] - featureMeans[i]);
            }
        }
        for (auto &stdDev : stdDeviation)
        {
            stdDev = sqrt(stdDev / trainingDb.size());
        }
    }
}

/*
    Computes the normalized Euclidean distance between two feature vectors.
    This function applies feature normalization using precomputed means
    and standard deviations before computing the distance.
*/
float scalarEuclidean(const vector<float> &a, const vector<float> &b)
{
    float dist = 0;
    for (int i = 0; i < a.size(); i++)
    {
        float diff = (a[i] - featureMeans[i]) / stdDeviation[i] - (b[i] - featureMeans[i]) / stdDeviation[i];
        dist += diff * diff;
    }

    return sqrt(dist);
}

/*
    Classifies a feature vector using the nearest neighbor approach.
    Function compares the input feature vector against stored training data
    using the normalized Euclidean distance and returns the label of the closest match.
    If no close match is found, it returns "Unknown".
*/
string classify(const vector<float> &features)
{
    if (trainingDb.empty())
    {
        return "Unknown";
    }

    float minDist = FLT_MAX;
    string label = "Unknown";

    for (const auto &entry : trainingDb)
    {
        float dist = scalarEuclidean(features, entry.features);

        if (dist < minDist)
        {
            minDist = dist;
            label = entry.label;
        }
    }
    if (minDist < 0.1)
        return "Unknown";

    return label;
}

/*
    Classifies a feature vector using the K nearest neighbor approach  where k = 3.
    This function computes the Euclidean distance between the input feature vector
    and all training samples, selects the K nearest neighbors, and performs
    weighted voting to determine the most likely class label.
    Inputs are Feature vector and k number and output is predicted label of based on kNN algorithm
*/
string knnClassify(const vector<float> &features, int K)
{
    if (trainingDb.empty())
        return "Unknown";

    vector<pair<float, string>> distances;
    for (const auto &entry : trainingDb)
    {
        float dist = scalarEuclidean(features, entry.features);
        distances.push_back({dist, entry.label});
    }

    sort(distances.begin(), distances.end());

    map<string, float> labelWeights;
    for (int i = 0; i < K && i < distances.size(); i++)
    {
        float weight = 1.0 / (distances[i].first + 1e-6);
        labelWeights[distances[i].second] += weight;
    }

    string label = "Unknown";
    float maxWeight = 0.5;
    for (const auto &pair : labelWeights)
    {
        if (pair.second > maxWeight)
        {
            maxWeight = pair.second;
            label = pair.first;
        }
    }

    return label;
}