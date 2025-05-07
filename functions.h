/*
    Sarthak Uday Talwadkar
    Date - February 13, 2025
*/

#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <numeric>
#include <fstream>
#include <cmath>
#include <sstream>

using namespace cv;
using namespace std;

struct regionFeatures
{
    float percentage;
    float aspectRatio;
    float eccentricity;
    Point2f centroid;
    RotatedRect orientedBox;
    float orientation;
};

struct trainingData
{
    string label;
    vector<float> features;
};

extern vector<trainingData> trainingDb;
extern vector<float> featureMeans;
extern vector<float> stdDeviation;

int preProcessImage(const Mat &src, Mat &dst);
int applyThreshold(const Mat &srx, Mat &dst);
int calculateThreshold(const Mat &src);
int erosion(const Mat &src, Mat &dst, Mat &filter);
int dilation(const Mat &src, Mat &dst, Mat &filter);
regionFeatures computeFeatures(const Mat &regionMap, int regionID);
void drawRegionFeatures(Mat &dst, const regionFeatures &features);
void loadTrainingData(string &filename);
float scalarEuclidean(const vector<float> &a, const vector<float> &b);
string classify(const vector<float> &features);
string knnClassify(const vector<float> &features, int K);
