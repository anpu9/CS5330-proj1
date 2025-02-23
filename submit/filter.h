//
// Created by Yuyang Tian on 2025/1/19.
// Header for image filter
//

#ifndef PROJECT1_FILTER_H
#define PROJECT1_FILTER_H
#include <opencv2/opencv.hpp>

// Declare the myGrayscale function
int greyscale(const cv::Mat& src, cv::Mat& dst);
int sepiaTone(const cv::Mat& src, cv::Mat& dst);
int vignetting(const cv::Mat& src, cv::Mat& dst);
int blur5x5_1( cv::Mat &src, cv::Mat &dst);
int blur5x5_2( cv::Mat &src, cv::Mat &dst);
int sobelX3x3( cv::Mat &src, cv::Mat &dst);
int sobelY3x3( cv::Mat &src, cv::Mat &dst);
int magnitude( cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);
int blurQuantize( cv::Mat &src, cv::Mat &dst, int levels = 10);
int getDepthMap(cv::Mat &src, cv::Mat &dst);
int applyFog( cv::Mat &src, cv::Mat &depth, cv::Mat &dst);
int backgroundBlur( cv::Mat &src, std::vector<cv::Rect> &faces, cv::Mat &dst);
int backgroundGrey( cv::Mat &src, std::vector<cv::Rect> &faces, cv::Mat &dst);


#endif //PROJECT1_FILTER_H
