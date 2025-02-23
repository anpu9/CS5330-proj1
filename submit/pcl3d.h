//
// Created by Yuyang Tian on 2025/1/21.
// To generate a PCD (point cloud data) for a image
//
#ifndef PROJECT1_PCL3D_H
#define PROJECT1_PCL3D_H
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
void generatePointCloud(const cv::Mat& rgbImage, const cv::Mat& depthImage,
                        const cv::Mat& cameraMatrix, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud);
int visualizePointCloud(const std::string& fileName);
#endif //PROJECT1_PCL3D_H
