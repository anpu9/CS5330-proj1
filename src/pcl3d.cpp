//
// Created by Yuyang Tian on 2025/1/21.
//
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <thread>
#include <chrono>

using namespace cv;
using namespace std;

// Function to generate point cloud
void generatePointCloud(const cv::Mat& rgbImage, const cv::Mat& depthImage,
                        const cv::Mat& cameraMatrix, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    // Get camera intrinsic parameters
    float fx = cameraMatrix.at<double>(0, 0);
    float fy = cameraMatrix.at<double>(1, 1);
    float cx = cameraMatrix.at<double>(0, 2);
    float cy = cameraMatrix.at<double>(1, 2);
    //
    for (int v = 0; v < depthImage.rows; ++v) {
        for (int u = 0; u < depthImage.cols; ++u) {
            float depth = depthImage.at<float>(v, u);  // Assuming depth is in meters
            if (depth <= 0) continue;  // Skip invalid depth

            // Back-project to 3D
            float x = (u - cx) * depth / fx;
            float y = (v - cy) * depth / fy;
            float z = depth;

            // Get the color from the RGB image
            cv::Vec3b color = rgbImage.at<cv::Vec3b>(v, u);

            // Create a point
            pcl::PointXYZRGB point;
            point.x = x;
            point.y = y;
            point.z = z;
            point.r = color[2]; // OpenCV uses BGR format
            point.g = color[1];
            point.b = color[0];

            // Add to the point cloud
            cloud->points.push_back(point);
        }
    }
    // Set cloud metadata
    cloud->width = cloud->points.size(); // Point cloud width
    cloud->height = 1;                   // Unorganized point cloud
    cloud->is_dense = false;             // Set to false if there are invalid points
}
int visualizePointCloud(const string& inputFilename) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(inputFilename, *cloud) == -1) {
        PCL_ERROR("Error: Couldn't read the PCD file %s\n", inputFilename.c_str());
        return -1;
    }

    // Normalize and fix color values for each point
    for (auto& point : cloud->points) {
        // Unpack the RGB values
        uint32_t rgb = *reinterpret_cast<int*>(&point.rgb);
        uint8_t r = (rgb >> 16) & 0x0000ff;
        uint8_t g = (rgb >> 8)  & 0x0000ff;
        uint8_t b = (rgb)       & 0x0000ff;

        // Repack properly
        uint32_t color = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
        point.rgb = *reinterpret_cast<float*>(&color);
    }

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);

    // Add the point cloud with RGB color handling
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "sample cloud");

    // Increase point size for better visibility
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");

    // Add coordinate system with smaller size
    viewer->addCoordinateSystem(0.1); // Reduced size to 0.1 from 1.0

    viewer->initCameraParameters();

    // Set a better initial viewpoint
    viewer->setCameraPosition(-1, -1, -1,    // camera position
                              0,  0,  0,    // viewpoint
                              0,  0,  1);   // up vector

    // Enable point picking (optional, for debugging)
    viewer->registerPointPickingCallback([](const pcl::visualization::PointPickingEvent& event) {
        float x, y, z;
        event.getPoint(x, y, z);
        std::cout << "Picked point: " << x << " " << y << " " << z << std::endl;
    });

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}