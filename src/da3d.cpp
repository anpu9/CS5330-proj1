//
// Created by Yuyang Tian on 2025/1/21.
//
#include <cstdio>
#include <opencv2/opencv.hpp>
#include "../include/DA2Network.hpp"
#include "../include/pcl3d.h"

using namespace std;
using namespace cv;
int main() {

    std::string outputDir = "../outputs";
    std::string outputFilename = outputDir + "/output.pcd";

    visualizePointCloud(outputFilename);
    std::cout << "Termating";
    return 0;
}