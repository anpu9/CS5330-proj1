/*
  Bruce A. Maxwell
  January 2025

  Modified by Yuyang Tian. 21 Jan
  User type 'c' to generate and display 3D point cloud of a image

*/

#include <cstdio>
#include <opencv2/opencv.hpp>
#include "../include/DA2Network.hpp"
#include "../include/pcl3d.h"


// opens a video stream and runs it through the depth anything network
// displays both the original video stream and the depth stream
int main(int argc, char *argv[]) {
  cv::VideoCapture *capdev;
  cv::Mat src; 
  cv::Mat dst;
  cv::Mat pclFrame;
  cv::Mat dst_vis;
  char filename[256]; // a string for the filename
  const float reduction = 0.5;
  const char* NETWORK_PATH = "../include/model_fp16.onnx";
  std::string outputDir = "../outputs";
  std::string outputFilename = outputDir + "/output.pcd";

  // make a DANetwork object
  DA2Network da_net(NETWORK_PATH);

  // open the video device
  capdev = new cv::VideoCapture(1);
  if( !capdev->isOpened() ) {
    printf("Unable to open video device\n");
    return(-1);
  }

  cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
		 (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));

  printf("Expected size: %d %d\n", refS.width, refS.height);

  float scale_factor = 256.0 / (refS.height*reduction);
  printf("Using scale factor %.2f\n", scale_factor);

  cv::namedWindow( "Video", 1 );
  cv::namedWindow( "Depth", 2 );

  for(;;) {
    // capture the next frame
    *capdev >> src;
    if( src.empty()) {
      printf("frame is empty\n");
      break;
    }
    // for speed purposes, reduce the size of the input frame by half
    cv::resize( src, src, cv::Size(), reduction, reduction );

    // set the network input
    da_net.set_input( src, scale_factor );

    // run the network
    da_net.run_network( dst, src.size());
    da_net.run_network_pcl( pclFrame, src.size());

    // apply a color map to the depth output to get a good visualization
    cv::applyColorMap(dst, dst_vis, cv::COLORMAP_INFERNO );

    // if you want to modify the src image based on the depth image, do that here
    /*
    for(int i=0;i<src.rows;i++) {
      for(int j=0;j<src.cols;j++) {
	if( dst.at<unsigned char>(i, j) < 128 ) {
	  src.at<cv::Vec3b>(i,j) = cv::Vec3b( 128, 100, 140 );
	}
      }
    }
    */

    // display the images
    cv::imshow("video", src);
    cv::imshow("depth", dst_vis);

    // terminate if the user types 'q'
    // Showing point cloud if the user types 'c'
    int key = cv::waitKey(10);
    if(key == 'q' ) {
      break;
    } else if (key == 'c') {
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 525.0, 0.0, 319.5,
                0.0, 525.0, 239.5,
                0.0, 0.0, 1.0);
//        Create a PCL point cloud object
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        generatePointCloud(src, pclFrame, cameraMatrix, cloud);
        // Save the point cloud to a PCD file
        if (pcl::io::savePCDFileASCII(outputFilename, *cloud) == 0) {
            std::cout << "Saved " << cloud->points.size() << " points to " << outputFilename << std::endl;
        } else {
            std::cerr << "Error: Could not save PCD file." << std::endl;
            return -1;
        }
        visualizePointCloud(outputFilename);
        break;
    }
  }

  printf("Terminating\n");

  return(0);
}

