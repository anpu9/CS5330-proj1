//
// Created by Yuyang Tian on 2025/1/19.
// For different filters implementation
#include <opencv2/core.hpp>
#include <cmath>
#include "../include/filter.h"
#include "../include/DA2Network.hpp"

using namespace cv;
using namespace std;

int greyscale(const Mat& src, Mat& dst) {
    // Check if the input image is not empty
    if (src.empty()) {
        cerr << "Input frame is empty!" << endl;
        return -1;
    }

    // Convert the input frame to grayscale
    dst.create(src.size(), CV_8UC1);
    // Accessing each pixels
    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            Vec3b bgrPixel = src.at<Vec3b>(y,x);
            uchar B = bgrPixel[0];
            uchar G = bgrPixel[1];
            uchar R = bgrPixel[2];
            float Y = 0.005 * R + 0.8 * G + 0.195 * B;
            dst.at<uchar>(y,x) = static_cast<uchar>(Y);
        }
    }
    return 1;
}
int sepiaTone(const Mat& src, Mat& dst) {
    // Check if the input image is not empty
    if (src.empty()) {
        cerr << "Input frame is empty!" << endl;
        return -1;
    }
    // Create an empty destination image of the same size as the source
    dst.create(src.size(), src.type());
    // Accessing each pixel in the image
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            // Extract BGR pixel value
            Vec3b bgrPixel = src.at<Vec3b>(y, x);
            // Original BGR values
            uchar B = bgrPixel[0];
            uchar G = bgrPixel[1];
            uchar R = bgrPixel[2];

            // Apply the sepia filter using the matrix coefficients
            uchar newB = static_cast<uchar>(min(0.272f * R + 0.534f * G + 0.131f * B, 255.0f));
            uchar newG = static_cast<uchar>(min(0.349f * R + 0.686f * G + 0.168f * B, 255.0f));
            uchar newR = static_cast<uchar>(min(0.393f * R + 0.769f * G + 0.189f * B, 255.0f));

            // Assign the new pixel values to the destination image
            dst.at<Vec3b>(y, x) = Vec3b(newB, newG, newR);
        }
    }
    return 1;
}
int vignetting(const Mat& src, Mat& dst) {
    dst = src.clone();
    int centerX = src.cols / 2;
    int centerY = src.rows / 2;
    float maxDistance = sqrt(pow(centerX, 2) + pow(centerY, 2));

    for (int y = 0; y < src.rows; y++) {
        for (int x = 0; x < src.cols; x++) {
            float dist = sqrt(pow(x - centerX, 2) + pow(y - centerY, 2));
            float factor = 1.0f - dist / maxDistance;
            factor = max(factor, 0.0f); // Ensure the factor stays between 0 and 1

            Vec3b pixel = src.at<Vec3b>(y, x);
            for (int c = 0; c < 3; ++c) {
                pixel[c] = saturate_cast<uchar>(pixel[c] * factor);
            }
            dst.at<Vec3b>(y, x) = pixel;
        }
    }
    return 0;
}
int blur5x5_1(Mat &src, Mat &dst) {
    // Check if the input image is not empty
    if (src.empty()) {
        cerr << "Input frame is empty!" << endl;
        return -1;
    }

    src.copyTo(dst);

    // The 5x5 Gaussian kernel (integer approximation)
    int kernel[5][5] = {
            {1, 2, 4, 2, 1},
            {2, 4, 8, 4, 2},
            {4, 8, 16, 8, 4},
            {2, 4, 8, 4, 2},
            {1, 2, 4, 2, 1}
    };
    int kernelSum = 100;
    // Accessing each pixel except border
    for (int i = 2; i < src.rows - 2; i++) {
        for (int j = 2; j < src.cols - 2; j++) {

            // process each color channal
            for (int c = 0; c < 3; c++) {
                int sum = 0;
                // Apply kernel
                for (int ki = -2; ki <= 2; ki++) {
                    for (int kj = -2; kj <= 2; kj++) {
                        sum += src.at<Vec3b>(i + ki,j + kj)[c] * kernel[ki + 2][kj + 2];
                    }
                }
                dst.at<Vec3b>(i, j)[c] = static_cast<uchar>(sum / kernelSum);
            }
        }
    }
    return 0;
}
int blur5x5_2(Mat &src, Mat &dst) {
    if (src.empty()) {
        return -1;
    }

    // Create temporary Mat for intermediate results
    Mat temp(src.size(), src.type());
    dst.create(src.size(), src.type());

    // 1D separable kernels
    const int kernel[5] = {1, 2, 4, 2, 1};
    const int kernelSum = 10; // Sum of kernel values

    // Horizontal pass using pointer arithmetic
    for (int i = 0; i < src.rows; i++) {
        // Get row pointers
        const cv::Vec3b* srcRow = src.ptr<cv::Vec3b>(i);
        cv::Vec3b* tempRow = temp.ptr<cv::Vec3b>(i);

        for (int j = 2; j < src.cols - 2; j++) {
            cv::Vec3i sum(0, 0, 0);

            // Apply horizontal kernel
            for (int k = -2; k <= 2; k++) {
                const cv::Vec3b& pixel = srcRow[j + k];
                for (int c = 0; c < 3; c++) {
                    sum[c] += pixel[c] * kernel[k + 2];
                }
            }

            // Store intermediate result
            for (int c = 0; c < 3; c++) {
                tempRow[j][c] = static_cast<uchar>(sum[c] / kernelSum);
            }
        }

        // Handle border cases by copying source pixels
        for (int j = 0; j < 2; j++) {
            tempRow[j] = srcRow[j];
            tempRow[src.cols - 1 - j] = srcRow[src.cols - 1 - j];
        }
    }

    // Vertical pass using pointer arithmetic
    for (int j = 0; j < src.cols; j++) {
        for (int i = 2; i < src.rows - 2; i++) {
            cv::Vec3i sum(0, 0, 0);

            // Apply vertical kernel
            for (int k = -2; k <= 2; k++) {
                const cv::Vec3b& pixel = temp.ptr<cv::Vec3b>(i + k)[j];
                for (int c = 0; c < 3; c++) {
                    sum[c] += pixel[c] * kernel[k + 2];
                }
            }

            // Store final result
            cv::Vec3b& dstPixel = dst.ptr<cv::Vec3b>(i)[j];
            for (int c = 0; c < 3; c++) {
                dstPixel[c] = static_cast<uchar>(sum[c] / kernelSum);
            }
        }
    }

    // Handle top and bottom borders
    temp.rowRange(0, 2).copyTo(dst.rowRange(0, 2));
    temp.rowRange(src.rows - 2, src.rows).copyTo(dst.rowRange(src.rows - 2, src.rows));

    return 0;
}
int sobelX3x3(Mat &src, Mat &dst) {
    if (src.empty()) {
        cerr << "Input frame is empty!" << endl;
        return -1;
    }

    // Create temporary and destination matrices
    Mat temp(src.rows, src.cols, CV_16SC3);
    dst.create(src.size(), CV_16SC3);

    // Separable kernels for X
    const int kernelV[3] = {1, 2, 1};     // Vertical (Gaussian smoothing)
    const int kernelH[3] = {1, 0, -1};    // Horizontal (Derivative)

    // Step 1: Vertical pass (Gaussian smoothing)
    for (int j = 0; j < src.cols; j++) {
        for (int i = 0; i < src.rows; i++) {
            for (int c = 0; c < 3; c++) {
                // Handle border conditions
                short up = (i == 0) ? src.ptr<Vec3b>(0)[j][c] : src.ptr<Vec3b>(i-1)[j][c];
                short down = (i == src.rows-1) ? src.ptr<Vec3b>(src.rows-1)[j][c] : src.ptr<Vec3b>(i+1)[j][c];

                // Apply vertical kernel
                short sum = up * kernelV[0] +
                            src.ptr<Vec3b>(i)[j][c] * kernelV[1] +
                            down * kernelV[2];

                temp.ptr<Vec3s>(i)[j][c] = sum / 4;  // Normalize by sum of kernel weights (1+2+1 = 4)
            }
        }
    }

    // Step 2: Horizontal pass (Derivative)
    for (int i = 0; i < src.rows; i++) {
        Vec3s* tempRow = temp.ptr<Vec3s>(i);
        Vec3s* dstRow = dst.ptr<Vec3s>(i);

        for (int j = 0; j < src.cols; j++) {
            for (int c = 0; c < 3; c++) {
                // Handle border conditions
                short left = (j == 0) ? tempRow[0][c] : tempRow[j-1][c];
                short right = (j == src.cols-1) ? tempRow[src.cols-1][c] : tempRow[j+1][c];

                // Apply horizontal kernel
                short sum = left * kernelH[0] +
                            tempRow[j][c] * kernelH[1] +
                            right * kernelH[2];

                dstRow[j][c] = sum / 2;  // Normalize by sum of absolute kernel values (1+1 = 2)
            }
        }
    }

    return 0;
}
int sobelY3x3(cv::Mat &src, Mat &dst) {
    if (src.empty()) {
        cerr << "Input frame is empty!" << endl;
        return -1;
    }
    // Create an empty destination image of the same size as the source
    Mat temp(src.rows, src.cols, CV_16SC3);
    dst.create(src.size(), CV_16SC3);

    // Separable kernels for Y
    const int kernelH[3] = {1, 2, 1};    // Horizontal Gaussian smoothing
    const int kernelV[3] = {1, 0, -1};     // Vertical derivative

    // Accesing each pixel
    // Horizontal
    for (int i = 0; i < src.rows; i++) {
        const Vec3b* srcRow = src.ptr<Vec3b>(i);
        Vec3s* tempRow = temp.ptr<Vec3s>(i);
        for (int j = 0; j < src.cols; j++) {
            // apply kenel
            for (int c = 0; c < 3; c++) {
                short left = (j == 0) ? srcRow[1][c] : srcRow[j-1][c];
                short right = (j == src.cols-1) ? srcRow[src.cols-2][c] : srcRow[j+1][c];

                short sum = left * kernelH[0] +
                            srcRow[j][c] * kernelH[1] +
                            right * kernelH[2];
                tempRow[j][c] = sum / 4;
            }
        }
    }
    // Vertical
    for (int j = 0; j < src.cols; j++) {
        for (int i = 0; i < src.rows; i++) {
            for (int c = 0; c < 3; c++) {
                short up = i == 0? temp.ptr<Vec3s>(1)[j][c] : temp.ptr<Vec3s>(i-1)[j][c];
                short down = i == src.rows-1? temp.ptr<Vec3s>(src.rows-2)[j][c] : temp.ptr<Vec3s>(i+1)[j][c];
                short sum = up * kernelV[0] +
                            temp.ptr<cv::Vec3s>(i)[j][c] * kernelV[1] +
                            down * kernelV[2];
                dst.ptr<cv::Vec3s>(i)[j][c] = sum / 2;
            }
        }
    }
    return 0;
}
int magnitude(cv::Mat &sx, Mat &sy, Mat &dst) {
    if (sx.empty() || sy.empty()) {
        cerr << "Input frame is empty!" << endl;
        return -1;
    }
    if (sx.size() != sy.size() || sx.type() != sy.type()) {
        std::cerr << "Error: sx and sy must have the same size and type." << std::endl;
        return -1; // Error code
    }

    // Create temporary and destination matrices
    dst.create(sx.size(), sx.type());

    // Using Euclidean distance for magnitude
    for (int i = 0; i < sx.rows; i++) {
        Vec3s* rowX = sx.ptr<Vec3s>(i);
        Vec3s* rowY = sy.ptr<Vec3s>(i);
        Vec3s* rowDst = dst.ptr<Vec3s>(i);

        for (int j = 0; j < sx.cols; j++) {
            for (int c = 0; c < 3; c++) {
                short gradX = rowX[j][c];
                short gradY = rowY[j][c];
                rowDst[j][c] = sqrt(gradX * gradX + gradY * gradY);
            }
        }
    }
    return 0;
}
int blurQuantize(cv::Mat &src, cv::Mat &dst, int levels) {
    if (src.empty()) {
        cerr << "Input frame is empty!" << endl;
        return -1;
    }
    // Create temporary and destination matrices
    dst.create(src.size(), src.type());

    // Processing color bucket
    double bucket = 255 / levels;
    float bucketColor[levels];

    for (int c = 0; c < levels; c++) {
        bucketColor[c] = (c + 0.1) * bucket;
    }

    // Processing each pixel
    for (int i = 0; i < src.rows; i++) {
        Vec3b* rowSrc = src.ptr<Vec3b>(i);
        Vec3b* rowDst = dst.ptr<Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            for (int c = 0; c < 3; c++) {
                rowDst[j][c] = bucketColor[static_cast<int>(rowSrc[j][c] / bucket)];
            }
        }
    }
    return 0;
}
int getDepthMap(cv::Mat &src, cv::Mat &dst) {
    if (src.empty()) {
        cerr << "Input frame is empty!" << endl;
        return -1;
    }
    // Create temporary and destination matrices
    dst.create(src.size(), src.type());
    DA2Network da_net( "../include/model_fp16.onnx" );

    // scale the network input so it's not larger than 512 on the small side
    const float reduction = 0.8;
    float scale_factor = 256.0 / (src.size().height*reduction);
    scale_factor = scale_factor > 1.0 ? 1.0 : scale_factor;
    printf("Scale factor: %.2f\n", scale_factor);

    // set up the network input
    da_net.set_input( src, scale_factor );

    // see how big the network input image is
    printf("Finished setting input: %d %d\n", da_net.in_height(), da_net.in_width() );

    // run the network
    da_net.run_network( dst, src.size());
}
int applyFog( cv::Mat &src, cv::Mat &depth, cv::Mat &dst) {
    if (src.empty() || depth.empty()) {
        cerr << "Input frame is empty!" << endl;
        return -1;
    }
    if (src.size() != depth.size()) {
        std::cerr << "Error: src and depth must have the same size" << std::endl;
        return -1; // Error code
    }

    dst = src.clone();

    cv::Mat normalizedDepth;
    cv::normalize(depth, normalizedDepth, 0, 1, cv::NORM_MINMAX, CV_32F);
    float fogDensity = 1.0f; // Adjust this value to control fog intensity

    // Apply fog effect on each pixels
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            float depthValue = normalizedDepth.at<float>(i,j);
            float invertedDepth = 1.0f - depthValue; // Invert the depth value
            float fogFactor = 1.0f - exp(-invertedDepth * invertedDepth * fogDensity);
            Vec3b& pixel = dst.at<Vec3b>(i, j);
            pixel = (1.0 - fogFactor) * pixel + Vec3b(255, 255, 255) * fogFactor;
        }
    }
    return 0;

}
int backgroundBlur(cv::Mat &src, std::vector<cv::Rect> &faces, cv::Mat &dst) {
    // Create a black mask for face regions (all zeros)
    Mat mask = Mat::zeros(src.size(), CV_8UC1);

    // Mark face regions as white in the mask
    for (const auto& face : faces) {
        Point center(face.x + face.width/2, face.y + face.height/2);
        // Calculate radius (you can adjust this calculation)
        int radius = std::min(face.width, face.height) / 2;

        // Draw white circle for face
        circle(mask, center, radius, Scalar(255), -1);  // -1 means filled circle
    }

    // Create a blurred version of the entire image
    Mat blurred;
    blur5x5_2(src, blurred);

    // Combine original and blurred images using the mask
    dst = src.clone();
    // Now blurred will be copied to non-face regions
    blurred.copyTo(dst, ~mask);  // Invert mask so faces stay original

    return 0;
}
int backgroundGrey(cv::Mat &src, std::vector<cv::Rect> &faces, cv::Mat &dst) {
    // Create black mask (all zeros)
    Mat mask = Mat::zeros(src.size(), CV_8UC1);

    // Set face regions to white (1) instead of black (0)
    for (const auto& face : faces) {
        Point center(face.x + face.width/2, face.y + face.height/2);

        // Calculate radius (you can adjust this calculation)
        int radius = max(face.width, face.height) / 2;

        // Draw white circle for face
        circle(mask, center, radius, Scalar(255), -1);  // -1 means filled circle
    }

    // Create a greyscale version of the entire image
    Mat grey;
    cvtColor(src, grey, COLOR_BGR2GRAY);
    cvtColor(grey, grey, COLOR_GRAY2BGR);  // Convert back to 3-channel for blending

    // Original
    dst = src.clone();

    // blend - now grayscale will be copied only outside face regions
    grey.copyTo(dst, ~mask);  // Note: you can also invert the mask here instead of changing initial values

    return 0;
}