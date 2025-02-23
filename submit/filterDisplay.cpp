//
// Created by Yuyang Tian on 2025/1/19.
// To display filtered image

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <sys/time.h>
#include "../src/filter.cpp"  // Assuming the filters are in filter.h

using namespace cv;
using namespace std;

// Apply Sepia filter, display and save
double getTime() {
    struct timeval cur;

    gettimeofday( &cur, NULL );
    return( cur.tv_sec + cur.tv_usec / 1000000.0 );
}
void processSepia(const Mat& original) {
    Mat sepiaImage;
    sepiaTone(original, sepiaImage);
    imshow("Sepia", sepiaImage);

    int key = waitKey(0);
    if (key == 's') { // Save Sepia image
        imwrite("../outputs/photoshot0-sepia.png", sepiaImage);
        cout << "Saved sepia image." << endl;
    }
}

// Apply Vignetting filter, display and save
void processVignetting(const Mat& original) {
    Mat vigImage;
    vignetting(original, vigImage);
    imshow("Vignetting", vigImage);

    int key = waitKey(0);
    if (key == 's') { // Save Vignetting image
        imwrite("../outputs/photoshot0-vignetting.png", vigImage);
        cout << "Saved vignetting image." << endl;
    }
}
void processQuantization(Mat& original, int level) {
    Mat quan;
    blurQuantize(original, quan, level);
    imshow("Quantization", quan);

    int key = waitKey(0);
    if (key == 's') { // Save Vignetting image
        string file = "../outputs/photoshot0-quantize-" + to_string(level) + ".png";
        imwrite(file, quan);
        cout << "Saved vignetting image." << endl;
    }
}

// Apply Blur filter, display and save
void processBlur(Mat& src) {
    cout << "enter blur\n";
    Mat dst;

    const int Ntimes = 20;

    //////////////////////////////
    // set up the timing for version 1
    double startTime = getTime();

    // execute the file on the original image a couple of times
    for(int i=0;i<Ntimes;i++) {
        blur5x5_1( src, dst);
    }
    imshow("Version1", dst);
    // end the timing
    double endTime = getTime();

    // compute the time per image
    double difference = (endTime - startTime) / Ntimes;

    int key = waitKey(0);
    if (key == 's') { // Save Vignetting image
        imwrite("../outputs/photoshot0-blur1.png", dst);
        cout << "Saved blurred version1 image." << endl;
    }

    // print the results
    printf("Time per image (1): %.4lf seconds\n", difference );

    //////////////////////////////
    // set up the timing for version 2
    startTime = getTime();

    // execute the file on the original image a couple of times
    for(int i=0;i<Ntimes;i++) {
        blur5x5_2( src, dst );
    }

    // end the timing
    endTime = getTime();

    // compute the time per image
    difference = (endTime - startTime) / Ntimes;

    // print the results
    printf("Time per image (2): %.4lf seconds\n", difference);
    imshow("Version2", dst);
    key = waitKey(0);
    if (key == 's') { // Save Vignetting image
        imwrite("../outputs/photoshot0-blur2.png", dst);
        cout << "Saved blurred version2 image." << endl;
    }
    // terminate the program
    printf("Terminating\n");
}

int main() {
    string image_path = "../outputs/photoshot0.png";
    Mat original = imread(image_path);

    if (original.empty()) {
        cerr << "Could not read the image: " << image_path << endl;
        return 1;
    }


    // Process each filter
//    processSepia(original);
//    processVignetting(original);
//    processBlur(original);
    processQuantization(original,5);
    cout << "Filters applied and processed successfully.\n";
    return 0;
}
