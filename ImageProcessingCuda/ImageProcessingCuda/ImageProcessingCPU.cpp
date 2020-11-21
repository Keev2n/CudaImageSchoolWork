#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ImageProcessing.h"

using namespace std;
using namespace cv;

void Teszt(Mat img, Mat grayImage) {
	int width = img.cols;
	int height = img.rows;
	int channels = img.channels();

	for (int x = 0; x < width; x++){
		for (int y = 0; y < height; y++){
			int grayOffset = width * y + x;
			int rgbOffset = channels * grayOffset;
			uchar b = img.data[rgbOffset];
			uchar g = img.data[rgbOffset + 1];
			uchar r = img.data[rgbOffset + 2];

			grayImage.data[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
		}
	}
}


int main() {
	Mat img = imread("test7.png");
	Mat grayImage(img.rows, img.cols, CV_8UC1);

	if (img.empty()) {
		std::cout << "Could not read the image: " << endl;
		return 1;
	}
	else {
		cout << "Height:" << img.rows << ", Width: " << img.cols << ", Channels: " << img.channels() << endl;
	}

	image_toGrayScale_Cuda(img.data, img.rows, img.cols, img.channels());

	imwrite("Inverted_Image.png", img);
	system("pause");
}
