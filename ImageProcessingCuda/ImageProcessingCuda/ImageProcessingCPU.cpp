#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ImageProcessing.h"

using namespace std;
using namespace cv;

void rgb2GRAYbasic(Mat img, Mat grayImage) {
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

void Teszt2(Mat grayImage, Mat gaussianFilter){
	/*int pixelertekMegvaltoztatando = grayImage.at<uchar>(1, 1) * 4;
	int balfelsoertek = grayImage.at<uchar>(0, 0);
	int kozepsofelso = grayImage.at<uchar>(0, 1) * 2;
	int jobbfelso = grayImage.at<uchar>(0, 2);
	int balkozepso = grayImage.at<uchar>(1, 0) * 2;
	int jobbkozepso = grayImage.at<uchar>(1, 2) * 2;
	int balalso = grayImage.at<uchar>(2, 0);
	int kozepsoalso = grayImage.at<uchar>(2, 1) * 2;
	int jobbalso = grayImage.at<uchar>(2, 3);

	for (int x = 0; x < 3; x++){
		for (int y = 0; y < 3; y++){
			int pixel = grayImage.at<uchar>(x, y);
			cout << pixel << " ";
		}
		cout << "\n";
	}

	int kozepsopixel = (pixelertekMegvaltoztatando + balfelsoertek + kozepsofelso + jobbfelso + balkozepso + jobbkozepso + balalso +
		kozepsoalso + jobbalso) / 16;

	cout << kozepsopixel;
	*/
	/*for (int i = 0; i < grayImage.rows; i++){
		for (int y = 0; y < 20; y++){
			gaussianFilter.at<uchar>(i, y) = grayImage.at<uchar>(i, y);
		}
	}
	*/
	for (int x = 1; x < (grayImage.cols - 1); x++){
		for (int y = 1; y < (grayImage.rows - 1); y++){
			int balfelsoertek = grayImage.at<uchar>(y - 1, x - 1);
			int kozepsofelso = grayImage.at<uchar>(y - 1, x) * 2;
			int jobbfelso = grayImage.at<uchar>(y - 1, x + 1);
			int balkozepso = grayImage.at<uchar>(y, x - 1) * 2;
			int jobbkozepso = grayImage.at<uchar>(y, x + 1) * 2;
			int balalso = grayImage.at<uchar>(y + 1, x - 1);
			int kozepsoalso = grayImage.at<uchar>(y + 1, x) * 2;
			int jobbalso = grayImage.at<uchar>(y + 1, x + 1);
			int jelenlegipixel = grayImage.at<uchar>(y, x) * 4;

			gaussianFilter.at<uchar>(y, x) = (jelenlegipixel + balfelsoertek + kozepsofelso + jobbfelso + balkozepso + jobbkozepso + balalso +
				kozepsoalso + jobbalso) / 16;
		}
	}

	imshow("Original", grayImage);
	imshow("GaussianFilter", gaussianFilter);

	waitKey(0);
}

int main() {
	Mat img = imread("test7.png");
	Mat grayImage(img.rows, img.cols, CV_8UC1);
	Mat gaussianFilter(img.rows, img.cols, CV_8UC1);

	if (img.empty()) {
		std::cout << "Could not read the image: " << endl;
		return 1;
	}
	else {
		cout << "Height:" << img.rows << ", Width: " << img.cols << ", Channels: " << img.channels() << endl;
	}

	//image_toGrayScale_Cuda(img.data, img.rows, img.cols, img.channels(), grayImage.data);
	// cout << "Height:" << grayImage.rows << ", Width: " << grayImage.cols << ", Channels: " << grayImage.channels() << endl;
	Teszt(img, grayImage);
	Teszt2(grayImage, gaussianFilter);


	imwrite("GraySclae_IMG.png", grayImage);
	system("pause");
}
