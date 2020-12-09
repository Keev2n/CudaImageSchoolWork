#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "ImageProcessing.h"

using namespace std;
using namespace cv;

void rgb2GRAYbasic(unsigned char* rgbImage, unsigned char* grayImage, int Col, int Row, int Channels) {
	//auto begin = std::chrono::high_resolution_clock::now();

	for (int x = 0; x < Col; x++) {
		for (int y = 0; y < Row; y++) {
			int grayOffset = Col * y + x;
			int rgbOffset = Channels * grayOffset;
			unsigned char b = rgbImage[rgbOffset];
			unsigned char g = rgbImage[rgbOffset + 1];
			unsigned char r = rgbImage[rgbOffset + 2];

		}
	}

	//auto end = std::chrono::high_resolution_clock::now();
	//auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	//std::cout << "Elapsed time: " << elapsed << "milliseonds" << std::endl;
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

void SobelEdge(unsigned char* gaussImage, unsigned char* sobelEdgeImage, int Col, int Row) {
	for (int x = 0; x < Col; x++) {
		int lastRowPosition = Col * (Row - 1) + x;
		sobelEdgeImage[x] = 0;
		sobelEdgeImage[lastRowPosition] = 0;
	}

	for (int y = 0; y < Row; y++) {
		int firstCol = y * Col;
		int lastCol = firstCol + Col - 1;
		sobelEdgeImage[firstCol] = 0;
		sobelEdgeImage[lastCol] = 0;
	}

	for (int x = 1; x < Col - 1; x++) {
		for (int y = 1; y < Row - 1; y++) {
			int currentPixelPosition = y * Col + x;
			int currentLeftPixelPosition = currentPixelPosition - 1;
			int currentRightPixelPoisiton = currentPixelPosition + 1;
			int upperPosition = currentPixelPosition - Col;
			int upperPositionleft = upperPosition - 1;
			int upperPositionRight = upperPosition + 1;
			int lowerPosition = currentPixelPosition + Col;
			int lowerPositonLeft = lowerPosition - 1;
			int lowerPositionRight = lowerPosition + 1;

			unsigned char current = gaussImage[currentPixelPosition];
			unsigned char currentLeft = gaussImage[currentLeftPixelPosition];
			unsigned char currentRight = gaussImage[currentRightPixelPoisiton];
			unsigned char currentUpper = gaussImage[upperPosition];
			unsigned char currentUpperLeft = gaussImage[upperPositionleft];
			unsigned char currentUpperRight = gaussImage[upperPositionRight];
			unsigned char currentLower = gaussImage[lowerPosition];
			unsigned char currentLowerLeft = gaussImage[lowerPositonLeft];
			unsigned char currentLowerRight = gaussImage[lowerPositionRight];

			int GY = (currentUpper * -2) + (currentUpperLeft * -1) + (currentUpperRight * -1)
				+ (currentLower * 2) + currentLowerLeft + currentLowerRight;

			int GX = (currentLeft * -2) + (currentRight * 2) + (currentUpperLeft * -1)
				+ (currentUpperRight)+(currentLowerLeft * -1) + (currentLowerRight);

			GX = abs(GX);
			GY = abs(GY);
			int G = GX * GX + GY * GY;
			G = sqrt(G);

			if (G >= 50) {
				sobelEdgeImage[currentPixelPosition] = 100;
			}
			else {
				sobelEdgeImage[currentPixelPosition] = 0;
			}

		}
	}
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
