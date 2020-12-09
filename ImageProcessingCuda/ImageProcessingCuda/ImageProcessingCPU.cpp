#include <iostream>
#include <fstream>
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

			grayImage[grayOffset] = 0.299f * r + 0.587f * g + 0.114f * b;
		}
	}

	//auto end = std::chrono::high_resolution_clock::now();
	//auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	//std::cout << "Elapsed time: " << elapsed << "milliseonds" << std::endl;
}

void gaussianBlur(unsigned char* grayImage, unsigned char* gaussianBlurImage, int Col, int Row){
	//auto begin = std::chrono::high_resolution_clock::now();
	for (int x = 0; x < Col; x++) {
		int lastRowPosition = Col * (Row - 1) + x;
		gaussianBlurImage[x] = grayImage[x];
		gaussianBlurImage[lastRowPosition] = grayImage[lastRowPosition];
	}

	for (int y = 0; y < Row; y++) {
		int firstCol = y * Col;
		int lastCol = firstCol + Col - 1;
		gaussianBlurImage[firstCol] = grayImage[firstCol];
		gaussianBlurImage[lastCol] = grayImage[lastCol];
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

			unsigned char current = grayImage[currentPixelPosition];
			unsigned char currentLeft = grayImage[currentLeftPixelPosition];
			unsigned char currentRight = grayImage[currentRightPixelPoisiton];
			unsigned char currentUpper = grayImage[upperPosition];
			unsigned char currentUpperLeft = grayImage[upperPositionleft];
			unsigned char currentUpperRight = grayImage[upperPositionRight];
			unsigned char currentLower = grayImage[lowerPosition];
			unsigned char currentLowerLeft = grayImage[lowerPositonLeft];
			unsigned char currentLowerRight = grayImage[lowerPositionRight];

			unsigned char result = (current * 4 + currentLeft * 2 + currentRight * 2 + currentUpper * 2 + currentUpperLeft
				+ currentUpperRight + currentLower * 2 + currentLowerLeft + currentLowerRight) / 16;
			gaussianBlurImage[currentPixelPosition] = result;
		}
	}

	//auto end = std::chrono::high_resolution_clock::now();
	//auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	//std::cout << "Elapsed time: " << elapsed << "milliseonds" << std::endl;
}

void SobelEdge(unsigned char* gaussImage, unsigned char* sobelEdgeImage, int Col, int Row) {
	//auto begin = std::chrono::high_resolution_clock::now();
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

			int G = abs(GX) + abs(GY);

			if (G >= 50) {
				sobelEdgeImage[currentPixelPosition] = 125;
			}
			else {
				sobelEdgeImage[currentPixelPosition] = 0;
			}

		}
	}

	//auto end = std::chrono::high_resolution_clock::now();
	//auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

	//std::cout << "Elapsed time: " << elapsed << "milliseonds" << std::endl;
}

void writeToFile(Mat Img) {
	ofstream myfile("Image.xls");
	for (int y = 0; y < Img.rows; y++) {
		for (int x = 0; x < Img.cols; x++) {
			int number = Img.at<uchar>(y, x);
			myfile << number << "\t";
		}
		myfile << "\n";
	}
	myfile.close();
}

void dummyDataTest() {
	uint8_t greyArr[11][12] = {
		{ 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255, 10 },
		{ 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255, 10 },
		{ 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255, 10 },
		{ 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255, 10 },
		{ 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255, 10 },
		{ 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255, 10 },
		{ 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255, 10 },
		{ 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255, 10 },
		{ 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255, 10 },
		{ 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255, 10 },
		{ 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 255, 10 },
	};

	Mat greyImg = cv::Mat(11, 12, CV_8UC1, &greyArr);
	Mat gaussanFilter(greyImg.rows, greyImg.cols, CV_8UC1);
	Mat sobelEdgeFilteredImage(greyImg.rows, greyImg.cols, CV_8UC1);

	gaussianBlur(greyImg.data, gaussanFilter.data, greyImg.cols, greyImg.rows);
	//imageProcessingCUDA(greyImg.data, greyImg.rows, greyImg.cols, 3, greyImg.data, gaussanFilter.data);
	SobelEdge(gaussanFilter.data, sobelEdgeFilteredImage.data, gaussanFilter.cols, gaussanFilter.rows);
	for (int y = 0; y < greyImg.rows; y++) {
		for (int x = 0; x < greyImg.cols; x++) {
			int number = greyImg.at<uchar>(y, x);
			std::cout << number << " ";
		}
		std::cout << std::endl;
	}

	std::cout << "****************************************" << std::endl;
	for (int y = 0; y < gaussanFilter.rows; y++) {
		for (int x = 0; x < gaussanFilter.cols; x++) {
			int number = gaussanFilter.at<uchar>(y, x);
			std::cout << number << " ";
		}
		std::cout << std::endl;
	}

	std::cout << "****************************************" << std::endl;
	for (int y = 0; y < sobelEdgeFilteredImage.rows; y++) {
		for (int x = 0; x < sobelEdgeFilteredImage.cols; x++) {
			int number = sobelEdgeFilteredImage.at<uchar>(y, x);
			std::cout << number << " ";
		}
		std::cout << std::endl;
	}

	//writeToFile(greyImg, gaussanFilter);

}

int main() {
	Mat rgbImage = imread("test9.jpg");
	Mat grayImage(rgbImage.rows, rgbImage.cols, CV_8UC1);
	Mat gaussianFilterImage(rgbImage.rows, rgbImage.cols, CV_8UC1);
	Mat sobelEdgeFilteredImage(rgbImage.rows, rgbImage.cols, CV_8UC1);

	if (rgbImage.empty()) {
		std::cout << "Could not read the image: " << endl;
		return 1;
	}
	else {
		cout << "Height:" << rgbImage.rows << ", Width: " << rgbImage.cols << ", Channels: " << rgbImage.channels() << endl;

		auto ossz = 0;
		for (int i = 0; i < 10; i++)
		{
			auto begin = std::chrono::high_resolution_clock::now();
			rgb2GRAYbasic(rgbImage.data, grayImage.data, rgbImage.cols, rgbImage.rows, rgbImage.channels());
			gaussianBlur(grayImage.data, gaussianFilterImage.data, grayImage.cols, grayImage.rows);
			SobelEdge(gaussianFilterImage.data, sobelEdgeFilteredImage.data, gaussianFilterImage.cols, gaussianFilterImage.rows);
			auto end = std::chrono::high_resolution_clock::now();
			auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
			ossz = ossz + elapsed;
			std::cout << "Elapsed time: " << elapsed << "milliseonds" << std::endl;
		}
		std::cout << (ossz / 10) << std::endl;
		//imageProcessingCUDA(rgbImage.data, rgbImage.rows, rgbImage.cols, rgbImage.channels(), grayImage.data, gaussianFilterImage.data, sobelEdgeFilteredImage.data);
		//cout << "Height:" << grayImage.rows << ", Width: " << grayImage.cols << ", Channels: " << grayImage.channels() << endl;
	}
	//dummyDataTest();
	//writeToFile(sobelEdgeFilteredImage);
	imwrite("GrayScale.png", grayImage);
	imwrite("gaussanFiltered_IMG.png", gaussianFilterImage);
	imwrite("sobelEdge.png", sobelEdgeFilteredImage);
	system("pause");
}
