
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ImageProcessing.h"
#include <math.h>
#include <iostream>


using namespace std;

__global__ void ImageToGrayScale_CUDA(unsigned char* RGBimage, int Row, int Col, int Channels, unsigned char* GrayImage);
__global__ void GaussianFilter_CUDA(unsigned char* GrayImage, int Row, int Col, unsigned char* GaussFilteredImage);
__global__ void SobelEdge_CUDA(unsigned char* gaussImage, unsigned char* sobelEdgeImage, int Col, int Row);


void imageProcessingCUDA(unsigned char* Image, int Row, int Col, int Channels, unsigned char* Image2) {
	unsigned char* dev_Image = NULL;
	unsigned char* dev_Image2 = NULL;

	cudaMalloc((void**)&dev_Image, Row * Col * Channels);
	cudaMalloc((void**)&dev_Image2, Row * Col);

	cudaMemcpy(dev_Image, Image, Row * Col * Channels, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Image2, Image2, Row * Col, cudaMemcpyHostToDevice);

	int threadNumber = 16;

	int blockX = (Col / threadNumber) + 1;
	int blockY = (Row / threadNumber) + 1;

	ImageToGrayScale_CUDA <<< dim3(blockX,blockY), dim3(threadNumber, threadNumber) >> > (dev_Image, Row, Col, Channels, dev_Image2);
	cout << cudaGetLastError() << endl;

	cudaMemcpy(Image, dev_Image, Row * Col * Channels, cudaMemcpyDeviceToHost);
	cudaMemcpy(Image2, dev_Image2, Row * Col, cudaMemcpyDeviceToHost);

	cudaFree(Image);
	cudaFree(Image2);
}

__global__ void ImageToGrayScale_CUDA(unsigned char* RGBimage, int Row, int Col, int Channels, unsigned char* GrayImage) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < Col && y < Row) {
		int grayOffset = y * Col + x;
		int rgbOffset = grayOffset * Channels;

		unsigned char b = RGBimage[rgbOffset];
		unsigned char g = RGBimage[rgbOffset + 1];
		unsigned char r = RGBimage[rgbOffset + 2];

		GrayImage[grayOffset] = 0.299f * r + 0.587f * g + 0.114f * b;
	}
}

__global__ void GaussianFilter_CUDA(unsigned char* GrayImage, int Row, int Col, unsigned char* GaussFilteredImage) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > (Col - 1) || y > (Row - 1)) {
		return;
	}

	if (x == 0 || y == 0 || x == (Col - 1) || y == (Row - 1)) {
		int currentPixelPosition = y * Col + x;
		GaussFilteredImage[currentPixelPosition] = GrayImage[currentPixelPosition];
		return;
	}

	int currentPixelPosition = y * Col + x;
	int currentLeftPixelPosition = currentPixelPosition - 1;
	int currentRightPixelPoisiton = currentPixelPosition + 1;
	int upperPosition = currentPixelPosition - Col;
	int upperPositionleft = upperPosition - 1;
	int upperPositionRight = upperPosition + 1;
	int lowerPosition = currentPixelPosition + Col;
	int lowerPositonLeft = lowerPosition - 1;
	int lowerPositionRight = lowerPosition + 1;

	unsigned char current = GrayImage[currentPixelPosition];
	unsigned char currentLeft = GrayImage[currentLeftPixelPosition];
	unsigned char currentRight = GrayImage[currentRightPixelPoisiton];
	unsigned char currentUpper = GrayImage[upperPosition];
	unsigned char currentUpperLeft = GrayImage[upperPositionleft];
	unsigned char currentUpperRight = GrayImage[upperPositionRight];
	unsigned char currentLower = GrayImage[lowerPosition];
	unsigned char currentLowerLeft = GrayImage[lowerPositonLeft];
	unsigned char currentLowerRight = GrayImage[lowerPositionRight];

	unsigned char result = (current * 4 + currentLeft * 2 + currentRight * 2 + currentUpper * 2 + currentUpperLeft
		+ currentUpperRight + currentLower * 2 + currentLowerLeft + currentLowerRight) / 16;
	GaussFilteredImage[currentPixelPosition] = result;
}

__global__ void SobelEdge_CUDA(unsigned char* gaussImage, unsigned char* sobelEdgeImage, int Col, int Row) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x > (Col - 1) || y > (Row - 1)) {
		return;
	}

	if (x == 0 || y == 0 || x == (Col - 1) || y == (Row - 1)) {
		int currentPixelPosition = y * Col + x;
		sobelEdgeImage[currentPixelPosition] = 0;
		return;
	}

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

	GX = fabsf(GX);
	GY = fabsf(GY);
	int G = GX * GX + GY * GY;
	G = sqrtf(G);

	if (G >= 50) {
		sobelEdgeImage[currentPixelPosition] = 100;
	}
	else {
		sobelEdgeImage[currentPixelPosition] = 0;
	}

}