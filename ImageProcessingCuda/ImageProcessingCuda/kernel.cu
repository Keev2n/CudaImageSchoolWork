
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ImageProcessing.h"
#include <math.h>
#include <iostream>


using namespace std;

__global__ void ImageToGrayScale_CUDA(unsigned char* RGBimage, int Row, int Col, int Channels, unsigned char* GrayImage);
__global__ void GaussianFilter_CUDA(unsigned char* GrayImage, int Row, int Col, unsigned char* GaussFilteredImage);

void image_toGrayScale_Cuda(unsigned char* Image, int Row, int Col, int Channels, unsigned char* Image2) {
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

__global__ void ImageToGrayScale_CUDA(unsigned char* Image, int Row, int Col, int Channels, unsigned char* Image2) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < Col && y < Row) {
		int grayOffset = y * Col + x;
		int rgbOffset = grayOffset * Channels;

		unsigned char b = Image[rgbOffset];
		unsigned char g = Image[rgbOffset + 1]; 
		unsigned char r = Image[rgbOffset + 2];


		Image2[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
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