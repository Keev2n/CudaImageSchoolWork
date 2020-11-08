
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "ImageProcessing.h"
#include <math.h>

__global__ void ImageToGrayScale_CUDA(unsigned char* Image, int Channels, int Width, int Height);

void image_toGrayScale_Cuda(unsigned char* Image, int Height, int Width, int Channels) {
	unsigned char* dev_Image = NULL;

	cudaMalloc((void**)&dev_Image, Height * Width * Channels);

	cudaMemcpy(dev_Image, Image, Height * Width * Channels, cudaMemcpyHostToDevice);

	int pixelNumbers = Height * Width;
	double param, result;
	param = 1024;
	result = sqrt(param);

	int blockNumbers = pixelNumbers / param;
	int blockXY = sqrt(blockNumbers);
	dim3 Grid_Image(blockXY, blockXY);
	dim3 Grid_Image_Thread(result, result);
	ImageToGrayScale_CUDA <<<Grid_Image, Grid_Image_Thread >> > (dev_Image, Channels, Width, Height);

	cudaMemcpy(Image, dev_Image, Height * Width * Channels, cudaMemcpyDeviceToHost);

	cudaFree(Image);
}

__global__ void ImageToGrayScale_CUDA(unsigned char* Image, int Channels, int Width, int Height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x < Width && y < Height) {
		int grayOffset = y * Width + x;
		int rgbOffset = grayOffset * Channels;

		unsigned char r = Image[rgbOffset];
		unsigned char g = Image[rgbOffset + 2]; 
		unsigned char b = Image[rgbOffset + 3];

		Image[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
	}

}