#include <iostream>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "ImageProcessing.h"

using namespace std;
using namespace cv;

int main() {
	Mat img = imread("test.png");

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
