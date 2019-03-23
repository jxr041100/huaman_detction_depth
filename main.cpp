#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <fstream>      
#include <algorithm>    

using namespace cv;
using namespace std;

#define DEBUG
double medianMat(Mat depthImg) {
	Mat Input = depthImg;
	Input = Input.reshape(0, 1); // spread Input Mat to single row
	vector<double> vecFromMat;
	Input.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat    
	sort(vecFromMat.begin(), vecFromMat.end()); // sort vecFromMat
	if (vecFromMat.size() % 2 == 0) { return (vecFromMat[vecFromMat.size() / 2 - 1] + vecFromMat[vecFromMat.size() / 2]) / 2; } // in case of even-numbered matrix
	return vecFromMat[(vecFromMat.size() - 1) / 2]; // odd-number of elements in matrix
}

Mat readDepthImg(string filename)
{
	ifstream file(filename);
	Mat depthImg;
	int rows = 0;
	std::string line;
	while (std::getline(file, line)) {
		std::istringstream stream(line);
		double x;
		while (stream >> x) {
			depthImg.push_back(x);
		}
		rows++;
	}
	// reshape to 2d:
	depthImg = depthImg.reshape(1, rows);               
	return depthImg;
}

Mat denoise(Mat depthImg, float near, float far)
{
	int width = depthImg.cols;
	int height = depthImg.rows;
	Mat denoised = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
	int cnt = 0, cnt_zeros = 0, cnt_255 = 0;
	for(int i = 0;i<height;i++)
		for (int j = 0; j < width; j++)
		{
			double depth = depthImg.at<double>(i, j);
			if (depth < near)
				denoised.at<uint8_t>(i, j) = 0, cnt_zeros++;
			else if (depth > far)
				denoised.at<uint8_t>(i, j) = 255,cnt_255++;
			else
				denoised.at<uint8_t>(i, j) = depth,cnt++;
		}
#ifdef DEBUG
	namedWindow("Erosion Demo", WINDOW_AUTOSIZE);
	imshow("after denoise", denoised);
#endif
	return denoised;
}

/**  @function Erosion  */
Mat Erosion(int erosion_elem, Mat depthImg)
{
	int erosion_type;
	if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
	else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
	else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
	int erosion_size = 1;
	Mat element = getStructuringElement(erosion_type,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));

	/// Apply the erosion operation
	Mat erosionImg;
	erode(depthImg, erosionImg, element, Point(-1,-1),2);
#ifdef DEBUG
	namedWindow("Erosion Demo", WINDOW_AUTOSIZE);
	imshow("Erosion Demo", erosionImg);
#endif
	return erosionImg;
}

Mat Dilation(int dilation_elem, Mat depthImg)
{
	int dilation_type;
	if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
	else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
	else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }

	int dilation_size = 1;

	Mat element = getStructuringElement(dilation_type,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	/// Apply the dilation operation
	Mat dilationImg;
	dilate(depthImg, dilationImg, element, Point(-1,-1),2);
#ifdef DEBUG
	namedWindow("Dilation Demo", WINDOW_AUTOSIZE);
	imshow("Dilation Demo", dilationImg);
#endif
	return dilationImg;
}

Mat cannyEdgeDetection(Mat depthImg,double sigma = 0.33)
{	
	Mat edgeDepth;
	GaussianBlur(depthImg, edgeDepth, Size(5, 5),0);
	double median = medianMat(edgeDepth);
	int lower = int(max(0.0, (1.0 - sigma) * median));
	int upper = int(min(255.0, (1.0 + sigma) * median));
	Canny(edgeDepth, edgeDepth, lower, upper);
#ifdef DEBUG
	namedWindow("depthImg after edge", WINDOW_AUTOSIZE);
	imshow("depthImg after edge", edgeDepth);
#endif
	return edgeDepth;
}


Mat cropped(Mat depthImg)
{
	int y0 = 30, y1 = 108, x0 = 60, x1 = 120;
	int width = depthImg.cols;
	int height = depthImg.rows;
	Mat cropped = cv::Mat::zeros(cv::Size(width, height), CV_8UC1);
	for (int y = y0; y < y1; y++)
		for (int x = x0; x < x1; x++)
			cropped.at<uint8_t>(y, x) = depthImg.at<uint8_t>(y, x);
#ifdef DEBUG
	namedWindow("depthImg cropping", WINDOW_AUTOSIZE);
	imshow("depthImg cropping", cropped);
#endif
	return cropped;
}

Rect findContours(Mat depthImg)
{
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	
	/// Find contours
	findContours(depthImg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;

	for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
	{
		double a = contourArea(contours[i], false);  //  Find the area of contour
		if (a > largest_area) {
			largest_area = a;
			largest_contour_index = i;                //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}
	}
#ifdef DEBUG	
	/// Show in a window
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	rectangle(depthImg, bounding_rect, Scalar(255, 0, 0), 1, 8, 0);
	imshow("Contours", depthImg);
#endif
	return bounding_rect;
}


int main(int argc, char** argv)
{
	if (argc < 2)
	{
		cout << "please provide the filename for depth file" << endl;
		return -1;
	}
	string filename = argv[1];
	//std::string filename = "D:\\Documents\\hiAIV200\\3DVision\\code\\depthHuman\\human_corridor_2.txt";
	
	//STEP1: read depth image
	Mat depthImg = 	readDepthImg(filename);
	
	//STEP2: denoising 
	Mat deonised_depthImg = denoise(depthImg, 1.0, 4.0);

	//STEP3: edge detection 
	Mat edge_depthImg = cannyEdgeDetection(deonised_depthImg);

	//STEP 4:dilation 
	Mat dilation_depthImg = Dilation(0, edge_depthImg);

	//STEP 5:erosion
	Mat erosion_depthImg = Erosion(0, dilation_depthImg);
	
	//STEP 6:cropped area
	Mat crop = cropped(erosion_depthImg);

	//STEP 7:find the contour
	Rect rect = findContours(crop);

	// apply person height/width ratio
	if (rect.height / rect.width < 2)
	{
		cout << "no person was detected, maybe some objects found" << endl;
		return -1;
	}

	//ASSUMMING THE ROBOT ALWAYS IN THE MIDDLE OF CORRIDOR
	//cout << rect.x << "," << rect.y << "," << rect.width << "," << rect.height << ";" << endl;
	int left = rect.x;
	int right = rect.x + rect.width;
	left = left - 60;
	right = 120 - right;

	if (left >= right)
		cout << "going left" << endl;
	else
		cout << "going right" << endl;

	waitKey(0);  
	return 0;
}