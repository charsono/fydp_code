#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Global variable describing person characteristics
Scalar PERSON_HSV;
Mat PERSON_TMPLT;
Scalar LOWER_BOUND, UPPER_BOUND;

Mat resize_img(Mat img, int size) {
	double r = size*1.0 / img.cols;
	Mat result;
	resize(img, result, Size(), r, r);
	return result;
}

Mat load_hsv_image(Mat image_bgr, int size=500) {
	Mat result;
	cvtColor(resize_img(image_bgr, size), result, CV_BGR2HSV);
	return result;
}

Mat filter_green_background(Mat image_hsv) {
	Mat mask, result;
	inRange(image_hsv, Scalar(40, 0, 0), Scalar(55, 255, 255), mask);
	mask = 255 - mask;
	bitwise_and(image_hsv, image_hsv, result, mask = mask);

	return result;
}

Mat get_person_with_color(Mat img) {
	Mat filtered_image, mask;
	inRange(img, LOWER_BOUND, UPPER_BOUND, mask);

	//morphological opening (remove small objects from the foreground)
	erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

	//morphological closing (fill small holes in the foreground)
	dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	bitwise_and(img, img, filtered_image, mask = mask);

	return filtered_image;
}

Mat get_person_with_tmplt(Mat img) {
	Point matchLoc;
	double maxVal_found = NULL;	Point maxLoc_found; double r_found;

	/* Resize original image to fit it with template. Continuously match template */
	for (int i = 20; i >= 0; i--) {
		float scale = 0.04*i + 0.2;
		Mat resized = resize_img(img, (int)(scale*img.cols));
		if (resized.rows < PERSON_TMPLT.rows || resized.cols < PERSON_TMPLT.cols) {
			break;
		}

		double r = img.cols / resized.cols; // Resizing ratio

		// Match Template, and get coordinate found
		Mat result;
		matchTemplate(img, PERSON_TMPLT, result, TM_CCOEFF);
		double minVal; double maxVal; Point minLoc; Point maxLoc;
		minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

		// If box found has better correlation, assign as new box
		if (maxVal_found == NULL || maxVal > maxVal_found) {
			maxVal_found = maxVal;
			maxLoc_found = maxLoc;
			r_found = r;
		}
	}
	// Box coordinate of the person
	Point start_coord = Point((int)(maxLoc_found.x*r_found), (int)(maxLoc_found.y*r_found));
	Point end_coord = Point((int)((maxLoc_found.x + PERSON_TMPLT.cols)*r_found), (int)((maxLoc_found.y + PERSON_TMPLT.rows)*r_found));

	// Create Mask
	Mat mask = Mat::zeros(img.size(), img.type());
	rectangle(mask, start_coord, end_coord, Scalar(255, 255, 255), CV_FILLED);

	Mat result;
	bitwise_and(img, mask, result);

	return result;
}

Mat hsv2gray(Mat img) {
	Mat img_BGR, result;
	cvtColor(img, img_BGR, CV_HSV2BGR);
	cvtColor(img, result, CV_BGR2GRAY);
	return result;
}

void find_person_in_img(Mat img, String descrption = "") {
	Mat input_image = load_hsv_image(img);

	// Apply filtering
	Mat color_filtered_image = get_person_with_color(input_image);
	Mat tmplt_filtered_image = get_person_with_tmplt(color_filtered_image);

	// Get person location
	Moments person_moments = moments(hsv2gray(tmplt_filtered_image), true);
	cout << descrption << endl;
	cout << "Area: " << person_moments.m00 << endl;
	cout << "x: " << person_moments.m10 / person_moments.m00 << endl;
	cout << "y: " << person_moments.m01 / person_moments.m00 << endl << endl;

	// Plot location in image
	rectangle(tmplt_filtered_image, Point(323, 110), Point(327, 115), Scalar(255, 255, 255), CV_FILLED);
	imshow("Image", tmplt_filtered_image);
	waitKey(0);

}

int main(int argc, char** argv)
{
	/* Part 1 */
	Mat person_original = load_hsv_image(imread("input/benchmark_cropped.jpg"));
	Mat person = resize_img(filter_green_background(person_original), 200);
	Mat color = load_hsv_image(imread("input/color_benchmark.jpg"));

	// Setting global variable describing person characteristics
	// For Color
	PERSON_HSV = mean(color);
	int hue_avg = (int)ceil(PERSON_HSV.val[0]);
	int s_avg = (int)ceil(PERSON_HSV.val[1]);
	/*int v_avg = (int)ceil(avg_hsv.val[2]);*/
	LOWER_BOUND = Scalar(std::max(hue_avg - 5, 0), std::max(s_avg - 40, 0), 0);
	UPPER_BOUND = Scalar(std::min(hue_avg + 5, 255), std::min(s_avg + 40, 255), 255);
	// For person template
	PERSON_TMPLT = get_person_with_color(person);


	/* Part 2 */
	for (int i = 0; i < 71; i++) {
		// Load Image
		String file_name;
		if (i < 10) {
			file_name = "input/IMG_290" + to_string(i) + ".jpg";
		}
		else {
			file_name = "input/IMG_29" + to_string(i) + ".jpg";
		}
		 
		Mat input_img = imread(file_name);
		find_person_in_img(input_img, file_name);	
	}

	return 0;
}
