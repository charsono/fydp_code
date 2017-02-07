#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

// Global variable describing person characteristics
Scalar PERSON_HSV;
Mat PERSON_TMPLT;
//Scalar LOWER_BOUND, UPPER_BOUND;
int HUE_AVG;

struct MoveData {
	int x;
	int y;
	int area;
};

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

vector<int> get_non_zero_bins(Mat hist) {
	vector<int> peak_values;
	for (int h = 0; h < hist.rows; h++)
	{
		if (hist.at<float>(h) > 0) {
			peak_values.push_back(h);
		}
	}
	return peak_values;
}

vector<int> get_peak_values(Mat hist) {
	vector<int> peak_values;
	vector<pair<int, int>> peak_values_dict;
	int window[9] = {};
	for (int h = 4; h < hist.rows; h++)
	{
		for (int i = 4; i >= 0; i--) {
			// Filling sliding window
			window[i] = hist.at<float>(h + i - 4);
			if (i != 4) {
				window[8 - i] = hist.at<float>(h - i + 4);
				if (window[4] < window[i] || window[4] < window[8 - i]) {
					break;
				}
			}
			if (i == 0 && hist.at<float>(h) > 0) {
				peak_values_dict.push_back(make_pair(h, hist.at<float>(h)));
			}
		}
	}
	sort(peak_values_dict.begin(), peak_values_dict.end(), [=](std::pair<int, int>& a, std::pair<int, int>& b)
	{
		return a.second > b.second;
	}
	);

	for (vector<pair<int,int>>::iterator it = peak_values_dict.begin(); it != peak_values_dict.end(); ++it)
	{
		if (peak_values.size() < 11) {
			peak_values.push_back(it->first);
		}
	}

	return peak_values;
}

std::vector<Mat> get_hsv_histogram(Mat image, String desc, bool is_save=false) {
	vector<Mat> hsv_planes;
	split(image, hsv_planes);

	int histSize = 180;
	int histSizeSV = 256;
	float range[] = {0, 180};
	const float* histRange = { range };
	float range_sv[] = { 0, 256};
	const float* histRange_sv = { range_sv };

	Mat h_hist, s_hist, v_hist;

	calcHist(&hsv_planes[0], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, true, false);
	calcHist(&hsv_planes[1], 1, 0, Mat(), s_hist, 1, &histSizeSV, &histRange_sv, true, false);
	calcHist(&hsv_planes[2], 1, 0, Mat(), v_hist, 1, &histSizeSV, &histRange_sv, true, false);

	vector<Mat> histograms = { h_hist, s_hist, v_hist };

	if (is_save) {
		// Output peak values

		ofstream out(desc + ".txt");
		for (int h = 0; h < h_hist.rows; h++)
		{
			out << h << " " << h_hist.at<float>(h) << endl;
		}
		out << endl;
		std::vector<int> peak_values = get_peak_values(h_hist);
		for (int i = 0; i < peak_values.size(); i++) {
			out << peak_values[i] << endl;
		}
		out.close();

		// Draw the histograms for H, S and V
		int hist_w = 360; int hist_h = 200;
		int bin_w = cvRound((double)hist_w / histSize);
		Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

		int hist_w_sv = 512; int hist_h_sv = 400;
		int bin_w_sv = cvRound((double)hist_w_sv / histSizeSV);
		Mat histImageS(hist_h_sv, hist_w_sv, CV_8UC3, Scalar(0, 0, 0));
		Mat histImageV(hist_h_sv, hist_w_sv, CV_8UC3, Scalar(0, 0, 0));

		normalize(h_hist, h_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		normalize(s_hist, s_hist, 0, histImageS.rows, NORM_MINMAX, -1, Mat());
		normalize(v_hist, v_hist, 0, histImageV.rows, NORM_MINMAX, -1, Mat());

		/// Draw for each channel
		for (int i = 1; i < histSize; i++)
		{
			line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(h_hist.at<float>(i - 1))),
				Point(bin_w*(i), hist_h - cvRound(h_hist.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
			line(histImageS, Point(bin_w_sv*(i - 1), hist_h_sv - cvRound(s_hist.at<float>(i - 1))),
				Point(bin_w_sv*(i), hist_h_sv - cvRound(s_hist.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
			line(histImageV, Point(bin_w_sv*(i - 1), hist_h_sv - cvRound(v_hist.at<float>(i - 1))),
				Point(bin_w_sv*(i), hist_h_sv - cvRound(v_hist.at<float>(i))),
				Scalar(255, 0, 0), 2, 8, 0);
		}

		imwrite(desc + "_h.jpg", histImage);
		imwrite(desc + "_s.jpg", histImageS);
		imwrite(desc + "_v.jpg", histImageV);
	}
	return histograms;
}

Mat filter_green_background(Mat image_hsv, int green_value) {
	Mat mask, result;
	inRange(image_hsv, Scalar(green_value - 7, 0, 0), Scalar(green_value + 7, 255, 255), mask);
	mask = 255 - mask;
	bitwise_and(image_hsv, image_hsv, result, mask = mask);

	return result;
}

Mat get_person_with_color(Mat img) {
	vector<Mat> segmented_hists = get_hsv_histogram(img, "img");
	std::vector<int> segmented_value = get_non_zero_bins(segmented_hists[0]);

	int diff = 180;
	int hue_avg = HUE_AVG;
	
	for (int i = 0; i < segmented_value.size(); i++) {
		if (abs(segmented_value[i]-HUE_AVG) < diff) {
			diff = abs(segmented_value[i]-HUE_AVG);
			hue_avg = segmented_value[i];
		}
	}
	
	Scalar lower_bound = Scalar(std::max(hue_avg - 3, 0), 0, 0);
	Scalar upper_bound = Scalar(std::min(hue_avg + 3, 255), 255, 255);

	Mat filtered_image, mask;
	inRange(img, lower_bound, upper_bound, mask);

	//morphological opening (remove small objects from the foreground)
	erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

	//morphological closing (fill small holes in the foreground)
	dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));
	erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

	bitwise_and(img, img, filtered_image, mask = mask);

	return filtered_image;
}

Mat get_person_with_tmplt(Mat img) {
	Point matchLoc;
	double maxVal_found = NULL; Point maxLoc_found;

	// Match Template, and get coordinate found
	Mat result;
	matchTemplate(img, PERSON_TMPLT, result, TM_CCOEFF_NORMED);
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

	// If box found has better correlation, assign as new box
	if (maxVal_found == NULL || maxVal > maxVal_found) {
		maxVal_found = maxVal;
		maxLoc_found = maxLoc;
	}
	
	// Box coordinate of the person
	Point start_coord = Point((int)(maxLoc_found.x), (int)(maxLoc_found.y));
	Point end_coord = Point((int)((maxLoc_found.x + PERSON_TMPLT.cols)), (int)((maxLoc_found.y + PERSON_TMPLT.rows)));

	// Create Mask
	Mat mask = Mat::zeros(img.size(), img.type());
	rectangle(mask, start_coord, end_coord, Scalar(255, 255, 255), CV_FILLED);

	Mat final_result;
	bitwise_and(img, mask, final_result);

	return final_result;
}

Mat hsv2gray(Mat img) {
	Mat img_BGR, result;
	cvtColor(img, img_BGR, CV_HSV2BGR);
	cvtColor(img, result, CV_BGR2GRAY);
	return result;
}

Mat mask_image(Mat input_image, Mat mask_image) {
	Mat hsv_channels[3], mask, result;
	split(mask_image, hsv_channels);
	threshold(hsv_channels[2], mask, 1, 1, THRESH_BINARY);
	bitwise_and(input_image, input_image, result, mask);

	return result;
}

MoveData find_person_in_img(Mat input_image, Mat original_image, Mat rgb_image, String description = "test") {
	int start_s = clock();
	// Apply filtering
	Mat color_filtered_image = get_person_with_color(input_image, description);
	imwrite(description + "_color_filtered.jpg", color_filtered_image);

	//waitKey(0);
	Mat tmplt_filtered_image = get_person_with_tmplt(color_filtered_image);

	// Get person location
	Moments person_moments = moments(hsv2gray(tmplt_filtered_image), true);
	MoveData result = {};
	result.x = (int)(person_moments.m10 / person_moments.m00);
	result.y = (int)(person_moments.m01 / person_moments.m00);
	result.area = (int)person_moments.m00;

	Mat final_image = mask_image(rgb_image, tmplt_filtered_image);
	// Plot location in image
	rectangle(final_image, Point((int)(result.x-5),
		(int)(result.y- 5)), Point((int)(result.x + 5), (int)(result.y + 5)), 
		Scalar(0, 255, 0), CV_FILLED);


	imwrite(description + "_final.jpg", final_image);
	return result;
}

Mat k_means(Mat input_image, int clusterCount) {
	//step 1 : map the src to the samples
	Mat samples(input_image.total(), 3, CV_32F);
	auto samples_ptr = samples.ptr<float>(0);
	for (int row = 0; row != input_image.rows; ++row) {
		auto src_begin = input_image.ptr<uchar>(row);
		auto src_end = src_begin + input_image.cols * input_image.channels();
		//auto samples_ptr = samples.ptr<float>(row * src.cols);
		while (src_begin != src_end) {
			samples_ptr[0] = src_begin[0];
			samples_ptr[1] = src_begin[1];
			samples_ptr[2] = src_begin[2];
			samples_ptr += 3; src_begin += 3;
		}
	}

	//step 2 : apply kmeans to find labels and centers
	Mat labels;
	int attempts = 5;
	Mat centers;
	kmeans(samples, clusterCount, labels,
		TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS,
			10, 0.01),
		attempts, KMEANS_PP_CENTERS, centers);

	//step 3 : map the centers to the output
	Mat new_image(input_image.size(), input_image.type());
	for (int row = 0; row != input_image.rows; ++row) {
		auto new_image_begin = new_image.ptr<uchar>(row);
		auto new_image_end = new_image_begin + new_image.cols * 3;
		auto labels_ptr = labels.ptr<int>(row * input_image.cols);

		while (new_image_begin != new_image_end) {
			int const cluster_idx = *labels_ptr;
			auto centers_ptr = centers.ptr<float>(cluster_idx);
			new_image_begin[0] = centers_ptr[0];
			new_image_begin[1] = centers_ptr[1];
			new_image_begin[2] = centers_ptr[2];
			new_image_begin += 3; ++labels_ptr;
		}
	}
	return new_image;
}

Mat sharpen_image(Mat input_image) {
	Mat frame, output;
	GaussianBlur(input_image, frame, Size(7, 7), 0);
	addWeighted(frame, -1, input_image, 2, 0, output);
	return output;
}

int get_hue_avg(vector<int> h_values) {
	int result = h_values[0];
	for (int i = 1; i < std::min(4, (int)h_values.size()); i++) {
		int value = h_values[i];
		if (value < result + 11 && value > result - 11) {
			result = (result * i + value) / (i + 1);
		}
	}
	return result;
}

Mat take_picture(VideoCapture cap) {
	Mat streamImg;
	while (1) {
		cap >> streamImg;
		imshow("cam", streamImg);
		char k = waitKey(1);
		if (k == 'a') {
			break;
		}
	}
	destroyWindow("cam");
	return streamImg;
}

int main(int argc, char** argv)
{
	// Continuously Take picture for testing
	/*VideoCapture cap(0);

	//cap.set(CV_CAP_PROP_FRAME_WIDTH,364);
	//cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
	if (!cap.isOpened()) {
		return -1;
	}
	else {
		printf("Camera is open /n ");
	}
	
	int i = 218;
	while (1) {
		Mat img = take_picture(cap);
		//imshow("result", img);

		//char k = waitKey(0);
		//if (k == 's') {
			imwrite("out/img_" + to_string(i) + ".jpg", img);
			i++;
		//}
		//else if (k == 'c') {
			continue;
		//}
		//destroyWindow("result");
	}*/

	/* Part 1 */
	
	// Take picture against green background
	Mat original_image = imread("out/david_1.jpg"); //take_picture(cap);
	original_image = sharpen_image(original_image);
	Mat person_original = load_hsv_image(original_image);
	
	// Filter Green Background
	vector<Mat> person_histograms = get_hsv_histogram(person_original, "dc_out/david_tmplt");
	std::vector<int> person_original_h_pv = get_peak_values(person_histograms[0]);
	int green_value = 70;
	for (int i = 0; i < person_original_h_pv.size(); i++) {
		if (person_original_h_pv[i] > 45 && person_original_h_pv[i] < 85) {
			green_value = person_original_h_pv[i];
			break;
		}
	}
	
	Mat person = filter_green_background(resize_img(person_original, 100), green_value);

	// Getting Person Template
	person_histograms = get_hsv_histogram(person, "filtered_person_0");
	Mat person_segmented = k_means(person, 2);
	person_histograms = get_hsv_histogram(person, "dc_v1/filtered_david");
	
	person_original_h_pv = get_peak_values(person_histograms[0]);

	HUE_AVG = get_hue_avg(person_original_h_pv);
	cout << "Hue avg:" << HUE_AVG << endl;

	// For person template
	PERSON_TMPLT = get_person_with_color(person_segmented);

	/* Part 2 */
	ofstream fout("dc_v2/result.txt");
	for (int i = 1; i < 2004; i+=3) {
		int start_2 = clock();
		Mat rgb_image = resize_img(imread("dc/img_" + to_string(i) + ".jpg"), 250);
		Mat image = sharpen_image(rgb_image);
		image = load_hsv_image(image, 250);

		Mat new_image = k_means(image, 3);
		//get_hsv_histogram(image, "dc_out/img_" + to_string(i));
		//imwrite("dc_out/segmented_image_"+to_string(i)+".jpg", new_image);

		//waitKey(0);
		MoveData move_data = find_person_in_img(new_image, image, rgb_image, "dc_v2/image_"+to_string(i));
		
		fout << "Segmented_image_" + to_string(i) << endl;
		fout << "Curr. Hue Avg: " << HUE_AVG << endl;
		fout << "Area: " << move_data.area << ", x, y:" << move_data.x << ", " << move_data.y << endl;
		int stop_2 = clock();
		fout << "time: " << double(stop_2 - start_2) / double(CLOCKS_PER_SEC) * 1000 << " ms"<< endl << endl;
	}
	fout.close();

	return 0;
}
