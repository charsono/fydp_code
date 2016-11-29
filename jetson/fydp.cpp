#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;


Mat load_hsv_image(Mat image_bgr){
	Mat result;
        cvtColor(image_bgr, result, CV_BGR2HSV);
	return result;
}

Mat filter_green_background(Mat image_hsv){
	Mat mask, result;
	inRange(image_hsv, Scalar(60, 0, 0), Scalar(75, 255, 255), mask);
	mask = 255 - mask;
	bitwise_and(image_hsv, image_hsv, result, mask=mask);

	return result;
}


int main() {
   VideoCapture cap(0);

 //  cap.set(CV_CAP_PROP_FRAME_WIDTH,544);
 //  cap.set(CV_CAP_PROP_FRAME_HEIGHT, 288);
   if (!cap.isOpened()){
	printf("Fail to open camera /n");
	return -1;
    }
    
    Mat streamImg;
    while(1) {

	cap.read(streamImg);
	imshow("cam", streamImg);
	char k = waitKey(1);	
	if (k=='a'){
		break;	
	}
    }

    Mat hsv_img = load_hsv_image(streamImg);

    Vec3b hsv = hsv_img.at<Vec3b>(0, 0);
    cout << "H value " << (int)hsv.val[0] << endl;

    Mat person = filter_green_background(hsv_img);
    cap.release();
    imshow("Hello!", person);
    waitKey();
}
