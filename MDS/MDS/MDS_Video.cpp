#include <iostream>

#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\tracking.hpp"

int main()
{
	cv::VideoCapture video(0);

	if (!video.isOpened()) return -1;

	cv::Mat frame;

	int frameWidth = video.get(cv::CAP_PROP_FRAME_WIDTH);
	int frameHeigth = video.get(cv::CAP_PROP_FRAME_HEIGHT);

	cv::VideoWriter output("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(frameWidth, frameHeigth));

	while (video.read(frame)) {

		cv::imshow("Video feed", frame);

		output.write(frame);

		if (cv::waitKey(25) >= 0) break;

	}

	output.release();
	video.release();

	cv::destroyAllWindows();

	return 0;
}