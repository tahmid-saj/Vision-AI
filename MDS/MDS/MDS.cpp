#include <iostream>

#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\video\tracking.hpp"

int main()
{
	cv::VideoCapture video(0);

	if (!video.isOpened()) return -1;

	cv::Mat frame;

	int frameWidth = video.get(cv::CAP_PROP_FRAME_WIDTH);
	int frameHeigth = video.get(cv::CAP_PROP_FRAME_HEIGHT);
	cv::VideoWriter output("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(frameWidth, frameHeigth));
	cv::Ptr<cv::Tracker> tracker = cv::TrackerKCF::create();
	video.read(frame);
	cv::Rect2d trackingBox = cv::selectROI(frame, false);
	tracker->init(frame, trackingBox);

	while (video.read(frame)) {
		if (tracker->update(frame, trackingBox)) {
			cv::rectangle(frame, trackingBox, cv::Scalar(255, 0, 0), 2, 8);
		}

		cv::imshow("Video feed", frame);

		output.write(frame);

		if (cv::waitKey(25) >= 0) break;

	}

	output.release();
	video.release();

	cv::destroyAllWindows();

	return 0;
}