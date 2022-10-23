#include <iostream>

#include "opencv2\opencv.hpp"
#include "opencv2\highgui.hpp"
#include "opencv2\tracking.hpp"

int main()
{
	cv::VideoCapture video("video.mp4");

	if (!video.isOpened()) return -1;

	cv::Mat frame;

	int frameWidth = video.get(cv::CAP_PROP_FRAME_WIDTH);
	int frameHeigth = video.get(cv::CAP_PROP_FRAME_HEIGHT);

	cv::VideoWriter output("output.avi",
		cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
		30,
		cv::Size(frameWidth, frameHeigth));

	video.read(frame);
	cv::Ptr<cv::MultiTracker> multiTracker = cv::MultiTracker::create();
	std::vector<cv::Rect> boundingBoxes;
	cv::selectROIs("Video feed", frame, boundingBoxes, false);

	if (boundingBoxes.size() < 1) return 0;

	for (const auto& boundingBox : boundingBoxes) {
		multiTracker->add(cv::TrackerKCF::create(), frame, boundingBox);
	}

	while (video.read(frame)) {

		multiTracker->update(frame);
		for (const auto& object : multiTracker->getObjects()) {
			cv::rectangle(frame, object, cv::Scalar(255, 0, 0), 2, 8);
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