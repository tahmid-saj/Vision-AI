#include <iostream>

#include "opencv\opencv.hpp"
#include "opencv\highgui.hpp"
#include "opencv\tracking.hpp"

void ResizeBoxes(cv::Rect& box) {
	box.x += cvRound(box.width * 0.1);
	box.width = cvRound(box.width * 0.8);
	box.y += cvRound(box.height * 0.06);
	box.height = cvRound(box.height * 0.8);
}


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

	cv::HOGDescriptor hog;
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

	std::vector<cv::Rect> detections;
	hog.detectMultiScale(frame, detections, 0, cv::Size(8, 8), cv::Size(32, 32), 1.2, 2);

	for (auto& detection : detections) {
		ResizeBoxes(detection);
		multiTracker->add(cv::TrackerKCF::create(), frame, detection);
	}

	int frameNumber = 1;
	while (video.read(frame)) {
		frameNumber++;

		if (frameNumber % 15 == 0) {
			detections.clear();
			hog.detectMultiScale(frame, detections, 0, cv::Size(8, 8), cv::Size(32, 32), 1.2, 2);
			cv::Ptr<cv::MultiTracker> multiTrackerTemp = cv::MultiTracker::create();
			multiTracker = multiTrackerTemp;
			for (auto& detection : detections) {
				ResizeBoxes(detection);
				multiTracker->add(cv::TrackerKCF::create(), frame, detection);
			}
		}
		else {
			multiTracker->update(frame);
		}

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