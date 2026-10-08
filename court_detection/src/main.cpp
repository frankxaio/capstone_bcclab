#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdexcept>
#include "TimeMeasurement.h"
#include "CourtLinePixelDetector.h"
#include "CourtLineCandidateDetector.h"
#include "BadmintonCourtFitter.h"

// Headless CLI matching the exercise pipeline's Windows detector arguments.
int main(int argc, char** argv) {
  if (argc < 4 || argc > 5) {
    std::cerr << "Usage: court-detection VIDEO_OR_IMAGE COURT_TXT OVERLAY_PNG [FRAME_INDEX]\n";
    return 1;
  }
  try {
    TimeMeasurement::debug = false;
    CourtLinePixelDetector::debug = false;
    CourtLineCandidateDetector::debug = false;
    BadmintonCourtFitter::debug = false;
    cv::Mat frame;
    if (argc == 4) frame = cv::imread(argv[1]);
    if (frame.empty()) {
      cv::VideoCapture video(argv[1]);
      if (!video.isOpened()) throw std::runtime_error("Cannot open input video");
      const int count = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));
      int index = count / 2;
      if (argc == 5) {
        size_t consumed = 0;
        index = std::stoi(argv[4], &consumed);
        if (consumed != std::string(argv[4]).size()) throw std::runtime_error("Invalid frame index");
      }
      if (index < 0 || index >= count) throw std::runtime_error("Frame index outside video");
      video.set(cv::CAP_PROP_POS_FRAMES, index);
      if (!video.read(frame)) throw std::runtime_error("Cannot decode selected frame");
    }
    CourtLinePixelDetector pixels;
    CourtLineCandidateDetector lines;
    BadmintonCourtFitter fitter;
    cv::Mat binary = pixels.run(frame);
    auto candidates = lines.run(binary, frame);
    if (candidates.size() < 4) throw std::runtime_error("Too few court line candidates");
    auto model = fitter.run(candidates, binary, frame);
    model.writeToFile(argv[2]);
    model.drawModel(frame);
    if (!cv::imwrite(argv[3], frame)) throw std::runtime_error("Cannot write overlay image");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Court detection failed: " << error.what() << '\n';
    return 2;
  }
}
