#ifndef AIPROCESSOR_H
#define AIPROCESSOR_H

#include <QString>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>

struct AIResult {
    int class_id;
    float confidence;
    cv::Rect box;
    std::vector<float> mask_coeffs;
};

class AIProcessor {
public:
    AIProcessor();
    ~AIProcessor();

    bool loadDetectionModel(const QString& modelPath);
    bool loadSegmentationModel(const QString& modelPath);

    bool isDetectionModelLoaded() const { return isDetModelLoaded; }
    bool isSegmentationModelLoaded() const { return isSegModelLoaded; }

    // Returns image with drawn bounding boxes
    cv::Mat runObjectDetection(const cv::Mat& inputImage);
    
    // Returns image with drawn segmentation masks
    cv::Mat runSegmentation(const cv::Mat& inputImage);

private:
    cv::dnn::Net detNet;
    cv::dnn::Net segNet;
    bool isDetModelLoaded;
    bool isSegModelLoaded;
};

#endif // AIPROCESSOR_H
