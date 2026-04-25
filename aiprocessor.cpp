#include "aiprocessor.h"
#include <QDebug>

AIProcessor::AIProcessor() : isDetModelLoaded(false), isSegModelLoaded(false) {}

AIProcessor::~AIProcessor() {}

bool AIProcessor::loadDetectionModel(const QString& modelPath) {
    try {
        detNet = cv::dnn::readNetFromONNX(modelPath.toStdString());
        isDetModelLoaded = !detNet.empty();
        if(isDetModelLoaded) {
            detNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            detNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            qDebug() << "Detection model loaded successfully:" << modelPath;
        } else {
            qWarning() << "Failed to load detection model:" << modelPath;
        }
        return isDetModelLoaded;
    } catch (const cv::Exception& e) {
        qWarning() << "OpenCV Exception during loading detection model:" << e.what();
        isDetModelLoaded = false;
        return false;
    }
}

bool AIProcessor::loadSegmentationModel(const QString& modelPath) {
    try {
        segNet = cv::dnn::readNetFromONNX(modelPath.toStdString());
        isSegModelLoaded = !segNet.empty();
        if(isSegModelLoaded) {
            segNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            segNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            qDebug() << "Segmentation model loaded successfully:" << modelPath;
        } else {
            qWarning() << "Failed to load segmentation model:" << modelPath;
        }
        return isSegModelLoaded;
    } catch (const cv::Exception& e) {
        qWarning() << "OpenCV Exception during loading segmentation model:" << e.what();
        isSegModelLoaded = false;
        return false;
    }
}

cv::Mat AIProcessor::runObjectDetection(const cv::Mat& inputImage) {
    if (!isDetModelLoaded || inputImage.empty()) {
        qWarning() << "Detection model not loaded or input image is empty.";
        return inputImage;
    }

    const float SCORE_THRESHOLD = 0.5f;
    const float NMS_THRESHOLD = 0.45f;
    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;

    cv::Mat blob;
    cv::dnn::blobFromImage(inputImage, blob, 1.0/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    detNet.setInput(blob);

    std::vector<cv::Mat> outputs;
    detNet.forward(outputs, detNet.getUnconnectedOutLayersNames());

    cv::Mat outMat = outputs[0];
    if(outMat.dims == 3) {
        outMat = cv::Mat(outMat.size[1], outMat.size[2], CV_32F, outMat.ptr<float>());
    }
    outMat = outMat.t();

    int num_preds = outMat.rows;
    int num_classes = outMat.cols - 4;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float x_factor = inputImage.cols / (float)INPUT_WIDTH;
    float y_factor = inputImage.rows / (float)INPUT_HEIGHT;

    for (int i = 0; i < num_preds; ++i) {
        float* data = outMat.ptr<float>(i);
        
        float max_score = -1.f;
        int class_id = -1;
        for (int j = 0; j < num_classes; ++j) {
            if (data[4 + j] > max_score) {
                max_score = data[4 + j];
                class_id = j;
            }
        }

        if (max_score >= SCORE_THRESHOLD) {
            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((cx - 0.5f * w) * x_factor);
            int top = int((cy - 0.5f * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            classIds.push_back(class_id);
            confidences.push_back(max_score);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    cv::Mat resultImage = inputImage.clone();
    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        cv::rectangle(resultImage, box, cv::Scalar(0, 255, 0), 2);
        
        QString label = QString("ID %1: %2%").arg(classIds[idx]).arg(int(confidences[idx] * 100));
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label.toStdString(), cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        cv::rectangle(resultImage, cv::Point(box.x, box.y - labelSize.height - baseLine),
                      cv::Point(box.x + labelSize.width, box.y), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(resultImage, label.toStdString(), cv::Point(box.x, box.y - baseLine),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    return resultImage;
}

cv::Mat AIProcessor::runSegmentation(const cv::Mat& inputImage) {
    if (!isSegModelLoaded || inputImage.empty()) {
        qWarning() << "Segmentation model not loaded or input image is empty.";
        return inputImage;
    }

    const float SCORE_THRESHOLD = 0.5f;
    const float NMS_THRESHOLD = 0.45f;
    const float MASK_THRESHOLD = 0.5f;
    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;

    cv::Mat blob;
    cv::dnn::blobFromImage(inputImage, blob, 1.0/255.0, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    segNet.setInput(blob);

    std::vector<cv::Mat> outputs;
    segNet.forward(outputs, segNet.getUnconnectedOutLayersNames());

    cv::Mat boxes_mat, proto_mat;
    if (outputs[0].dims == 4) {
        proto_mat = outputs[0];
        boxes_mat = outputs[1];
    } else {
        proto_mat = outputs[1];
        boxes_mat = outputs[0];
    }

    if(boxes_mat.dims == 3) {
        boxes_mat = cv::Mat(boxes_mat.size[1], boxes_mat.size[2], CV_32F, boxes_mat.ptr<float>());
    }
    boxes_mat = boxes_mat.t();

    int num_preds = boxes_mat.rows;
    int mask_coeffs_count = proto_mat.size[1];
    int num_classes = boxes_mat.cols - 4 - mask_coeffs_count;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> masks_coeffs_list;

    float x_factor = inputImage.cols / (float)INPUT_WIDTH;
    float y_factor = inputImage.rows / (float)INPUT_HEIGHT;

    for (int i = 0; i < num_preds; ++i) {
        float* data = boxes_mat.ptr<float>(i);
        
        float max_score = -1.f;
        int class_id = -1;
        for (int j = 0; j < num_classes; ++j) {
            if (data[4 + j] > max_score) {
                max_score = data[4 + j];
                class_id = j;
            }
        }

        if (max_score >= SCORE_THRESHOLD) {
            float cx = data[0];
            float cy = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((cx - 0.5f * w) * x_factor);
            int top = int((cy - 0.5f * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            classIds.push_back(class_id);
            confidences.push_back(max_score);
            boxes.push_back(cv::Rect(left, top, width, height));

            std::vector<float> coeffs(data + 4 + num_classes, data + 4 + num_classes + mask_coeffs_count);
            masks_coeffs_list.push_back(coeffs);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    cv::Mat resultImage = inputImage.clone();
    cv::Mat color_mask = cv::Mat::zeros(resultImage.size(), CV_8UC3);

    int proto_h = proto_mat.size[2];
    int proto_w = proto_mat.size[3];
    cv::Mat proto(mask_coeffs_count, proto_h * proto_w, CV_32F, proto_mat.ptr<float>());

    for (int idx : indices) {
        cv::Rect box = boxes[idx];
        
        cv::Mat coeffs_mat(1, mask_coeffs_count, CV_32F, masks_coeffs_list[idx].data());
        cv::Mat mask_mat = coeffs_mat * proto;
        mask_mat = mask_mat.reshape(1, proto_h);

        cv::exp(-mask_mat, mask_mat);
        mask_mat = 1.0f / (1.0f + mask_mat);

        cv::resize(mask_mat, mask_mat, resultImage.size());
        cv::Mat binary_mask = mask_mat > MASK_THRESHOLD;

        cv::Rect img_rect(0, 0, resultImage.cols, resultImage.rows);
        cv::Rect valid_box = box & img_rect;

        if (valid_box.area() > 0) {
            cv::Mat cropped_mask = cv::Mat::zeros(binary_mask.size(), CV_8U);
            binary_mask(valid_box).copyTo(cropped_mask(valid_box));
            
            color_mask.setTo(cv::Scalar(0, 0, 255), cropped_mask);
        }
        
        cv::rectangle(resultImage, box, cv::Scalar(0, 0, 255), 2);
    }
    
    cv::addWeighted(resultImage, 1.0, color_mask, 0.5, 0, resultImage);

    return resultImage;
}
