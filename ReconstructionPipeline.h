#ifndef RECONSTRUCTION_PIPELINE_H
#define RECONSTRUCTION_PIPELINE_H

#include <vector>
#include <QString>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

struct CameraParams {
    QString imageName;
    cv::Mat K;  // 3x3
    cv::Mat R;  // 3x3
    cv::Mat t;  // 3x1
    cv::Mat P;  // 3x4 = K * [R|t]
};

class ReconstructionPipeline
{
public:
    ReconstructionPipeline();
    ~ReconstructionPipeline();

    void setImages(const std::vector<QString> &imagePaths);
    bool loadCameraParams(const QString &paramsFilePath);
    bool reconstruct();

    std::vector<cv::Point3f> getPointCloud() const;
    std::vector<cv::Vec3b> getPointColors() const;

private:
    // Dữ liệu
    std::vector<cv::Mat> images;
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;
    cv::Ptr<cv::Feature2D> detector;

    cv::Mat K_fallback;    // intrinsic khi không có params
    cv::Mat distCoeffs;    // (bỏ qua distortion)

    // Ground truth params
    std::vector<CameraParams> camParams;
    bool hasGroundTruthParams = false;

    // Kết quả
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Vec3b> colors;

    // Hàm nội bộ
    void extractFeatures(int idx);
    void matchFeatures(int idx1, int idx2, std::vector<cv::DMatch> &goodMatches);
    bool estimatePoseFromMatches(const std::vector<cv::Point2f> &pts1,
                                 const std::vector<cv::Point2f> &pts2,
                                 cv::Mat &R, cv::Mat &t);
    void doTriangulate(const cv::Mat &P0, const cv::Mat &P1,
                       const std::vector<cv::Point2f> &pts0,
                       const std::vector<cv::Point2f> &pts1,
                       std::vector<cv::Point3f> &outPts);
    double computeReprojectionError(const cv::Mat &P,
                                    const cv::Point3f &pt3d,
                                    const cv::Point2f &pt2d);
    void filterOutliersByDensity(float radius, int minNeighbors);

    // Hai phương thức reconstruct riêng
    bool reconstructWithGroundTruth();
    bool reconstructWithEstimatedPose();
};

#endif
