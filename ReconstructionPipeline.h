#pragma once
#include <vector>
#include <QString>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>

struct CameraParams {
    QString  imageName;
    cv::Mat  K;   // 3x3 intrinsic
    cv::Mat  R;   // 3x3 rotation    (world → camera)
    cv::Mat  t;   // 3x1 translation (world → camera)
    cv::Mat  P;   // 3x4 projection  = K * [R | t]
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
    std::vector<cv::Vec3b>   getPointColors() const;

private:
    void extractFeatures(int idx);
    void matchFeatures(int idx1, int idx2, std::vector<cv::DMatch> &goodMatches);
    void doTriangulate(const cv::Mat &P0, const cv::Mat &P1,
                       const std::vector<cv::Point2f> &pts0,
                       const std::vector<cv::Point2f> &pts1,
                       std::vector<cv::Point3f> &outPts);

    // Xử lý một cặp ảnh (i, j) → thêm điểm vào points3D/colors
    void processImagePair(size_t i, size_t j);

    // Statistical Outlier Removal
    void removeOutliers(int kNeighbors, float stdDevMult);

    bool estimatePoseFromMatches(const std::vector<cv::Point2f> &pts1,
                                 const std::vector<cv::Point2f> &pts2,
                                 cv::Mat &R, cv::Mat &t);

    std::vector<cv::Mat>                   images;
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat>                   descriptors;
    std::vector<CameraParams>              camParams;
    bool                                   hasGroundTruthParams = false;

    cv::Ptr<cv::SIFT> detector;
    cv::Mat           K_fallback;
    cv::Mat           distCoeffs;

    std::vector<cv::Point3f> points3D;
    std::vector<cv::Vec3b>   colors;

    // Bounding box TempleRing mở rộng nhẹ
    static constexpr float xMin = -0.12f, xMax = 0.22f;
    static constexpr float yMin = -0.18f, yMax = 0.28f;
    static constexpr float zMin = -0.35f, zMax = 0.12f;
};
