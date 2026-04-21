#ifndef RECONSTRUCTIONPIPELINE_H
#define RECONSTRUCTIONPIPELINE_H

#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <qstring>
#include <vector>

class ReconstructionPipeline
{
public:
    ReconstructionPipeline();
    ~ReconstructionPipeline();

    // Đầu vào: danh sách đường dẫn ảnh
    void setImages(const std::vector<QString> &imagePaths);

    // Chạy toàn bộ pipeline: feature, matching, pose, triangulation
    bool reconstruct();

    // Lấy kết quả point cloud (dạng vector<cv::Point3f>)
    std::vector<cv::Point3f> getPointCloud() const;
    std::vector<cv::Vec3b> getPointColors() const; // màu từ ảnh

private:
    std::vector<cv::Mat> images;
    std::vector<std::vector<cv::KeyPoint>> keypoints;
    std::vector<cv::Mat> descriptors;
    cv::Ptr<cv::Feature2D> detector;    // SIFT hoặc ORB

    // Thông tin camera (giả sử đã biết nội tại)
    cv::Mat K;      // ma trận intrinsic 3x3
    cv::Mat distCoeffs;

    // Kết quả
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Vec3b> colors;

    // Các hàm nội bộ
    void extractFeatures(int idx);
    void matchFeatures(int idx1, int idx2, std::vector<cv::DMatch> &goodMatches);
    bool estimatePose(const std::vector<cv::Point2f> &pts1,
                      const std::vector<cv::Point2f> &pts2,
                      cv::Mat &R, cv::Mat &t);
    void triangulatePoints(const cv::Mat &P1, const cv::Mat &P2,
                           const std::vector<cv::Point2f> &pts1,
                           const std::vector<cv::Point2f> &pts2,
                           std::vector<cv::Point3f> &points3D);
};

#endif // RECONSTRUCTION_PIPELINE_H
