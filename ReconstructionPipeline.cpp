#include "reconstructionpipeline.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QDebug>

ReconstructionPipeline::ReconstructionPipeline()
{
    detector = cv::SIFT::create();

    // Giả sử camera đã được calibrate, thông số phù hợp với ảnh 640x480
    // Bạn có thể điều chỉnh theo kích thước ảnh thực tế
    double fx = 600.0, fy = 600.0, cx = 320.0, cy = 240.0;
    K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    distCoeffs = cv::Mat::zeros(4,1,CV_64F);
}

ReconstructionPipeline::~ReconstructionPipeline() {}

void ReconstructionPipeline::setImages(const std::vector<QString> &imagePaths)
{
    images.clear();
    for (const auto &path : imagePaths) {
        // Đọc ảnh màu (3 kênh BGR)
        cv::Mat img = cv::imread(path.toStdString(), cv::IMREAD_COLOR);
        if (!img.empty()) {
            images.push_back(img);
        } else {
            qWarning() << "Cannot load image:" << path;
        }
    }
    qDebug() << "Loaded" << images.size() << "images";
}

bool ReconstructionPipeline::reconstruct()
{
    if (images.size() < 2) {
        qWarning() << "Cần ít nhất 2 ảnh!";
        return false;
    }

    // Bước 1: Trích xuất feature cho tất cả ảnh
    keypoints.resize(images.size());
    descriptors.resize(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        extractFeatures(i);
        qDebug() << "Image" << i << "keypoints:" << keypoints[i].size();
    }

    // Bước 2: Chỉ làm việc với cặp ảnh đầu tiên (0 và 1)
    std::vector<cv::DMatch> matches;
    matchFeatures(0, 1, matches);
    qDebug() << "Matches between 0 and 1:" << matches.size();

    if (matches.size() < 50) {
        qWarning() << "Không đủ matches giữa 2 ảnh đầu!";
        return false;
    }

    // Lấy tọa độ điểm 2D tương ứng
    std::vector<cv::Point2f> pts0, pts1;
    for (const auto &m : matches) {
        pts0.push_back(keypoints[0][m.queryIdx].pt);
        pts1.push_back(keypoints[1][m.trainIdx].pt);
    }

    // Bước 3: Ước lượng pose (R, t)
    cv::Mat R, t;
    if (!estimatePose(pts0, pts1, R, t)) {
        qWarning() << "Không thể ước lượng pose!";
        return false;
    }

    // Bước 4: Tạo ma trận projection
    cv::Mat P0 = K * cv::Mat::eye(3,4,CV_64F);          // ảnh đầu: [I|0]
    cv::Mat RT;
    cv::hconcat(R, t, RT);
    cv::Mat P1 = K * RT;

    // Bước 5: Triangulation
    triangulatePoints(P0, P1, pts0, pts1, points3D);
    qDebug() << "Triangulated points:" << points3D.size();

    // Bước 6: Gán màu từ ảnh thứ nhất (dùng keypoints của ảnh 0)
    colors.clear();
    for (const auto &kp : keypoints[0]) {
        int x = cvRound(kp.pt.x);
        int y = cvRound(kp.pt.y);
        if (x >= 0 && y >= 0 && x < images[0].cols && y < images[0].rows) {
            cv::Vec3b bgr = images[0].at<cv::Vec3b>(y, x);
            colors.push_back(bgr);
        } else {
            // Nếu keypoint nằm ngoài ảnh, gán màu xám
            colors.push_back(cv::Vec3b(128, 128, 128));
        }
    }
    qDebug() << "Colors assigned:" << colors.size();

    // Đảm bảo số lượng điểm 3D và màu bằng nhau (chỉ lấy số điểm tối thiểu)
    size_t minSize = std::min(points3D.size(), colors.size());
    points3D.resize(minSize);
    colors.resize(minSize);

    return true;
}

void ReconstructionPipeline::extractFeatures(int idx)
{
    cv::Mat gray;
    cv::cvtColor(images[idx], gray, cv::COLOR_BGR2GRAY);
    detector->detectAndCompute(gray, cv::noArray(), keypoints[idx], descriptors[idx]);
}

void ReconstructionPipeline::matchFeatures(int idx1, int idx2, std::vector<cv::DMatch> &goodMatches)
{
    // Chuyển descriptor về CV_32F nếu cần (FlannBasedMatcher yêu cầu)
    if (descriptors[idx1].type() != CV_32F)
        descriptors[idx1].convertTo(descriptors[idx1], CV_32F);
    if (descriptors[idx2].type() != CV_32F)
        descriptors[idx2].convertTo(descriptors[idx2], CV_32F);

    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(descriptors[idx1], descriptors[idx2], knnMatches, 2);

    for (const auto &knn : knnMatches) {
        if (knn.size() == 2 && knn[0].distance < 0.75 * knn[1].distance) {
            goodMatches.push_back(knn[0]);
        }
    }
}

bool ReconstructionPipeline::estimatePose(const std::vector<cv::Point2f> &pts1,
                                          const std::vector<cv::Point2f> &pts2,
                                          cv::Mat &R, cv::Mat &t)
{
    if (pts1.size() < 8 || pts2.size() < 8) return false;
    cv::Mat E, mask;
    E = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, mask);
    if (E.empty()) return false;
    int inliers = cv::recoverPose(E, pts1, pts2, K, R, t, mask);
    return (inliers > 20);
}

void ReconstructionPipeline::triangulatePoints(const cv::Mat &P0, const cv::Mat &P1,
                                               const std::vector<cv::Point2f> &pts0,
                                               const std::vector<cv::Point2f> &pts1,
                                               std::vector<cv::Point3f> &points3D)
{
    cv::Mat points4D;
    cv::triangulatePoints(P0, P1, pts0, pts1, points4D);
    points3D.clear();

    // Đảm bảo points4D là CV_64F
    if (points4D.type() != CV_64F) {
        points4D.convertTo(points4D, CV_64F);
    }

    for (int i = 0; i < points4D.cols; ++i) {
        double w = points4D.at<double>(3, i);
        if (w != 0.0) {
            double x = points4D.at<double>(0, i) / w;
            double y = points4D.at<double>(1, i) / w;
            double z = points4D.at<double>(2, i) / w;
            points3D.push_back(cv::Point3f(static_cast<float>(x),
                                           static_cast<float>(y),
                                           static_cast<float>(z)));
        }
    }
}

std::vector<cv::Point3f> ReconstructionPipeline::getPointCloud() const { return points3D; }
std::vector<cv::Vec3b> ReconstructionPipeline::getPointColors() const { return colors; }
