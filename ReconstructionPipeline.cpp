#include "reconstructionpipeline.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QDebug>

ReconstructionPipeline::ReconstructionPipeline()
{
    detector = cv::SIFT::create();

    double fx = 600.0, fy = 600.0, cx = 320.0, cy = 240.0;
    K = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    distCoeffs = cv::Mat::zeros(4,1,CV_64F);
}

ReconstructionPipeline::~ReconstructionPipeline() {}

void ReconstructionPipeline::setImages(const std::vector<QString> &imagePaths)
{
    images.clear();
    for (const auto &path : imagePaths) {
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
    if (images.size() < 2) return false;

    keypoints.resize(images.size());
    descriptors.resize(images.size());

    for (size_t i = 0; i < images.size(); i++) {
        extractFeatures(i);
        qDebug() << "Image" << i << "keypoints:" << keypoints[i].size();
    }

    points3D.clear();
    colors.clear();

    // ================= INIT =================
    std::vector<cv::DMatch> matches;
    matchFeatures(0, 1, matches);

    if (matches.size() < 50) {
        qWarning() << "Init pair failed";
        return false;
    }

    std::vector<cv::Point2f> pts0, pts1;
    for (auto &m : matches) {
        pts0.push_back(keypoints[0][m.queryIdx].pt);
        pts1.push_back(keypoints[1][m.trainIdx].pt);
    }

    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(pts0, pts1, K, cv::RANSAC, 0.999, 0.5, mask);

    if (E.empty()) return false;

    cv::Mat R, t;
    int inliers = cv::recoverPose(E, pts0, pts1, K, R, t, mask);

    if (inliers < 30) return false;

    cv::Mat P0 = K * cv::Mat::eye(3,4,CV_64F);
    cv::Mat RT;
    cv::hconcat(R, t, RT);
    cv::Mat P1 = K * RT;

    std::vector<cv::Point2f> inlier0, inlier1;
    std::vector<cv::DMatch> inlierMatches;

    for (size_t i = 0; i < matches.size(); i++) {
        if (mask.at<uchar>(i)) {
            inlier0.push_back(pts0[i]);
            inlier1.push_back(pts1[i]);
            inlierMatches.push_back(matches[i]);
        }
    }

    std::vector<cv::Point3f> pts3d;
    triangulatePoints(P0, P1, inlier0, inlier1, pts3d);

    for (size_t i = 0; i < pts3d.size() && i < inlierMatches.size(); i++) {
        if (pts3d[i].z < 0 || pts3d[i].z > 50) continue;

        points3D.push_back(pts3d[i]);

        auto pt = keypoints[0][inlierMatches[i].queryIdx].pt;
        int x = cvRound(pt.x), y = cvRound(pt.y);

        colors.push_back(images[0].at<cv::Vec3b>(y,x));
    }

    qDebug() << "Initial cloud:" << points3D.size();

    // ================= INCREMENTAL =================
    cv::Mat prevR = R.clone();
    cv::Mat prevT = t.clone();

    for (size_t i = 2; i < images.size(); i++) {

        std::vector<cv::DMatch> matches;
        matchFeatures(i-1, i, matches);

        if (matches.size() < 50) {
            qDebug() << "Skip image" << i;
            continue;
        }

        std::vector<cv::Point2f> ptsPrev, ptsCurr;
        for (auto &m : matches) {
            ptsPrev.push_back(keypoints[i-1][m.queryIdx].pt);
            ptsCurr.push_back(keypoints[i][m.trainIdx].pt);
        }

        cv::Mat mask;
        cv::Mat E = cv::findEssentialMat(ptsPrev, ptsCurr, K, cv::RANSAC, 0.999, 0.5, mask);

        if (E.empty()) continue;

        cv::Mat R, t;
        int inliers = cv::recoverPose(E, ptsPrev, ptsCurr, K, R, t, mask);

        if (inliers < 30) {
            qDebug() << "Skip image" << i << "(low inliers)";
            continue;
        }

        // 👉 accumulate pose (VERY IMPORTANT)
        cv::Mat currR = prevR * R;
        cv::Mat currT = prevT + prevR * t;

        cv::Mat RT;
        cv::hconcat(currR, currT, RT);
        cv::Mat P_curr = K * RT;

        cv::Mat RT_prev;
        cv::hconcat(prevR, prevT, RT_prev);
        cv::Mat P_prev = K * RT_prev;

        std::vector<cv::Point2f> inlierPrev, inlierCurr;
        std::vector<cv::DMatch> inlierMatches;

        for (size_t k = 0; k < matches.size(); k++) {
            if (mask.at<uchar>(k)) {
                inlierPrev.push_back(ptsPrev[k]);
                inlierCurr.push_back(ptsCurr[k]);
                inlierMatches.push_back(matches[k]);
            }
        }

        std::vector<cv::Point3f> newPts;
        triangulatePoints(P_prev, P_curr, inlierPrev, inlierCurr, newPts);

        for (size_t k = 0; k < newPts.size() && k < inlierMatches.size(); k++) {

            if (newPts[k].z < 0 || newPts[k].z > 50)
                continue;

            points3D.push_back(newPts[k]);

            auto pt = keypoints[i][inlierMatches[k].trainIdx].pt;
            int x = cvRound(pt.x), y = cvRound(pt.y);

            colors.push_back(images[i].at<cv::Vec3b>(y,x));
        }

        prevR = currR;
        prevT = currT;

        qDebug() << "After image" << i << "cloud size:" << points3D.size();
    }

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

void ReconstructionPipeline::triangulatePoints(const cv::Mat &P0, const cv::Mat &P1,
                                               const std::vector<cv::Point2f> &pts0,
                                               const std::vector<cv::Point2f> &pts1,
                                               std::vector<cv::Point3f> &points3D)
{
    cv::Mat points4D;
    cv::triangulatePoints(P0, P1, pts0, pts1, points4D);
    points3D.clear();

    if (points4D.type() != CV_64F) {
        points4D.convertTo(points4D, CV_64F);
    }

    for (int i = 0; i < points4D.cols; ++i) {
        double w = points4D.at<double>(3, i);
        if (w != 0.0) {
            double x = points4D.at<double>(0, i) / w;
            double y = points4D.at<double>(1, i) / w;
            double z = points4D.at<double>(2, i) / w;

            // ✅ loại điểm phía sau camera
            if (z > 0) {
                points3D.push_back(cv::Point3f(
                    static_cast<float>(x),
                    static_cast<float>(y),
                    static_cast<float>(z)
                    ));
            }
        }
    }
}

std::vector<cv::Point3f> ReconstructionPipeline::getPointCloud() const { return points3D; }
std::vector<cv::Vec3b> ReconstructionPipeline::getPointColors() const { return colors; }
