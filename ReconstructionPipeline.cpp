#include "reconstructionpipeline.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <QDebug>
#include <QFile>
#include <QTextStream>
#include <QFileInfo>
#include <QRegularExpression>
#include <cmath>
#include <set>
#include <algorithm>

// ------------------------------------------------------------------
// Constructor / Destructor
// ------------------------------------------------------------------
ReconstructionPipeline::ReconstructionPipeline()
{
    detector = cv::SIFT::create(0, 3, 0.04, 10, 1.6);
    K_fallback = (cv::Mat_<double>(3,3)
                      << 1520.4, 0.0,    302.32,
                  0.0,    1525.9, 246.87,
                  0.0,    0.0,    1.0);
    distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
}

ReconstructionPipeline::~ReconstructionPipeline() {}

// ------------------------------------------------------------------
// loadCameraParams (giữ nguyên)
// ------------------------------------------------------------------
bool ReconstructionPipeline::loadCameraParams(const QString &paramsFilePath)
{
    QFile file(paramsFilePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Không mở được file params:" << paramsFilePath;
        return false;
    }
    camParams.clear();
    QTextStream in(&file);
    QString firstLine = in.readLine().trimmed();
    bool ok = false;
    int numImages = firstLine.toInt(&ok);
    if (!ok) {
        qWarning() << "Dòng đầu file params không phải số nguyên:" << firstLine;
        file.close();
        return false;
    }
    qDebug() << "File params khai báo" << numImages << "camera(s)";
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty()) continue;
        QStringList tok = line.split(QRegularExpression("\\s+"), Qt::SkipEmptyParts);
        if (tok.size() < 22) {
            qWarning() << "Dòng không đủ 22 token, bỏ qua:" << line.left(60);
            continue;
        }
        CameraParams cp;
        cp.imageName = tok[0];
        cp.K = cv::Mat(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                cp.K.at<double>(r, c) = tok[1 + r*3 + c].toDouble();
        cp.R = cv::Mat(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                cp.R.at<double>(r, c) = tok[10 + r*3 + c].toDouble();
        cp.t = cv::Mat(3, 1, CV_64F);
        for (int r = 0; r < 3; ++r)
            cp.t.at<double>(r, 0) = tok[19 + r].toDouble();
        cv::Mat RT;
        cv::hconcat(cp.R, cp.t, RT);
        cp.P = cp.K * RT;
        camParams.push_back(cp);
    }
    file.close();
    if ((int)camParams.size() != numImages) {
        qWarning() << "Số camera đọc được" << camParams.size()
            << "≠ khai báo" << numImages;
    }
    hasGroundTruthParams = !camParams.empty();
    qDebug() << "Đã load" << camParams.size() << "camera params thành công";
    if (!camParams.empty()) {
        const auto &c0 = camParams[0];
        qDebug() << "Camera[0]:" << c0.imageName;
        qDebug() << "  K: fx=" << c0.K.at<double>(0,0)
                 << "fy=" << c0.K.at<double>(1,1)
                 << "cx=" << c0.K.at<double>(0,2)
                 << "cy=" << c0.K.at<double>(1,2);
    }
    return true;
}

// ------------------------------------------------------------------
// setImages
// ------------------------------------------------------------------
void ReconstructionPipeline::setImages(const std::vector<QString> &imagePaths)
{
    images.clear();
    for (const auto &path : imagePaths) {
        cv::Mat img = cv::imread(path.toStdString(), cv::IMREAD_COLOR);
        if (!img.empty())
            images.push_back(img);
        else
            qWarning() << "Không đọc được ảnh:" << path;
    }
    qDebug() << "Loaded" << images.size() << "images";
}

// ------------------------------------------------------------------
// extractFeatures
// ------------------------------------------------------------------
void ReconstructionPipeline::extractFeatures(int idx)
{
    cv::Mat gray;
    cv::cvtColor(images[idx], gray, cv::COLOR_BGR2GRAY);
    detector->detectAndCompute(gray, cv::noArray(),
                               keypoints[idx], descriptors[idx]);
}

// ------------------------------------------------------------------
// matchFeatures (Lowe's ratio test 0.75 + cross-check)
// ------------------------------------------------------------------
void ReconstructionPipeline::matchFeatures(int idx1, int idx2,
                                           std::vector<cv::DMatch> &goodMatches)
{
    cv::Mat d1 = descriptors[idx1], d2 = descriptors[idx2];
    if (d1.type() != CV_32F) d1.convertTo(d1, CV_32F);
    if (d2.type() != CV_32F) d2.convertTo(d2, CV_32F);
    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knn1, knn2;
    matcher.knnMatch(d1, d2, knn1, 2);
    matcher.knnMatch(d2, d1, knn2, 2);
    std::set<int> goodIndices1, goodIndices2;
    for (size_t i = 0; i < knn1.size(); ++i) {
        if (knn1[i].size() == 2 && knn1[i][0].distance < 0.75f * knn1[i][1].distance) {
            goodIndices1.insert(i);
        }
    }
    for (size_t i = 0; i < knn2.size(); ++i) {
        if (knn2[i].size() == 2 && knn2[i][0].distance < 0.75f * knn2[i][1].distance) {
            goodIndices2.insert(knn2[i][0].trainIdx);
        }
    }
    goodMatches.clear();
    for (size_t i = 0; i < knn1.size(); ++i) {
        if (goodIndices1.count(i) && goodIndices2.count(knn1[i][0].trainIdx)) {
            goodMatches.push_back(knn1[i][0]);
        }
    }
}

// ------------------------------------------------------------------
// estimatePoseFromMatches (dùng essential matrix)
// ------------------------------------------------------------------
bool ReconstructionPipeline::estimatePoseFromMatches(
    const std::vector<cv::Point2f> &pts1,
    const std::vector<cv::Point2f> &pts2,
    cv::Mat &R, cv::Mat &t)
{
    if ((int)pts1.size() < 8) return false;
    cv::Mat mask;
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K_fallback,
                                     cv::RANSAC, 0.999, 1.0, mask);
    if (E.empty()) return false;
    int inliers = cv::recoverPose(E, pts1, pts2, K_fallback, R, t, mask);
    return (inliers > 20);
}

// ------------------------------------------------------------------
// doTriangulate
// ------------------------------------------------------------------
void ReconstructionPipeline::doTriangulate(
    const cv::Mat &P0, const cv::Mat &P1,
    const std::vector<cv::Point2f> &pts0,
    const std::vector<cv::Point2f> &pts1,
    std::vector<cv::Point3f> &outPts)
{
    cv::Mat pts4D;
    cv::triangulatePoints(P0, P1, pts0, pts1, pts4D);
    outPts.clear();
    outPts.reserve(pts4D.cols);
    if (pts4D.type() != CV_64F)
        pts4D.convertTo(pts4D, CV_64F);
    for (int i = 0; i < pts4D.cols; ++i) {
        double w = pts4D.at<double>(3, i);
        if (std::abs(w) < 1e-9) continue;
        outPts.push_back(cv::Point3f(
            (float)(pts4D.at<double>(0, i) / w),
            (float)(pts4D.at<double>(1, i) / w),
            (float)(pts4D.at<double>(2, i) / w)));
    }
}

// ------------------------------------------------------------------
// computeReprojectionError
// ------------------------------------------------------------------
double ReconstructionPipeline::computeReprojectionError(
    const cv::Mat &P, const cv::Point3f &pt3d, const cv::Point2f &pt2d)
{
    cv::Mat pt4d = (cv::Mat_<double>(4,1) << pt3d.x, pt3d.y, pt3d.z, 1.0);
    cv::Mat proj = P * pt4d;
    double inv_w = 1.0 / proj.at<double>(2);
    double u = proj.at<double>(0) * inv_w;
    double v = proj.at<double>(1) * inv_w;
    double dx = u - pt2d.x;
    double dy = v - pt2d.y;
    return std::sqrt(dx*dx + dy*dy);
}

// ------------------------------------------------------------------
// filterOutliersByDensity
// ------------------------------------------------------------------
void ReconstructionPipeline::filterOutliersByDensity(float radius, int minNeighbors)
{
    if (points3D.empty()) return;
    std::vector<bool> keep(points3D.size(), false);
    for (size_t i = 0; i < points3D.size(); ++i) {
        int neighbors = 0;
        for (size_t j = 0; j < points3D.size(); ++j) {
            if (i == j) continue;
            float dx = points3D[i].x - points3D[j].x;
            float dy = points3D[i].y - points3D[j].y;
            float dz = points3D[i].z - points3D[j].z;
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist < radius) ++neighbors;
        }
        if (neighbors >= minNeighbors) keep[i] = true;
    }
    std::vector<cv::Point3f> newPts;
    std::vector<cv::Vec3b> newColors;
    for (size_t i = 0; i < points3D.size(); ++i) {
        if (keep[i]) {
            newPts.push_back(points3D[i]);
            newColors.push_back(colors[i]);
        }
    }
    qDebug() << "Lọc density: giữ lại" << newPts.size() << "/" << points3D.size();
    points3D.swap(newPts);
    colors.swap(newColors);
}

// ------------------------------------------------------------------
// reconstructWithGroundTruth
// ------------------------------------------------------------------
bool ReconstructionPipeline::reconstructWithGroundTruth()
{
    qDebug() << "=== Chế độ: GROUND-TRUTH camera params ===";
    const float xMin = -0.10f, xMax = 0.20f;
    const float yMin = -0.15f, yMax = 0.25f;
    const float zMin = -0.30f, zMax = 0.10f;

    for (size_t i = 0; i < images.size(); ++i) {
        for (size_t j = i+1; j < images.size(); ++j) {
            std::vector<cv::DMatch> matches;
            matchFeatures(i, j, matches);
            if (matches.size() < 50) continue;

            std::vector<cv::Point2f> pts_i, pts_j;
            pts_i.reserve(matches.size());
            pts_j.reserve(matches.size());
            for (const auto &m : matches) {
                pts_i.push_back(keypoints[i][m.queryIdx].pt);
                pts_j.push_back(keypoints[j][m.trainIdx].pt);
            }

            std::vector<cv::Point3f> newPts;
            doTriangulate(camParams[i].P, camParams[j].P, pts_i, pts_j, newPts);

            int kept = 0;
            for (size_t k = 0; k < newPts.size() && k < matches.size(); ++k) {
                const cv::Point3f &pt = newPts[k];
                if (pt.x < xMin || pt.x > xMax) continue;
                if (pt.y < yMin || pt.y > yMax) continue;
                if (pt.z < zMin || pt.z > zMax) continue;

                cv::Mat pw = (cv::Mat_<double>(3,1) << pt.x, pt.y, pt.z);
                cv::Mat depth_i = cv::Mat(camParams[i].R.row(2)) * pw;
                cv::Mat depth_j = cv::Mat(camParams[j].R.row(2)) * pw;
                double z_i = depth_i.at<double>(0,0) + camParams[i].t.at<double>(2);
                double z_j = depth_j.at<double>(0,0) + camParams[j].t.at<double>(2);
                if (z_i <= 0 || z_j <= 0) continue;

                double err_i = computeReprojectionError(camParams[i].P, pt, pts_i[k]);
                double err_j = computeReprojectionError(camParams[j].P, pt, pts_j[k]);
                if (err_i > 2.0 || err_j > 2.0) continue;

                cv::Point2f kpPt = keypoints[i][matches[k].queryIdx].pt;
                int x = cvRound(kpPt.x), y = cvRound(kpPt.y);
                cv::Vec3b color(128,128,128);
                if (x>=0 && y>=0 && x<images[i].cols && y<images[i].rows)
                    color = images[i].at<cv::Vec3b>(y, x);

                points3D.push_back(pt);
                colors.push_back(color);
                ++kept;
            }
            qDebug() << "Cặp" << i << "-" << j << "matches=" << matches.size()
                     << "triangulated=" << newPts.size() << "kept=" << kept;
        }
    }
    filterOutliersByDensity(0.02f, 3);
    qDebug() << "=== Tổng điểm 3D (ground truth):" << points3D.size() << "===";
    return !points3D.empty();
}

// ------------------------------------------------------------------
// reconstructWithEstimatedPose
// ------------------------------------------------------------------
bool ReconstructionPipeline::reconstructWithEstimatedPose()
{
    qDebug() << "=== Chế độ: ESTIMATED pose ===";
    // Chọn cặp cơ sở tốt nhất
    int best_i = 0, best_j = 1;
    size_t bestMatches = 0;
    for (size_t i = 0; i < images.size(); ++i) {
        for (size_t j = i+1; j < images.size(); ++j) {
            std::vector<cv::DMatch> tmp;
            matchFeatures(i, j, tmp);
            if (tmp.size() > bestMatches) {
                bestMatches = tmp.size();
                best_i = i;
                best_j = j;
            }
        }
    }
    if (bestMatches < 100) {
        qWarning() << "Không tìm thấy cặp ảnh cơ sở đủ tốt!";
        return false;
    }
    qDebug() << "Cặp cơ sở:" << best_i << "-" << best_j << "matches=" << bestMatches;

    std::vector<cv::DMatch> baseMatches;
    matchFeatures(best_i, best_j, baseMatches);
    std::vector<cv::Point2f> pts_i, pts_j;
    for (const auto &m : baseMatches) {
        pts_i.push_back(keypoints[best_i][m.queryIdx].pt);
        pts_j.push_back(keypoints[best_j][m.trainIdx].pt);
    }

    cv::Mat R, t;
    if (!estimatePoseFromMatches(pts_i, pts_j, R, t)) {
        qWarning() << "Không thể estimate pose cho cặp cơ sở";
        return false;
    }

    cv::Mat P0 = K_fallback * cv::Mat::eye(3,4,CV_64F);
    cv::Mat RT;
    cv::hconcat(R, t, RT);
    cv::Mat P1 = K_fallback * RT;

    std::vector<cv::Point3f> basePoints3D;
    doTriangulate(P0, P1, pts_i, pts_j, basePoints3D);

    struct Point3DCorr {
        cv::Point3f pt;
        int idx_i, idx_j;
    };
    std::vector<Point3DCorr> baseCorr;
    for (size_t k = 0; k < basePoints3D.size() && k < baseMatches.size(); ++k) {
        if (basePoints3D[k].z > 0.01f && basePoints3D[k].z < 10.0f) {
            baseCorr.push_back({basePoints3D[k], baseMatches[k].queryIdx, baseMatches[k].trainIdx});
        }
    }
    qDebug() << "Base triangulated points (filtered):" << baseCorr.size();

    points3D.clear();
    colors.clear();
    for (const auto &corr : baseCorr) {
        points3D.push_back(corr.pt);
        cv::Point2f kpPt = keypoints[best_i][corr.idx_i].pt;
        int x = cvRound(kpPt.x), y = cvRound(kpPt.y);
        cv::Vec3b color(128,128,128);
        if (x>=0 && y>=0 && x<images[best_i].cols && y<images[best_i].rows)
            color = images[best_i].at<cv::Vec3b>(y, x);
        colors.push_back(color);
    }

    std::vector<cv::Mat> poses_R, poses_t;
    std::vector<int> pose_img_indices;
    poses_R.push_back(cv::Mat::eye(3,3,CV_64F));
    poses_t.push_back(cv::Mat::zeros(3,1,CV_64F));
    pose_img_indices.push_back(best_i);
    poses_R.push_back(R.clone());
    poses_t.push_back(t.clone());
    pose_img_indices.push_back(best_j);

    // Bổ sung các ảnh còn lại
    for (size_t idx = 0; idx < images.size(); ++idx) {
        if (idx == best_i || idx == best_j) continue;
        int bestRef = -1;
        size_t maxMatches = 0;
        for (size_t r = 0; r < pose_img_indices.size(); ++r) {
            std::vector<cv::DMatch> tmp;
            matchFeatures(idx, pose_img_indices[r], tmp);
            if (tmp.size() > maxMatches) {
                maxMatches = tmp.size();
                bestRef = r;
            }
        }
        if (bestRef == -1 || maxMatches < 50) {
            qDebug() << "Bỏ qua ảnh" << idx << "do không đủ matches";
            continue;
        }

        std::vector<cv::DMatch> matches;
        matchFeatures(idx, pose_img_indices[bestRef], matches);
        std::vector<cv::Point2f> pts_cur, pts_ref;
        for (const auto &m : matches) {
            pts_cur.push_back(keypoints[idx][m.queryIdx].pt);
            pts_ref.push_back(keypoints[pose_img_indices[bestRef]][m.trainIdx].pt);
        }

        cv::Mat R_rel, t_rel;
        if (!estimatePoseFromMatches(pts_cur, pts_ref, R_rel, t_rel)) {
            qDebug() << "Không thể estimate pose cho ảnh" << idx;
            continue;
        }

        cv::Mat R_abs = poses_R[bestRef] * R_rel;
        cv::Mat t_abs = poses_R[bestRef] * t_rel + poses_t[bestRef];

        poses_R.push_back(R_abs.clone());
        poses_t.push_back(t_abs.clone());
        pose_img_indices.push_back(idx);

        cv::Mat RT_ref, RT_cur;
        cv::hconcat(poses_R[bestRef], poses_t[bestRef], RT_ref);
        cv::hconcat(R_abs, t_abs, RT_cur);
        cv::Mat P_ref = K_fallback * RT_ref;
        cv::Mat P_cur = K_fallback * RT_cur;

        std::vector<cv::Point3f> newPts;
        doTriangulate(P_ref, P_cur, pts_ref, pts_cur, newPts);

        int added = 0;
        for (size_t k = 0; k < newPts.size() && k < matches.size(); ++k) {
            if (newPts[k].z <= 0 || newPts[k].z > 10.0f) continue;
            double err_ref = computeReprojectionError(P_ref, newPts[k], pts_ref[k]);
            double err_cur = computeReprojectionError(P_cur, newPts[k], pts_cur[k]);
            if (err_ref > 2.0 || err_cur > 2.0) continue;

            points3D.push_back(newPts[k]);
            cv::Point2f kpPt = keypoints[idx][matches[k].queryIdx].pt;
            int x = cvRound(kpPt.x), y = cvRound(kpPt.y);
            cv::Vec3b color(128,128,128);
            if (x>=0 && y>=0 && x<images[idx].cols && y<images[idx].rows)
                color = images[idx].at<cv::Vec3b>(y, x);
            colors.push_back(color);
            ++added;
        }
        qDebug() << "Ảnh" << idx << "thêm" << added << "điểm mới";
    }

    filterOutliersByDensity(0.02f, 3);
    qDebug() << "=== Tổng điểm 3D (estimated):" << points3D.size() << "===";
    return !points3D.empty();
}

PointCloudT::Ptr ReconstructionPipeline::convertToPCLCloud(const std::vector<cv::Point3f>& pts,
                                   const std::vector<cv::Vec3b>& cols) const {
    PointCloudT::Ptr cloud(new PointCloudT);
    cloud->reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        PointT p;
        p.x = pts[i].x;
        p.y = pts[i].y;
        p.z = pts[i].z;
        // Gán màu (OpenCV BGR -> PCL RGB)
        p.b = cols[i][0];
        p.g = cols[i][1];
        p.r = cols[i][2];
        cloud->push_back(p);
    }
    return cloud;
}

void ReconstructionPipeline::convertFromPCLCloud(PointCloudT::Ptr cloud,
                         std::vector<cv::Point3f>& pts,
                         std::vector<cv::Vec3b>& cols) {
    pts.clear();
    cols.clear();
    pts.reserve(cloud->size());
    cols.reserve(cloud->size());
    for (const auto& p : cloud->points) {
        pts.emplace_back(p.x, p.y, p.z);
        cols.emplace_back(p.b, p.g, p.r);
    }
}
// ------------------------------------------------------------------
// reconstruct (public)
// ------------------------------------------------------------------
bool ReconstructionPipeline::reconstruct()
{
    if (images.size() < 2) {
        qWarning() << "Cần ít nhất 2 ảnh!";
        return false;
    }

    points3D.clear();
    colors.clear();

    keypoints.resize(images.size());
    descriptors.resize(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        extractFeatures(i);
        qDebug() << "Image" << i << "keypoints:" << keypoints[i].size();
    }

    if (hasGroundTruthParams && camParams.size() >= images.size())
       /* return*/ reconstructWithGroundTruth();
    else
        /*return */reconstructWithEstimatedPose();

    // === CẢI THIỆN CHẤT LƯỢNG ĐÁM MÂY ĐIỂM ===
    qDebug() << "Bắt đầu hậu xử lý point cloud...";

    // filterOutliersByDensity(0.02f, 3);

    // 1. Lọc nhiễu thống kê (xóa điểm có khoảng cách lân cận bất thường)
    statisticalOutlierFilter(50, 1.0);

    // 2. Lọc theo bán kính (xóa điểm cô lập)
    radiusOutlierFilter(0.02, 3);

    // 3. Giảm mẫu bằng voxel grid (tùy chọn, làm đều mật độ)
    voxelGridDownsample(0.005);

    qDebug() << "Point cloud sau xử lý:" << points3D.size() << "điểm";
    return true;
}

// ------------------------------------------------------------------
// Getters
// ------------------------------------------------------------------
std::vector<cv::Point3f> ReconstructionPipeline::getPointCloud() const
{ return points3D; }

std::vector<cv::Vec3b> ReconstructionPipeline::getPointColors() const
{ return colors; }

// reconstructionpipeline.cpp

void ReconstructionPipeline::statisticalOutlierFilter(float meanK, float stdDevMulThresh) {
    if (points3D.empty()) return;
    int K = (int)meanK;
    size_t n = points3D.size();
    if (n < (size_t)K + 1) return;

    // Tính khoảng cách trung bình đến K láng giềng gần nhất cho mỗi điểm
    std::vector<double> meanDists(n, 0.0);

    for (size_t i = 0; i < n; ++i) {
        // Thu thập khoảng cách đến tất cả điểm khác
        std::vector<double> dists;
        dists.reserve(n - 1);
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            double dx = points3D[i].x - points3D[j].x;
            double dy = points3D[i].y - points3D[j].y;
            double dz = points3D[i].z - points3D[j].z;
            dists.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
        }
        // Lấy K láng giềng gần nhất
        std::partial_sort(dists.begin(),
                          dists.begin() + std::min(K, (int)dists.size()),
                          dists.end());
        double sum = 0.0;
        int cnt = std::min(K, (int)dists.size());
        for (int k = 0; k < cnt; ++k) sum += dists[k];
        meanDists[i] = (cnt > 0) ? sum / cnt : 0.0;
    }

    // Tính mean và stddev của meanDists
    double globalMean = 0.0;
    for (double d : meanDists) globalMean += d;
    globalMean /= (double)n;

    double variance = 0.0;
    for (double d : meanDists) {
        double diff = d - globalMean;
        variance += diff * diff;
    }
    double stdDev = std::sqrt(variance / (double)n);
    double threshold = globalMean + stdDevMulThresh * stdDev;

    // Giữ lại điểm có meanDist < threshold
    std::vector<cv::Point3f> newPts;
    std::vector<cv::Vec3b>   newCols;
    for (size_t i = 0; i < n; ++i) {
        if (meanDists[i] < threshold) {
            newPts.push_back(points3D[i]);
            newCols.push_back(colors[i]);
        }
    }
    qDebug() << "SOR: giữ lại" << newPts.size() << "/" << n << "điểm";
    points3D.swap(newPts);
    colors.swap(newCols);
}

void ReconstructionPipeline::radiusOutlierFilter(float radius, int minNeighbors) {
    if (points3D.empty()) return;
    size_t n = points3D.size();
    std::vector<cv::Point3f> newPts;
    std::vector<cv::Vec3b>   newCols;

    for (size_t i = 0; i < n; ++i) {
        int neighbors = 0;
        for (size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            float dx = points3D[i].x - points3D[j].x;
            float dy = points3D[i].y - points3D[j].y;
            float dz = points3D[i].z - points3D[j].z;
            if (std::sqrt(dx*dx + dy*dy + dz*dz) < radius) {
                ++neighbors;
                if (neighbors >= minNeighbors) break; // early exit
            }
        }
        if (neighbors >= minNeighbors) {
            newPts.push_back(points3D[i]);
            newCols.push_back(colors[i]);
        }
    }
    qDebug() << "ROR: giữ lại" << newPts.size() << "/" << n << "điểm";
    points3D.swap(newPts);
    colors.swap(newCols);
}

void ReconstructionPipeline::voxelGridDownsample(float leafSize) {
    if (points3D.empty()) return;

    // Tìm bounding box
    float xMin = points3D[0].x, xMax = xMin;
    float yMin = points3D[0].y, yMax = yMin;
    float zMin = points3D[0].z, zMax = zMin;
    for (const auto &p : points3D) {
        xMin = std::min(xMin, p.x); xMax = std::max(xMax, p.x);
        yMin = std::min(yMin, p.y); yMax = std::max(yMax, p.y);
        zMin = std::min(zMin, p.z); zMax = std::max(zMax, p.z);
    }

    // Gán mỗi điểm vào voxel
    // key = (ix, iy, iz), value = (tổng xyz, tổng màu, số điểm)
    struct VoxelData {
        double sx = 0, sy = 0, sz = 0;
        double sr = 0, sg = 0, sb = 0;
        int count = 0;
    };
    auto voxelKey = [&](const cv::Point3f &p) -> std::tuple<int,int,int> {
        int ix = (int)std::floor((p.x - xMin) / leafSize);
        int iy = (int)std::floor((p.y - yMin) / leafSize);
        int iz = (int)std::floor((p.z - zMin) / leafSize);
        return {ix, iy, iz};
    };

    std::map<std::tuple<int,int,int>, VoxelData> voxelMap;
    for (size_t i = 0; i < points3D.size(); ++i) {
        auto key = voxelKey(points3D[i]);
        auto &vd = voxelMap[key];
        vd.sx += points3D[i].x; vd.sy += points3D[i].y; vd.sz += points3D[i].z;
        vd.sr += colors[i][2];  vd.sg += colors[i][1];  vd.sb += colors[i][0];
        vd.count++;
    }

    std::vector<cv::Point3f> newPts;
    std::vector<cv::Vec3b>   newCols;
    newPts.reserve(voxelMap.size());
    newCols.reserve(voxelMap.size());
    for (const auto &kv : voxelMap) {
        const VoxelData &vd = kv.second;
        double inv = 1.0 / vd.count;
        newPts.push_back(cv::Point3f(
            (float)(vd.sx * inv),
            (float)(vd.sy * inv),
            (float)(vd.sz * inv)));
        newCols.push_back(cv::Vec3b(
            (uchar)(vd.sb * inv),  // B
            (uchar)(vd.sg * inv),  // G
            (uchar)(vd.sr * inv))); // R
    }
    qDebug() << "VoxelGrid: giữ lại" << newPts.size() << "/" << points3D.size() << "điểm";
    points3D.swap(newPts);
    colors.swap(newCols);
}

// // Lọc Statistical Outlier Removal (SOR)
// void ReconstructionPipeline::statisticalOutlierFilter(float meanK, float stdDevMulThresh) {
//     if (points3D.empty()) return;
//     PointCloudT::Ptr cloud = convertToPCLCloud(points3D, colors);
//     PointCloudT::Ptr filtered(new PointCloudT);  // ← Output riêng

//     pcl::StatisticalOutlierRemoval<PointT> sor;
//     sor.setInputCloud(cloud);
//     sor.setMeanK(meanK);
//     sor.setStddevMulThresh(stdDevMulThresh);
//     sor.filter(*filtered);  // ← filter vào biến mới

//     convertFromPCLCloud(filtered, points3D, colors);
//     qDebug() << "SOR: giữ lại" << points3D.size() << "điểm";
// }

// // Lọc Radius Outlier Removal (ROR)
// void ReconstructionPipeline::radiusOutlierFilter(float radius, int minNeighbors) {
//     if (points3D.empty()) return;
//     PointCloudT::Ptr cloud = convertToPCLCloud(points3D, colors);
//     PointCloudT::Ptr filtered(new PointCloudT);  // ← Output riêng

//     pcl::RadiusOutlierRemoval<PointT> ror;
//     ror.setInputCloud(cloud);
//     ror.setRadiusSearch(radius);
//     ror.setMinNeighborsInRadius(minNeighbors);
//     ror.filter(*filtered);  // ← filter vào biến mới

//     convertFromPCLCloud(filtered, points3D, colors);
//     qDebug() << "ROR: giữ lại" << points3D.size() << "điểm";
// }

// // Giảm mẫu bằng Voxel Grid
// void ReconstructionPipeline::voxelGridDownsample(float leafSize) {
//     if (points3D.empty()) return;
//     PointCloudT::Ptr cloud = convertToPCLCloud(points3D, colors);
//     PointCloudT::Ptr filtered(new PointCloudT);  // ← Output riêng biệt

//     pcl::VoxelGrid<PointT> vg;
//     vg.setInputCloud(cloud);
//     vg.setLeafSize(leafSize, leafSize, leafSize);
//     vg.filter(*filtered);  // ← filter vào biến mới, không overwrite input

//     convertFromPCLCloud(filtered, points3D, colors);
//     qDebug() << "VoxelGrid: giữ lại" << points3D.size() << "điểm";
// }
