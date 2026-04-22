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

// ─────────────────────────────────────────────
// Constructor / Destructor
// ─────────────────────────────────────────────
ReconstructionPipeline::ReconstructionPipeline()
{
    detector = cv::SIFT::create(0, 3, 0.04, 10, 1.6);

    // Fallback intrinsics – chỉ dùng khi KHÔNG có file params
    // (với TempleRing thực tế là fx=1520.4, fy=1525.9, cx=302.32, cy=246.87)
    K_fallback = (cv::Mat_<double>(3,3)
                      << 1520.4, 0.0,    302.32,
                  0.0,    1525.9, 246.87,
                  0.0,    0.0,    1.0);
    distCoeffs = cv::Mat::zeros(4, 1, CV_64F);
}

ReconstructionPipeline::~ReconstructionPipeline() {}

// ─────────────────────────────────────────────
// loadCameraParams
// Đọc file templeR_par.txt (Middlebury format):
//   Dòng 1: số lượng ảnh (47)
//   Mỗi dòng tiếp: imageName k11 k12 ... k33 r11 ... r33 t1 t2 t3
//   Tổng 22 token mỗi dòng dữ liệu
// ─────────────────────────────────────────────
bool ReconstructionPipeline::loadCameraParams(const QString &paramsFilePath)
{
    QFile file(paramsFilePath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qWarning() << "Không mở được file params:" << paramsFilePath;
        return false;
    }

    camParams.clear();
    QTextStream in(&file);

    // Dòng đầu tiên: số lượng ảnh
    QString firstLine = in.readLine().trimmed();
    bool ok = false;
    int numImages = firstLine.toInt(&ok);
    if (!ok) {
        qWarning() << "Dòng đầu file params không phải số nguyên:" << firstLine;
        file.close();
        return false;
    }
    qDebug() << "File params khai báo" << numImages << "camera(s)";

    // Đọc từng dòng camera
    while (!in.atEnd()) {
        QString line = in.readLine().trimmed();
        if (line.isEmpty()) continue;

        // Tách theo khoảng trắng
        QStringList tok = line.split(QRegularExpression("\\s+"),
                                     Qt::SkipEmptyParts);
        // Cần đúng 22 token: 1 tên + 9 K + 9 R + 3 t
        if (tok.size() < 22) {
            qWarning() << "Dòng không đủ 22 token, bỏ qua:" << line.left(60);
            continue;
        }

        CameraParams cp;
        cp.imageName = tok[0];

        // --- K (3×3) ---
        cp.K = cv::Mat(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                cp.K.at<double>(r, c) = tok[1 + r*3 + c].toDouble();

        // --- R (3×3) ---
        cp.R = cv::Mat(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                cp.R.at<double>(r, c) = tok[10 + r*3 + c].toDouble();

        // --- t (3×1) ---
        cp.t = cv::Mat(3, 1, CV_64F);
        for (int r = 0; r < 3; ++r)
            cp.t.at<double>(r, 0) = tok[19 + r].toDouble();

        // --- P = K * [R | t] (tính sẵn, dùng nhiều lần) ---
        cv::Mat RT;
        cv::hconcat(cp.R, cp.t, RT);   // 3×4
        cp.P = cp.K * RT;              // 3×4

        camParams.push_back(cp);
    }

    file.close();

    if ((int)camParams.size() != numImages) {
        qWarning() << "Số camera đọc được" << camParams.size()
            << "≠ khai báo" << numImages;
    }

    hasGroundTruthParams = !camParams.empty();
    qDebug() << "Đã load" << camParams.size() << "camera params thành công";

    // In thông số camera đầu tiên để kiểm tra
    if (!camParams.empty()) {
        const auto &c0 = camParams[0];
        qDebug() << "Camera[0]:" << c0.imageName;
        qDebug() << "  K: fx=" << c0.K.at<double>(0,0)
                 << "fy="      << c0.K.at<double>(1,1)
                 << "cx="      << c0.K.at<double>(0,2)
                 << "cy="      << c0.K.at<double>(1,2);
    }

    return true;
}

// ─────────────────────────────────────────────
// setImages
// ─────────────────────────────────────────────
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

// ─────────────────────────────────────────────
// reconstruct
// ─────────────────────────────────────────────
bool ReconstructionPipeline::reconstruct()
{
    if (images.size() < 2) {
        qWarning() << "Cần ít nhất 2 ảnh!";
        return false;
    }

    points3D.clear();
    colors.clear();

    // Bước 1: Trích xuất features cho TẤT CẢ ảnh
    keypoints.resize(images.size());
    descriptors.resize(images.size());
    for (size_t i = 0; i < images.size(); ++i) {
        extractFeatures(i);
        qDebug() << "Image" << i << "keypoints:" << keypoints[i].size();
    }

    // ═══════════════════════════════════════════════
    // CASE A: Có ground-truth params từ *_par.txt
    //   → Dùng trực tiếp P = K*[R|t] đã tính sẵn
    //   → Không bị drift, scale chính xác
    // ═══════════════════════════════════════════════
    if (hasGroundTruthParams && camParams.size() >= images.size())
    {
        qDebug() << "=== Chế độ: GROUND-TRUTH camera params ===";

        // Bounding box của TempleRing (từ README):
        //   min: (-0.023121, -0.038009, -0.091940)
        //   max: ( 0.078626,  0.121636, -0.017395)
        // Dùng để lọc outlier nằm ngoài bbox mở rộng 3×
        const float xMin = -0.10f, xMax = 0.20f;
        const float yMin = -0.15f, yMax = 0.25f;
        const float zMin = -0.30f, zMax = 0.10f;

        for (size_t i = 0; i + 1 < images.size(); ++i)
        {
            // --- Match features ---
            std::vector<cv::DMatch> matches;
            matchFeatures(i, i + 1, matches);

            if ((int)matches.size() < 30) {
                qWarning() << "Cặp" << i << "-" << (i+1)
                           << ": chỉ" << matches.size()
                           << "matches → bỏ qua";
                continue;
            }

            // --- Lấy điểm 2D tương ứng ---
            std::vector<cv::Point2f> ptsA, ptsB;
            ptsA.reserve(matches.size());
            ptsB.reserve(matches.size());
            for (const auto &m : matches) {
                ptsA.push_back(keypoints[i  ][m.queryIdx].pt);
                ptsB.push_back(keypoints[i+1][m.trainIdx].pt);
            }

            // --- Triangulation dùng P đã tính sẵn ---
            std::vector<cv::Point3f> newPts;
            doTriangulate(camParams[i].P, camParams[i+1].P,
                          ptsA, ptsB, newPts);

            // --- Lọc điểm hợp lệ ---
            int kept = 0;
            for (size_t j = 0; j < newPts.size() && j < matches.size(); ++j)
            {
                const cv::Point3f &pt = newPts[j];

                // 1) Điểm phải nằm trong bbox mở rộng
                if (pt.x < xMin || pt.x > xMax) continue;
                if (pt.y < yMin || pt.y > yMax) continue;
                if (pt.z < zMin || pt.z > zMax) continue;

                // 2) Điểm phải nằm TRƯỚC cả 2 camera (depth > 0)
                //    depth = R[2,:] · pt + t[2]
                cv::Mat pw = (cv::Mat_<double>(3,1) << pt.x, pt.y, pt.z);
                // Ép MatExpr → cv::Mat trước khi dùng .at<>()
                cv::Mat depth_i_mat = cv::Mat(camParams[i  ].R.row(2)) * pw;
                cv::Mat depth_j_mat = cv::Mat(camParams[i+1].R.row(2)) * pw;
                double z_i = depth_i_mat.at<double>(0,0) + camParams[i  ].t.at<double>(2);
                double z_j = depth_j_mat.at<double>(0,0) + camParams[i+1].t.at<double>(2);
                if (z_i <= 0 || z_j <= 0) continue;

                // 3) Gán màu từ keypoint tương ứng trên ảnh i
                cv::Point2f kpPt = keypoints[i][matches[j].queryIdx].pt;
                int x = cvRound(kpPt.x);
                int y = cvRound(kpPt.y);
                cv::Vec3b color(128, 128, 128);
                if (x >= 0 && y >= 0 && x < images[i].cols && y < images[i].rows)
                    color = images[i].at<cv::Vec3b>(y, x);

                points3D.push_back(pt);
                colors.push_back(color);
                ++kept;
            }

            qDebug() << "Cặp" << i << "-" << (i+1)
                     << ": matches=" << matches.size()
                     << "  triangulated=" << newPts.size()
                     << "  kept=" << kept;
        }
    }
    // ═══════════════════════════════════════════════
    // CASE B: Không có params → ước lượng pose
    // ═══════════════════════════════════════════════
    else
    {
        qDebug() << "=== Chế độ: ESTIMATED pose (không có file params) ===";
        qWarning() << "Khuyến nghị: đặt file templeR_par.txt cùng thư mục ảnh";

        cv::Mat R_accum = cv::Mat::eye(3, 3, CV_64F);
        cv::Mat t_accum = cv::Mat::zeros(3, 1, CV_64F);

        for (size_t i = 0; i + 1 < images.size(); ++i)
        {
            std::vector<cv::DMatch> matches;
            matchFeatures(i, i + 1, matches);

            if ((int)matches.size() < 100) {
                qWarning() << "Cặp" << i << "-" << (i+1)
                           << ": không đủ matches (" << matches.size() << ")";
                continue;
            }

            std::vector<cv::Point2f> ptsA, ptsB;
            for (const auto &m : matches) {
                ptsA.push_back(keypoints[i  ][m.queryIdx].pt);
                ptsB.push_back(keypoints[i+1][m.trainIdx].pt);
            }

            cv::Mat R_rel, t_rel;
            if (!estimatePoseFromMatches(ptsA, ptsB, R_rel, t_rel)) {
                qWarning() << "Không estimate được pose cặp" << i << "-" << (i+1);
                continue;
            }

            cv::Mat R_global = R_accum * R_rel;
            cv::Mat t_global = R_accum * t_rel + t_accum;

            cv::Mat RT_i, RT_next;
            cv::hconcat(R_accum,  t_accum,  RT_i);
            cv::hconcat(R_global, t_global, RT_next);
            cv::Mat P_i    = K_fallback * RT_i;
            cv::Mat P_next = K_fallback * RT_next;

            std::vector<cv::Point3f> newPts;
            doTriangulate(P_i, P_next, ptsA, ptsB, newPts);

            int kept = 0;
            for (size_t j = 0; j < newPts.size() && j < matches.size(); ++j) {
                if (newPts[j].z <= 0 || newPts[j].z > 50.f) continue;
                points3D.push_back(newPts[j]);
                cv::Point2f kpPt = keypoints[i][matches[j].queryIdx].pt;
                int x = cvRound(kpPt.x), y = cvRound(kpPt.y);
                if (x >= 0 && y >= 0 && x < images[i].cols && y < images[i].rows)
                    colors.push_back(images[i].at<cv::Vec3b>(y, x));
                else
                    colors.push_back(cv::Vec3b(128, 128, 128));
                ++kept;
            }

            qDebug() << "Cặp" << i << "-" << (i+1)
                     << ": kept=" << kept;

            R_accum = R_global.clone();
            t_accum = t_global.clone();
        }
    }

    qDebug() << "=== Tổng điểm 3D:" << points3D.size() << "===";
    return !points3D.empty();
}

// ─────────────────────────────────────────────
// extractFeatures
// ─────────────────────────────────────────────
void ReconstructionPipeline::extractFeatures(int idx)
{
    cv::Mat gray;
    cv::cvtColor(images[idx], gray, cv::COLOR_BGR2GRAY);
    detector->detectAndCompute(gray, cv::noArray(),
                               keypoints[idx], descriptors[idx]);
}

// ─────────────────────────────────────────────
// matchFeatures  (Lowe's ratio test 0.75)
// ─────────────────────────────────────────────
void ReconstructionPipeline::matchFeatures(int idx1, int idx2,
                                           std::vector<cv::DMatch> &goodMatches)
{
    // SIFT descriptors đã là CV_32F, nhưng convert phòng hờ
    cv::Mat d1 = descriptors[idx1], d2 = descriptors[idx2];
    if (d1.type() != CV_32F) d1.convertTo(d1, CV_32F);
    if (d2.type() != CV_32F) d2.convertTo(d2, CV_32F);

    cv::FlannBasedMatcher matcher;
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(d1, d2, knnMatches, 2);

    goodMatches.clear();
    for (const auto &knn : knnMatches)
        if (knn.size() == 2 && knn[0].distance < 0.75f * knn[1].distance)
            goodMatches.push_back(knn[0]);
}

// ─────────────────────────────────────────────
// estimatePoseFromMatches  (dùng cho CASE B)
// ─────────────────────────────────────────────
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

// ─────────────────────────────────────────────
// doTriangulate
// ─────────────────────────────────────────────
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
        if (std::abs(w) < 1e-9) continue;  // tránh chia cho 0
        outPts.push_back(cv::Point3f(
            (float)(pts4D.at<double>(0, i) / w),
            (float)(pts4D.at<double>(1, i) / w),
            (float)(pts4D.at<double>(2, i) / w)
            ));
    }
}

// ─────────────────────────────────────────────
// Getters
// ─────────────────────────────────────────────
std::vector<cv::Point3f> ReconstructionPipeline::getPointCloud() const
{ return points3D; }

std::vector<cv::Vec3b> ReconstructionPipeline::getPointColors() const
{ return colors; }
