#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(
  const Mat &img_1, const Mat &img_2,
  std::vector<KeyPoint> &keypoints_1,
  std::vector<KeyPoint> &keypoints_2,
  std::vector<DMatch> &matches);

void pose_estimation_2d2d(
  const std::vector<KeyPoint> &keypoints_1,
  const std::vector<KeyPoint> &keypoints_2,
  const std::vector<DMatch> &matches,
  Mat &R, Mat &t);

void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points
);

inline cv::Scalar get_color(float depth) {
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    if (depth > up_th) depth = up_th;
    if (depth < low_th) depth = low_th;
    return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

Point2f pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {
    if (argc != 3) {
        cout << "usage: triangulation img1 img2" << endl;
        return 1;
    }

    Mat img_1 = imread(argv[1], 1);
    Mat img_2 = imread(argv[2], 1);

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    vector<Point3d> points;
    triangulation(keypoints_1, keypoints_2, matches, R, t, points);

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    Mat img1_plot = img_1.clone();
    Mat img2_plot = img_2.clone();

    for (int i = 0; i < matches.size(); i++) {
        float depth1 = points[i].z;   // points就是重建后的3D点
        std::cout << "depth: " << depth1 << std::endl;
        cv::Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);
        Point2f pixel1 = keypoints_1[matches[i].queryIdx].pt;

        Point2f residual1;//图像1残差
        residual1.x = pt1_cam.x - pixel1.x;
        residual1.y = pt1_cam.y - pixel1.y;
        cout << "图像1像素点的深度为" << depth1 << "平移单位 "<< endl;
        cout<<"图像1的残差为: " << residual1.x << ", " << residual1.y << ")像素单位 " << endl;


        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t; // 做变换的不应该是图像1中的像素坐标吗
        float depth2 = pt2_trans.at<double>(2, 0);
        cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
        Point2f pixel2 = keypoints_2[matches[i].trainIdx].pt;
        Point2f residual2;//图像2残差
        residual2.x = pt2_trans.at<double>(0, 0) - pixel2.x;
        residual2.y = pt2_trans.at<double>(1, 0) - pixel2.y;
        cout << "图像2像素点的深度为" << depth2 << "平移单位 "<< endl;
        cout<<"图像2的残差为: " << residual2.x << ", " << residual2.y << ")像素单位 " << endl;
    }
    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey();

    return 0;
}

void find_feature_matches(
    const cv::Mat &img1, const cv::Mat &img2,
    std::vector<cv::KeyPoint> &kp1, 
    std::vector<cv::KeyPoint> &kp2,
    std::vector<cv::DMatch> &matches 
) {
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img1, kp1);
    detector->detect(img2, kp2);

    descriptor->compute(img1, kp1, descriptors_1);
    descriptor->compute(img2, kp2, descriptors_2);

    std::vector<cv::DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    double min_dist = 10000, max_dist = 0.0;
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist > max_dist) max_dist = dist;
        if (dist < min_dist) min_dist = dist;
    }
    printf("-- MAX: %f \n", max_dist);
    printf("-- MIN: %f \n", min_dist);

    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= std::max(min_dist * 2, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

void pose_estimation_2d2d(
    const std::vector<KeyPoint> &keypoints_1,
    const std::vector<KeyPoint> &keypoints_2,
    const std::vector<DMatch> &matches,
    Mat &R, Mat &t) {
    
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    std::vector<cv::Point2d> points1, points2;
    for (int i = 0; i < matches.size(); i++) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }
    
    // 计算本质矩阵
    Point2d principal_point(325.1, 249.7);      
    int focal_length = 521;           
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
}

void triangulation(
  const vector<KeyPoint> &keypoint_1,
  const vector<KeyPoint> &keypoint_2,
  const std::vector<DMatch> &matches,
  const Mat &R, const Mat &t,
  vector<Point3d> &points
) {
    Mat T1 = (Mat_<float>(3, 4) <<
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0);
    Mat T2 = (Mat_<float>(3, 4) <<
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
    );

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    std::vector<cv::Point2d> pts_1, pts_2;
    for (auto m : matches) {
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
        pts_2.push_back(pixel2cam(keypoint_2[m.queryIdx].pt, K));
    }
    cv::Mat pts_4d;
    cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

    for (int i = 0; i < pts_4d.cols; i++) {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3, 0); // 归一化
        Point3d p(
        x.at<float>(0, 0),
        x.at<float>(1, 0),
        x.at<float>(2, 0)
        );
        points.push_back(p);
    }
}

Point2f pixel2cam(const Point2d &p, const Mat &K) {
  return Point2f
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}
