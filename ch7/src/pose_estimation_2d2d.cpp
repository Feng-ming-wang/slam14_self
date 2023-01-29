#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/imgcodecs/legacy/constants_c.h"

void find_feature_matches(
    const cv::Mat &img1, const cv::Mat &img2,
    std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<cv::DMatch> &matches
);

void pose_estimation_2d2d(
    std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<cv::DMatch> &matches,
    cv::Mat &R, cv::Mat &t
);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << "usage: pose_estimation_2d2d img1 img2. " << std::endl;
        return 1;
    }
    cv::Mat img1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img1.data && img2.data && "couldn't load image.");

    // 要找两幅图像中的特征点吧
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(img1, img2, keypoints_1, keypoints_2, matches);
    std::cout << "一共找到了 " << matches.size() << " 组匹配点." << std::endl;

    cv::Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    //-- 验证E=t^R*scale
    cv::Mat t_x =
    (cv::Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
      t.at<double>(2, 0), 0, -t.at<double>(0, 0),
      -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    std::cout << "t^R=" << std::endl << t_x * R << std::endl;

    //-- 验证对极约束
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (cv::DMatch m: matches) {
        cv::Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        cv::Mat y1 = (cv::Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        cv::Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        cv::Mat y2 = (cv::Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        cv::Mat d = y2.t() * t_x * R * y1;
        std::cout << "epipolar constraint = " << d << std::endl;
    }
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

    double min_dist = 10000, max_dist = 0;

    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    printf("-- Max_dist: %f \n", max_dist);
    printf("-- Min_dist: %f \n", min_dist);

    for (int i = 0; i < descriptors_1.rows; i++)
    {
        if (match[i].distance <= std::max(min_dist*2, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
    return cv::Point2d(
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void pose_estimation_2d2d(
    std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<cv::DMatch> &matches,
    cv::Mat &R, cv::Mat &t
) {
    cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    std::vector<cv::Point2d> points1, points2;
    for (int i = 0; i < matches.size(); i++) {
        points1.push_back(kp1[matches[i].queryIdx].pt);
        points2.push_back(kp2[matches[i].trainIdx].pt);
    }

    // 计算基础矩阵
    cv::Mat fundamental_matrix;
    fundamental_matrix = cv::findFundamentalMat(points1, points2, cv::FM_8POINT);
    std::cout << "fundamental_matrix is " << std::endl << fundamental_matrix << std::endl;

    cv::Point2d principal_point(325.1, 249.7);  
    double focal_length = 521;     
    cv::Mat essential_matrix;
    essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);
    std::cout << "essential_matrix is " << std::endl << essential_matrix << std::endl;

    cv::Mat homography_matrix;
    homography_matrix = cv::findHomography(points1, points2, cv::RANSAC, 3);
    std::cout << "homography_matrix is " << std::endl << homography_matrix << std::endl;

    cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    std::cout << "R is " << std::endl << R << std::endl;
    std::cout << "t is " << std::endl << t << std::endl;
}