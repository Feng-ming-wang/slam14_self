#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/imgcodecs/legacy/constants_c.h"
#include <chrono>

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "usage: feature_extraction img1 img2" << std::endl;
        return 1;
    }
    // 读取图像
    cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    // 初始化
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // 第一步，检测 Oriented FAST 角点位置
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    // 第二步，根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "Extract ORB cost: " << time_used.count() << " .seconds" << std::endl;

    cv::Mat outimg1;
    cv::drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    // cv::namedWindow("ORB features", 0);
    // cv::imshow("ORB features", outimg1);
    // cv::waitKey();

    // 第三步，对两幅图像中的BRIEF描述子进行匹配，使用Hamming距离
    std::vector<cv::DMatch> matches;
    t1 = std::chrono::steady_clock::now();
    matcher->match(descriptors_1, descriptors_2, matches);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "match ORB cost: " << time_used.count() << " seconds." << std::endl;

    // 第四步，匹配点对筛选
    // 计算最小和最大距离
    auto min_max = minmax_element(matches.begin(), matches.end(), [](const cv::DMatch &m1, const cv::DMatch &m2) {return m1.distance < m2.distance;});
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    printf("-- Max dist: %f \n", max_dist);
    printf("-- Min dist: %f \n", min_dist);

    //当描述子距离小于2倍的最小距离时，认为是好的匹配，为了防止最小距离很小的情况，设置一个阈值 30.0(经验值)
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    // 绘制匹配结果
    cv::Mat img_match;
    cv::Mat good_match;
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
    cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, good_match);
    cv::imshow("all matches", img_match);
    cv::imshow("good matches", good_match);
    cv::waitKey();

    return 0;
}