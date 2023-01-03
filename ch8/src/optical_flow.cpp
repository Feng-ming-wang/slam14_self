#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

std::string file_1 = "../data/LK1.png";
std::string file_2 = "../data/LK2.png";

class OpticalFlowTracker {
public:
    OpticalFlowTracker(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const std::vector<cv::KeyPoint> &kp1_,
        std::vector<cv::KeyPoint> &kp2_,
        std::vector<bool> &success_,
        bool inverse_ = true, bool has_initial_ = false) : 
        img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
        has_initial(has_initial_) {}  // 不清楚为什么要这样写

    void calculateOpticalFlow(const cv::Range &range);

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const std::vector<cv::KeyPoint> &kp1;
    std::vector<cv::KeyPoint> &kp2;
    std::vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

void OpticalFlowSingleLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse = false,
    bool has_initial = false
);

void OpticalFlowMultiLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse = false
);

inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);
    
    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
    + xx * (1 - yy) * img.at<uchar>(y, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}

int main(int argc, char **argv)
{
    cv::Mat img1 = cv::imread(file_1, 0);
    cv::Mat img2 = cv::imread(file_2, 0);

    std::cout << img1.size() << std::endl;
    std::cout << img2.size() << std::endl;

    std::vector<cv::KeyPoint> kp1;
    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20);
    detector->detect(img1, kp1);

    // 单层光流
    std::vector<cv::KeyPoint> kp2_single;
    std::vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);

    // 多层光流
    // std::vector<cv::KeyPoint> kp2_multi;
    // std::vector<bool> success_multi;
    // OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi);

    cv::Mat img2_single;
    cv::cvtColor(img2, img2_single, cv::COLOR_GRAY2BGR);
    for (int k; k < kp2_single.size(); k++) {
        if (success_single[k])
        {
            cv::circle(img2_single, kp2_single[k].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[k].pt, kp2_single[k].pt, cv::Scalar(0, 250, 0));
        }
    }

    // cv::Mat img2_multi;
    // cv::cvtColor(img2, img2_multi, cv::COLOR_GRAY2BGR);
    // for (int i = 0; i < kp2_multi.size(); i++)
    // {
    //     if (success_multi[i])
    //     {
    //         cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
    //         cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
    //     }
    // }
    
    cv::imshow("single LK tracking", img2_single);
    // cv::imshow("multi LK tracking", img2_multi);
    cv::waitKey(0);

    return 0;
}

void OpticalFlowSingleLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse,
    bool has_initial){
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    cv::parallel_for_(cv::Range(0, kp1.size()), std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, std::placeholders::_1));
}

void OpticalFlowTracker::calculateOpticalFlow(const cv::Range &range) {
    int half_patch_size = 4;
    int iterations = 10;
    for (int i = range.start; i < range.end; i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true;

        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        Eigen::Vector2d J;
        for (int iter = 0; iter < iterations; iter++) {
            if (inverse == false) {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);;  // Jacobian
                    if (inverse == false) {
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                   GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                   GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                    } else if (iter == 0) {
                        // in inverse mode, J keeps same for all iterations
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                   GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                   GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }
                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();
                    }
                }
                Eigen::Vector2d update = H.ldlt().solve(b);

                if (std::isnan(update[0])) {
                    // sometimes occurred when we have a black or white patch and H is irreversible
                    std::cout << "update is nan" << std::endl;
                    succ = false;
                    break;
                }

                if (iter > 0 && cost > lastCost) {
                    break;
                }

                // update dx, dy
                dx += update[0];
                dy += update[1];
                lastCost = cost;
                succ = true;

                if (update.norm() < 1e-2) {
                    // converge
                    break;
                }
            }
            success[i] = succ;
            // set kp2
            kp2[i].pt = kp.pt + cv::Point2f(dx, dy);
        }     
}

void OpticalFlowMultiLevel(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const std::vector<cv::KeyPoint> &kp1,
    std::vector<cv::KeyPoint> &kp2,
    std::vector<bool> &success,
    bool inverse
) {
    // 先构建金字塔
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    std::vector<cv::Mat> pyr1, pyr2;
    for (int i = 0; i < pyramids; i++)
    {
        if (i == 0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr, cv::Size(pyr1[i - 1].cols * pyramid_scale, 
                pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr, cv::Size(pyr2[i - 1].cols * pyramid_scale, 
                pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    // 由粗到精 计算每层的特征点
    // 先得到顶层(粗的)的特征点
    std::vector<cv::KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp : kp1) {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }
    for (int level = pyramids - 1; level >= 0; level++)
    {
        success.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        if (level > 0) //提前准备好下一层的关键点，
        // 就是讲上一层的关键点坐标除以采样率，这里除以0.5，即将关键点坐标乘以2.
        {
            for (auto &kp: kp1_pyr)
                kp.pt /= pyramid_scale;
            for (auto &kp: kp2_pyr)
                kp.pt /= pyramid_scale;
        }
    }
    for (auto &kp: kp2_pyr)
        kp2.push_back(kp); 
}
