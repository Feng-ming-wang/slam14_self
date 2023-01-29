#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    std::cout << "reading images..." << std::endl;
    std::vector<cv::Mat> images;
    for (int i = 0; i < 10; i++)
    {
        std::string path = "../data/" + std::to_string(i + 1) + ".png";
        images.push_back(cv::imread(path));
    }

    std::cout << "detecting ORB features ... " << std::endl;
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    std::vector<cv::Mat> descriptors;
    for (cv::Mat &image : images)
    {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptor;
        detector->detectAndCompute(image, cv::Mat(), keypoints, descriptor);
        descriptors.push_back(descriptor);
    }

    std::cout << "creating vocabulary ... " << std::endl;
    DBoW3::Vocabulary vocabulary;
    vocabulary.create(descriptors);
    std::cout << "vocabulary info: " << vocabulary << std::endl;
    vocabulary.save("../data/vocabulary.yml.gz");
    std::cout << "done" << std::endl;
    
    return 0;
}
