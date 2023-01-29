#include <opencv2/opencv.hpp>
#include <sophus/se3.h>
#include <boost/format.hpp>
// #include <pangolin/pangolin.h>
// #include <Eigen/Core>
// #include <Eigen/Dense>

typedef std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
double baseline = 0.573;

std::string left_file = "/home/fmw/gitClone/mySelf/slam14_self/ch8/data/left.png";
std::string disparity_file = "/home/fmw/gitClone/mySelf/slam14_self/ch8/data/disparity.png";
boost::format fmt_others("/home/fmw/gitClone/mySelf/slam14_self/ch8/data/%06d.png");  

class JacobianAccumulator {
public:
    JacobianAccumulator(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const VecVector2d &px_ref_,
        const std::vector<double> depth_ref_,
        Sophus::SE3 &T21_) : img1(img1_), img2(img2_), px_ref(px_ref_), depth_ref(depth_ref_), T21(T21_) {
        projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
    }

    void accumulate_jacobian(const cv::Range &range);
    Matrix6d hessian() const {return H;}
    Vector6d bias() const {return b;}
    double cost_func() const {return cost;}
    VecVector2d projected_points() const {return projection;}

    void reset() {
        H = Matrix6d::Zero();
        b = Vector6d::Zero();
        cost = 0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const std::vector<double> depth_ref;
    Sophus::SE3 &T21;
    VecVector2d projection; // projected points

    std::mutex hessian_mutex;
    Matrix6d H = Matrix6d::Zero();
    Vector6d b = Vector6d::Zero();
    double cost = 0;
};

void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const std::vector<double> depth_ref,
    Sophus::SE3 &T21
);

void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const std::vector<double> depth_ref,
    Sophus::SE3 &T21
);

inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}

int main(int argc, char **argv)
{
    cv::Mat left_image = cv::imread(left_file, 0);
    cv::Mat disparity_image = cv::imread(disparity_file, 0);

    cv::RNG rng;
    int nPoints = 2000;
    int boarder = 20;
    VecVector2d pixels_ref;
    std::vector<double> depth_ref;
    for (int i = 0; i < nPoints; i++)
    {
        int x = rng.uniform(boarder, left_image.cols - boarder);
        int y = rng.uniform(boarder, left_image.rows - boarder);
        int disparity = disparity_image.at<uchar>(y, x);
        double depth = fx * baseline / disparity;
        depth_ref.push_back(depth);
        pixels_ref.push_back(Eigen::Vector2d(x, y));
    }

    Sophus::SE3 T_cur_ref;
    for (int i = 1; i < 6; i++)
    {
        cv::Mat img = cv::imread((fmt_others % i).str(), 0);
        DirectPoseEstimationMultiLayer(left_image, img, pixels_ref, depth_ref, T_cur_ref);
    } 
    return 0;
}

void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const std::vector<double> depth_ref,
    Sophus::SE3 &T21) {
    
    int iterations = 10;
    double cost = 0, last_cost = 0;
    auto t1 = std::chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);
    for (int iter = 0; iter < iterations; iter++) {
        jaco_accu.reset();
        cv::parallel_for_(cv::Range(0, px_ref.size()), std::bind(&JacobianAccumulator::accumulate_jacobian, &jaco_accu, std::placeholders::_1));

        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();
        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3::exp(update) * T21;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0])) {
            std::cout << "update is nan" << std::endl;
            break;
        }
        if (iter > 0 && cost > last_cost) {
            std::cout << "cost increased: " << cost << ", " << last_cost << std::endl;
            break;
        }
        if (update.norm() < 1e-3) {
            break;
        }
        last_cost = cost;
            
    }
    std::cout << "T21 = \n" << T21.matrix() << std::endl;
    auto t2 = std::chrono::steady_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "direct method for single layer: " << time_used.count() << std::endl;

}

void JacobianAccumulator::accumulate_jacobian(const cv::Range &range) {
    const int half_patch_size = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    int cnt_good = 0;
    double cost_tmp = 0.0;

    for (size_t i = range.start; i < range.end; i++)
    {
        Eigen::Vector3d point_ref = depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
        Eigen::Vector3d point_cur = T21 * point_ref;
        if (point_cur[2] < 0) continue;

        float u = point_cur[0] * fx / point_cur[2] + cx, v = point_cur[1] * fy / point_cur[2] + cy;
        if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size || v > img2.rows - half_patch_size)
            continue;
        
        projection[i] = Eigen::Vector2d(u, v);
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2], Z2 = Z * Z, Z_inv = 1 / Z, Z2_inv = 1 / Z2;
        cnt_good++;

        for (int x = -half_patch_size; x <= half_patch_size; x++) {
            for (int y = -half_patch_size; y <= half_patch_size; y++) {
                double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) - GetPixelValue(img2, u + x, v + y);
                
                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y), - GetPixelValue(img2, u + x, v - 1 + y))
                );
                Vector6d J = -1 * (J_img_pixel.transpose() * J_pixel_xi).transpose();
                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
        }    
    }
    if (cnt_good)
    {
        std::unique_lock<std::mutex> lck(hessian_mutex);
        hessian += hessian;
        bias += bias;
        cost += cost_tmp / cnt_good;
    } 
}

void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const std::vector<double> depth_ref,
    Sophus::SE3 &T21) {
    
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // 创建图像金字塔
    std::vector<cv::Mat> pyr1, pyr2;
    for (int i = 0; i < pyramids; i++)
    {
        if (i == 1) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr, cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr, cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }  
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;
    for (int level = pyramids - 1; level >= 0; level--) {
        std::cout  << "pyr level = " << level << "-------------------------------------------" << std::endl;
        VecVector2d px_ref_pyr;
        for (auto px : px_ref) {
            px_ref_pyr.push_back(scales[level] * px);
        }
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];  // 这些内参在哪里用到了

        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }
    
    

}
