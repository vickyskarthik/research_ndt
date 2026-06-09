#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_representation.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <omp.h>
#include <vector>

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <stdlib.h>
#include <iomanip>
#include <thread>
#include <sys/types.h>
#include <dirent.h>
#include <cmath>

#include "../include/pw_ndt_radar_scan_matching/ellipse.cc"
#include "../include/pw_ndt_radar_scan_matching/pw_ndt.h"
#include "../include/pw_ndt_radar_scan_matching/ndt.h"

using namespace std;
using namespace cv;
using namespace std::chrono_literals;

//-- global variable --//
vector<Eigen::Matrix4f> init_guess_vec;
vector<Eigen::Matrix4f> gt_vec;
nav_msgs::Path gt_path_msg;

float begin_t = 0;
double old_delta_dis = 0;
bool init = false;  int Index = 0;
float old_vel = 0; float old_ang_vel = 0; double old_t = 0;
float curr_vel = 0; float curr_ang_vel = 0; double curr_t = 0;
pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc (new pcl::PointCloud<pcl::PointXYZI>);
Eigen::Matrix4f curr_tf;

// ── C2 global ──────────────────────────────────────────────────
bool use_distortion_correction_ = false; // C2: per-azimuth motion distortion compensation
// ── C4/C5 globals ──────────────────────────────────────────────────
bool   log_covariance_       = false;   // Config B/C/D: write _cov.csv
bool   log_degeneracy_       = false;   // Config C/D:   write _deg.csv
bool   use_constrained_update_ = false; // Config D only: constrain Newton step
double epsilon_reg_          = 1e-6;    // Tikhonov regularisation for C4
double tau_degeneracy_       = 0.0;     // degeneracy threshold (set from B1)
std::ofstream cov_log_;                 // per-frame covariance CSV
std::ofstream deg_log_;                 // per-frame degeneracy CSV
// ───────────────────────────────────────────────────────────────────

string directory = "DATA_PATH";
string gt_directory = "GT_PATH";
string save_directory = "SAVE_PATH";
string init_directory = "no_init_directory";
string bag_name = "no_specific_bag";

//-- NDT param --//
float intensity_filter = 0;
int image_width = 125;
int ndt2d_grid_extent_ = int(image_width/2);
float ndt2d_grid_resolution_ = 0.125;
int ndt2d_max_it_ = 300;
double ndt2d_eps_ = 0.003;
double max_step_size_ = 0.03;

bool use_pw = true;
bool use_imu = false;

//-- Debug param --//
int global_viz = 0;
int start_index = 0;
int save_res = 0;
string res_file_name = "_res.txt";

uint64_t curr_time_ns = 0;   // nanosecond timestamp of current frame (for CSV)

//////////////////////////////////////

float global_threshold;
float global_bias;
float grid_size1, grid_size2, grid_size3, grid_size4;
float eps1, eps2, eps3, eps4;
double max_step_size1, max_step_size2, max_step_size3, max_step_size4;
float global_mask_flag;

ros::Publisher old_pc_pub;
ros::Publisher new_pc_pub;
ros::Publisher out_pc_pub;
ros::Publisher nd_map_pub;

ros::Publisher gt_path_pub;
ros::Publisher gt_pose_pub;
ros::Publisher res_pose_pub;

ros::Publisher err_pub;

pcl::PointCloud<pcl::PointXYZI>::Ptr match_scan(new pcl::PointCloud<pcl::PointXYZI>);
sensor_msgs::PointCloud2 old_pc;
sensor_msgs::PointCloud2 new_pc;
sensor_msgs::PointCloud2 out_pc;

template <class Container>
void SplitStr(const std::string& str, Container& cont, char delim = ' ') {
  std::size_t current, previous = 0;
  current = str.find(delim);
  while (current != std::string::npos) {
    cont.push_back(str.substr(previous, current - previous));
    previous = current + 1;
    current = str.find(delim, previous);
  }
  cont.push_back(str.substr(previous, current - previous));
}

pcl::PointCloud<pcl::PointXYZI>::Ptr ImageToPointCloud(Mat& image_raw, Mat& image_mask){
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
  int mid = (image_raw.rows - 1)/2;
  pcl::PointXYZI point;
  if(global_mask_flag == 1){
    for(int i=0; i<image_raw.rows; i++){
      for(int j=0; j<image_raw.cols; j++){
        float point_value = (int)image_raw.at<uchar>(i,j);
        float check_point = (int)image_mask.at<uchar>(i,j);
        point.x = -(i - mid)*ndt2d_grid_resolution_;
        point.y = -(j - mid)*ndt2d_grid_resolution_;
        point.z = 0;
        point.intensity = point_value;
        if(check_point > 0 && point_value > global_threshold){
          pc->points.push_back(point);
        }
      }
    }
  } else {
    for(int i=0; i<image_raw.rows; i++){
      for(int j=0; j<image_raw.cols; j++){
        float point_value = (int)image_raw.at<uchar>(i,j);
        point.x = -(i - mid)*ndt2d_grid_resolution_;
        point.y = -(j - mid)*ndt2d_grid_resolution_;
        point.z = 0;
        point.intensity = point_value;
        if(point_value > global_threshold){
          pc->points.push_back(point);
        }
      }
    }
  }
  return pc;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr ImageToPointCloud(Mat& image_raw){
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc(new pcl::PointCloud<pcl::PointXYZI>);
  int mid = (image_raw.rows - 1)/2;
  pcl::PointXYZI point;
  for(int i=0; i<image_raw.rows; i++){
    for(int j=0; j<image_raw.cols; j++){
      float point_value = (int)image_raw.at<uchar>(i,j);
      point.x = -(i - mid)*ndt2d_grid_resolution_;
      point.y = -(j - mid)*ndt2d_grid_resolution_;
      point.z = 0;
      point.intensity = point_value;
      pc->points.push_back(point);
    }
  }
  return pc;
}

bool isNanInMatrix(Eigen::Matrix4f& T){
  for(int i=0; i<4; i++){
    for(int j=0; j<4; j++){
      if(T(i,j) != T(i,j) ){ return true; }
    }
  }
  return false;
}

vector<Eigen::Matrix4f> loadPoses(string file_name) {
  vector<Eigen::Matrix4f> poses;
  Eigen::Matrix4f P = Eigen::Matrix4f::Identity();
  poses.push_back(P); // id=0
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp) return poses;
  while (!feof(fp)) {
    Eigen::Matrix4f P = Eigen::Matrix4f::Identity();
    if (fscanf(fp, "%f %f %f %f %f %f %f %f %f %f %f %f",
                   &P(0,0), &P(0,1), &P(0,2), &P(0,3),
                   &P(1,0), &P(1,1), &P(1,2), &P(1,3),
                   &P(2,0), &P(2,1), &P(2,2), &P(2,3) )==12) {
      poses.push_back(P);
    }
  }
  fclose(fp);
  return poses;
}

///////////////////////////////////////////////////////////////////////////
void pub_odom_msg(ros::Publisher pose_pub, Eigen::Matrix4f P, string child_frame_id);
void Callback(pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc);
void read_directory(const std::string& name, std::vector<std::string>& v);
void WriteToFile(Eigen::Matrix4f& T, string& file_path);
pcl::PointCloud<pcl::PointXYZI>::Ptr make2DPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_3d);
pcl::PointCloud<pcl::PointXYZI>::Ptr RadarPointCloudFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in, float thres);
float GetDistanceFromT(Eigen::Matrix4f T);
float VelocityCal(Eigen::Matrix4f T, double delta_t);
float AngularVelocityCal(Eigen::Matrix4f T, double delta_t);
void DoPWNDT(const pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc, pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud, float grid_step_, double eps_, double max_step_size_, float bias_,
              Eigen::Matrix4f &T, double &score,
              Eigen::Matrix<pcl::ndt2d::NormalDist<pcl::PointXYZI>, Eigen::Dynamic, Eigen::Dynamic> &normal_distributions_map,
              Eigen::Matrix3d &H_raw_out, Eigen::Matrix3d &H_reg_out, bool &H_valid_out);
nav_msgs::Path gen_path_msg(vector<Eigen::Matrix4f> pose_vec);
bool CheckAcc(Eigen::Matrix4f& T);
void PrintAcc(Eigen::Matrix4f& T);
void NDTMultiStage(const pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc, Eigen::Matrix4f &T_final, pcl::PointCloud<pcl::PointXYZI>::Ptr &output_cloud,
                   Eigen::Matrix3d &H_raw_out, Eigen::Matrix3d &H_reg_out, bool &H_valid_out);
void Vizsualization(bool viz,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud,
                    const pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc);
void deleteNDMap();
void pubNDMap(Eigen::Matrix< pcl::ndt2d::NormalDist<pcl::PointXYZI> , Eigen::Dynamic, Eigen::Dynamic> normal_distributions_map,
              double scale);
pcl::PointCloud<pcl::PointXYZI>::Ptr RadarPolarToCartesian(
    const cv::Mat& raw_polar_image,
    float vel_mps = 0.0f,
    float ang_vel_rads = 0.0f,
    double sweep_dt = 0.0);
///////////////////////////////////////////////////////////////////////////

// --- helpers ---
inline double safe_dt(double dt) { return (std::isfinite(dt) && dt > 1e-9) ? dt : 1e-9; }

pcl::PointCloud<pcl::PointXYZI>::Ptr removeInfinePoint(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud){
  pcl::PointCloud<pcl::PointXYZI>::iterator it = cloud->points.begin();
  while (it != cloud->points.end()) {
    float x = it->x, y = it->y, z = it->z;
    if (!pcl_isfinite(x) || !pcl_isfinite(y) || !pcl_isfinite(z)) it = cloud->points.erase(it);
    else ++it;
  }
  return cloud;
}

#include <filesystem>
namespace fs = std::filesystem;

void OpenBag(std::string bag_name) {
  //==================== Load GT & (optional) init guess ====================//
  gt_vec = loadPoses(gt_directory);  // make sure this param points to the *_gt.txt FILE
  std::cout << "[INFO] GT file size: " << gt_vec.size() << std::endl;
  gt_path_msg = gen_path_msg(gt_vec);

  if (init_directory != "no_init_directory") {
    const std::string init_file = init_directory + "/" + bag_name + "_res.txt";
    std::cout << "[INFO] Using initial guess file: " << init_file << std::endl;
    init_guess_vec = loadPoses(init_file);
    std::cout << "[INFO] Initial guess file size: " << init_guess_vec.size() << std::endl;
  }

  //==================== Enumerate radar PNGs ====================//
  const std::string radar_dir = directory;
  struct Frame { uint64_t stamp_us; std::string path; };
  std::vector<Frame> frames; frames.reserve(10000);

  try {
    for (const auto& entry : fs::directory_iterator(radar_dir)) {
      if (!entry.is_regular_file()) continue;
      if (entry.path().extension() != ".png") continue;
      const std::string stem = entry.path().stem().string();
      try {
        uint64_t us = std::stoull(stem);  // Oxford: microseconds
        frames.push_back({us, entry.path().string()});
      } catch (...) { /* skip */ }
    }
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] Failed to iterate radar directory '" << radar_dir
              << "': " << e.what() << std::endl;
    return;
  }

  if (frames.empty()) {
    std::cerr << "[ERROR] No .png radar images found in: " << radar_dir << std::endl;
    return;
  }

  std::sort(frames.begin(), frames.end(),
            [](const Frame& a, const Frame& b){ return a.stamp_us < b.stamp_us; });

  std::cout << "[INFO] Processing " << frames.size()
            << " radar frames from: " << radar_dir << std::endl;

  //==================== Open C4/C5 CSV logs ====================//
  // CSV stems derived from res_file_name so each config writes separate files:
  //   _res_configC.txt  →  _configC_cov.csv  /  _configC_deg.csv
  //   _res_configD.txt  →  _configD_cov.csv  /  _configD_deg.csv
  std::string res_stem = res_file_name; // e.g. "_res_configC.txt"
  // strip leading "_res" prefix if present
  if (res_stem.size() > 4 && res_stem.substr(0,4) == "_res")
    res_stem = res_stem.substr(4);         // → "_configC.txt"
  // strip extension
  auto dot = res_stem.rfind('.');
  if (dot != std::string::npos) res_stem = res_stem.substr(0, dot); // → "_configC"
  if (res_stem.empty()) res_stem = "_default";

  if (log_covariance_) {
    const std::string cov_path = save_directory + "/" + bag_name + res_stem + "_cov.csv";
    cov_log_.open(cov_path);
    cov_log_ << "frame_id,stamp_us,Sxx,Sxy,Sxt,Syy,Syt,Stt,trace,cond_num\n";
    std::cout << "[C4] Covariance log: " << cov_path << std::endl;
  }
  if (log_degeneracy_) {
    const std::string deg_path = save_directory + "/" + bag_name + res_stem + "_deg.csv";
    deg_log_.open(deg_path);
    deg_log_ << "frame_id,stamp_us,lambda_min,lambda_max,cond_num,is_degenerate,D0,D1,D2,H_valid\n";
    std::cout << "[C5] Degeneracy log: " << deg_path << std::endl;
  }

  //==================== Reset pipeline state ====================//
  curr_tf.setIdentity();
  init = false;
  Index = 0;
  curr_vel = 0.0f;
  curr_ang_vel = 0.0f;
  old_delta_dis = 0.0;

  //==================== Main loop ====================//
  for (const auto& f : frames) {
    curr_time_ns = f.stamp_us * 1000ULL;                   // μs → ns
    curr_t       = static_cast<double>(f.stamp_us) * 1e-6; // μs → s

    if (Index < start_index) { ++Index; continue; }

    cv::Mat image_raw = cv::imread(f.path, cv::IMREAD_GRAYSCALE);
    if (image_raw.empty()) {
      std::cerr << "[WARNING] Failed to read image: " << f.path << std::endl;
      ++Index; continue;
    }

    // C2: pass previous-frame velocity for per-azimuth undistortion
    // Using old_vel/old_ang_vel (from frame k-1) to undistort frame k.
    // safe_dt guards against zero/invalid dt on the first frame.
    const double sweep_dt = safe_dt(curr_t - old_t);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cart_pc = RadarPolarToCartesian(
        image_raw,
        use_distortion_correction_ ? old_vel : 0.0f,
        use_distortion_correction_ ? (old_ang_vel * static_cast<float>(M_PI) / 180.0f) : 0.0f,
        use_distortion_correction_ ? sweep_dt : 0.0);
    if (!cart_pc || cart_pc->empty()) {
      std::cerr << "[WARNING] Polar2Cart produced empty cloud for: " << f.path << std::endl;
      ++Index; continue;
    }

    Callback(cart_pc);
    gt_path_pub.publish(gt_path_msg);

    ++Index;
  }

  std::cout << "[INFO] Finished processing " << Index << " radar frames." << std::endl;

  // Close C4/C5 CSV logs
  if (cov_log_.is_open()) { cov_log_.flush(); cov_log_.close(); }
  if (deg_log_.is_open()) { deg_log_.flush(); deg_log_.close(); }
}

void Callback(pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc){
  // clean input point cloud from NaNs
  if (!in_pc->is_dense){
    in_pc->is_dense = false;
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*in_pc, *in_pc, indices);
  }

  if(!init){ init = true; }
  else{
    cout << "-----------------------------------" << endl;
    cout << "[Index]: " << Index << " t:" << begin_t+curr_t << endl;

    if(old_delta_dis > 3.5){
      cout << "{ERROR} [index] " << " t:" << begin_t+curr_t << " " << Index << " old_delta_dis: " << old_delta_dis << "\n";
    }

    Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
    pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud (new pcl::PointCloud<pcl::PointXYZI>);
    // C4/C5: Hessian outputs from Stage 3 NDT
    Eigen::Matrix3d H_raw  = Eigen::Matrix3d::Zero();
    Eigen::Matrix3d H_reg  = Eigen::Matrix3d::Zero();
    bool            H_valid = false;

    // empty cloud safety
    if (!old_in_pc || old_in_pc->empty() || !in_pc || in_pc->empty()) {
      std::cout << "[WARN] Empty cloud — skipping NDT at index " << Index << "\n";
    } else if (init_directory != "no_init_directory" &&
               Index > 0 &&
               Index < static_cast<int>(init_guess_vec.size())) {
      // Use init guess with bounds check
      Eigen::Matrix4f T_init(Eigen::Matrix4f::Identity());
      const Eigen::Matrix4f& P_new = init_guess_vec[Index];
      const Eigen::Matrix4f& P_old = init_guess_vec[Index-1];
      T_init = P_new.inverse() * P_old;
      cout << " Load init guess ";

      const double dt = safe_dt(curr_t - old_t);
      float init_curr_vel = VelocityCal(T_init, dt);
      if(init_curr_vel < 0.5f){
        T = T_init;
        curr_vel = VelocityCal(T, dt);
        curr_ang_vel = AngularVelocityCal(T, dt);
      } else {
        T = T_init;
        in_pc = removeInfinePoint(in_pc);
        old_in_pc = removeInfinePoint(old_in_pc);
        NDTMultiStage(old_in_pc, in_pc, T, output_cloud, H_raw, H_reg, H_valid);
      }
    } else {
      // No init guess; do NDT if clouds are valid
      in_pc = removeInfinePoint(in_pc);
      old_in_pc = removeInfinePoint(old_in_pc);
      if (!old_in_pc->empty() && !in_pc->empty()) {
        NDTMultiStage(old_in_pc, in_pc, T, output_cloud, H_raw, H_reg, H_valid);
      } else {
        std::cout << "[WARN] Empty cloud after filtering — skipping NDT at index " << Index << "\n";
      }
    }

    //-- Pub pointcloud msg --//
    pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc_flt (new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc_flt (new pcl::PointCloud<pcl::PointXYZI>);
    in_pc_flt = RadarPointCloudFilter(in_pc, 85);
    old_in_pc_flt = RadarPointCloudFilter(old_in_pc, 85);
    pcl::toROSMsg(*in_pc_flt, new_pc);
    pcl::toROSMsg(*old_in_pc_flt, old_pc);
    pcl::toROSMsg(*output_cloud, out_pc);
    old_pc.header.frame_id = "res_odom";
    new_pc.header.frame_id = "res_odom";
    out_pc.header.frame_id = "res_odom";
    old_pc_pub.publish(old_pc);
    new_pc_pub.publish(new_pc);
    out_pc_pub.publish(out_pc);

    //-- Show result --//
    cout << " [RES] ";
    PrintAcc(T);

    //-- Read GT (bounds-checked) --//
    bool have_gt = (Index > 0 && Index < static_cast<int>(gt_vec.size()));
    Eigen::Matrix4f P_gt_new = Eigen::Matrix4f::Identity();
    if (have_gt) {
      const Eigen::Matrix4f& P_new = gt_vec[Index];
      const Eigen::Matrix4f& P_old = gt_vec[Index-1];
      Eigen::Matrix4f T_gt = P_new.inverse() * P_old;

      cout << " [GT] ";
      PrintAcc(T_gt);

      const float x_gt = T_gt(0,3), y_gt = T_gt(1,3);
      const float th_gt = std::atan2(T_gt(1,0), T_gt(0,0));
      const float x_res = T(0,3),  y_res = T(1,3);
      const float th_res = std::atan2(T(1,0), T(0,0));
      cout << " [ERROR] xytheta: " << std::abs(x_res-x_gt) << " m "
                                   << std::abs(y_res-y_gt) << " m "
                                   << std::abs((th_res-th_gt)*180.0/ M_PI) << " deg" << endl;

      // publish error
      nav_msgs::Odometry err_msg;
      err_msg.header.seq = Index;
      err_msg.pose.pose.position.x = std::abs(x_res-x_gt);
      err_msg.pose.pose.position.y = std::abs(y_res-y_gt);
      err_msg.pose.pose.position.z = std::abs((th_res-th_gt)*180.0/ M_PI);
      err_pub.publish(err_msg);

      P_gt_new = P_new;
      pub_odom_msg(gt_pose_pub, P_gt_new, "gt_odom");
    } else {
      std::cout << " [GT] (skipped: Index " << Index << " out of GT range " << gt_vec.size() << ")\n";
    }

    // Update motion / pose
    const double dt = safe_dt(curr_t - old_t);
    curr_vel = VelocityCal(T, dt);
    curr_ang_vel = AngularVelocityCal(T, dt);
    old_delta_dis = GetDistanceFromT(T);
    curr_tf = T * curr_tf;
    Eigen::Matrix4f curr_tf_inv =  curr_tf.inverse();

    //-- Write result to file --//
    string file_path = save_directory + "/" + bag_name + res_file_name;
    if (save_res != 0){
      WriteToFile(curr_tf_inv, file_path);
    }

    // ── C4: Covariance logging ─────────────────────────────────────────────────
    if (log_covariance_ && cov_log_.is_open() && H_valid) {
      // Tikhonov regularisation: H_reg + epsilon*I  (prevents near-singular inversion)
      Eigen::Matrix3d H_for_cov = H_reg + epsilon_reg_ * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d Sigma      = H_for_cov.inverse();
      double trace_s             = Sigma.trace();
      // Condition number from regularised H
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_c(H_reg);
      double cond_num = es_c.eigenvalues()(2) / (std::abs(es_c.eigenvalues()(0)) + 1e-10);
      // Write: frame_id, stamp_us, Sxx,Sxy,Sxt, Syy,Syt, Stt, trace, cond_num
      cov_log_ << Index << "," << (curr_time_ns / 1000ULL) << ","
               << Sigma(0,0) << "," << Sigma(0,1) << "," << Sigma(0,2) << ","
               << Sigma(1,1) << "," << Sigma(1,2) << ","
               << Sigma(2,2) << "," << trace_s << "," << cond_num << "\n";
    }

    // ── C5: Degeneracy logging ─────────────────────────────────────────────────
    if (log_degeneracy_ && deg_log_.is_open()) {
      // Use H_raw (before PD correction) for true degenerate direction detection
      // H_raw == zero matrix when H_valid==false (early exit frames) — log as invalid
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es_d(H_raw);
      double lmin      = es_d.eigenvalues()(0);
      double lmax      = es_d.eigenvalues()(2);
      double cond_d    = lmax / (std::abs(lmin) + 1e-10);
      bool   is_deg    = H_valid && (tau_degeneracy_ > 0.0) && (lmin < tau_degeneracy_);
      Eigen::Vector3d D_diag = Eigen::Vector3d::Zero();
      if (tau_degeneracy_ > 0.0) {
        for (int k=0; k<3; k++)
          D_diag(k) = (es_d.eigenvalues()(k) >= tau_degeneracy_) ? 1.0 : 0.0;
      }
      // Write: frame_id, stamp_us, lambda_min, lambda_max, cond_num, is_deg, D0,D1,D2, H_valid
      deg_log_ << Index << "," << (curr_time_ns / 1000ULL) << ","
               << lmin << "," << lmax << "," << cond_d << ","
               << (int)is_deg << ","
               << D_diag(0) << "," << D_diag(1) << "," << D_diag(2) << ","
               << (int)H_valid << "\n";
    }
    // ──────────────────────────────────────────────────────────────────

    //-- Pub odom msg --//
    pub_odom_msg(res_pose_pub, curr_tf_inv, "res_odom");

    Vizsualization(global_viz, in_pc, output_cloud, old_in_pc);
  }

  // Update persistent state for next step
  old_in_pc = in_pc;
  old_vel = curr_vel;
  old_ang_vel = curr_ang_vel;
  old_t = curr_t;
}

nav_msgs::Path gen_path_msg(vector<Eigen::Matrix4f> pose_vec){
  nav_msgs::Path path_msg;
  path_msg.header.frame_id = "origin";
  geometry_msgs::PoseStamped p;
  for (int i=0; i<pose_vec.size(); i++){
    Eigen::Matrix4f P = pose_vec[i];
    tf::Vector3 origin;
    origin.setValue(static_cast<double>(P(0,3)),static_cast<double>(P(1,3)),static_cast<double>(P(2,3)));
    tf::Matrix3x3 tf3d;
    tf3d.setValue(static_cast<double>(P(0,0)), static_cast<double>(P(0,1)), static_cast<double>(P(0,2)),
                  static_cast<double>(P(1,0)), static_cast<double>(P(1,1)), static_cast<double>(P(1,2)),
                  static_cast<double>(P(2,0)), static_cast<double>(P(2,1)), static_cast<double>(P(2,2)));
    tf::Quaternion tfqt;
    tf3d.getRotation(tfqt);
    tf::Transform transform;
    transform.setOrigin(origin);
    transform.setRotation(tfqt);

    tf::poseTFToMsg(transform, p.pose);
    path_msg.poses.push_back(p);
  }
  return path_msg;
}

void pub_odom_msg(ros::Publisher pose_pub, Eigen::Matrix4f P, string child_frame_id){
  tf::Vector3 origin;
  origin.setValue(static_cast<double>(P(0,3)),static_cast<double>(P(1,3)),static_cast<double>(P(2,3)));
  tf::Matrix3x3 tf3d;
  tf3d.setValue(static_cast<double>(P(0,0)), static_cast<double>(P(0,1)), static_cast<double>(P(0,2)),
                static_cast<double>(P(1,0)), static_cast<double>(P(1,1)), static_cast<double>(P(1,2)),
                static_cast<double>(P(2,0)), static_cast<double>(P(2,1)), static_cast<double>(P(2,2)));
  tf::Quaternion tfqt;
  tf3d.getRotation(tfqt);
  tf::Transform transform;
  transform.setOrigin(origin);
  transform.setRotation(tfqt);

  nav_msgs::Odometry  odom_msg;
  odom_msg.header.frame_id = "origin";   // no leading slash — matches TF broadcast
  odom_msg.child_frame_id = child_frame_id;
  tf::poseTFToMsg(transform, odom_msg.pose.pose);
  pose_pub.publish(odom_msg);

  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;
  transformStamped.header.frame_id = "origin";
  transformStamped.child_frame_id = child_frame_id;
  transformStamped.transform.translation.x = odom_msg.pose.pose.position.x;
  transformStamped.transform.translation.y = odom_msg.pose.pose.position.y;
  transformStamped.transform.translation.z = odom_msg.pose.pose.position.z;
  transformStamped.transform.rotation.x = odom_msg.pose.pose.orientation.x;
  transformStamped.transform.rotation.y = odom_msg.pose.pose.orientation.y;
  transformStamped.transform.rotation.z = odom_msg.pose.pose.orientation.z;
  transformStamped.transform.rotation.w = odom_msg.pose.pose.orientation.w;
  br.sendTransform(transformStamped);
}

void NDTMultiStage(const pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc, Eigen::Matrix4f &T, pcl::PointCloud<pcl::PointXYZI>::Ptr &output_cloud,
                   Eigen::Matrix3d &H_raw_out, Eigen::Matrix3d &H_reg_out, bool &H_valid_out){
  double score;
  // Dummy Hessians for stages 1 and 2 (discarded — only Stage 3 Hessian is exposed)
  Eigen::Matrix3d H_raw_dummy, H_reg_dummy;  bool H_valid_dummy;

  pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc_flt (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc_flt (new pcl::PointCloud<pcl::PointXYZI>);
  in_pc_flt = RadarPointCloudFilter(in_pc, global_threshold);
  old_in_pc_flt = RadarPointCloudFilter(old_in_pc, global_threshold);

  Eigen::Matrix< pcl::ndt2d::NormalDist<pcl::PointXYZI> , Eigen::Dynamic, Eigen::Dynamic> normal_distributions_map1, normal_distributions_map2, normal_distributions_map3;

  if (curr_vel<0.5f){
    cout << "[[[ low speed mode ]]]" << endl;
    // Low-speed: single stage — Hessian from this stage is the final one
    DoPWNDT(old_in_pc_flt, in_pc_flt, output_cloud, 1.5f, 0.006, 0.0065, 0, T, score, normal_distributions_map3,
            H_raw_out, H_reg_out, H_valid_out);
  } else{
    // Stage 1 and 2: discard Hessians (coarse stages, not representative)
    DoPWNDT(old_in_pc_flt, in_pc_flt, output_cloud, grid_size1, eps1, max_step_size1, 0, T, score, normal_distributions_map1,
            H_raw_dummy, H_reg_dummy, H_valid_dummy);
    DoPWNDT(old_in_pc_flt, in_pc_flt, output_cloud, grid_size2, eps2, max_step_size2, global_bias, T, score, normal_distributions_map2,
            H_raw_dummy, H_reg_dummy, H_valid_dummy);
    in_pc_flt = RadarPointCloudFilter(in_pc, 85);
    old_in_pc_flt = RadarPointCloudFilter(old_in_pc, 85);
    // Stage 3 (finest): this Hessian is exposed for C4/C5 analysis
    DoPWNDT(old_in_pc_flt, in_pc_flt, output_cloud, grid_size3, eps3, max_step_size3, 85, T, score, normal_distributions_map3,
            H_raw_out, H_reg_out, H_valid_out);
  }

  deleteNDMap();
  pubNDMap(normal_distributions_map3, 1.5);
}

int main(int argc, char** argv)
{
  ros::init (argc, argv, "odometry_intensity_test");
  ros::NodeHandle nh("~");

  old_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("vis/old_pc", 10);
  new_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("vis/new_pc", 10);
  out_pc_pub = nh.advertise<sensor_msgs::PointCloud2>("vis/out_pc", 10);
  nd_map_pub = nh.advertise<visualization_msgs::MarkerArray>("vis/ndmap", 10);
  gt_path_pub = nh.advertise<nav_msgs::Path>("vis/gt_path", 10);
  gt_pose_pub = nh.advertise<nav_msgs::Odometry>("vis/gt_pose", 10);
  res_pose_pub = nh.advertise<nav_msgs::Odometry>("vis/res_pose", 10);
  err_pub = nh.advertise<nav_msgs::Odometry>("err", 10);

  if (!nh.getParam("directory", directory)) ROS_WARN("[%s] Failed to get param 'directory', use default setting: %s", ros::this_node::getName().c_str(), directory.c_str());
  if (!nh.getParam("gt_directory", gt_directory)) ROS_WARN("[%s] Failed to get param 'gt_directory', use default setting: %s", ros::this_node::getName().c_str(), gt_directory.c_str());
  if (!nh.getParam("save_directory", save_directory)) ROS_WARN("[%s] Failed to get param 'save_directory', use default setting: %s", ros::this_node::getName().c_str(), save_directory.c_str());
  if (!nh.getParam("init_directory", init_directory)) ROS_WARN("[%s] Failed to get param 'init_directory', use default setting: %s", ros::this_node::getName().c_str(), init_directory.c_str());
  if (!nh.getParam("bag_name", bag_name)) ROS_WARN("[%s] Failed to get param 'bag_name', use default setting: %s", ros::this_node::getName().c_str(), bag_name.c_str());
  if (!nh.getParam("res_file_name", res_file_name)) ROS_WARN("[%s] Failed to get param 'res_file_name', use default setting: %s", ros::this_node::getName().c_str(), res_file_name.c_str());
  if (!nh.getParam("global_viz", global_viz)) ROS_WARN("[%s] Failed to get param 'global_viz', use default setting: %d", ros::this_node::getName().c_str(), global_viz);
  if (!nh.getParam("save_res", save_res)) ROS_WARN("[%s] Failed to get param 'save_res', use default setting: %d", ros::this_node::getName().c_str(), save_res);
  if (!nh.getParam("start_index", start_index)) ROS_WARN("[%s] Failed to get param 'start_index', use default setting: %d", ros::this_node::getName().c_str(), start_index);
  if (!nh.getParam("global_threshold", global_threshold)) ROS_WARN("[%s] Failed to get param 'global_threshold', use default setting: %f", ros::this_node::getName().c_str(), global_threshold);
  if (!nh.getParam("global_bias", global_bias)) ROS_WARN("[%s] Failed to get param 'global_bias', use default setting: %f", ros::this_node::getName().c_str(), global_bias);
  if (!nh.getParam("global_mask_flag", global_mask_flag)) ROS_WARN("[%s] Failed to get param 'global_mask_flag', use default setting: %f", ros::this_node::getName().c_str(), global_mask_flag);
  // if (!nh.getParam("use_pw", use_pw)) ROS_WARN("[%s] Failed to get param 'use_pw', use default setting: %d", ros::this_node::getName().c_str(), use_pw);
  // if (!nh.getParam("use_imu", use_imu)) ROS_WARN("[%s] Failed to get param 'use_imu', use default setting: %d", ros::this_node::getName().c_str(), use_imu);
  // if (!nh.getParam("ndt2d_grid_resolution", ndt2d_grid_resolution_)) ROS_WARN("[%s] Failed to get param 'ndt2d_grid_resolution', use default setting: %f", ros::this_node::getName().c_str(), ndt2d_grid_resolution_);
  // if (!nh.getParam("image_width", image_width)) ROS_WARN("[%s] Failed to get param 'image_width', use default setting: %d", ros::this_node::getName().c_str(), image_width);
  // if (!nh.getParam("radar_resolution", radar_resolution)) ROS_WARN("[%s] Failed to get param 'radar_resolution', use default setting: %f", ros::this_node::getName().c_str(), radar_resolution);
  // if (!nh.getParam())

  if (!nh.getParam("grid_size1", grid_size1)) ROS_WARN("[%s] Failed to get param 'grid_size1', use default setting: %f", ros::this_node::getName().c_str(), grid_size1);
  if (!nh.getParam("grid_size2", grid_size2)) ROS_WARN("[%s] Failed to get param 'grid_size2', use default setting: %f", ros::this_node::getName().c_str(), grid_size2);
  if (!nh.getParam("grid_size3", grid_size3)) ROS_WARN("[%s] Failed to get param 'grid_size3', use default setting: %f", ros::this_node::getName().c_str(), grid_size3);
  if (!nh.getParam("eps1", eps1)) ROS_WARN("[%s] Failed to get param 'eps1', use default setting: %f", ros::this_node::getName().c_str(), eps1);
  if (!nh.getParam("eps2", eps2)) ROS_WARN("[%s] Failed to get param 'eps2', use default setting: %f", ros::this_node::getName().c_str(), eps2);
  if (!nh.getParam("eps3", eps3)) ROS_WARN("[%s] Failed to get param 'eps3', use default setting: %f", ros::this_node::getName().c_str(), eps3);
  if (!nh.getParam("max_step_size1", max_step_size1)) ROS_WARN("[%s] Failed to get param 'max_step_size1', use default setting: %f", ros::this_node::getName().c_str(), max_step_size1);
  if (!nh.getParam("max_step_size2", max_step_size2)) ROS_WARN("[%s] Failed to get param 'max_step_size2', use default setting: %f", ros::this_node::getName().c_str(), max_step_size2);
  if (!nh.getParam("max_step_size3", max_step_size3)) ROS_WARN("[%s] Failed to get param 'max_step_size3', use default setting: %f", ros::this_node::getName().c_str(), max_step_size3);

  // C2 param
  nh.param<bool>("use_distortion_correction", use_distortion_correction_, false);
  ROS_INFO("[C2] use_distortion_correction=%d", use_distortion_correction_);

  // C4/C5 params
  nh.param<bool>("log_covariance",       log_covariance_,        false);
  nh.param<bool>("log_degeneracy",       log_degeneracy_,        false);
  nh.param<bool>("use_constrained_update", use_constrained_update_, false);
  nh.param<double>("epsilon_reg",        epsilon_reg_,           1e-6);
  nh.param<double>("tau_degeneracy",     tau_degeneracy_,        0.0);
  ROS_INFO("[C4/C5] log_cov=%d log_deg=%d constrained=%d eps_reg=%.2e tau=%.4f",
           log_covariance_, log_degeneracy_, use_constrained_update_, epsilon_reg_, tau_degeneracy_);

  OpenBag(bag_name);
  return 0;
}

void read_directory(const std::string& name, std::vector<std::string>& v)
{
  DIR* dirp = opendir(name.c_str());
  struct dirent * dp;
  while ((dp = readdir(dirp)) != NULL) {
    v.push_back(dp->d_name);
  }
  closedir(dirp);
}

void WriteToFile(Eigen::Matrix4f& T, string& file_path){
  FILE *fp;
  fp = fopen(file_path.c_str(),"a");
  fprintf(fp,"%f %f %f %f %f %f %f %f %f %f %f %f\n",
          T(0,0),T(0,1),0.0 ,T(0,3),
          T(1,0),T(1,1),0.0 ,T(1,3),
          0.0   ,0.0   ,1.0 ,T(2,3));
  fclose(fp);
}

pcl::PointCloud<pcl::PointXYZI>::Ptr make2DPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_3d){
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_2d (new pcl::PointCloud<pcl::PointXYZI>);
  pcl::copyPointCloud(*pc_3d, *pc_2d);
  for(int i=0 ; i<pc_2d->size(); i++){
    pc_2d->points[i].z = 0;
  }
  return pc_2d;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr RadarPointCloudFilter(pcl::PointCloud<pcl::PointXYZI>::Ptr pc_in, float thres){
  pcl::PointCloud<pcl::PointXYZI>::Ptr pc_out (new pcl::PointCloud<pcl::PointXYZI>);
  for(int i=0 ; i<pc_in->size(); i++){
    if(pc_in->points[i].intensity > thres){
      pc_out->push_back(pc_in->points[i]);
    }
  }
  return pc_out;
}

float GetDistanceFromT(Eigen::Matrix4f T){
  return std::sqrt( T(0,3)*T(0,3) + T(1,3)*T(1,3) );
}

float VelocityCal(Eigen::Matrix4f T, double delta_t){
  if (!std::isfinite(delta_t) || delta_t <= 1e-9) return 0.0f;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      if (!std::isfinite(T(i,j))) return 0.0f;

  const double dx = static_cast<double>(T(0,3));
  const double dy = static_cast<double>(T(1,3));
  return static_cast<float>(std::hypot(dx, dy) / delta_t);
}

float AngularVelocityCal(Eigen::Matrix4f T, double delta_t){
  if (!std::isfinite(delta_t) || delta_t <= 1e-9) return 0.0f;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      if (!std::isfinite(T(i,j))) return 0.0f;

  const double r00 = static_cast<double>(T(0,0));
  const double r10 = static_cast<double>(T(1,0));
  const double sy  = std::hypot(r00, r10);
  const double delta_yaw = (sy >= 1e-9) ? std::atan2(r10, r00) : 0.0;  // [rad]
  const double ang_vel_deg_s = (-delta_yaw * 180.0 / M_PI) / delta_t;  // [deg/s]
  return std::isfinite(ang_vel_deg_s) ? static_cast<float>(ang_vel_deg_s) : 0.0f;
}

void DoPWNDT(const pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc, const pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc, pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud, float grid_step_, double eps_, double max_step_size_, float bias_,
              Eigen::Matrix4f &T, double &score,
              Eigen::Matrix< pcl::ndt2d::NormalDist<pcl::PointXYZI> , Eigen::Dynamic, Eigen::Dynamic>& normal_distributions_map,
              Eigen::Matrix3d &H_raw_out, Eigen::Matrix3d &H_reg_out, bool &H_valid_out){

  pcl::PWNormalDistributionsTransform2D<pcl::PointXYZI, pcl::PointXYZI> pw_ndt;

  if (use_pw){
    pw_ndt.setMaximumIterations (ndt2d_max_it_);
    pw_ndt.setTransformationEpsilon (eps_);
    // C5: pass constrained-update settings to the NDT object
    pw_ndt.setUseConstrainedUpdate(use_constrained_update_);
    pw_ndt.setTauDegeneracy(tau_degeneracy_);
    Eigen::Vector2f grid_center;	grid_center << 0, 0;
    pw_ndt.setGridCentre (grid_center);
    Eigen::Vector2f grid_extent;	grid_extent << ndt2d_grid_extent_, ndt2d_grid_extent_;
    pw_ndt.setGridExtent (grid_extent);
    Eigen::Vector2f grid_step;	grid_step << grid_step_, grid_step_;
    pw_ndt.setGridStep (grid_step);
    pw_ndt.setMaxStepSize(max_step_size_);
    pw_ndt.setIntensityBias(bias_);
    pw_ndt.setInputSource (old_in_pc);
    pw_ndt.setInputTarget (in_pc);
    const Eigen::Matrix4f T_initial_guess = T;  // save before align() overwrites T
    pw_ndt.align (*output_cloud, T);
    score = pw_ndt.getFitnessScore ();
    T = pw_ndt.getFinalTransformation();
    normal_distributions_map = pw_ndt.getNormalDistributionMap();
    // C4/C5: retrieve Hessian from NDT object
    H_raw_out   = pw_ndt.getFinalHessianRaw();
    H_reg_out   = pw_ndt.getFinalHessianRegularised();
    H_valid_out = pw_ndt.isFinalHessianValid();

    // ── C5: Degeneracy-aware post-alignment constraint ────────────────────
    // Applied ONCE after full convergence, not per-Newton-step.
    // Uses eigenvectors of H_raw_out (true degenerate directions before PD
    // correction) to zero out the transformation delta along degenerate axes.
    // Reference: Zhang, Kaess & Singh (2016), adapted for NDT radar odometry.
    if (use_constrained_update_ && H_valid_out && tau_degeneracy_ > 0.0) {
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(H_raw_out);
      const Eigen::Vector3d& evals = es.eigenvalues();   // ascending
      // Only trigger if lambda_min is POSITIVE and below tau.
      // Negative lambda_min = PD-corrected frame, not geometric degeneracy.
      if (evals(0) > 0.0 && evals(0) < tau_degeneracy_) {
        const Eigen::Matrix3d& evecs = es.eigenvectors();
        Eigen::Vector3d D_diag;
        for (int _k=0; _k<3; _k++)
          D_diag(_k) = (evals(_k) >= tau_degeneracy_) ? 1.0 : 0.0;
        Eigen::Matrix3d P_safe = evecs * D_diag.asDiagonal() * evecs.transpose();

        // Extract delta between initial guess and aligned result in [x,y,theta]
        // T_initial_guess saved before pw_ndt.align(); T is now the final result
        const Eigen::Matrix4f& T_in_ref = T_initial_guess;
        const Eigen::Matrix4f  T_final  = T;  // T = pw_ndt.getFinalTransformation() already
        Eigen::Matrix4f delta_mat = T_in_ref.inverse() * T_final;
        float dx    = delta_mat(0,3);
        float dy    = delta_mat(1,3);
        float dth   = std::atan2(static_cast<float>(delta_mat(1,0)),
                                 static_cast<float>(delta_mat(0,0)));
        Eigen::Vector3d delta_xyt(dx, dy, dth);
        Eigen::Vector3d constrained = P_safe * delta_xyt;

        // Reconstruct constrained transformation from T_in_ref + constrained delta
        float c = std::cos(constrained(2)), s = std::sin(constrained(2));
        Eigen::Matrix4f T_constrained = Eigen::Matrix4f::Identity();
        T_constrained(0,0)=c; T_constrained(0,1)=-s; T_constrained(0,3)=constrained(0);
        T_constrained(1,0)=s; T_constrained(1,1)= c; T_constrained(1,3)=constrained(1);
        T = T_in_ref * T_constrained;
      }
    }
    // ─────────────────────────────────────────────────────────────────────
  } else {
    pcl::NormalDistributionsTransform<pcl::PointXYZI, pcl::PointXYZI> ndt;
    ndt.setTransformationEpsilon (0.01);
    ndt.setStepSize (1);
    ndt.setResolution (10);
    ndt.setMaximumIterations (300);
    pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc_flat (new pcl::PointCloud<pcl::PointXYZI>);
    old_in_pc_flat = make2DPointCloud(old_in_pc);
    pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc_flat (new pcl::PointCloud<pcl::PointXYZI>);
    in_pc_flat = make2DPointCloud(in_pc);
    ndt.setInputSource (old_in_pc_flat);
    ndt.setInputTarget (in_pc_flat);
    ndt.align (*output_cloud);
    std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged ()
              << " score: " << ndt.getFitnessScore () << std::endl;
    T = ndt.getFinalTransformation ();
  }
}

void deleteNDMap(){
  MarkerArray ret;
  Marker m;
  m.action = Marker::DELETEALL;
  ret.markers.push_back(m);
  nd_map_pub.publish(ret);
}

void pubNDMap(Eigen::Matrix< pcl::ndt2d::NormalDist<pcl::PointXYZI> , Eigen::Dynamic, Eigen::Dynamic> normal_distributions_map,
              double scale){
  vector<Marker> marker_vec;
  for (size_t i=0; i<normal_distributions_map.rows(); i++){
    for (size_t j=0; j<normal_distributions_map.cols(); j++){
      Eigen::Vector2d u = normal_distributions_map(i,j).getMean();
      Eigen::Matrix2d cov = normal_distributions_map(i,j).getCovar();
      if(u[0] != 0 && u[1] != 0){
        Marker marker;
        marker = MarkerOfEllipse(u, cov, Color::kAqua, 0.25, scale*2);
        marker_vec.push_back(marker);
        marker = MarkerOfEllipse(u, cov, Color::kAqua, 0.25, scale*3);
        marker_vec.push_back(marker);
        marker = MarkerOfEllipse(u, cov, Color::kAqua, 0.25, scale*4);
        marker_vec.push_back(marker);
      }
    }
  }
  MarkerArray marker_array;
  marker_array = JoinMarkers(marker_vec);
  nd_map_pub.publish(marker_array);
}

bool CheckAcc(Eigen::Matrix4f& T){
  const double dt = safe_dt(curr_t - old_t);

  const float v   = VelocityCal(T, dt);
  const float acc = (v - old_vel) / dt;

  const float w   = AngularVelocityCal(T, dt);
  const float ang_acc = (w - old_ang_vel) / dt;

  if (!std::isfinite(v) || !std::isfinite(w)) return true;

  if (std::abs(acc) > 8.0 || std::abs(ang_acc) > 50.0){
    ROS_WARN("[%d] abnormal acc: %f", Index, acc);
    return true;
  }
  return false;
}

void PrintAcc(Eigen::Matrix4f& T){
  const double dt = safe_dt(curr_t - old_t);

  const float x = T(0,3); const float y = T(1,3);
  const float yaw_deg = static_cast<float>(std::atan2(T(1,0), T(0,0)) * 180.0 / M_PI);
  std::cout << " xytheta: " << x << " " << y << " " << yaw_deg << std::endl;

  const float v = VelocityCal(T, dt);
  const float w = AngularVelocityCal(T, dt);
  const float acc = (v - old_vel) / dt;
  const float ang_acc = (w - old_ang_vel) / dt;

  std::cout << " vel: " << v << " ang_vel: " << w << " "
            << " acc: " << acc << " ang_acc: " << ang_acc << std::endl;
}

/*
1. extract azimuth angles
2. extract and normalize FFT data
3. create a cartesian grid
4. calculate where each cartesian pixel maps to the polar radar image
5. convert from polar image to cartesian radar image
6. convert cartesian as point cloud 
*/
pcl::PointCloud<pcl::PointXYZI>::Ptr RadarPolarToCartesian(
    const cv::Mat& raw_polar_image,
    float vel_mps,
    float ang_vel_rads,
    double sweep_dt) {
    const float radar_resolution = 0.0432f;
    const int encoder_size = 5600;
    const float cart_resolution = 0.125f;
    const int cart_pixel_width = 2001;
    const bool interpolate_crossover = true;

    const int azimuth_rows = raw_polar_image.rows;
    const int fft_columns = raw_polar_image.cols - 11;

    // Step 1: Extract azimuths
    std::vector<float> azimuths(azimuth_rows);
    for (int i = 0; i < azimuth_rows; ++i) {
        uint16_t az_raw = *reinterpret_cast<const uint16_t*>(&raw_polar_image.at<uchar>(i, 8));
        azimuths[i] = static_cast<float>(az_raw) / encoder_size * 2.0f * static_cast<float>(M_PI);
    }
    float azimuth_step = (azimuths.size() >= 2) ? (azimuths[1] - azimuths[0]) : 0.0f;
    if (!std::isfinite(azimuth_step) || std::abs(azimuth_step) < 1e-6f) {
        azimuth_step = 2.0f * static_cast<float>(M_PI) / 4000.0f; // safe fallback
    }

    // Step 2: Normalize FFT values
    cv::Mat fft_data(azimuth_rows, fft_columns, CV_32F);
    for (int i = 0; i < azimuth_rows; ++i) {
        for (int j = 0; j < fft_columns; ++j) {
            fft_data.at<float>(i, j) = static_cast<float>(raw_polar_image.at<uchar>(i, 11 + j)) / 255.0f;
        }
    }

    // Interpolate crossover
    if (interpolate_crossover) {
        cv::Mat top = fft_data.row(fft_data.rows - 1);
        cv::Mat bottom = fft_data.row(0);
        cv::vconcat(top, fft_data, fft_data);
        cv::vconcat(fft_data, bottom, fft_data);
    }

    // Step 3: Cartesian grid
    float cart_min_range = cart_pixel_width / 2 * cart_resolution;
    std::vector<float> coords(cart_pixel_width);
    for (int i = 0; i < cart_pixel_width; ++i) {
        coords[i] = -cart_min_range + i * cart_resolution;
    }

    cv::Mat X, Y;
    cv::repeat(cv::Mat(coords).t(), cart_pixel_width, 1, X);
    cv::repeat(cv::Mat(coords), 1, cart_pixel_width, Y);

    cv::Mat range, angle;
    cv::magnitude(X, Y, range);
    cv::phase(X, Y, angle, false);

    cv::Mat negative_mask;
    cv::compare(angle, 0, negative_mask, cv::CMP_LT);
    negative_mask.convertTo(negative_mask, CV_32F, 2.0 * CV_PI);
    angle += negative_mask;

    // Step 4: Generate remap coords
    cv::Mat u = (range - radar_resolution / 2.0f) / radar_resolution;
    cv::Mat v = (angle - azimuths[0]) / azimuth_step;
    if (interpolate_crossover) v += 1.0f;

    // ── C2: Per-azimuth motion distortion compensation ──────────────────────────
    // Each output pixel maps to an azimuth index (v). Points at higher v were
    // captured later in the sweep when the vehicle had moved further.
    // We correct by transforming the world coordinate of each pixel forward
    // by the vehicle’s displacement at that azimuth’s capture time, then
    // re-derive the range to sample the correct polar bin.
    //
    // alpha = fractional sweep time: 0 at azimuth 0, 1 at final azimuth
    // Correction transform (small angle, body frame: x=right, y=forward in Oxford NDT):
    //   x_sensor = x_world + dx*sin(a_world) - dy*cos(a_world)  (cylindrical shift)
    // Simplified to range-only shift (dominant term, valid for straight driving):
    //   range_corrected = range + dx_alpha  where dx_alpha = vel * alpha * dt
    // The range shift moves the sample bin along the azimuth ray direction.
    // ───────────────────────────────────────────────────────────────────────────
    if (std::abs(vel_mps) > 0.01f && sweep_dt > 1e-6) {
        // ── C2: Range-based per-azimuth undistortion ─────────────────────────
        // Physics: vehicle moves forward by d = vel*alpha*dt during the sweep.
        // A world point at true range r0 appears at r_i ≈ r0 - d*cos(phys_az).
        // Recovery: r0 = r_i + d * cos(phys_az)
        //
        // Grid convention: X=right, Y=backward, grid angle 3π/2 = forward.
        // cos(phys_az) = cos(angle_grid - 3π/2) = -sin(angle_grid)
        // → range_corr = range - d * alpha * sin(angle_grid)
        //
        // Verification:
        //  Forward  (angle=3π/2): sin=-1 → range_corr = range + d  ✓
        //  Backward (angle=π/2):  sin=+1 → range_corr = range - d  ✓
        //  Sideways (angle=0,π):  sin=0  → no correction            ✓
        // ─────────────────────────────────────────────────────────────────────
        const float n_az = static_cast<float>(azimuth_rows);
        cv::Mat v_raw = v.clone();
        if (interpolate_crossover) v_raw -= 1.0f;
        cv::Mat alpha = v_raw / n_az;
        alpha.setTo(0.0f, alpha < 0.0f);
        alpha.setTo(1.0f, alpha > 1.0f);

        const float total_d = vel_mps * static_cast<float>(sweep_dt);

        // sin(angle) per pixel via polarToCart(mag=1): x=cos(angle), y=sin(angle)
        cv::Mat sin_angle, dummy_cos;
        cv::polarToCart(cv::Mat::ones(angle.size(), CV_32F), angle, dummy_cos, sin_angle);

        // r0 = r_i - d * alpha * sin(angle_grid)
        cv::Mat range_corr = range - total_d * alpha.mul(sin_angle);
        u = (range_corr - radar_resolution / 2.0f) / radar_resolution;

        std::cout << "[C2] range undistortion: vel=" << vel_mps << " m/s"
                  << "  max_shift=" << total_d << " m" << std::endl;
    }
    // ───────────────────────────────────────────────────────────────────────────

    // Clamp remap indices to valid ranges
    cv::Mat u_max = cv::Mat::ones(u.size(), u.type()) * (fft_columns - 1);
    cv::Mat v_max = cv::Mat::ones(v.size(), v.type()) * (fft_data.rows - 1);
    u.setTo(0, u < 0);
    cv::min(u, u_max, u);
    v.setTo(0, v < 0);
    cv::min(v, v_max, v);

    // Step 5: Remap
    cv::Mat cart_img;
    cv::remap(fft_data, cart_img, u, v, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    // Step 6: Convert to point cloud (OpenMP parallelised — thread-local buffers merged)
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    {
        const int mid = cart_img.rows / 2;
        #pragma omp parallel
        {
            pcl::PointCloud<pcl::PointXYZI> local_cloud;
            #pragma omp for nowait schedule(dynamic, 32)
            for (int i = 0; i < cart_img.rows; ++i) {
                for (int j = 0; j < cart_img.cols; ++j) {
                    float intensity = cart_img.at<float>(i, j) * 255.0f;
                    if (intensity < 70.0f) continue;
                    pcl::PointXYZI pt;
                    pt.x = (j - mid) * cart_resolution;
                    pt.y = -(i - mid) * cart_resolution;
                    pt.z = 0.0f;
                    pt.intensity = intensity;
                    local_cloud.points.push_back(pt);
                }
            }
            #pragma omp critical
            *cloud += local_cloud;
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    return cloud;
}

// OptimizeWithCeres removed — IMU/Ceres pipeline not used in this research

///--- Visualization ---///
void Vizsualization(bool viz,
  const pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc,
  const pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud,
  const pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc){
if(!viz) return;

// Initializing point cloud visualizer
pcl::visualization::PCLVisualizer::Ptr
viewer_final (new pcl::visualization::PCLVisualizer ("3D Viewer"));
viewer_final->setBackgroundColor (0.1, 0.1, 0.1);

// Target cloud (red).
pcl::PointCloud<pcl::PointXYZI>::Ptr in_pc_flat (new pcl::PointCloud<pcl::PointXYZI>);
in_pc_flat = make2DPointCloud(in_pc);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
target_color (in_pc_flat, 255, 0, 0);
viewer_final->addPointCloud<pcl::PointXYZI> (in_pc_flat, target_color, "target_cloud");

// Transformed cloud (green).
pcl::PointCloud<pcl::PointXYZI>::Ptr output_cloud_flat (new pcl::PointCloud<pcl::PointXYZI>);
output_cloud_flat = make2DPointCloud(output_cloud);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
output_color (output_cloud_flat, 0, 255, 0);
viewer_final->addPointCloud<pcl::PointXYZI> (output_cloud_flat, output_color, "output_cloud");

// Previous cloud (blue).
pcl::PointCloud<pcl::PointXYZI>::Ptr old_in_pc_flat (new pcl::PointCloud<pcl::PointXYZI>);
old_in_pc_flat = make2DPointCloud(old_in_pc);
pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI>
input_color (old_in_pc_flat, 0, 30, 255);
viewer_final->addPointCloud<pcl::PointXYZI> (old_in_pc_flat, input_color, "input_cloud");

// Viewer config
viewer_final->addCoordinateSystem (1.0, "global");
viewer_final->initCameraParameters ();
viewer_final->setCameraPosition (0,0,400,0,0,0,0,0,0);

// Spin until closed.
while (!viewer_final->wasStopped ()) {
viewer_final->spinOnce (100);
std::this_thread::sleep_for(std::chrono::milliseconds(5));
}
}
