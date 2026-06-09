#ifndef PCL_NDT_2D_VISUALIZATION_H_
#define PCL_NDT_2D_VISUALIZATION_H_

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>

namespace pcl {
namespace visualization {

/**
 * Fast Gaussian heatmap visualization for NDT2D
 * Optimized with bounding boxes and parallel processing
 */


// template<typename PointT>
// void VisualizeNDTGridHeatmap(
//     const Eigen::Matrix<
//         pcl::ndt2d::NormalDist<PointT>,
//         Eigen::Dynamic, Eigen::Dynamic
//     >& gmap,
//     const std::string& out_path,
//     float map_range = 100.0f,
//     int   img_size  = 1000,
//     float cutoff_std = 4.0f,
//     bool  normalize = true,
//     bool  show_cells = true,
//     bool  show_grid_lines = true,
//     bool  show_cell_centers = false,
//     const Eigen::Vector2f& grid_step = Eigen::Vector2f(1.0f, 1.0f)  // Cell size in meters
// )
// {
//     if (gmap.rows() == 0 || gmap.cols() == 0) {
//         std::cerr << "[WARN] Empty Gaussian map!" << std::endl;
//         return;
//     }

//     cv::Mat heat = cv::Mat::zeros(img_size, img_size, CV_32FC1);
    
//     float half = map_range / 2.0f;
//     float scale = img_size / map_range;
    
//     // Precompute all pixel world coordinates
//     std::vector<float> world_x(img_size), world_y(img_size);
//     #pragma omp parallel for
//     for (int i = 0; i < img_size; i++) {
//         world_x[i] = (i / scale) - half;
//         world_y[i] = half - (i / scale);
//     }
    
//     int rows = gmap.rows();
//     int cols = gmap.cols();
//     int total_cells = rows * cols;
//     int processed_cells = 0;
    
//     // Create a separate image for grid visualization
//     cv::Mat grid_overlay = cv::Mat::zeros(img_size, img_size, CV_8UC3);
    
//     // Calculate cell boundaries in world coordinates
//     // Assuming grid starts at (-half, -half) in world coordinates
//     float cell_width_world = grid_step[0];
//     float cell_height_world = grid_step[1];
    
//     // Process each cell
//     #pragma omp parallel for reduction(+:processed_cells) collapse(2)
//     for (int r = 0; r < rows; r++) {
//         for (int c = 0; c < cols; c++) {
//             const auto& g = gmap(r, c);
            
//             // Skip invalid Gaussians but still draw their cell if requested
//             Eigen::Vector2d mu = g.getMean();
//             bool valid_gaussian = (mu.norm() >= 1e-6);
            
//             if (valid_gaussian) {
//                 Eigen::Matrix2d cov = g.getCovar();
//                 if (cov.determinant() >= 1e-6) {
//                     processed_cells++;
//                     Eigen::Matrix2d inv_cov = cov.inverse();
                    
//                     // Get eigenvalue decomposition for bounding box
//                     Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(cov);
//                     Eigen::Vector2d eigenvalues = eigensolver.eigenvalues();
//                     float max_eigenvalue = eigenvalues.maxCoeff();
                    
//                     // Use cutoff for bounding box
//                     float bound = cutoff_std * std::sqrt(max_eigenvalue);
                    
//                     // Convert to pixel bounds
//                     int px_min = static_cast<int>((mu.x() - bound + half) * scale);
//                     int px_max = static_cast<int>((mu.x() + bound + half) * scale);
//                     int py_min = static_cast<int>((half - (mu.y() + bound)) * scale);
//                     int py_max = static_cast<int>((half - (mu.y() - bound)) * scale);
                    
//                     px_min = std::max(0, std::min(img_size - 1, px_min));
//                     px_max = std::max(0, std::min(img_size - 1, px_max));
//                     py_min = std::max(0, std::min(img_size - 1, py_min));
//                     py_max = std::max(0, std::min(img_size - 1, py_max));
                    
//                     // Precompute covariance terms
//                     double a = inv_cov(0, 0);
//                     double b = inv_cov(0, 1);
//                     double d = inv_cov(1, 1);
                    
//                     // Evaluate Gaussian within bounding box
//                     for (int py = py_min; py <= py_max; py++) {
//                         float wy = world_y[py];
//                         float dy_base = wy - mu.y();
//                         float* row_ptr = heat.ptr<float>(py);
                        
//                         for (int px = px_min; px <= px_max; px++) {
//                             float wx = world_x[px];
//                             float dx = wx - mu.x();
//                             float dy = dy_base;
                            
//                             double mahalanobis = dx * (a * dx + b * dy) + dy * (b * dx + d * dy);
                            
//                             if (mahalanobis > cutoff_std * cutoff_std) continue;
                            
//                             row_ptr[px] += static_cast<float>(std::exp(-0.5 * mahalanobis));
//                         }
//                     }
//                 }
//             }
            
//             // Draw cell boundaries (serial section for drawing)
//             #pragma omp critical
//             if (show_cells) {
//                 // Calculate cell corners in world coordinates
//                 float cell_left = -half + c * cell_width_world;
//                 float cell_right = cell_left + cell_width_world;
//                 float cell_top = half - r * cell_height_world;  // Note: Y flipped
//                 float cell_bottom = cell_top - cell_height_world;
                
//                 // Convert to pixel coordinates
//                 int px_left = static_cast<int>((cell_left + half) * scale);
//                 int px_right = static_cast<int>((cell_right + half) * scale);
//                 int py_top = static_cast<int>((half - cell_top) * scale);
//                 int py_bottom = static_cast<int>((half - cell_bottom) * scale);
                
//                 // Clamp to image bounds
//                 px_left = std::max(0, std::min(img_size - 1, px_left));
//                 px_right = std::max(0, std::min(img_size - 1, px_right));
//                 py_top = std::max(0, std::min(img_size - 1, py_top));
//                 py_bottom = std::max(0, std::min(img_size - 1, py_bottom));
                
//                 // Draw cell rectangle
//                 cv::Scalar cell_color;
//                 if (valid_gaussian) {
//                     // Valid cells in blue
//                     cell_color = cv::Scalar(255, 150, 0);  // BGR: Orange
//                 } else {
//                     // Empty cells in gray
//                     cell_color = cv::Scalar(100, 100, 100);
//                 }
                
//                 cv::rectangle(grid_overlay, 
//                             cv::Point(px_left, py_top),
//                             cv::Point(px_right, py_bottom),
//                             cell_color, 1);
                
//                 // Draw cell center if requested
//                 if (show_cell_centers) {
//                     float cell_center_x = (cell_left + cell_right) / 2.0f;
//                     float cell_center_y = (cell_top + cell_bottom) / 2.0f;
//                     int px_center = static_cast<int>((cell_center_x + half) * scale);
//                     int py_center = static_cast<int>((half - cell_center_y) * scale);
                    
//                     if (px_center >= 0 && px_center < img_size && 
//                         py_center >= 0 && py_center < img_size) {
//                         cv::circle(grid_overlay, 
//                                   cv::Point(px_center, py_center),
//                                   2, cv::Scalar(0, 255, 255), -1);  // Cyan centers
//                     }
//                 }
                
//                 // Draw Gaussian mean if valid
//                 if (valid_gaussian) {
//                     int px_mean = static_cast<int>((mu.x() + half) * scale);
//                     int py_mean = static_cast<int>((half - mu.y()) * scale);
                    
//                     if (px_mean >= 0 && px_mean < img_size && 
//                         py_mean >= 0 && py_mean < img_size) {
//                         cv::circle(grid_overlay, 
//                                   cv::Point(px_mean, py_mean),
//                                   3, cv::Scalar(0, 0, 255), -1);  // Red means
//                     }
//                 }
//             }
//         }
//     }
    
//     std::cout << "[INFO] Processed " << processed_cells << "/" << total_cells 
//               << " valid Gaussians" << std::endl;
//     std::cout << "[INFO] Grid size: " << rows << "x" << cols 
//               << " cells, Cell size: " << grid_step[0] << "x" << grid_step[1] 
//               << " meters" << std::endl;
    
//     // Convert heatmap to 8-bit and apply color map
//     cv::Mat heat_display;
//     if (normalize) {
//         cv::normalize(heat, heat_display, 0, 255, cv::NORM_MINMAX);
//         heat_display.convertTo(heat_display, CV_8UC1);
//     } else {
//         heat.convertTo(heat_display, CV_8UC1, 255.0);
//     }
    
//     cv::Mat heat_color;
//     cv::applyColorMap(heat_display, heat_color, cv::COLORMAP_JET);
    
//     // Draw overall grid lines if requested
//     if (show_grid_lines) {
//         // Main grid lines (every N cells for clarity)
//         int grid_line_interval = std::max(1, std::min(rows, cols) / 10);
        
//         // Vertical grid lines
//         for (int i = 0; i <= cols; i += grid_line_interval) {
//             float world_x = -half + i * cell_width_world;
//             int px = static_cast<int>((world_x + half) * scale);
//             if (px >= 0 && px < img_size) {
//                 cv::line(heat_color, cv::Point(px, 0), cv::Point(px, img_size),
//                         cv::Scalar(255, 255, 255), 2);
//             }
//         }
        
//         // Horizontal grid lines
//         for (int j = 0; j <= rows; j += grid_line_interval) {
//             float world_y = half - j * cell_height_world;
//             int py = static_cast<int>((half - world_y) * scale);
//             if (py >= 0 && py < img_size) {
//                 cv::line(heat_color, cv::Point(0, py), cv::Point(img_size, py),
//                         cv::Scalar(255, 255, 255), 2);
//             }
//         }
        
//         // Draw boundary of the entire grid
//         float grid_left = -half;
//         float grid_right = -half + cols * cell_width_world;
//         float grid_top = half;
//         float grid_bottom = half - rows * cell_height_world;
        
//         int px_left = static_cast<int>((grid_left + half) * scale);
//         int px_right = static_cast<int>((grid_right + half) * scale);
//         int py_top = static_cast<int>((half - grid_top) * scale);
//         int py_bottom = static_cast<int>((half - grid_bottom) * scale);
        
//         cv::rectangle(heat_color,
//                      cv::Point(px_left, py_top),
//                      cv::Point(px_right, py_bottom),
//                      cv::Scalar(255, 255, 0), 2);  // Yellow boundary
//     }
    
//     // Combine heatmap with grid overlay
//     if (show_cells) {
//         // Blend grid overlay with heatmap (alpha blending)
//         cv::addWeighted(heat_color, 0.7, grid_overlay, 0.3, 0, heat_color);
//     }
    
//     // Add legend/text overlay
//     cv::putText(heat_color, 
//                 "NDT Grid: " + std::to_string(rows) + "x" + std::to_string(cols) + 
//                 " cells, " + std::to_string(processed_cells) + " valid Gaussians",
//                 cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
//                 cv::Scalar(255, 255, 255), 2);
    
//     cv::putText(heat_color, 
//                 "Cell size: " + std::to_string(grid_step[0]) + "x" + 
//                 std::to_string(grid_step[1]) + "m",
//                 cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
//                 cv::Scalar(255, 255, 255), 2);
    
//     // Save image
//     cv::imwrite(out_path, heat_color);
//     std::cout << "[INFO] Saved NDT Grid Heatmap: " << out_path << std::endl;
// }

// template<typename PointT>
// void VisualizeNDTGridHeatmap(
//     const Eigen::Matrix<pcl::ndt2d::NormalDist<PointT>, Eigen::Dynamic, Eigen::Dynamic>& gmap,
//     const std::string& out_path,
//     const Eigen::Vector2f& grid_center,        // <-- REQUIRED
//     const Eigen::Vector2f& grid_extent,        // <-- REQUIRED
//     const Eigen::Vector2f& grid_step,          // <-- REQUIRED
//     float map_range = 100.0f,
//     int img_size = 1000,
//     float cutoff_std = 4.0f,
//     bool normalize = true,
//     bool show_cells = true,
//     bool show_grid_lines = true,
//     bool show_cell_centers = false
// )
// {
//     if (gmap.rows() == 0 || gmap.cols() == 0) {
//         std::cerr << "[WARN] Empty Gaussian map!" << std::endl;
//         return;
//     }

//     int rows = gmap.rows();
//     int cols = gmap.cols();

//     float half = map_range * 0.5f;
//     float scale = img_size / map_range;

//     // Heatmap (float intensity)
//     cv::Mat heat = cv::Mat::zeros(img_size, img_size, CV_32FC1);

//     // Precompute world coordinates for each pixel
//     std::vector<float> world_x(img_size), world_y(img_size);

//     #pragma omp parallel for
//     for (int i = 0; i < img_size; i++) {
//         world_x[i] = (i / scale) - half;
//         world_y[i] = half - (i / scale);  // Y inverted
//     }

//     // Compute grid origin in world coordinate
//     float x_min = grid_center.x() - grid_extent.x() * 0.5f;
//     float y_min = grid_center.y() - grid_extent.y() * 0.5f;

//     cv::Mat grid_overlay = cv::Mat::zeros(img_size, img_size, CV_8UC3);

//     int processed_cells = 0;

//     // MAIN LOOP
//     #pragma omp parallel for collapse(2) reduction(+:processed_cells)
//     for (int r = 0; r < rows; r++) {
//         for (int c = 0; c < cols; c++) {

//             const auto& g = gmap(r, c);
//             Eigen::Vector2d mu = g.getMean();

//             bool valid_gaussian = (mu.norm() > 1e-6);

//             if (valid_gaussian) {
//                 Eigen::Matrix2d cov = g.getCovar();
//                 if (cov.determinant() >= 1e-6) {
//                     processed_cells++;

//                     Eigen::Matrix2d inv_cov = cov.inverse();
//                     double a = inv_cov(0,0);
//                     double b = inv_cov(0,1);
//                     double d = inv_cov(1,1);

//                     // bounding box
//                     Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
//                     float maxev = es.eigenvalues().maxCoeff();
//                     float radius = cutoff_std * std::sqrt(maxev);

//                     int px_min = std::clamp<int>((mu.x() - radius + half) * scale, 0, img_size-1);
//                     int px_max = std::clamp<int>((mu.x() + radius + half) * scale, 0, img_size-1);
//                     int py_min = std::clamp<int>((half - (mu.y() + radius)) * scale, 0, img_size-1);
//                     int py_max = std::clamp<int>((half - (mu.y() - radius)) * scale, 0, img_size-1);

//                     // Evaluate Gaussian
//                     for (int py = py_min; py <= py_max; py++) {
//                         float wy = world_y[py];
//                         float dy_base = wy - mu.y();
//                         float* row_ptr = heat.ptr<float>(py);

//                         for (int px = px_min; px <= px_max; px++) {
//                             float wx = world_x[px];
//                             float dx = wx - mu.x();
//                             float dy = dy_base;

//                             double m = dx*(a*dx + b*dy) + dy*(b*dx + d*dy);
//                             if (m > cutoff_std*cutoff_std) continue;

//                             row_ptr[px] += std::exp(-0.5 * m);
//                         }
//                     }
//                 }
//             }

//             // -------- CELL DRAWING (Corrected) ----------
//             #pragma omp critical
//             if (show_cells) 
//             {
//                 float cell_left   = x_min + c * grid_step.x();
//                 float cell_right  = cell_left + grid_step.x();
//                 float cell_bottom = y_min + r * grid_step.y();
//                 float cell_top    = cell_bottom + grid_step.y();

//                 int px_left  = std::clamp<int>((cell_left  + half) * scale, 0, img_size-1);
//                 int px_right = std::clamp<int>((cell_right + half) * scale, 0, img_size-1);
//                 int py_top   = std::clamp<int>((half - cell_top) * scale, 0, img_size-1);
//                 int py_bot   = std::clamp<int>((half - cell_bottom) * scale, 0, img_size-1);

//                 cv::Scalar col = valid_gaussian ?
//                     cv::Scalar(255,150,0) :
//                     cv::Scalar(100,100,100);

//                 cv::rectangle(grid_overlay,
//                               cv::Point(px_left, py_top),
//                               cv::Point(px_right, py_bot),
//                               col, 1);

//                 if (show_cell_centers) {
//                     float cx = (cell_left + cell_right)*0.5f;
//                     float cy = (cell_top + cell_bottom)*0.5f;

//                     int px = (cx + half)*scale;
//                     int py = (half - cy)*scale;

//                     if (px >= 0 && px < img_size && py >= 0 && py < img_size)
//                         cv::circle(grid_overlay, {px,py}, 2, cv::Scalar(0,255,255), -1);
//                 }

//                 // Gaussian mean marker
//                 if (valid_gaussian) {
//                     int px = (mu.x() + half) * scale;
//                     int py = (half - mu.y()) * scale;

//                     if (px>=0 && px<img_size && py>=0 && py<img_size)
//                         cv::circle(grid_overlay, {px, py}, 3, cv::Scalar(0,0,255), -1);
//                 }
//             }
//         }
//     }

//     // -------- Normalize heatmap --------
//     cv::Mat heat_u8;
//     if (normalize) {
//         cv::normalize(heat, heat_u8, 0, 255, cv::NORM_MINMAX);
//         heat_u8.convertTo(heat_u8, CV_8UC1);
//     } else {
//         heat.convertTo(heat_u8, CV_8UC1);
//     }

//     cv::Mat heat_color;
//     cv::applyColorMap(heat_u8, heat_color, cv::COLORMAP_JET);

//     // Overlay cells
//     if (show_cells)
//         cv::addWeighted(heat_color, 0.75, grid_overlay, 0.25, 0, heat_color);

//     // -------- Save --------
//     cv::putText(heat_color,
//                 "Cells: " + std::to_string(rows) + "x" + std::to_string(cols),
//                 {10,30}, 1, 0.8, {255,255,255}, 2);

//     cv::imwrite(out_path, heat_color);

//     std::cout << "[INFO] Saved: " << out_path << "\n";
// }

// template<typename PointT>
// void VisualizeNDTGridHeatmap(
//     const Eigen::Matrix<pcl::ndt2d::NormalDist<PointT>, Eigen::Dynamic, Eigen::Dynamic>& gmap,
//     const std::string& out_path,
//     const Eigen::Vector2f& grid_center,        // center of the NDT grid
//     const Eigen::Vector2f& grid_extent,        // full size of the grid
//     const Eigen::Vector2f& grid_step,          // cell size
//     float map_range = 100.0f,
//     int img_size = 1000,
//     float cutoff_std = 4.0f,
//     bool normalize = true,
//     bool show_cells = true,
//     bool show_grid_lines = true,
//     bool show_cell_centers = false
// )
// {
//     if (gmap.rows() == 0 || gmap.cols() == 0) {
//         std::cerr << "[WARN] Empty Gaussian map!" << std::endl;
//         return;
//     }

//     int rows = gmap.rows();
//     int cols = gmap.cols();

//     float half = map_range * 0.5f;
//     float scale = img_size / map_range;

//     cv::Mat heat = cv::Mat::zeros(img_size, img_size, CV_32FC1);

//     // Precompute world coordinates
//     std::vector<float> world_x(img_size), world_y(img_size);
//     for (int i = 0; i < img_size; i++) {
//         world_x[i] = (i / scale) - half;
//         world_y[i] = half - (i / scale); // invert Y
//     }

//     // Compute grid origin in world coordinates
//     float x_min = grid_center.x() - grid_extent.x() * 0.5f;
//     float y_min = grid_center.y() - grid_extent.y() * 0.5f;

//     cv::Mat grid_overlay = cv::Mat::zeros(img_size, img_size, CV_8UC3);

//     int processed_cells = 0;

//     // Iterate over all grid cells
//     for (int r = 0; r < rows; r++) {
//         for (int c = 0; c < cols; c++) {

//             const auto& g = gmap(r, c);
//             Eigen::Vector2d mu = g.getMean();

//             bool valid_gaussian = (mu.norm() > 1e-6);
//             if (valid_gaussian) {
//                 Eigen::Matrix2d cov = g.getCovar();
//                 if (cov.determinant() >= 1e-6) {
//                     processed_cells++;

//                     Eigen::Matrix2d inv_cov = cov.inverse();
//                     double a = inv_cov(0,0), b = inv_cov(0,1), d = inv_cov(1,1);

//                     // Bounding box
//                     Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
//                     float maxev = es.eigenvalues().maxCoeff();
//                     float radius = cutoff_std * std::sqrt(maxev);

//                     int px_min = std::clamp<int>((mu.x() - radius + half) * scale, 0, img_size-1);
//                     int px_max = std::clamp<int>((mu.x() + radius + half) * scale, 0, img_size-1);
//                     int py_min = std::clamp<int>((half - (mu.y() + radius)) * scale, 0, img_size-1);
//                     int py_max = std::clamp<int>((half - (mu.y() - radius)) * scale, 0, img_size-1);

//                     for (int py = py_min; py <= py_max; py++) {
//                         float wy = world_y[py];
//                         float dy_base = wy - mu.y();
//                         float* row_ptr = heat.ptr<float>(py);

//                         for (int px = px_min; px <= px_max; px++) {
//                             float wx = world_x[px];
//                             float dx = wx - mu.x();
//                             double m = dx*(a*dx + b*dy_base) + dy_base*(b*dx + d*dy_base);
//                             if (m > cutoff_std*cutoff_std) continue;
//                             row_ptr[px] += std::exp(-0.5 * m);
//                         }
//                     }
//                 }
//             }

//             // -------- CELL DRAWING ----------
//             if (show_cells) {
//                 float cell_left   = x_min + c * grid_step.x();
//                 float cell_right  = cell_left + grid_step.x();
//                 float cell_bottom = y_min + r * grid_step.y();
//                 float cell_top    = cell_bottom + grid_step.y();

//                 int px_left  = std::clamp<int>((cell_left + half) * scale, 0, img_size-1);
//                 int px_right = std::clamp<int>((cell_right + half) * scale, 0, img_size-1);
//                 int py_top   = std::clamp<int>((half - cell_top) * scale, 0, img_size-1);
//                 int py_bot   = std::clamp<int>((half - cell_bottom) * scale, 0, img_size-1);

//                 cv::Scalar col = valid_gaussian ? cv::Scalar(255,150,0) : cv::Scalar(100,100,100);

//                 if (show_grid_lines)
//                     cv::rectangle(grid_overlay,
//                                   cv::Point(px_left, py_top),
//                                   cv::Point(px_right, py_bot),
//                                   col, 1);

//                 // Cell centers
//                 if (show_cell_centers) {
//                     int cx = (int)((cell_left + cell_right)*0.5f + half) * scale;
//                     int cy = (int)(half - (cell_bottom + grid_step.y()*0.5f)) * scale;
//                     cv::circle(grid_overlay, {cx,cy}, 2, cv::Scalar(0,255,255), -1);
//                 }

//                 // Gaussian mean marker
//                 if (valid_gaussian) {
//                     int px = (int)((mu.x() + half) * scale);
//                     int py = (int)(half - mu.y()) * scale;
//                     if (px>=0 && px<img_size && py>=0 && py<img_size)
//                         cv::circle(grid_overlay, {px, py}, 3, cv::Scalar(0,0,255), -1);
//                 }
//             }
//         }
//     }

//     // Normalize heatmap
//     cv::Mat heat_u8;
//     if (normalize) {
//         cv::normalize(heat, heat_u8, 0, 255, cv::NORM_MINMAX);
//         heat_u8.convertTo(heat_u8, CV_8UC1);
//     } else {
//         heat.convertTo(heat_u8, CV_8UC1);
//     }

//     cv::Mat heat_color;
//     cv::applyColorMap(heat_u8, heat_color, cv::COLORMAP_JET);

//     if (show_cells)
//         cv::addWeighted(heat_color, 0.75, grid_overlay, 0.25, 0, heat_color);

//     cv::putText(heat_color,
//                 "Cells: " + std::to_string(rows) + "x" + std::to_string(cols),
//                 {10,30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255,255,255}, 2);

//     cv::imwrite(out_path, heat_color);
//     std::cout << "[INFO] Saved: " << out_path << "\n";
// }


template<typename PointT>
void VisualizeSingleNDTCell(
    const pcl::ndt2d::NormalDist<PointT>& cell,
    const Eigen::Vector2f& cell_min,    // bottom-left corner of the cell
    const Eigen::Vector2f& cell_max,    // top-right corner of the cell
    const std::string& out_path,
    int img_size = 200,
    float cutoff_std = 4.0f)
{
    cv::Mat img = cv::Mat::zeros(img_size, img_size, CV_8UC3);

    Eigen::Vector2d mu = cell.getMean();
    Eigen::Matrix2d cov = cell.getCovar();

    if (mu.norm() < 1e-6 || cov.determinant() < 1e-6) {
        std::cerr << "[WARN] Invalid Gaussian for this cell!" << std::endl;
        cv::imwrite(out_path, img);
        return;
    }

    Eigen::Matrix2d inv_cov = cov.inverse();

    float scale_x = img_size / (cell_max.x() - cell_min.x());
    float scale_y = img_size / (cell_max.y() - cell_min.y());

    // Draw Gaussian heatmap
    for (int y = 0; y < img_size; ++y) {
        double wy = cell_max.y() - y / scale_y;
        for (int x = 0; x < img_size; ++x) {
            double wx = cell_min.x() + x / scale_x;
            Eigen::Vector2d diff(wx - mu.x(), wy - mu.y());
            double val = std::exp(-0.5 * diff.transpose() * inv_cov * diff);
            uchar intensity = static_cast<uchar>(std::min(val * 255.0, 255.0));
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(intensity, intensity, intensity);
        }
    }

    // Draw mean as red dot
    int px = static_cast<int>((mu.x() - cell_min.x()) * scale_x);
    int py = static_cast<int>((cell_max.y() - mu.y()) * scale_y);
    cv::circle(img, {px, py}, 2, {0, 0, 255}, -1);

    // Draw covariance ellipse in green
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
    Eigen::Vector2d evals = es.eigenvalues();
    Eigen::Matrix2d evecs = es.eigenvectors();
    double angle = std::atan2(evecs(1, 1), evecs(0, 1)) * 180.0 / M_PI;
    cv::Size axes(cutoff_std * std::sqrt(evals[0]) * scale_x,
                  cutoff_std * std::sqrt(evals[1]) * scale_y);
    cv::ellipse(img, cv::Point(px, py), axes, angle, 0, 360, {0, 255, 0}, 1);

    cv::imwrite(out_path, img);
    std::cout << "[INFO] Saved single NDT cell: " << out_path << std::endl;
}

template<typename PointT>
void VisualizeMostPopulatedNDTCell(
    const Eigen::Matrix<pcl::ndt2d::NormalDist<PointT>, Eigen::Dynamic, Eigen::Dynamic>& gmap,
    const Eigen::Vector2f& grid_center,
    const Eigen::Vector2f& grid_extent,
    const Eigen::Vector2f& grid_step,
    const std::string& out_path,
    int img_size = 200,
    float cutoff_std = 4.0f)
{
    if (gmap.rows() == 0 || gmap.cols() == 0) {
        std::cerr << "[WARN] Empty Gaussian map!" << std::endl;
        return;
    }

    int rows = gmap.rows();
    int cols = gmap.cols();

    size_t max_points = 0;
    int best_r = -1, best_c = -1;

    // Find the most populated cell
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const auto& cell = gmap(r, c);
            if (cell.getNumPoints() > max_points) {
                max_points = cell.getNumPoints();
                best_r = r;
                best_c = c;
            }
        }
    }

    if (best_r < 0 || best_c < 0) {
        std::cerr << "[WARN] No populated NDT cells found!" << std::endl;
        return;
    }

    std::cout << "[INFO] Most populated cell: (" << best_r << "," << best_c << ") with "
              << max_points << " points.\n";

    // Compute cell bounds
    Eigen::Vector2f cell_min = grid_center - grid_extent + Eigen::Vector2f(best_c*grid_step.x(), best_r*grid_step.y());
    Eigen::Vector2f cell_max = cell_min + grid_step;

    // Visualize this cell
    auto& cell = gmap(best_r, best_c);
    VisualizeSingleNDTCell(cell, cell_min, cell_max, out_path, img_size, cutoff_std);
}

template<typename PointT>
void VisualizeNDTWithPoints(
    const Eigen::Matrix<pcl::ndt2d::NormalDist<PointT>, Eigen::Dynamic, Eigen::Dynamic>& gmap,
    const typename pcl::PointCloud<PointT>::ConstPtr points,
    const Eigen::Vector2f& grid_center,
    const Eigen::Vector2f& grid_extent,
    const Eigen::Vector2f& grid_step,
    const std::string& out_path,
    float map_range = 100.0f,
    int img_size = 1000,
    float cutoff_std = 4.0f)
{
    // First, generate heatmap using existing function logic
    cv::Mat heat = cv::Mat::zeros(img_size, img_size, CV_32FC1);
    float half = map_range * 0.5f;
    float scale = img_size / map_range;

    std::vector<float> world_x(img_size), world_y(img_size);
    for (int i = 0; i < img_size; i++) {
        world_x[i] = (i / scale) - half;
        world_y[i] = half - (i / scale);
    }

    int rows = gmap.rows();
    int cols = gmap.cols();

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const auto& g = gmap(r, c);
            Eigen::Vector2d mu = g.getMean();
            Eigen::Matrix2d cov = g.getCovar();
            if (mu.norm() < 1e-6 || cov.determinant() < 1e-6) continue;

            Eigen::Matrix2d inv_cov = cov.inverse();
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
            float radius = cutoff_std * std::sqrt(es.eigenvalues().maxCoeff());

            int px_min = std::clamp<int>((mu.x() - radius + half) * scale, 0, img_size-1);
            int px_max = std::clamp<int>((mu.x() + radius + half) * scale, 0, img_size-1);
            int py_min = std::clamp<int>((half - (mu.y() + radius)) * scale, 0, img_size-1);
            int py_max = std::clamp<int>((half - (mu.y() - radius)) * scale, 0, img_size-1);

            for (int py = py_min; py <= py_max; py++) {
                float wy = world_y[py];
                float* row_ptr = heat.ptr<float>(py);
                for (int px = px_min; px <= px_max; px++) {
                    float wx = world_x[px];
                    Eigen::Vector2d diff(wx - mu.x(), wy - mu.y());
                    double m = diff.transpose() * inv_cov * diff;
                    if (m > cutoff_std*cutoff_std) continue;
                    row_ptr[px] += std::exp(-0.5 * m);
                }
            }
        }
    }

    // Normalize and colorize heatmap
    cv::Mat heat_u8, heat_color;
    cv::normalize(heat, heat_u8, 0, 255, cv::NORM_MINMAX);
    heat_u8.convertTo(heat_u8, CV_8UC1);
    cv::applyColorMap(heat_u8, heat_color, cv::COLORMAP_JET);

    // Overlay aligned points
    for (const auto& p : points->points) {
        int px = static_cast<int>((p.x + half) * scale);
        int py = static_cast<int>((half - p.y) * scale);
        if (px >= 0 && px < img_size && py >= 0 && py < img_size)
            heat_color.at<cv::Vec3b>(py, px) = cv::Vec3b(255,255,255); // white dot
    }

    cv::imwrite(out_path, heat_color);
    std::cout << "[INFO] Saved heatmap with aligned points: " << out_path << std::endl;
}


template<typename PointT>
void VisualizeNDTGridHeatmap(
    const Eigen::Matrix<pcl::ndt2d::NormalDist<PointT>, Eigen::Dynamic, Eigen::Dynamic>& gmap,
    const std::string& out_path,
    const Eigen::Vector2f& grid_center,
    const Eigen::Vector2f& grid_extent,
    const Eigen::Vector2f& grid_step,
    float map_range = 100.0f,
    int img_size = 1000,
    float cutoff_std = 4.0f,
    bool normalize = true,
    bool show_cells = true,
    bool show_grid_lines = true,
    bool show_cell_centers = false,
    bool show_ellipses = true      // <-- NEW
)
{
    if (gmap.rows() == 0 || gmap.cols() == 0) {
        std::cerr << "[WARN] Empty Gaussian map!" << std::endl;
        return;
    }

    int rows = gmap.rows();
    int cols = gmap.cols();

    float half = map_range * 0.5f;
    float scale = img_size / map_range;

    cv::Mat heat = cv::Mat::zeros(img_size, img_size, CV_32FC1);

    // World coordinates precompute
    std::vector<float> world_x(img_size), world_y(img_size);
    for (int i = 0; i < img_size; i++) {
        world_x[i] = (i / scale) - half;
        world_y[i] = half - (i / scale);
    }

    float x_min = grid_center.x() - grid_extent.x() * 0.5f;
    float y_min = grid_center.y() - grid_extent.y() * 0.5f;

    cv::Mat grid_overlay = cv::Mat::zeros(img_size, img_size, CV_8UC3);

    int processed_cells = 0;

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {

            const auto& g = gmap(r, c);
            Eigen::Vector2d mu = g.getMean();
            bool valid_gaussian = (mu.norm() > 1e-6);

            Eigen::Matrix2d cov = g.getCovar();
            bool cov_valid = (cov.determinant() >= 1e-6);

            if (valid_gaussian && cov_valid)
                processed_cells++;

            // ---- Heatmap computation ----
            if (valid_gaussian && cov_valid) {
                Eigen::Matrix2d inv_cov = cov.inverse();
                double a = inv_cov(0,0), b = inv_cov(0,1), d = inv_cov(1,1);

                Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
                float maxev = es.eigenvalues().maxCoeff();
                float radius = cutoff_std * std::sqrt(maxev);

                int px_min = std::clamp<int>((mu.x() - radius + half) * scale, 0, img_size-1);
                int px_max = std::clamp<int>((mu.x() + radius + half) * scale, 0, img_size-1);
                int py_min = std::clamp<int>((half - (mu.y() + radius)) * scale, 0, img_size-1);
                int py_max = std::clamp<int>((half - (mu.y() - radius)) * scale, 0, img_size-1);

                for (int py = py_min; py <= py_max; py++) {
                    float wy = world_y[py];
                    float dy_base = wy - mu.y();
                    float* row_ptr = heat.ptr<float>(py);

                    for (int px = px_min; px <= px_max; px++) {
                        float wx = world_x[px];
                        float dx = wx - mu.x();
                        double m = dx*(a*dx + b*dy_base) + dy_base*(b*dx + d*dy_base);
                        if (m > cutoff_std*cutoff_std) continue;
                        row_ptr[px] += std::exp(-0.5 * m);
                    }
                }
            }

            // ---- Cell overlay ----
            if (show_cells) {
                float cell_left   = x_min + c * grid_step.x();
                float cell_right  = cell_left + grid_step.x();
                float cell_bottom = y_min + r * grid_step.y();
                float cell_top    = cell_bottom + grid_step.y();

                int px_left  = std::clamp<int>((cell_left + half) * scale, 0, img_size-1);
                int px_right = std::clamp<int>((cell_right + half) * scale, 0, img_size-1);
                int py_top   = std::clamp<int>((half - cell_top) * scale, 0, img_size-1);
                int py_bot   = std::clamp<int>((half - cell_bottom) * scale, 0, img_size-1);

                cv::Scalar col = valid_gaussian ? cv::Scalar(255,150,0) : cv::Scalar(100,100,100);

                if (show_grid_lines)
                    cv::rectangle(grid_overlay,
                                  cv::Point(px_left, py_top),
                                  cv::Point(px_right, py_bot),
                                  col, 1);

                if (show_cell_centers) {
                    int cx = (int)((cell_left + cell_right) * 0.5f + half) * scale;
                    int cy = (int)(half - (cell_bottom + grid_step.y() * 0.5f)) * scale;
                    cv::circle(grid_overlay, {cx,cy}, 2, cv::Scalar(0,255,255), -1);
                }

                // Gaussian mean marker
                if (valid_gaussian) {
                    int px = (int)((mu.x() + half) * scale);
                    int py = (int)(half - mu.y()) * scale;
                    if (px>=0 && px<img_size && py>=0 && py<img_size)
                        cv::circle(grid_overlay, {px, py}, 3, cv::Scalar(0,0,255), -1);

                    // ---- Draw covariance ellipse ----
                    if (show_ellipses) {
                        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(cov);
                        Eigen::Vector2d evals = es.eigenvalues();
                        Eigen::Matrix2d evecs = es.eigenvectors();

                        double angle = std::atan2(evecs(1,1), evecs(0,1)) * 180.0 / M_PI;
                        double scale_x = cutoff_std * std::sqrt(evals[0]) * scale;
                        double scale_y = cutoff_std * std::sqrt(evals[1]) * scale;

                        cv::ellipse(grid_overlay,
                                    cv::Point(px, py),
                                    cv::Size(scale_x, scale_y),
                                    angle,
                                    0, 360,
                                    cv::Scalar(0,255,0),
                                    1);
                    }
                }
            }
        }
    }

    // ---- Normalize and colorize heatmap ----
    cv::Mat heat_u8;
    if (normalize) {
        cv::normalize(heat, heat_u8, 0, 255, cv::NORM_MINMAX);
        heat_u8.convertTo(heat_u8, CV_8UC1);
    } else {
        heat.convertTo(heat_u8, CV_8UC1);
    }

    cv::Mat heat_color;
    cv::applyColorMap(heat_u8, heat_color, cv::COLORMAP_JET);

    if (show_cells)
        cv::addWeighted(heat_color, 0.75, grid_overlay, 0.25, 0, heat_color);

    cv::putText(heat_color,
                "Cells: " + std::to_string(rows) + "x" + std::to_string(cols),
                {10,30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255,255,255}, 2);

    cv::imwrite(out_path, heat_color);
    std::cout << "[INFO] Saved: " << out_path << "\n";
}




/**
 * Specialized function to visualize the 4 overlapping NDT grids
 * This matches your NDT2D implementation which uses 4 offset grids
 */
template<typename PointT>
void VisualizeNDTMultiGrid(
    const std::vector<Eigen::Matrix<
        pcl::ndt2d::NormalDist<PointT>,
        Eigen::Dynamic, Eigen::Dynamic
    >>& grid_maps,  // Vector of 4 grid maps
    const std::string& out_path,
    float map_range = 100.0f,
    int   img_size  = 1000,
    const Eigen::Vector2f& grid_step = Eigen::Vector2f(1.0f, 1.0f)
)
{
    if (grid_maps.size() != 4) {
        std::cerr << "[ERROR] Need exactly 4 grid maps for multi-grid visualization!" << std::endl;
        return;
    }
    
    cv::Mat final_image = cv::Mat::zeros(img_size, img_size, CV_8UC3);
    float half = map_range / 2.0f;
    float scale = img_size / map_range;
    
    // Colors for each of the 4 grids
    std::vector<cv::Scalar> grid_colors = {
        cv::Scalar(255, 0, 0),    // Blue - Grid 0
        cv::Scalar(0, 255, 0),    // Green - Grid 1
        cv::Scalar(0, 0, 255),    // Red - Grid 2
        cv::Scalar(255, 255, 0)   // Cyan - Grid 3
    };
    
    std::vector<std::string> grid_names = {
        "Base Grid", "X-offset Grid", "Y-offset Grid", "XY-offset Grid"
    };
    
    // Draw each grid
    for (int grid_idx = 0; grid_idx < 4; grid_idx++) {
        const auto& gmap = grid_maps[grid_idx];
        cv::Scalar color = grid_colors[grid_idx];
        
        int rows = gmap.rows();
        int cols = gmap.cols();
        
        // Calculate offset for this grid
        float offset_x = 0, offset_y = 0;
        if (grid_idx == 1) offset_x = grid_step[0] / 2.0f;
        if (grid_idx == 2) offset_y = grid_step[1] / 2.0f;
        if (grid_idx == 3) {
            offset_x = grid_step[0] / 2.0f;
            offset_y = grid_step[1] / 2.0f;
        }
        
        // Draw cells for this grid
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                const auto& g = gmap(r, c);
                Eigen::Vector2d mu = g.getMean();
                
                if (mu.norm() < 1e-6) continue;
                
                // Calculate cell boundaries with offset
                float cell_left = -half + c * grid_step[0] + offset_x;
                float cell_right = cell_left + grid_step[0];
                float cell_top = half - r * grid_step[1] + offset_y;
                float cell_bottom = cell_top - grid_step[1];
                
                // Convert to pixels
                int px_left = static_cast<int>((cell_left + half) * scale);
                int px_right = static_cast<int>((cell_right + half) * scale);
                int py_top = static_cast<int>((half - cell_top) * scale);
                int py_bottom = static_cast<int>((half - cell_bottom) * scale);
                
                // Clamp and draw
                px_left = std::max(0, std::min(img_size - 1, px_left));
                px_right = std::max(0, std::min(img_size - 1, px_right));
                py_top = std::max(0, std::min(img_size - 1, py_top));
                py_bottom = std::max(0, std::min(img_size - 1, py_bottom));
                
                // Draw cell boundary
                cv::rectangle(final_image,
                            cv::Point(px_left, py_top),
                            cv::Point(px_right, py_bottom),
                            color, 1);
                
                // Draw Gaussian mean
                int px_mean = static_cast<int>((mu.x() + half) * scale);
                int py_mean = static_cast<int>((half - mu.y()) * scale);
                
                if (px_mean >= 0 && px_mean < img_size && 
                    py_mean >= 0 && py_mean < img_size) {
                    cv::circle(final_image,
                              cv::Point(px_mean, py_mean),
                              2, color, -1);
                }
            }
        }
        
        // Add legend for this grid
        cv::putText(final_image, grid_names[grid_idx],
                   cv::Point(10, 30 * (grid_idx + 1)),
                   cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }
    
    // Add title
    cv::putText(final_image, "NDT2D Multi-Grid Structure (4 overlapping grids)",
               cv::Point(img_size/2 - 200, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
               cv::Scalar(255, 255, 255), 2);
    
    cv::imwrite(out_path, final_image);
    std::cout << "[INFO] Saved NDT Multi-Grid Visualization: " << out_path << std::endl;
}


template<typename PointT>
void VisualizeGaussianHeatmap(
    const Eigen::Matrix<
        pcl::ndt2d::NormalDist<PointT>,
        Eigen::Dynamic, Eigen::Dynamic
    >& gmap,
    const std::string& out_path,
    float map_range = 100.0f,
    int   img_size  = 2001,
    float cutoff_std = 4.0f,  // Standard deviations to consider
    bool  normalize = true,
    bool  show_cells = false  // Draw cell boundaries
)
{
    if (gmap.rows() == 0 || gmap.cols() == 0) {
        std::cerr << "[WARN] Empty Gaussian map!" << std::endl;
        return;
    }

    cv::Mat heat = cv::Mat::zeros(img_size, img_size, CV_32FC1);
    
    float half = map_range / 2.0f;
    float scale = img_size / map_range;
    
    // Precompute all pixel world coordinates
    std::vector<float> world_x(img_size), world_y(img_size);
    #pragma omp parallel for
    for (int i = 0; i < img_size; i++) {
        world_x[i] = (i / scale) - half;
        world_y[i] = half - (i / scale);  // Note: Y flipped for image coordinates
    }
    
    int rows = gmap.rows();
    int cols = gmap.cols();
    int total_cells = rows * cols;
    int processed_cells = 0;
    
    // Process each cell in parallel
    #pragma omp parallel for reduction(+:processed_cells) collapse(2)
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const auto& g = gmap(r, c);
            
            // Skip invalid Gaussians
            if (g.getMean().norm() < 1e-6) continue;
            
            Eigen::Matrix2d cov = g.getCovar();
            if (cov.determinant() < 1e-6) continue;
            
            processed_cells++;
            
            Eigen::Vector2d mu = g.getMean();
            Eigen::Matrix2d inv_cov = cov.inverse();
            
            // Get eigenvalue decomposition for bounding box
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(cov);
            Eigen::Vector2d eigenvalues = eigensolver.eigenvalues();
            float max_eigenvalue = eigenvalues.maxCoeff();
            
            // Use 4-sigma cutoff for bounding box
            float bound = cutoff_std * std::sqrt(max_eigenvalue);
            
            // Convert to pixel bounds (clamped to image size)
            int px_min = static_cast<int>((mu.x() - bound + half) * scale);
            int px_max = static_cast<int>((mu.x() + bound + half) * scale);
            int py_min = static_cast<int>((half - (mu.y() + bound)) * scale);  // Note: Y flipped
            int py_max = static_cast<int>((half - (mu.y() - bound)) * scale);
            
            px_min = std::max(0, std::min(img_size - 1, px_min));
            px_max = std::max(0, std::min(img_size - 1, px_max));
            py_min = std::max(0, std::min(img_size - 1, py_min));
            py_max = std::max(0, std::min(img_size - 1, py_max));
            
            // Precompute covariance terms for fast evaluation
            double a = inv_cov(0, 0);
            double b = inv_cov(0, 1);
            double d = inv_cov(1, 1);
            
            // Evaluate Gaussian within bounding box
            for (int py = py_min; py <= py_max; py++) {
                float wy = world_y[py];
                float dy_base = wy - mu.y();
                float* row_ptr = heat.ptr<float>(py);
                
                for (int px = px_min; px <= px_max; px++) {
                    float wx = world_x[px];
                    float dx = wx - mu.x();
                    float dy = dy_base;
                    
                    // Compute Mahalanobis distance squared
                    double mahalanobis = dx * (a * dx + b * dy) + dy * (b * dx + d * dy);
                    
                    // Early termination if beyond cutoff
                    if (mahalanobis > cutoff_std * cutoff_std) continue;
                    
                    // Add Gaussian contribution
                    row_ptr[px] += static_cast<float>(std::exp(-0.5 * mahalanobis));
                }
            }
        }
    }
    
    std::cout << "[INFO] Processed " << processed_cells << "/" << total_cells 
              << " valid Gaussians" << std::endl;
    
    // Convert to 8-bit and apply color map
    cv::Mat heat_display;
    if (normalize) {
        cv::normalize(heat, heat_display, 0, 255, cv::NORM_MINMAX);
        heat_display.convertTo(heat_display, CV_8UC1);
    } else {
        // Just clip to 0-255 range
        heat.convertTo(heat_display, CV_8UC1, 255.0);
    }
    
    // Apply color map
    cv::Mat heat_color;
    cv::applyColorMap(heat_display, heat_color, cv::COLORMAP_JET);
    
    // Optionally draw cell boundaries
    if (show_cells && rows > 0 && cols > 0) {
        float cell_width = map_range / rows;
        float cell_height = map_range / cols;
        float pixel_per_cell_x = img_size / static_cast<float>(rows);
        float pixel_per_cell_y = img_size / static_cast<float>(cols);
        
        // Draw vertical lines
        for (int i = 0; i <= rows; i++) {
            int x = static_cast<int>(i * pixel_per_cell_x);
            cv::line(heat_color, cv::Point(x, 0), cv::Point(x, img_size), 
                    cv::Scalar(255, 255, 255), 1);
        }
        
        // Draw horizontal lines
        for (int j = 0; j <= cols; j++) {
            int y = static_cast<int>(j * pixel_per_cell_y);
            cv::line(heat_color, cv::Point(0, y), cv::Point(img_size, y), 
                    cv::Scalar(255, 255, 255), 1);
        }
    }
    
    // Save image
    cv::imwrite(out_path, heat_color);
    std::cout << "[INFO] Saved Gaussian Heatmap: " << out_path 
              << " (" << img_size << "x" << img_size << ")" << std::endl;
}

/**
 * Alternative: Visualize only Gaussian means as points
 * Much faster for debugging
 */
template<typename PointT>
void VisualizeGaussianMeans(
    const Eigen::Matrix<
        pcl::ndt2d::NormalDist<PointT>,
        Eigen::Dynamic, Eigen::Dynamic
    >& gmap,
    const std::string& out_path,
    float map_range = 100.0f,
    int   img_size  = 1000,
    int   point_radius = 2
)
{
    cv::Mat image = cv::Mat::zeros(img_size, img_size, CV_8UC3);
    image.setTo(cv::Scalar(0, 0, 0));  // Black background
    
    float half = map_range / 2.0f;
    float scale = img_size / map_range;
    
    int rows = gmap.rows();
    int cols = gmap.cols();
    int valid_cells = 0;
    
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const auto& g = gmap(r, c);
            Eigen::Vector2d mu = g.getMean();
            
            // Skip invalid means
            if (mu.norm() < 1e-6) continue;
            
            // Convert to pixel coordinates
            int px = static_cast<int>((mu.x() + half) * scale);
            int py = static_cast<int>((half - mu.y()) * scale);  // Y flipped
            
            // Check bounds
            if (px >= 0 && px < img_size && py >= 0 && py < img_size) {
                // Color based on cell position or covariance determinant
                Eigen::Matrix2d cov = g.getCovar();
                double det = cov.determinant();
                
                // Use determinant to determine color intensity
                int intensity = static_cast<int>(std::min(255.0, det * 1000.0));
                cv::Scalar color(0, intensity, 255 - intensity);  // Blue to green gradient
                
                cv::circle(image, cv::Point(px, py), point_radius, color, -1);
                valid_cells++;
            }
        }
    }
    
    cv::imwrite(out_path, image);
    std::cout << "[INFO] Saved Gaussian Means: " << out_path 
              << " (" << valid_cells << " points)" << std::endl;
}

/**
 * Visualize covariance ellipses for each Gaussian
 * Shows orientation and uncertainty
 */
template<typename PointT>
void VisualizeGaussianEllipses(
    const Eigen::Matrix<
        pcl::ndt2d::NormalDist<PointT>,
        Eigen::Dynamic, Eigen::Dynamic
    >& gmap,
    const std::string& out_path,
    float map_range = 100.0f,
    int   img_size  = 1000,
    float ellipse_scale = 2.0f  // Draw ellipses at 2-sigma
)
{
    cv::Mat image = cv::Mat::zeros(img_size, img_size, CV_8UC3);
    image.setTo(cv::Scalar(0, 0, 0));  // Black background
    
    float half = map_range / 2.0f;
    float scale = img_size / map_range;
    
    int rows = gmap.rows();
    int cols = gmap.cols();
    
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            const auto& g = gmap(r, c);
            Eigen::Vector2d mu = g.getMean();
            
            if (mu.norm() < 1e-6) continue;
            
            Eigen::Matrix2d cov = g.getCovar();
            if (cov.determinant() < 1e-6) continue;
            
            // Convert mean to pixel coordinates
            int px = static_cast<int>((mu.x() + half) * scale);
            int py = static_cast<int>((half - mu.y()) * scale);
            
            if (px < 0 || px >= img_size || py < 0 || py >= img_size) continue;
            
            // Get ellipse parameters from covariance
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigensolver(cov);
            Eigen::Vector2d eigenvalues = eigensolver.eigenvalues();
            Eigen::Matrix2d eigenvectors = eigensolver.eigenvectors();
            
            // Scale eigenvalues for visualization
            double major_axis = 2.0 * ellipse_scale * std::sqrt(eigenvalues[0]);
            double minor_axis = 2.0 * ellipse_scale * std::sqrt(eigenvalues[1]);
            
            // Convert to pixel units
            int major_pixels = static_cast<int>(major_axis * scale);
            int minor_pixels = static_cast<int>(minor_axis * scale);
            
            // Get angle in degrees (from first eigenvector)
            double angle = std::atan2(eigenvectors(1, 0), eigenvectors(0, 0)) * 180.0 / M_PI;
            
            // Draw ellipse
            cv::Scalar color(0, 255, 0);  // Green ellipses
            cv::ellipse(image, cv::Point(px, py), 
                       cv::Size(major_pixels, minor_pixels), 
                       angle, 0, 360, color, 1);
            
            // Draw center point
            cv::circle(image, cv::Point(px, py), 2, cv::Scalar(255, 0, 0), -1);
        }
    }
    
    cv::imwrite(out_path, image);
    std::cout << "[INFO] Saved Gaussian Ellipses: " << out_path << std::endl;
}

} // namespace visualization
} // namespace pcl

#endif // PCL_NDT_2D_VISUALIZATION_H_