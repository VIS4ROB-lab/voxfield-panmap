#include "panoptic_mapping/common/lidar.h"

#include <unordered_map>
#include <vector>

namespace panoptic_mapping {

void Lidar::Config::checkParams() const {
  // great than check (> 0)
  checkParamGT(width, 0, "width");
  checkParamGT(height, 0, "height");
  checkParamGT(min_range, 0.f, "min_range");
  checkParamGT(max_range, 0.f, "max_range");
  checkParamGT(smooth_thre_m, 0.f, "smooth_thre_m");
  checkParamGT(smooth_thre_ratio, 0.f, "smooth_thre_ratio");
  checkParamLT(ground_min_z, 0.f, "ground_min_z");
  checkParamCond(max_range > min_range,
                 "'max_range' is expected > 'min_range'.");
  checkParamCond(fov_up > fov_down,
                 "'fov_up' is expected > 'fov_down'.");
}

void Lidar::Config::initializeDependentVariableDefaults() {
    // deg
    fov = std::abs(fov_down) + std::abs(fov_up);
    
    // rad
    fov_up_rad = fov_up / 180.0f * M_PI;
    fov_down_rad = fov_down / 180.0f * M_PI;
    fov_rad = fov / 180.0f * M_PI;

    // degree
    res_h = 360.0 / width;
    res_v = fov / height;

    // rad
    res_h_rad = res_h / 180.0f * M_PI;
    res_v_rad = res_v / 180.0f * M_PI;

    // height (m)
    max_z = max_range * std::sin(fov_up_rad);
    min_z = max_range * std::sin(fov_down_rad);
}

void Lidar::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("width", &width, "px");
  setupParam("height", &height, "px");
  setupParam("fov_up", &fov_up, "deg");
  setupParam("fov_down", &fov_down, "deg");
  setupParam("max_range", &max_range, "m");
  setupParam("min_range", &min_range, "m");
  setupParam("ground_min_z", &ground_min_z, "m");
  setupParam("smooth_thre_m", &smooth_thre_m, "m");
  setupParam("smooth_thre_ratio", &smooth_thre_ratio);
}

// TODO (py): add contents
Lidar::Lidar(const Config& config) : config_(config.checkValid()) {
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();
}

// check if the point is in vicinity of the sensor
// for Lidar, it's not really a frustum
bool Lidar::pointIsInViewFrustum(const Point& point_C, 
                               float inflation_distance) const{
  
  // Try if we do not have such limit
  float depth = point_C.norm();
  if (depth > config_.max_range + inflation_distance) { //out of range
    return false;
  }
  //inflation_distance = 0.0;
  if (point_C.z() > depth * std::sin(config_.fov_up_rad) + inflation_distance) {
    return false;
  }
  if (point_C.z() < depth * std::sin(config_.fov_down_rad) - inflation_distance) {
    return false;
  }
  return true;
}

bool Lidar::submapIsInViewFrustum(const Submap& submap,
                                   const Transformation& T_M_C) const {
  const float radius = submap.getBoundingVolume().getRadius();

  const Point center_C = T_M_C.inverse() * submap.getT_M_S() *
                         submap.getBoundingVolume().getCenter();
                         
  return pointIsInViewFrustum(center_C, radius);
}

bool Lidar::blockIsInViewFrustum(const Submap& submap,
                                  const voxblox::BlockIndex& block_index,
                                  const Transformation& T_M_C) const {
  const Transformation T_C_S =
      T_M_C.inverse() * submap.getT_M_S();  // p_C = T_C_M * T_M_S * p_S
  const FloatingPoint block_size = submap.getTsdfLayer().block_size();
  const FloatingPoint block_diag_half = std::sqrt(3.0f) * block_size / 2.0f;
  return blockIsInViewFrustum(submap, block_index, T_C_S, block_size,
                              block_diag_half);
}

bool Lidar::blockIsInViewFrustum(const Submap& submap,
                                  const voxblox::BlockIndex& block_index,
                                  const Transformation& T_C_S, float block_size,
                                  float block_diag_half) const {
  auto& block = submap.getTsdfLayer().getBlockByIndex(block_index);
  const Point p_C =
      T_C_S * (block.origin() +
               Point(1, 1, 1) * block_size / 2.0);  // center point of the block
  return pointIsInViewFrustum(p_C, block_diag_half);
}

bool Lidar::projectPointCloudToImagePlane(const Pointcloud& ptcloud_C,
                                          const Colors& colors, 
                                          const Labels& labels,
                                          cv::Mat &vertex_map,   // corresponding point
                                          cv::Mat &depth_image,  // Float depth image (CV_32FC1).
                                          cv::Mat &color_image,
                                          cv::Mat &id_image)     // panoptic label
                                          const {
    bool label_available = false;
    if (labels.size() == ptcloud_C.size())
      label_available = true;

    // TODO(py): consider to calculate in parallel to speed up
    for (int i=0; i<ptcloud_C.size(); i++)
    {
      int u, v;
      float depth = projectPointToImagePlane(ptcloud_C[i], &u, &v); //colum, row
      if (depth > 0.0)
      { 
        float old_depth = depth_image.at<float>(v,u);
        if (old_depth <= 0.0 || old_depth > depth) // save only nearest point for each pixel
        {
          for (int k=0; k<=2; k++) // should be 3f instead of 3b
          {
            vertex_map.at<cv::Vec3f>(v,u)[k] = ptcloud_C[i](k);
          }

          depth_image.at<float>(v,u) = depth;
          
          //BGR default order
          color_image.at<cv::Vec3b>(v,u)[0] = colors[i].b;
          color_image.at<cv::Vec3b>(v,u)[1] = colors[i].g;
          color_image.at<cv::Vec3b>(v,u)[2] = colors[i].r;
          
          if (label_available)
            id_image.at<int>(v,u) = labels[i].id_label; // it's better to have another one for visualization

          // sem_image.at<int>(v,u) = 
        }
      }
    }
    return false;                                         
}

// point should be in the LiDAR's coordinate system
// if the return value is -1.0,  
float Lidar::projectPointToImagePlane(const Point& p_C, int* u,
                                     int* v) const {
  // All values are ceiled and floored to guarantee that the resulting points
  // will be valid for any integer conversion.
  float depth = std::sqrt(p_C.x() * p_C.x() + p_C.y() * p_C.y() + p_C.z() * p_C.z());
  float yaw = std::atan2(p_C.y(), p_C.x());
  float pitch = std::asin(p_C.z() / depth);
  // projections in im coor (percentage)
  float proj_x = 0.5 * (yaw / M_PI + 1.0); //[0-1]
  float proj_y = 1.0 - (pitch - config_.fov_down_rad) / config_.fov_rad; //[0-1]
  // scale to image size 
  proj_x *= config_.width;  //[0-W]
  proj_y *= config_.height; //[0-H]
  // round for integer index
  CHECK_NOTNULL(u);
  *u = std::round(proj_x);
  if (*u == config_.width)
    *u = 0;

  CHECK_NOTNULL(v);
  *v = std::round(proj_y);
  if (std::ceil(proj_y) > config_.height - 1 || std::floor(proj_y) < 0) {
    return (-1.0);
  }
  return depth;
}

bool Lidar::projectPointToImagePlane(const Point& p_C, float* u,
                                     float* v) const {
  // All values are ceiled and floored to guarantee that the resulting points
  // will be valid for any integer conversion.
  float depth = p_C.norm(); //std::sqrt(p_C.x() * p_C.x() + p_C.y() * p_C.y() + p_C.z() * p_C.z());
  float yaw = std::atan2(p_C.y(), p_C.x()); //-pi, pi
  float pitch = std::asin(p_C.z() / depth); 
  // projections in im coor (percentage)
  float proj_x = 0.5 * (yaw / M_PI + 1.0); //[0-1]
  float proj_y = 1.0 - (pitch - config_.fov_down_rad) / config_.fov_rad; //[0-1]
  // scale to image size 
  proj_x *= config_.width;  //[0-W]
  proj_y *= config_.height; //[0-H]
  // round for integer index
  CHECK_NOTNULL(u);
  *u = proj_x;

  CHECK_NOTNULL(v);
  *v = proj_y;
  if (std::ceil(proj_y) > config_.height - 1 || std::floor(proj_y) < 0) {
    return false;
  }
  return true;
}

cv::Mat Lidar::computeNormalImage(const cv::Mat &vertex_map,
                                  const cv::Mat &depth_image) const {
    
    cv::Mat normal_image(depth_image.size(), CV_32FC3, 0.0);
    for (int u=0; u < config_.width; u++){
      for (int v=0; v < config_.height; v++){
        Point p;
        p << vertex_map.at<cv::Vec3f>(v, u)[0], 
             vertex_map.at<cv::Vec3f>(v, u)[1], 
             vertex_map.at<cv::Vec3f>(v, u)[2];

        float d_p = depth_image.at<float>(v,u);
        float sign = 1.0; //sign of the normal vector

        if(d_p > 0)
        {
          // neighbor x (in ring)
          int n_x_u;
          if(u==config_.width-1)
            n_x_u = 0;
          else
            n_x_u = u + 1;
          Point n_x;
          n_x << vertex_map.at<cv::Vec3f>(v, n_x_u)[0], 
                 vertex_map.at<cv::Vec3f>(v, n_x_u)[1], 
                 vertex_map.at<cv::Vec3f>(v, n_x_u)[2];

          float d_n_x = depth_image.at<float>(v, n_x_u);
          if (d_n_x < 0)
            continue;
          if (std::abs(d_n_x - d_p) > config_.smooth_thre_ratio * d_p) //on the boundary, not continous
            continue;

          // neighbor y
          int n_y_v;
          if(v == config_.height-1){
            n_y_v = v - 1;
            sign *= -1.0;
          }
          else {
            n_y_v = v + 1;
          }
          Point n_y;
          n_y << vertex_map.at<cv::Vec3f>(n_y_v, u)[0], 
                 vertex_map.at<cv::Vec3f>(n_y_v, u)[1], 
                 vertex_map.at<cv::Vec3f>(n_y_v, u)[2];

          float d_n_y = depth_image.at<float>(n_y_v, u);
          if (d_n_y < 0)
            continue;
          if (std::abs(d_n_y - d_p) > config_.smooth_thre_ratio * d_p) //on the boundary, not continous
            continue;
          
          Point dx = n_x - p;
          Point dy = n_y - p;

          Point normal = (dx.cross(dy)).normalized() * sign; 
          cv::Vec3f& normals = normal_image.at<cv::Vec3f>(v, u);
          for (int k=0; k<=2; k++)
            normals[k] = normal(k);
        }
      }
    }
    return normal_image;
}

// Find submaps that locate in FOV
std::vector<int> Lidar::findVisibleSubmapIDs(const SubmapCollection& submaps,
                                              const Transformation& T_M_C,
                                              bool only_active_submaps,
                                              bool include_freespace) const {
  std::vector<int> result;
  for (const Submap& submap : submaps) {
    if (!submap.isActive() && only_active_submaps) { // non-active submaps are directly skipped
      continue;
    }
    if (submap.getLabel() == PanopticLabel::kFreeSpace && !include_freespace) { //free-space submap is also skipped
      continue;
    }
    if (!submapIsInViewFrustum(submap, T_M_C)) { //only keep those in FOV
      continue;
    }
    result.push_back(submap.getID());
  }
  //ROS_INFO("Find %d visible submaps in current pose", result.size());
  return result;
}

voxblox::BlockIndexList Lidar::findVisibleBlocks(const Submap& submap,
                                                  const Transformation& T_M_C,
                                                  const float max_range) const {
  // Setup.
  voxblox::BlockIndexList result;
  voxblox::BlockIndexList all_blocks;
  submap.getTsdfLayer().getAllAllocatedBlocks(&all_blocks);
  const Transformation T_C_S =
      T_M_C.inverse() * submap.getT_M_S();  // p_C = T_C_M * T_M_S * p_S
  const FloatingPoint block_size = submap.getTsdfLayer().block_size();
  // const FloatingPoint block_diag_half = std::sqrt(3.0f) * block_size / 2.0f;

  // Iterate through all blocks.
  for (auto& index : all_blocks) {
    auto& block = submap.getTsdfLayer().getBlockByIndex(index);
    const Point p_C =
        T_C_S * (block.origin() + Point(1, 1, 1) * block_size /
                                      2.0);  // Center point of the block.
    // if (max_range > 0 && p_C.norm() > config_.max_range + block_diag_half) {
    //   continue;
    // }

    // if (pointIsInViewFrustum(p_C, block_diag_half))
    //   result.push_back(index);

    if (pointIsInViewFrustum(p_C, block_size / 2.0f))                         
      result.push_back(index);
  }
  return result;
}

//used 
std::unordered_map<int, voxblox::BlockIndexList> Lidar::findVisibleBlocks(
    const SubmapCollection& submaps, const Transformation& T_M_C,
    const float max_range, bool only_active_submaps) const {
  std::unordered_map<int, voxblox::BlockIndexList> result;
  // for each submap
  for (const Submap& submap : submaps) {
    if (!submap.isActive() && only_active_submaps) { 
      continue;
    }
    if (!submapIsInViewFrustum(submap, T_M_C)) {
      continue;
    }
    voxblox::BlockIndexList block_list =
        findVisibleBlocks(submap, T_M_C, max_range);
    if (!block_list.empty()) {
      result[submap.getID()] = block_list;
    }
  }
  return result;
}

// TODO(py): update
cv::Mat Lidar::computeVertexMap(const cv::Mat& depth_image) const {
  // Corresponding Point for each pixel. 
  cv::Mat vertices(depth_image.size(), CV_32FC3);
  // const float fx_inv = 1.f / config_.fx;
  // const float fy_inv = 1.f / config_.fy;
  // for (int v = 0; v < depth_image.rows; v++) {
  //   for (int u = 0; u < depth_image.cols; u++) {
  //     cv::Vec3f& vertex = vertices.at<cv::Vec3f>(v, u);  // x, y, z
  //     vertex[2] = depth_image.at<float>(v, u);
  //     vertex[0] = (static_cast<float>(u) - config_.vx) * vertex[2] * fx_inv;
  //     vertex[1] = (static_cast<float>(v) - config_.vy) * vertex[2] * fy_inv;
  //   }
  // }
  return vertices;
}

cv::Mat Lidar::computeVertexMap(const Pointcloud& ptcloud_C, 
                                const cv::Mat& index_image) const {
  cv::Mat vertices(index_image.size(), CV_32FC3);
  for (int v = 0; v < index_image.rows; v++) {
     for (int u = 0; u < index_image.cols; u++) {
       cv::Vec3f& vertex = vertices.at<cv::Vec3f>(v, u);  // x, y, z
       int idx = index_image.at<int>(v, u);
       for (int k = 0; k<=2; k++){
         vertex[k] = ptcloud_C[idx](k);
       } 
     }
  }
  return vertices;
}

cv::Mat Lidar::computeValidityImage(const cv::Mat& depth_image) const {
  // Check whether the depth image is valid. Currently just checks for min and
  // max range.
  cv::Mat validity_image(depth_image.size(), CV_8UC1);
  for (int v = 0; v < depth_image.rows; v++) {
    for (int u = 0; u < depth_image.cols; u++) {
      float depth = depth_image.at<float>(v, u);
      validity_image.at<uchar>(v, u) = static_cast<uchar>(
          depth >= config_.min_range && depth <= config_.max_range);
    }
  }
  return validity_image;
}

}  // namespace panoptic_mapping
