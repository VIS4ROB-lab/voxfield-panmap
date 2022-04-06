#ifndef PANOPTIC_MAPPING_COMMON_LIDAR_H_
#define PANOPTIC_MAPPING_COMMON_LIDAR_H_

#include <unordered_map>
#include <vector>
#include <Eigen/Dense>

#include <opencv2/core/mat.hpp>
#include <ros/ros.h>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/map/submap.h"
#include "panoptic_mapping/map/submap_collection.h"

namespace panoptic_mapping {

/**
 * @brief Utility class bundling camera related operations and data. Currently
 * the camera pose T_M_C is not stored with a camera but provided by the caller.
 */
class Lidar {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 0;

    // RGBD-Camera
    // Camera Intrinsics in pixels.
    int width = 1800; // or 900
    int height = 64; 

    // horizontal resolution: 0.20 deg  or (0.40 deg)
    // vertical resolution:   0.43 deg

    // degree
    float fov_up = 3.0f; 
    float fov_down = -25.0f;

    // Maximum range (ray-length) in meters.
    float max_range = 50.0f;
    // Minimum range (ray-length) in meters.
    float min_range = 1.5f;
    // Expected minimum height of the ground
    float ground_min_z = -5.0f;

    // Max bearable depth difference between 
    // two adjacent pixels for a valid normal estimation
    float smooth_thre_m = 2.0f;
    
    // smooth_thre = smooth_thre_ratio * dist
    float smooth_thre_ratio = 0.05f;
    
    // extended variables
    float fov; //degree
    float fov_up_rad; //rad
    float fov_down_rad; //rad
    float fov_rad; //rad
    float res_h; //degree
    float res_v; //degree
    float res_h_rad; //rad
    float res_v_rad; //rad
    float max_z; //m 
    float min_z; //m

    Config() { setConfigName("Lidar"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
    void initializeDependentVariableDefaults() override;
  };

  explicit Lidar(const Config& config);
  virtual ~Lidar() = default;

  // Access.
  const Config& getConfig() const { return config_; }

  // Visibility checks.
  bool pointIsInViewFrustum(const Point& point_C,
                            float inflation_distance = 0.f) const;

  bool submapIsInViewFrustum(const Submap& submap,
                             const Transformation& T_M_C) const;

  bool blockIsInViewFrustum(const Submap& submap,
                            const voxblox::BlockIndex& block_index,
                            const Transformation& T_M_C) const;

  bool blockIsInViewFrustum(const Submap& submap,
                            const voxblox::BlockIndex& block_index,
                            const Transformation& T_C_S, float block_size,
                            float block_diag_half) const;

  // Visibility search.
  std::vector<int> findVisibleSubmapIDs(const SubmapCollection& submaps,
                                        const Transformation& T_M_C,
                                        bool only_active_submaps = true,
                                        bool include_freespace = false) const;

  voxblox::BlockIndexList findVisibleBlocks(const Submap& subamp,
                                            const Transformation& T_M_C,
                                            const float max_range = -1.f) const;

  std::unordered_map<int, voxblox::BlockIndexList> findVisibleBlocks(
      const SubmapCollection& submaps, const Transformation& T_M_C,
      const float max_range = -1.f, bool only_active_submaps = true) const;

  // Projection.
  // bool projectPointToImagePlane(const Point& p_C, float* u, float* v) const;

  bool projectPointCloudToImagePlane(const Pointcloud& ptcloud_C, 
                                     const Colors& colors,
                                     const Labels& labels,
                                     cv::Mat &vertex_map,
                                     cv::Mat &depth_image,
                                     cv::Mat &color_image,  
                                     cv::Mat &id_image) const;

  float projectPointToImagePlane(const Point& p_C, int* u, int* v) const;

  bool projectPointToImagePlane(const Point& p_C, float* u, float* v) const;
                                     
  cv::Mat computeVertexMap(const cv::Mat& depth_image) const;

  cv::Mat computeVertexMap(const Pointcloud& ptcloud_C, 
                           const cv::Mat& index_image) const;

  cv::Mat computeValidityImage(const cv::Mat& depth_image) const;

  cv::Mat computeRangeImage() const;

  cv::Mat computeNormalImage(const cv::Mat &vertex_map,
                             const cv::Mat &depth_image) const;

  cv::Mat computeSemanticImage() const;

  cv::Mat computeInstanceImage() const;

 private:
  const Config config_;

  // Pre-computed stored values.
  std::vector<Point> view_frustum_;  // Top, right, bottom, left plane normals.

};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_COMMON_LIDAR_H_
