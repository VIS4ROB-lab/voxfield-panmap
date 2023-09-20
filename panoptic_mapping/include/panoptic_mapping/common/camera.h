#ifndef PANOPTIC_MAPPING_COMMON_CAMERA_H_
#define PANOPTIC_MAPPING_COMMON_CAMERA_H_

#include <unordered_map>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/map/submap.h"
#include "panoptic_mapping/map/submap_collection.h"

namespace panoptic_mapping {

/**
 * @brief Utility class bundling camera related operations and data. Currently
 * the camera pose T_M_C is not stored with a camera but provided by the caller.
 */
class Camera {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 0;

    // RGBD-Camera
    // Camera Intrinsics in pixels.
    int width = 640;
    int height = 480;
    float vx = 320.f;  // Center point.
    float vy = 240.f;
    float fx = 320.f;  // Focal lengths.
    float fy = 320.f;

    float depth_unit = 1.f; // scale for the depth, for real-sense camera, the value is 1000.f

    // Maximum range (ray-length) in meters.
    float max_range = 5.f;

    // Minimum range (ray-length) in meters.
    float min_range = 0.1f;


    // Max bearable depth difference between 
    // two adjacent pixels for a valid normal estimation
    float smooth_thre_m = 1.0f;

    Config() { setConfigName("Camera"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  explicit Camera(const Config& config);
  virtual ~Camera() = default;

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
  bool projectPointCloudToImagePlane(const Pointcloud& ptcloud_C, const Colors& colors, 
                                     const Labels& labels, cv::Mat &vertex_map,   // corresponding point    
                                     cv::Mat &depth_image,  // Float depth image (CV_32FC1).     
                                     cv::Mat &color_image,
                                     cv::Mat &id_image) const;    
                                          
  bool projectPointToImagePlane(const Point& p_C, float* u, float* v) const;

  bool projectPointToImagePlane(const Point& p_C, int* u, int* v) const;

  cv::Mat computeVertexMap(const cv::Mat& depth_image) const;

  cv::Mat computeValidityImage(const cv::Mat& depth_image) const;

  cv::Mat computeNormalImage(const cv::Mat& vertex_map, const cv::Mat& depth_image) const;

 private:
  const Config config_;

  // Pre-computed stored values.
  std::vector<Point> view_frustum_;  // Top, right, bottom, left plane normals.
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_COMMON_CAMERA_H_
