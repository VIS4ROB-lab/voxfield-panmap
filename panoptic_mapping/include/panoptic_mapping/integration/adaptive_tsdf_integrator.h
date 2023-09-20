#ifndef PANOPTIC_MAPPING_INTEGRATION_ADAPTIVE_TSDF_INTEGRATOR_H_
#define PANOPTIC_MAPPING_INTEGRATION_ADAPTIVE_TSDF_INTEGRATOR_H_

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <ros/ros.h>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/common/camera.h"
#include "panoptic_mapping/common/lidar.h"
#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/integration/projection_interpolators.h"
#include "panoptic_mapping/integration/tsdf_integrator_base.h"

namespace panoptic_mapping {

/**
 * @brief Adaptively determine the method used for Tsdf integration
 * Select from projective and raycasting integrator to guarantee 
 * the efficiency of the algorithm
 */
class AdaptiveIntegrator : public TsdfIntegratorBase {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 4;

    // skip the free space submap, only do the panoptic reconstruction part
    bool skip_free_space_submap = false;

    // use "merged" (bundled) raycasting or the "simple" one
    bool merged_raycasting = true;

    // if "merged" raycasting is selected, we only apply it on freespace submap
    bool only_merge_freespace = false;

    // If true, drop off the weight behind the surface crossing.
    bool use_weight_dropoff = true;

    // Distance in meters where the weight dropp off reaches zero. Negative
    // values  are multiples of the voxel size.
    float weight_dropoff_epsilon = -1.f;

    // If true, use unitary (w=1) weights to update the TSDF. Otherwise use
    // weights as a function of the squared depth to approximate typical RGBD
    // sensor confidence.
    bool weight_reduction = false;

    // Weight reduction w.r.t the distance to the sensor (default: square)
    float weight_reduction_exp = 2.0f;

    // Maximum weight used for TSDF updates. High max weight keeps information
    // longer in memory, low max weight favors rapid updates.
    float max_weight = 1e5;

    // Interpolation method to be used when interpolating values in the images.
    // Supported are {nearest, bilinear, adaptive}.
    std::string interpolation_method = "adaptive";

    // If true, rays that don't belong to the submap ID are treated as clearing
    // rays.
    bool foreign_rays_clear = true;

    bool allow_clear = false;

    // If false, allocate blocks where points fall directly into. If true, also
    // allocate neighboring blocks if the voxels lie on the boundary of the
    // block.
    bool allocate_neighboring_blocks = false;

    // If true, overwrite voxels that have a distance update larger than 0.05 m.
    bool use_longterm_fusion = false;

    // If true, the entire length of a ray is integrated, 
    // if false only the region inside the trunaction distance is used.
    bool voxel_carving_enabled = false;

    bool freespace_carving_enabled = true;

    float max_ray_length_m = 100.0f;

    float min_ray_length_m = 0.5f;

    // Number of threads used to perform integration. Integration is
    // submap-parallel. (reasonable)
    int integration_threads = std::thread::hardware_concurrency();

    bool use_lidar = false;

    bool apply_normal_refine = false;

    bool apply_normal_refine_freespace = false;

    Config() { setConfigName("AdaptiveTsdfIntegrator"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  AdaptiveIntegrator(const Config& config, std::shared_ptr<Globals> globals,
                       bool print_config = true);
  ~AdaptiveIntegrator() override = default;

  void processInput(SubmapCollection* submaps, InputData* input) override;

 protected:
  /**
   * @brief Allocate all new blocks in all submaps.
   *
   * @param submaps Submap collection to allocate data in.
   * @param input Input measurements based on which blocks are allocated.
   */
  virtual void allocateNewBlocks(SubmapCollection* submaps,
                                 const InputData& input);

  virtual void updateSubmap(Submap* submap, InterpolatorBase* interpolator,
                            const voxblox::BlockIndexList& block_indices,
                            const InputData& input) const;

  virtual void updateBlock(Submap* submap, InterpolatorBase* interpolator,
                           const voxblox::BlockIndex& block_index,
                           const Transformation& T_C_S,
                           const InputData& input) const;

  /**
   * @brief Update a specific voxel based on the input (used for projective mapping)
   *
   * @param interpolator Projection interpolator to use. Interpolators are not
   thread safe.
   * @param voxel Tsdf voxel to be updated.
   * @param p_C Position of the voxel center in camera frame in meters.
   * @param input Input data used for the update.
   * @param submap_id SubmapID of the owning submap.
   * @param is_free_space_submap Whether the voxel belongs to a freespace map.
   * @param apply_normal_refine Whether use the normal vector to refine the projective SDF.
   * @param truncation_distance Truncation distance to be used.
   * @param voxel_size Voxel size of the TSDF layer.
   * @param class_voxel Optional: class voxel to be updated.

   * @return True if the voxel was updated.
   */
  //NOTE(py): add param "apply_normal_refine" and "T_C_S"
  virtual bool updateVoxel(InterpolatorBase* interpolator, TsdfVoxel* voxel,
                           const Point& p_C, const InputData& input,
                           const int submap_id, const bool is_free_space_submap,
                           const bool apply_normal_refine, const Transformation& T_C_S,
                           const float truncation_distance,
                           const float voxel_size,
                           ClassVoxel* class_voxel = nullptr) const;
  
  // Used for raycasting
  virtual bool updateVoxel(TsdfVoxel* voxel, const Point& origin, const Point& p_C, 
                          const Point& p_S, const Ray& n_C, 
                          const GlobalIndex& global_voxel_idx,
                          const Color& color, const bool is_free_space_submap,
                          const bool apply_normal_refine, const Transformation& T_C_S,
                          const float precompute_weight,
                          const float truncation_distance, const float voxel_size, 
                          ClassVoxel* class_voxel = nullptr) const;
  

  /**
   * @brief Sets up the interpolator and computes the signed distance.
   *
   * @param p_C Voxel center in camera frame in meters.
   * @param interpolator Interpolator to setup and use.
   * @return Whether the voxel is valid to continue processing.
   */
  virtual bool computeSignedDistance(const Point& p_C,
                                     InterpolatorBase* interpolator,
                                     float* sdf, float* u, float* v) const;

  virtual float computeSignedDistance(const Point& origin,
                                      const Point& point_G,
                                      const Point& voxel_center) const; 
  
  float computeVoxelWeight(const Point& p_C, const float voxel_size,
                           const float truncation_distance,
                           const float sdf, const bool projective,
                           const bool with_init_weight = false,
                           const float init_weight = 1.0) const override;
  
  void updateVoxelValues(TsdfVoxel* voxel, const float sdf,
                         const float weight,
                         const Color* color = nullptr) const override;

  void updateVoxelGradient(TsdfVoxel* voxel, const Ray normal,
                           const float weight) const override;

  Pointcloud extractSubmapPointCloud(const cv::Mat& vertex_map,
                                     const cv::Mat& id_image, int id, int down_rate = 1) const;

  Pointcloud extractSubmapNormals(const cv::Mat& normal_image,
                                  const cv::Mat& id_image, int id, int down_rate = 1) const;

  Colors extractSubmapColors(const cv::Mat& color_image,
                             const cv::Mat& id_image, int id, int down_rate = 1) const;

  // Thread safe.
  inline bool isPointValid(const Point& point_C, const bool freespace_point,
                           bool* is_clearing) const {
    DCHECK(is_clearing != nullptr);
    const FloatingPoint ray_distance = point_C.norm();
    if (ray_distance < config_.min_ray_length_m) { // Too close
      return false;
    } else if (ray_distance > config_.max_ray_length_m) { // Too far away
      if (config_.allow_clear || freespace_point) { 
        *is_clearing = true;
        return true;
      } else {
        return false;
      }
    } else {
      *is_clearing = freespace_point;
      return true;
    }
  }

  // Cached data.
  Eigen::MatrixXf range_image_;
  float max_range_in_image_ = 0.f;
  float max_range_;
  float min_range_;
  float max_z_;
  float min_z_;
  // const Camera::Config* cam_config_;
  // const Lidar::Config* lidar_config_;
  std::vector<std::unique_ptr<InterpolatorBase>>
      interpolators_;  // one for each thread.

 private:
  const Config config_;
  static config_utilities::Factory::RegistrationRos<
      TsdfIntegratorBase, AdaptiveIntegrator, std::shared_ptr<Globals>>
      registration_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_INTEGRATION_ADAPTIVE_TSDF_INTEGRATOR_H_
