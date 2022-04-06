#ifndef PANOPTIC_MAPPING_INTEGRATION_RAYCAST_TSDF_INTEGRATOR_H_
#define PANOPTIC_MAPPING_INTEGRATION_RAYCAST_TSDF_INTEGRATOR_H_

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <ros/ros.h>

#include <voxblox/integrator/integrator_utils.h>
#include <voxblox/utils/approx_hash_array.h>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/common/camera.h"
#include "panoptic_mapping/common/lidar.h"
#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/integration/projection_interpolators.h"
#include "panoptic_mapping/integration/tsdf_integrator_base.h"

namespace panoptic_mapping {

/**
 * @brief Raycast each 3D point and integrate the TSDF of the voxels along the ray
 */
class RaycastIntegrator : public TsdfIntegratorBase {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 4;

    bool use_range_image = true;

    // use "merged" raycasting method
    bool merged_raycasting = false;
    
    // if "merged" raycasting is enabled, only apply it to the freespace submap
    bool only_merge_freespace = false;

    // use multi-threads integration for the freespace submap independently (not together with other submaps)
    bool free_space_multi_threads = false;

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

    // If true, the entire length of a ray is integrated, 
    // if false only the region inside the trunaction distance is used.
    // NOTE (py): space carving is neccessary for the dynamic mapping
    // you need to explicity update the free area in case some part is moving 
    // or changing 
    // with the cost of longer consuming time
    // For panoptic mapping, we can enable the space voxel carving only for those
    // potential moving objects such as people and vehicles
    bool voxel_carving_enabled = false;
    bool freespace_carving_enabled = true; // carving needed for the esdf mapping

    float max_ray_length_m = 100.0f;
    float min_ray_length_m = 0.5f;

    bool allow_clear = false;

    // If true, rays that don't belong to the submap ID are treated as clearing
    // rays.
    bool foreign_rays_clear = true;

    // If true, overwrite voxels that have a distance update larger than 0.05 m.
    bool use_longterm_fusion = false;

    // Number of threads used to perform integration. Integration is
    // submap-parallel. (reasonable)
    int integration_threads = std::thread::hardware_concurrency();

    bool use_lidar = false;

    bool apply_normal_refine = false;

    bool apply_normal_refine_freespace = false;

    bool curve_assumption = false; // consume more time but get a more accurate non-projective sdf under the assumption of a continous curve representation of the surface

    bool behind_surface_reliable_band = false; // apply a reliable band also behind the surface

    float reliable_band_ratio = 2.0; // the reliable band distance threshold would be reliable_band_ratio * truncation_distance

    float reliable_normal_ratio_thre = 0.2; 

    bool skip_free_space_submap = false; // just for debugging, include the free space submap or not

    Config() { setConfigName("RaycastTsdfIntegrator"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  RaycastIntegrator(const Config& config, std::shared_ptr<Globals> globals,
                       bool print_config = true);
  ~RaycastIntegrator() override = default;

  void processInput(SubmapCollection* submaps, InputData* input) override;

 protected:

  virtual void updateSubmap(Submap* submap, const InputData& input);
                            
  virtual bool updateVoxel(TsdfVoxel* voxel, const Point& origin, const Point& p_C, 
                          const Point& p_S, const Ray& n_C, 
                          const GlobalIndex& global_voxel_idx,
                          const Color& color, const bool is_free_space_submap,
                          const bool apply_normal_refine, const Transformation& T_C_S,
                          const float init_weight,
                          const float truncation_distance, const float voxel_size, 
                          ClassVoxel* class_voxel = nullptr) const;

  float computeSignedDistance(const Point& origin,
                              const Point& point_G,
                              const Point& voxel_center) const;

  float computeVoxelWeight(const Point& p_C, const float voxel_size,
                           const float truncation_distance,
                           const float sdf, bool projective = false,
                           const bool with_init_weight = false, const float init_weight = 1.0) const override;
  
  void updateVoxelValues(TsdfVoxel* voxel, const float sdf,
                         const float weight,
                         const Color* color = nullptr) const override;

  void updateVoxelGradient(TsdfVoxel* voxel, const Point normal,
                                   const float weight) const override;

  Pointcloud extractSubmapPointCloud(const cv::Mat& vertex_map,
                                     const cv::Mat& id_image, int id) const;

  Pointcloud extractSubmapNormals(const cv::Mat& normal_image,
                                  const cv::Mat& id_image, int id) const;

  Colors extractSubmapColors(const cv::Mat& color_image,
                            const cv::Mat& id_image, int id) const;

  float getVoxelWeight(const Point& point_C) const; // mainly used for "merged" raycasting   

  /**
   * Will return a pointer to a voxel located at global_voxel_idx in the tsdf
   * layer. Thread safe.
   * Takes in the last_block_idx and last_block to prevent unneeded map lookups.
   * If this voxel belongs to a block that has not been allocated, a block in
   * temp_block_map_ is created/accessed and a voxel from this map is returned
   * instead. Unlike the layer, accessing temp_block_map_ is controlled via a
   * mutex allowing it to grow during integration.
   * These temporary blocks can be merged into the layer later by calling
   * updateLayerWithStoredBlocks
   */
  virtual TsdfVoxel* allocateStorageAndGetVoxelPtr(Submap* submap,
    const GlobalIndex& global_voxel_idx, TsdfBlock::Ptr* last_block,
    BlockIndex* last_block_idx, ClassBlock::Ptr* last_class_block = nullptr);

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

  std::mutex temp_block_mutex_;
  /**
   * Temporary block storage, used to hold blocks that need to be created while
   * integrating a new pointcloud
   */
  TsdfLayer::BlockHashMap temp_block_map_;

  /* We need to prevent simultaneous access to the voxels in the map. We could
   * put a single mutex on the map or on the blocks, but as voxel updating is
   * the most expensive operation in integration and most voxels are close
   * together, both strategies would bottleneck the system. We could make a
   * mutex per voxel, but this is too ram heavy as one mutex = 40 bytes.
   * Because of this we create an array that is indexed by the first n bits of
   * the voxels hash. Assuming a uniform hash distribution, this means the
   * chance of two threads needing the same lock for unrelated voxels is
   * (num_threads / (2^n)). For 8 threads and 12 bits this gives 0.2%.
   */
  voxblox::ApproxHashArray<12, std::mutex, GlobalIndex, LongIndexHash> mutexes_;

 private:
  const Config config_;
  static config_utilities::Factory::RegistrationRos<
      TsdfIntegratorBase, RaycastIntegrator, std::shared_ptr<Globals>>
      registration_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_INTEGRATION_RAYCAST_TSDF_INTEGRATOR_H_
