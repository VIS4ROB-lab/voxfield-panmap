#ifndef PANOPTIC_MAPPING_INTEGRATION_CLASS_ADAPTIVE_TSDF_INTEGRATOR_H_
#define PANOPTIC_MAPPING_INTEGRATION_CLASS_ADAPTIVE_TSDF_INTEGRATOR_H_

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/common/camera.h"
#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/integration/projection_interpolators.h"
#include "panoptic_mapping/integration/adaptive_tsdf_integrator.h"
#include "panoptic_mapping/integration/tsdf_integrator_base.h"

namespace panoptic_mapping {

/**
 * @brief Adaptively determine the method used for Tsdf integration
 * Select from projective and raycasting integrator to guarantee 
 * the efficiency of the algorithm.
 * Also update a separate class layer to estimate
 * which voxels belong to the current submap.
 */
class ClassAdaptiveIntegrator : public AdaptiveIntegrator {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 4;

    bool use_binary_classification = false;    // false: use a counter per id.
    bool use_instance_classification = false;  // false: use class id.
    bool update_only_tracked_submaps = true;

    // Integration params.
    AdaptiveIntegrator::Config ai_config;

    Config() { setConfigName("ClassAdaptiveTsdfIntegrator"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  ClassAdaptiveIntegrator(const Config& config,
                            std::shared_ptr<Globals> globals);
  ~ClassAdaptiveIntegrator() override = default;

  void processInput(SubmapCollection* submaps, InputData* input) override;

 protected:
  void updateSubmap(Submap* submap, InterpolatorBase* interpolator,
                    const voxblox::BlockIndexList& block_indices,
                    const InputData& input) const override;

  void updateBlock(Submap* submap, InterpolatorBase* interpolator,
                   const voxblox::BlockIndex& block_index,
                   const Transformation& T_C_S,
                   const InputData& input) const override;

  // simple raycasting
  void integratePointCloud(Submap* submap, Pointcloud &submap_points,
                           Colors &submap_colors, Pointcloud &submap_normals,
                           Transformation &T_S_C, const bool normal_refine_on,                       
                           const bool is_free_space_submap, const float voxel_size,                       
                           const int voxels_per_side, const float truncation_distance) const; 
  
  // bundle (merged) raycasting
  void integratePointCloudMerged(Submap* submap, Pointcloud &submap_points, 
                                 Colors &submap_colors, Pointcloud &submap_normals,
                                 Transformation &T_S_C, const bool normal_refine_on,                       
                                 const bool is_free_space_submap, const float voxel_size, 
                                 const int voxels_per_side, const float truncation_distance) const;                                 
                                 
                                                                                    
  
  // For projective mapping
  bool updateVoxel(InterpolatorBase* interpolator, TsdfVoxel* voxel,
                   const Point& p_C, const InputData& input,
                   const int submap_id, const bool is_free_space_submap,
                   const bool apply_normal_refine, const Transformation& T_C_S,
                   const float truncation_distance, const float voxel_size,
                   ClassVoxel* class_voxel = nullptr) const override;
  
  // For raycasting
  bool updateVoxel(TsdfVoxel* voxel, const Point& origin, const Point& p_C, 
                   const Point& p_S, const Ray& n_C, 
                   const GlobalIndex& global_voxel_idx,
                   const Color& color, const bool is_free_space_submap,
                   const bool apply_normal_refine, const Transformation& T_C_S,
                   const float precompute_weight, 
                   const float truncation_distance, const float voxel_size, 
                   ClassVoxel* class_voxel = nullptr) const override;

  void updateClassVoxel(InterpolatorBase* interpolator, ClassVoxel* voxel,
                        const InputData& input, const int submap_id) const;

  void bundleRays(const Transformation& T_S_C, const Pointcloud& submap_points,
                  const float voxel_size_inv,
                  voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type* voxel_map,                       
                  voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type* clear_map) const; 

  void integrateRay(Submap* submap, const Point &p_C, const Color &color,
                    const Ray &n_C, const float weight, const Transformation &T_S_C,
                    const bool normal_refine_on, const bool is_free_space_submap,
                    const bool is_clearing, const bool integrate_full_ray,
                    const float voxel_size, const int voxels_per_side, 
                    const float truncation_distance) const;                                             
                                                                  
  void integrateVoxelMerged(Submap *submap, const Pointcloud &submap_points,
                            const Colors &submap_colors, const Pointcloud &submap_normals,                      
                            const Transformation& T_S_C, const std::pair<GlobalIndex, AlignedVector<size_t>>& kv,                      
                            const bool clearing_ray, const bool normal_refine_on,
                            const bool is_free_space_submap, const bool integrate_full_ray,                        
                            const float voxel_size, const int voxels_per_side,                                              
                            const float truncation_distance) const;                            

  float getVoxelWeight(const Point& point_C) const;                
                                                                  

 private:
  const Config config_;
  static config_utilities::Factory::RegistrationRos<
      TsdfIntegratorBase, ClassAdaptiveIntegrator, std::shared_ptr<Globals>>
      registration_;

  // Cached data.
  std::unordered_map<int, int> id_to_class_; //<submap_id, submap_class>
  size_t num_classes_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_INTEGRATION_CLASS_ADAPTIVE_TSDF_INTEGRATOR_H_
