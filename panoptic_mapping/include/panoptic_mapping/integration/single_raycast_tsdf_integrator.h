#ifndef PANOPTIC_MAPPING_INTEGRATION_SINGLE_RAYCAST_TSDF_INTEGRATOR_H_
#define PANOPTIC_MAPPING_INTEGRATION_SINGLE_RAYCAST_TSDF_INTEGRATOR_H_

#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/common/camera.h"
#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/integration/raycast_tsdf_integrator.h"
#include "panoptic_mapping/map/classification/uncertainty.h"

namespace panoptic_mapping {

/**
 * @brief Integrator that integrates all data into a single submap to emulate a
 * monolithic approach. Combine this module with the SingleTsdfIDTracker.
 */
class SingleRaycastIntegrator : public RaycastIntegrator {
 public:
  struct Config : public config_utilities::Config<Config> {
    int verbosity = 4;

    // Standard integrator params.
    RaycastIntegrator::Config ri_config;

    // If true require a color image and update voxel colors.
    bool use_color = true;

    // If true require a segmentation image and integrate it into a class layer.
    bool use_segmentation = true;

    // If true require an uncertainty image and integrate it into an class layer
    // of type 'UncertaintyLayer'.
    bool use_uncertainty = false;

    // Decay rate in [0, 1] used to update uncertainty voxels. Only used if
    // 'use_uncertainty' is true.
    float uncertainty_decay_rate = 0.5f;

    Config() { setConfigName("SingleRaycastIntegrator"); }

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
  };

  SingleRaycastIntegrator(const Config& config, std::shared_ptr<Globals> globals);
  ~SingleRaycastIntegrator() override = default;

  void processInput(SubmapCollection* submaps, InputData* input) override;

 protected:
  void updateSubmap(Submap* submap, const InputData& input) override;

  // simple raycasting
  void integratePointCloudSimple(Submap* submap, Pointcloud &submap_points,
                                Colors &submap_colors, Pointcloud &submap_normals,
                                Transformation &T_S_C, const bool normal_refine_on,                       
                                const bool is_free_space_submap, const bool multi_threads, 
                                const float voxel_size, const int voxels_per_side, 
                                const float truncation_distance) const;                       
                                
  void integratePointSimple(Submap* submap, const Pointcloud &submap_points, const Colors &submap_colors,
                            const Pointcloud &submap_normals, const Transformation &T_S_C,
                            const bool normal_refine_on,                    
                            const bool is_free_space_submap, const bool integrate_full_ray,                      
                            const float voxel_size, const int voxels_per_side, 
                            const float truncation_distance, 
                            voxblox::ThreadSafeIndex* index_getter) const;                            
                                                                                                                                 
  // bundle (merged) raycasting
  void integratePointCloudMerged(Submap* submap, Pointcloud &submap_points, 
                                 Colors &submap_colors, Pointcloud &submap_normals,
                                 Transformation &T_S_C, const bool normal_refine_on,                       
                                 const bool is_free_space_submap, const bool multi_threads,
                                 const float voxel_size, const int voxels_per_side,
                                 const float truncation_distance) const;   

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

  void integrateVoxelsMerged(Submap *submap, const Pointcloud &submap_points,
                             const Colors &submap_colors, const Pointcloud &submap_normals,                      
                             const Transformation& T_S_C,   
                             const voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type& voxel_map,                  
                             const bool clearing_ray, const bool normal_refine_on,
                             const bool is_free_space_submap, const bool integrate_full_ray,
                             const float voxel_size, const int voxels_per_side,                                              
                             const float truncation_distance, const int thread_idx) const;  
                                                                         
  void integrateVoxelMerged(Submap *submap, const Pointcloud &submap_points,
                            const Colors &submap_colors, const Pointcloud &submap_normals,                      
                            const Transformation& T_S_C, const std::pair<GlobalIndex, AlignedVector<size_t>>& kv,                      
                            const bool clearing_ray, const bool normal_refine_on,
                            const bool is_free_space_submap, const bool integrate_full_ray,                        
                            const float voxel_size, const int voxels_per_side,                                              
                            const float truncation_distance) const;           

  bool updateVoxel(TsdfVoxel* voxel, const Point& origin, const Point& p_C, 
                   const Point& p_S, const Ray& n_C, 
                   const GlobalIndex& global_voxel_idx,
                   const Color& color, const bool is_free_space_submap,
                   const bool apply_normal_refine, const Transformation& T_C_S,
                   const float init_weight,
                   const float truncation_distance, const float voxel_size, 
                   ClassVoxel* class_voxel = nullptr) const override;                 

        

 private:
  const Config config_;
  static config_utilities::Factory::RegistrationRos<
      TsdfIntegratorBase, SingleRaycastIntegrator, std::shared_ptr<Globals>>
      registration_;

  void updateClassVoxel(InterpolatorBase* interpolator, const InputData& input,
                        ClassVoxel* class_voxel) const;

  void updateUncertaintyVoxel(InterpolatorBase* interpolator,
                              const InputData& input,
                              UncertaintyVoxel* class_voxel) const;

  int num_classes_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_INTEGRATION_SINGLE_RAYCAST_TSDF_INTEGRATOR_H_
