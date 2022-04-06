#ifndef PANOPTIC_MAPPING_INTEGRATION_TSDF_INTEGRATOR_BASE_H_
#define PANOPTIC_MAPPING_INTEGRATION_TSDF_INTEGRATOR_BASE_H_

#include <memory>
#include <utility>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/common/globals.h"
#include "panoptic_mapping/common/input_data_user.h"
#include "panoptic_mapping/map/submap_collection.h"

namespace panoptic_mapping {

/**
 * Interface for TSDF integrators.
 */
class TsdfIntegratorBase : public InputDataUser {
 public:
  explicit TsdfIntegratorBase(std::shared_ptr<Globals> globals)
      : globals_(std::move(globals)) {}
  ~TsdfIntegratorBase() override = default;
  
  virtual void processInput(SubmapCollection* submaps, InputData* input) = 0;

 protected:

  // pure virtual function interface

  /**
   * @brief Compute the measurement weight for a given voxel based on the
   * parameters set in the config.
   *
   * @param p_C Voxel center in camera frame in meters.
   * @param voxel_size The voxel size in meters.
   * @param truncation_distance The truncation distance in meters.
   * @param sdf The signed distance used for this update.
   * @param projective Using projective mapping or not (account for ray width)
   * @param with_init_weight Use a precompute init weight, directly enter the weight drop-off step
   * @param init_weight the precompute init weight
   * @return The measurement weight.
   */
  virtual float computeVoxelWeight(const Point& p_C, const float voxel_size,
                                   const float truncation_distance,
                                   const float sdf, bool projective,
                                   const bool with_init_weight, const float init_weight) const = 0;
  
  /**
   * @brief Update the values of a voxel in a weighted averaging fashion.
   *
   * @param voxel The voxel to be updated. The pointer is not checked for
   * validity.
   * @param sdf The signed distance measurement to be fused, already truncated
   * to the truncation distance.
   * @param weight The measurement weight to be used for the update.
   * @param color Optional pointer to a color to be fused.
   */
  virtual void updateVoxelValues(TsdfVoxel* voxel, const float sdf,
                                 const float weight,
                                 const Color* color = nullptr) const = 0;

  /**
   * @brief Update the TSDF gradient of a voxel in a weighted averaging fashion.
   *
   * @param voxel The voxel to be updated. The pointer is not checked for
   * validity.
   * @param normal The unit normal vector at the surface where the ray cast 
   * (in the submap's frame).
   * @param weight The measurement weight to be used for the update.
   */
  virtual void updateVoxelGradient(TsdfVoxel* voxel, const Ray normal,
                                   const float weight) const = 0;

  std::shared_ptr<Globals> globals_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_INTEGRATION_TSDF_INTEGRATOR_BASE_H_
