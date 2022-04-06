#include "panoptic_mapping/integration/class_projective_tsdf_integrator.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <voxblox/integrator/merge_integration.h>

#include "panoptic_mapping/common/index_getter.h"

namespace panoptic_mapping {

config_utilities::Factory::RegistrationRos<
    TsdfIntegratorBase, ClassProjectiveIntegrator, std::shared_ptr<Globals>>
    ClassProjectiveIntegrator::registration_("class_projective");

void ClassProjectiveIntegrator::Config::checkParams() const {
  checkParamConfig(pi_config);
}

void ClassProjectiveIntegrator::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("use_binary_classification", &use_binary_classification);
  setupParam("use_instance_classification", &use_instance_classification);
  setupParam("update_only_tracked_submaps", &update_only_tracked_submaps);
  setupParam("projective_integrator_config", &pi_config);
}

ClassProjectiveIntegrator::ClassProjectiveIntegrator(
    const Config& config, std::shared_ptr<Globals> globals)
    : config_(config.checkValid()),
      ProjectiveIntegrator(config.pi_config, std::move(globals), false) {
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();

  // Store class count.
  if (!config_.use_binary_classification &&
      !config_.use_instance_classification) {
    // The +1 is added because 0 is reserved for the belonging submap.
    num_classes_ = globals_->labelHandler()->numberOfLabels() + 1;
  }
}

void ClassProjectiveIntegrator::processInput(SubmapCollection* submaps,
                                             InputData* input) {
  
  //ROS_INFO("Debuging/Begin Class Projective Integration");
  CHECK_NOTNULL(submaps);  // Input is not used here and checked later.
  // Cache submap ids by class.
  id_to_class_.clear();

  for (const Submap& submap : *submaps) {
    id_to_class_.emplace(submap.getID(), submap.getClassID()); //<submap_id, submap_class>
  }
  id_to_class_[-1] = -1;  // Used for unknown classes.
  if (config_.use_instance_classification &&
      !config_.use_binary_classification) {
    // Track the number of classes (where classes in this case are instances).
    // NOTE(schmluk): This is dangerous if submaps are de-allocated and can grow
    // arbitrarily large.
    num_classes_ = id_to_class_.size() + 1;
  }

  // Run the integration.
  ProjectiveIntegrator::processInput(submaps, input);
}

//This function is called when ClassProjectiveIntegrator is selected
void ClassProjectiveIntegrator::updateBlock(
    Submap* submap, InterpolatorBase* interpolator,
    const voxblox::BlockIndex& block_index, const Transformation& T_C_S,
    const InputData& input) const {
  CHECK_NOTNULL(submap);
  // Set up preliminaries.

  if (!submap->getTsdfLayer().hasBlock(block_index)) {
    LOG_IF(WARNING, config_.verbosity >= 1)
        << "Tried to access inexistent block '" << block_index.transpose()
        << "' in submap " << submap->getID() << ".";
    return;
  }
  
  TsdfBlock& block = submap->getTsdfLayerPtr()->getBlockByIndex(block_index);

  const float voxel_size = block.voxel_size();
  const float truncation_distance = submap->getConfig().truncation_distance;

  const int submap_id = submap->getID();
  const bool is_free_space_submap =
      submap->getLabel() == PanopticLabel::kFreeSpace;
  bool was_updated = false;
  
  bool normal_reliable = submap->getNormalReliability(); // free space default true
  
  bool apply_normal_refine = is_free_space_submap ? 
                              config_.pi_config.apply_normal_refine_freespace :
                              config_.pi_config.apply_normal_refine;
  
  bool normal_refine_on = false;
  if (apply_normal_refine && normal_reliable)
    normal_refine_on = true;
    
  // Allocate the class block if not yet existent and get it.
  ClassBlock::Ptr class_block;
  if (submap->hasClassLayer() &&
      (!config_.update_only_tracked_submaps || submap->wasTracked())) { 
        //update_only_tracked_submaps: defualt true
    class_block =
        submap->getClassLayerPtr()->allocateBlockPtrByIndex(block_index);
  }

  // Update all voxels inside this block
  for (size_t i = 0; i < block.num_voxels(); ++i) {
    TsdfVoxel& voxel = block.getVoxelByLinearIndex(i);

    // Voxel center in camera frame.       
    const Point p_C = T_C_S * block.computeCoordinatesFromLinearIndex(i); 
                          
    ClassVoxel* class_voxel = nullptr;
    if (class_block) {
      class_voxel = &class_block->getVoxelByLinearIndex(i);
    }

    // update this voxel's sdf and weight (also consider the gradient)
    if (updateVoxel(interpolator, &voxel, p_C, input, submap_id,
                    is_free_space_submap, normal_refine_on, T_C_S,
                    truncation_distance, voxel_size,
                    class_voxel)) {
      was_updated = true;
    }
  }
  if (was_updated) {
    block.setUpdatedAll();
  }
}

// projective mapping for each voxel
bool ClassProjectiveIntegrator::updateVoxel(
    InterpolatorBase* interpolator, TsdfVoxel* voxel, const Point& p_C,
    const InputData& input, const int submap_id,
    const bool is_free_space_submap, const bool apply_normal_refine, 
    const Transformation& T_C_S, const float truncation_distance,
    const float voxel_size, ClassVoxel* class_voxel) const {
  
  // Compute the signed distance. Firstly we need to setup the interpolator.
  float sdf, u, v;

  const float distance_to_voxel = p_C.norm();

  // neither too far or too close
  if (distance_to_voxel < min_range_ ||
      distance_to_voxel > max_range_) 
    return false;

  // Project the current voxel into the range image, only count points that fall
  // fully into the image.
  if (config_.pi_config.use_lidar)
  {
    if (!globals_->lidar()->projectPointToImagePlane(p_C, &u, &v))
      return false;
  }
  else{ //camera
    if (p_C.z() < 0.0) 
       return false;
  
    if (!globals_->camera()->projectPointToImagePlane(p_C, &u, &v))
      return false;
  }

  // Set up the interpolator and compute the signed distance.
  // why not directly use the depth_image from input data (because the interpolate need eigen::mat as the input)
  interpolator->computeWeights(u, v, range_image_);  //according to u,v (float)

  // Check whether this ray is projected to another submap
  // if so, we discard this voxel
  const bool point_belongs_to_this_submap =
      interpolator->interpolateID(input.idImage()) == submap_id;
  if (!(point_belongs_to_this_submap || config_.pi_config.foreign_rays_clear ||
        is_free_space_submap)) { 
    //these three statements should all be false to finally return false
    return false;
  }

  // Interpolate depth on the range image 
  const float distance_to_surface =
      interpolator->interpolateRange(range_image_); 
  
  sdf = distance_to_surface - distance_to_voxel; //the projective sdf
  
  cv::Vec3f normal;
  Ray n_C;
  Ray v_C;
  bool update_gradient = false;

  bool in_the_reliable_band = true;
  if (sdf > truncation_distance * config_.pi_config.reliable_band_ratio)
    in_the_reliable_band = false;

  // normal refine
  if (apply_normal_refine && input.has(InputData::InputType::kNormalImage) 
      && in_the_reliable_band) {
    // current normal vector
    normal = input.normalImage().at<cv::Vec3f>(v, u);
    n_C = Ray(normal[0], normal[1], normal[2]); //in sensor(camera)'s frame

    float normal_ratio = 1.0f;
    if (voxel->gradient.norm() > kFloatEpsilon){ //use current un-updated normal because the weight is unknown
      v_C = T_C_S.getRotationMatrix() * voxel->gradient; //back to sensor(camera)'s frame
      normal_ratio = std::abs(v_C.dot(p_C)/p_C.norm());
    }
    else { //gradient not ready yet, use the first (current) normal vector
      if (n_C.norm() > kFloatEpsilon)  //current normal is valid
        normal_ratio = std::abs(n_C.dot(p_C)/p_C.norm());
    }
    if (normal_ratio < config_.pi_config.reliable_normal_ratio_thre) // NOTE(py): ruling out extremely large incidence angle
      return false;

    sdf *= normal_ratio; //get the non-projective sdf
    update_gradient = true;
  }

  if (sdf < -truncation_distance) {
    return false;
  }

  // Compute the weight of this measurement.
  const float weight = computeVoxelWeight(p_C, voxel_size, truncation_distance, sdf, true);

  // Only merge color and classification data near the surface.
  if (sdf > truncation_distance) {
    // far away (truncated part) or not used to be meshed (free submap), do not interpolate color 
    updateVoxelValues(voxel, sdf, weight);
    voxel->distance = std::min(truncation_distance, voxel->distance);
  } else {
    // Update voxel gradient
    if (update_gradient){
      Ray n_S = T_C_S.getRotationMatrix().inverse() * n_C;  //in submap's frame
      if (n_C.norm() > kFloatEpsilon) 
        updateVoxelGradient(voxel, n_S, weight); //direction issue
    }
    if (is_free_space_submap) {
      updateVoxelValues(voxel, sdf, weight);
    } else {
      const Color color = interpolator->interpolateColor(input.colorImage());
      updateVoxelValues(voxel, sdf, weight, &color);
    }
    // Update the class voxel. (the category)
    if (class_voxel) {
      updateClassVoxel(interpolator, class_voxel, input, submap_id);
    }
  }
  return true;
}

void ClassProjectiveIntegrator::updateClassVoxel(InterpolatorBase* interpolator,
                                                 ClassVoxel* voxel,
                                                 const InputData& input,
                                                 const int submap_id) const {
  if (config_.use_binary_classification) {
    // Use ID 0 for belongs, 1 for does not belong.
    if (config_.use_instance_classification) {
      // Just count how often the assignments were right.
      voxel->incrementCount(
          1 - static_cast<int>(interpolator->interpolateID(input.idImage()) ==
                               submap_id));
    } else {
      // Only the class needs to match.
      auto it = id_to_class_.find(submap_id);
      auto it2 =
          id_to_class_.find(interpolator->interpolateID(input.idImage()));
      if (it != id_to_class_.end() && it2 != id_to_class_.end()) {
        voxel->incrementCount(1 - static_cast<int>(it->second == it2->second));
      } else {
        voxel->incrementCount(1);
      }
    }
  } else {
    if (config_.use_instance_classification) {
      voxel->incrementCount(interpolator->interpolateID(input.idImage()));
    } else {
      // NOTE(schmluk): id_to_class should always exist since it's created based
      // on the input.
      voxel->incrementCount(
          id_to_class_.at(interpolator->interpolateID(input.idImage())));
    }
  }
}

}  // namespace panoptic_mapping
