#include "panoptic_mapping/integration/single_projective_tsdf_integrator.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <voxblox/integrator/merge_integration.h>

#include "panoptic_mapping/common/index_getter.h"

// TODO(py): add the lidar option
namespace panoptic_mapping {

config_utilities::Factory::RegistrationRos<
    TsdfIntegratorBase, SingleProjectiveIntegrator, std::shared_ptr<Globals>>
    SingleProjectiveIntegrator::registration_("single_projective");

void SingleProjectiveIntegrator::Config::checkParams() const {
  checkParamConfig(pi_config);
  if (use_uncertainty) {
    checkParamGE(uncertainty_decay_rate, 0.f, "uncertainty_decay_rate");
    checkParamLE(uncertainty_decay_rate, 1.f, "uncertainty_decay_rate");
  }
}

void SingleProjectiveIntegrator::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("projective_integrator", &pi_config);
  setupParam("use_color", &use_color);
  setupParam("use_segmentation", &use_segmentation);
  setupParam("use_uncertainty", &use_uncertainty);
  setupParam("uncertainty_decay_rate", &uncertainty_decay_rate);
  // Set this param to false since there is only one layer.
  pi_config.foreign_rays_clear = false; // only one map, no class
}

SingleProjectiveIntegrator::SingleProjectiveIntegrator(const Config& config,
                                           std::shared_ptr<Globals> globals)
    : config_(config.checkValid()),
      ProjectiveIntegrator(config.pi_config,
                           std::move(globals), false) {
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();
  // Cache num classes info.
  // num_classes_ = globals_->labelHandler()->numberOfLabels();
}

// the only map processed is the freespace submap
void SingleProjectiveIntegrator::processInput(SubmapCollection* submaps,
                                              InputData* input) {
  CHECK_NOTNULL(submaps);
  CHECK_NOTNULL(input);
  // CHECK_NOTNULL(globals_->camera().get());
  CHECK(inputIsValid(*input));

  // Allocate all blocks in the map.
  
  if (config_.pi_config.use_lidar)
     max_range_ = globals_->lidar()->getConfig().max_range;
  else
     max_range_ = globals_->camera()->getConfig().max_range;

  Submap* map = submaps->getSubmapPtr(submaps->getActiveFreeSpaceSubmapID());
  
  Timer alloc_timer("tsdf_integration/allocate_blocks");
  auto t1 = std::chrono::high_resolution_clock::now();
  allocateNewBlocks(map, input);
  auto t2 = std::chrono::high_resolution_clock::now();
  alloc_timer.Stop();

  Timer integrate_timer("tsdf_integration/integration");
  // Find all active blocks that are in the field of view.
  voxblox::BlockIndexList block_lists;
  if (config_.pi_config.use_lidar)
    block_lists = globals_->lidar()->findVisibleBlocks(
      *map, input->T_M_C(), max_range_in_image_);
  else
    block_lists = globals_->camera()->findVisibleBlocks(
      *map, input->T_M_C(), max_range_in_image_);
      
  std::vector<voxblox::BlockIndex> indices;
  indices.resize(block_lists.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    indices[i] = block_lists[i];
  }
  IndexGetter<voxblox::BlockIndex> index_getter(indices);
  const Transformation T_C_S = input->T_M_C().inverse() * map->getT_M_S();

  // Integrate in parallel for each block (only one map)
  std::vector<std::future<void>> threads; 
  for (int i = 0; i < config_.pi_config.integration_threads;
       ++i) {
    threads.emplace_back(std::async(
        std::launch::async, [this, &index_getter, map, input, i, T_C_S]() {
          voxblox::BlockIndex index;
          while (index_getter.getNextIndex(&index)) {
            this->updateBlock(map, interpolators_[i].get(), index, T_C_S,
                              *input);
          }
        }));
  }

  // Join all threads.
  for (auto& thread : threads) {
    thread.get();
  }
  auto t3 = std::chrono::high_resolution_clock::now();
  integrate_timer.Stop();

  LOG_IF(INFO, config_.verbosity >= 4)
      << "Allocate: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()
      << "ms, Integrate: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
      << "ms.";
}

void SingleProjectiveIntegrator::updateBlock(Submap* submap,
                                             InterpolatorBase* interpolator,
                                             const voxblox::BlockIndex& block_index,
                                             const Transformation& T_C_S,
                                             const InputData& input) const {
  // Set up preliminaries.
  if (!submap->getTsdfLayer().hasBlock(block_index)) {
    LOG_IF(WARNING, config_.verbosity >= 1)
        << "Tried to access inexistent block '" << block_index.transpose()
        << "' in submap " << submap->getID() << ".";
    return;
  }
  TsdfBlock& block = submap->getTsdfLayerPtr()->getBlockByIndex(block_index);
  bool was_updated = false;
  const float voxel_size = block.voxel_size();
  const float truncation_distance = submap->getConfig().truncation_distance;
  const int submap_id = submap->getID();
  ClassBlock::Ptr class_block;
  const bool use_class_layer = submap->hasClassLayer();
  if (use_class_layer) {
    if (!submap->getClassLayer().hasBlock(block_index)) {
      LOG_IF(WARNING, config_.verbosity >= 1)
          << "Tried to access inexistent class block '"
          << block_index.transpose() << "' in submap " << submap->getID()
          << ".";
      return;
    }
    class_block = submap->getClassLayerPtr()->getBlockPtrByIndex(block_index);
  }

  // Update all voxels.
  for (size_t i = 0; i < block.num_voxels(); ++i) {
    TsdfVoxel& voxel = block.getVoxelByLinearIndex(i);
    ClassVoxel* class_voxel = nullptr;
    if (use_class_layer) {
      class_voxel = &class_block->getVoxelByLinearIndex(i);
    }
    const Point p_C = T_C_S * block.computeCoordinatesFromLinearIndex(
                                  i);  // Voxel center in camera frame.
    if (updateVoxel(interpolator, &voxel, p_C, input, submap_id, true, 
                    config_.pi_config.apply_normal_refine_freespace, T_C_S,
                    truncation_distance, voxel_size, class_voxel)) { // only process the free-space submap as the single-submap
      was_updated = true;
    }
  }

  if (was_updated) {
    block.setUpdatedAll();
  }
}

bool SingleProjectiveIntegrator::updateVoxel(
    InterpolatorBase* interpolator, TsdfVoxel* voxel, const Point& p_C,
    const InputData& input, const int submap_id,
    const bool is_free_space_submap, const bool apply_normal_refine, 
    const Transformation& T_C_S, const float truncation_distance,
    const float voxel_size, ClassVoxel* class_voxel) const {
  
  float sdf, u, v;
  // Setup the interpolator for current voxel center and compute 
  // the signed distance. 
  if (!computeSignedDistance(p_C, interpolator, &sdf, &u, &v)) {
    return false;
  }

  cv::Vec3f normal;
  Ray n_C, v_C;
  bool update_gradient = false;

  bool in_the_reliable_band = true;
  if (sdf > truncation_distance * config_.pi_config.reliable_band_ratio)
    in_the_reliable_band = false;

  if(apply_normal_refine && 
    input.has(InputData::InputType::kNormalImage) && 
    in_the_reliable_band) {

    float normal_ratio = 1.0f;
    if (voxel->gradient.norm() > kFloatEpsilon) { // use current un-updated normal because the weight is unknown
      normal = input.normalImage().at<cv::Vec3f>(v, u);
      n_C = Ray(normal[0], normal[1], normal[2]); // in sensor(camera)'s frame
      v_C = T_C_S.getRotationMatrix() * voxel->gradient; // back to sensor(camera)'s frame
      if (config_.pi_config.curve_assumption && n_C.norm() > kFloatEpsilon) {
        // method 1: curve assumption, [sin(theta+alpha)-sin(theta)]/sin(alpha)
        float cos_theta = std::abs(v_C.dot(p_C)/p_C.norm()); 
        float cos_alpha = std::abs(v_C.dot(n_C)/n_C.norm()); 
        float sin_theta = std::sqrt(1-cos_theta*cos_theta);
        float sin_alpha = std::sqrt(1-cos_alpha*cos_alpha);
        normal_ratio = std::abs(sin_theta * (cos_alpha-1) / sin_alpha + cos_theta);
        if (isnan(normal_ratio)) 
          normal_ratio = cos_theta;
        // LOG(INFO) << "ratio:" << normal_ratio;
      } else {
        // method 2: cos(theta)
        normal_ratio = std::abs(v_C.dot(p_C)/p_C.norm()); 
      }
    } else { // gradient not ready yet, use the first (current) normal vector
      // NOTE(py): kFloatEpsilon is a safe value in case of numerical rounding error
      if (n_C.norm() > kFloatEpsilon) { // current normal is valid
        normal_ratio = std::abs(n_C.dot(p_C)/p_C.norm());
      }
    }
    if (normal_ratio < config_.pi_config.reliable_normal_ratio_thre) // NOTE(py): ruling out extremely large incidence angle
      return false;
    sdf *= normal_ratio; // get the non-projective sdf, if it's still larger than truncation distance, the gradient would not be updated
    update_gradient = true;
  }
  if (sdf < -truncation_distance) {
    return false;
  }
  // Compute the weight of this measurement.
  const float weight = computeVoxelWeight(p_C, voxel_size, truncation_distance, sdf, true);

  if (sdf > truncation_distance) {
    // far away (truncated part) or not used to be meshed (free submap), do not interpolate color 
    updateVoxelValues(voxel, sdf, weight);
    voxel->distance = std::min(truncation_distance, voxel->distance);
  } else {
    // Update voxel gradient
    if (update_gradient) {
      Ray n_S = T_C_S.getRotationMatrix().inverse() * n_C;  //in submap's frame
      if (n_C.norm() > kFloatEpsilon) 
        updateVoxelGradient(voxel, n_S, weight); //direction issue
    }
    const Color color = interpolator->interpolateColor(input.colorImage());
    updateVoxelValues(voxel, sdf, weight, &color);
    // Update the semantic information if requested.
    if (class_voxel) {
      // Uncertainty voxels are handled differently.
      if (config_.use_uncertainty &&
          class_voxel->getVoxelType() == ClassVoxelType::kUncertainty) {
        updateUncertaintyVoxel(interpolator, input,
                               static_cast<UncertaintyVoxel*>(class_voxel));
      } else {
        updateClassVoxel(interpolator, input, class_voxel);
      }
    }
  }
  return true;
}

void SingleProjectiveIntegrator::allocateNewBlocks(Submap* map, InputData* input) {
  // This method also resets the depth image.
  range_image_.setZero();
  max_range_in_image_ = 0.f;

  const Transformation T_S_C = map->getT_S_M() * input->T_M_C();
  // Parse through each point to reset the depth image.
  for (int v = 0; v < input->depthImage().rows; v++) {
    for (int u = 0; u < input->depthImage().cols; u++) {
      const cv::Vec3f& vertex = input->vertexMap().at<cv::Vec3f>(v, u);
      const Point p_C(vertex[0], vertex[1], vertex[2]);
      const float ray_distance = p_C.norm();
      range_image_(v, u) = ray_distance;
      max_range_in_image_ = std::max(max_range_in_image_, ray_distance);
    }
  }
  max_range_in_image_ = std::min(max_range_in_image_, max_range_);

  // Allocate all potential blocks.
  const float block_size = map->getTsdfLayer().block_size();
  const float block_diag_half = std::sqrt(3.f) * block_size / 2.f;
  const Transformation T_C_S = T_S_C.inverse();
  const Point camera_S = T_S_C.getPosition();  // T_S_C
  const int max_steps = std::floor((max_range_in_image_ + block_diag_half) /
                                   map->getTsdfLayer().block_size());
  for (int x = -max_steps; x <= max_steps; ++x) {
    for (int y = -max_steps; y <= max_steps; ++y) {
      for (int z = -max_steps; z <= max_steps; ++z) {
        const Point offset(x, y, z);
        const Point candidate_S = camera_S + offset * block_size;

        bool point_in_view = false;
        if (config_.pi_config.use_lidar)
          point_in_view = globals_->lidar()->pointIsInViewFrustum(T_C_S * candidate_S, block_diag_half);                                       
        else
          point_in_view = globals_->camera()->pointIsInViewFrustum(T_C_S * candidate_S, block_diag_half);                                           
        
        if (point_in_view) {
          map->getTsdfLayerPtr()->allocateBlockPtrByCoordinates(candidate_S);
          if (map->hasClassLayer()) {
            map->getClassLayerPtr()->allocateBlockPtrByCoordinates(candidate_S);
          }
        }
      }
    }
  }

  // Update the bounding volume.
  map->updateBoundingVolume();
}

void SingleProjectiveIntegrator::updateClassVoxel(InterpolatorBase* interpolator,
                                            const InputData& input,
                                            ClassVoxel* class_voxel) const {
  // For the single TSDF case there is no belonging submap, just use the ID
  // directly.
  const int id = interpolator->interpolateID(input.idImage());
  class_voxel->incrementCount(id);
}

// used mainly for monolithic semantic mapping (with uncertainty)
void SingleProjectiveIntegrator::updateUncertaintyVoxel(
    InterpolatorBase* interpolator, const InputData& input,
    UncertaintyVoxel* class_voxel) const {
  // Do not update voxels which are assigned as groundtruth.
  if (class_voxel->is_ground_truth) {
    return;
  }
  // Update Uncertainty Voxel Part.
  const float uncertainty =
      interpolator->interpolateUncertainty(input.uncertaintyImage());

  // Magic uncertainty value which labels a voxel as groundtruth in the
  // uncertainty input..
  // TODO(zrene) find a better way to implement this.
  if (uncertainty == -1.0) {
    // Make sure GT voxels have zero uncertainty and entropy.
    class_voxel->counts =
        std::vector<ClassificationCount>(FixedCountVoxel::numCounts());
    // Update classification part.
    class_voxel->is_ground_truth = true;
    class_voxel->uncertainty = 0.f;
  } else {
    // Update uncertainty.
    class_voxel->uncertainty =
        config_.uncertainty_decay_rate * uncertainty +
        (1.f - config_.uncertainty_decay_rate) * class_voxel->uncertainty;
  }

  updateClassVoxel(interpolator, input, class_voxel);
}

}  // namespace panoptic_mapping
