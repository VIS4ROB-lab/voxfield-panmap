#include "panoptic_mapping/integration/adaptive_tsdf_integrator.h"

#include <algorithm>
#include <chrono>
#include <future>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <voxblox/integrator/merge_integration.h>

#include "panoptic_mapping/common/index_getter.h"

namespace panoptic_mapping {

config_utilities::Factory::RegistrationRos<
    TsdfIntegratorBase, AdaptiveIntegrator, std::shared_ptr<Globals>>
    AdaptiveIntegrator::registration_("adaptive");

void AdaptiveIntegrator::Config::checkParams() const {
  checkParamGT(integration_threads, 0, "integration_threads");
  checkParamGT(max_weight, 0.f, "max_weight");
  checkParamGT(weight_reduction_exp, 0.f, "weight_reduction_exp");
  if (use_weight_dropoff) {
    checkParamNE(weight_dropoff_epsilon, 0.f, "weight_dropoff_epsilon");
  }
}

void AdaptiveIntegrator::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("skip_free_space_submap", &skip_free_space_submap);
  setupParam("merged_raycasting", &merged_raycasting);
  setupParam("only_merge_freespace", &only_merge_freespace);
  setupParam("use_weight_dropoff", &use_weight_dropoff);
  setupParam("weight_dropoff_epsilon", &weight_dropoff_epsilon);
  setupParam("weight_reduction", &weight_reduction);
  setupParam("weight_reduction_exp", &weight_reduction_exp);
  setupParam("foreign_rays_clear", &foreign_rays_clear);
  setupParam("max_weight", &max_weight);
  setupParam("voxel_carving_enabled", &voxel_carving_enabled);
  setupParam("freespace_carving_enabled", &freespace_carving_enabled);
  setupParam("max_ray_length_m", &max_ray_length_m, "m");
  setupParam("min_ray_length_m", &min_ray_length_m, "m");
  setupParam("interpolation_method", &interpolation_method);
  setupParam("allocate_neighboring_blocks", &allocate_neighboring_blocks);
  setupParam("use_longterm_fusion", &use_longterm_fusion);
  setupParam("integration_threads", &integration_threads);
  setupParam("use_lidar", &use_lidar);
  setupParam("apply_normal_refine", &apply_normal_refine);
  setupParam("apply_normal_refine_freespace", &apply_normal_refine_freespace);
}

AdaptiveIntegrator::AdaptiveIntegrator(const Config& config,
                                       std::shared_ptr<Globals> globals,
                                       bool print_config)
    : config_(config.checkValid()), TsdfIntegratorBase(std::move(globals)) {
  LOG_IF(INFO, config_.verbosity >= 1 && print_config) << "\n"
                                                       << config_.toString();
  // Request all inputs.
  addRequiredInputs(
      {InputData::InputType::kColorImage, InputData::InputType::kDepthImage,
       InputData::InputType::kSegmentationImage,
       InputData::InputType::kVertexMap});
  // InputData::InputType::kValidityImage not used

  // Setup the interpolators (one for each thread).
  for (int i = 0; i < config_.integration_threads; ++i) {
    interpolators_.emplace_back(
        config_utilities::Factory::create<InterpolatorBase>(
            config_.interpolation_method));
  }

  // Allocate range image.
  if (config_.use_lidar)
    range_image_ = Eigen::MatrixXf(globals_->lidar()->getConfig().height,
                                   globals_->lidar()->getConfig().width);
  else
    range_image_ = Eigen::MatrixXf(globals_->camera()->getConfig().height,
                                   globals_->camera()->getConfig().width);
  
}

void AdaptiveIntegrator::processInput(SubmapCollection* submaps,
                                      InputData* input) {
  CHECK_NOTNULL(submaps);
  CHECK_NOTNULL(input);
  if (config_.use_lidar){
    CHECK_NOTNULL(globals_->lidar().get());
    max_range_ = globals_->lidar()->getConfig().max_range;
    min_range_ = globals_->lidar()->getConfig().min_range;
    max_z_ = globals_->lidar()->getConfig().max_z;
    min_z_ = std::max(globals_->lidar()->getConfig().min_z, 
              globals_->lidar()->getConfig().ground_min_z);
  }
  else {
    CHECK_NOTNULL(globals_->camera().get());
    max_range_ = globals_->camera()->getConfig().max_range;
    min_range_ = globals_->camera()->getConfig().min_range;
  }
  CHECK(inputIsValid(*input));

  // Allocate all blocks in corresponding submaps.
  Timer alloc_timer("tsdf_integration/allocate_blocks");
  
  allocateNewBlocks(submaps, *input);
  alloc_timer.Stop();
  
  if (config_.verbosity > 3)
    ROS_INFO("Debuging/Allocate all the new blocks");

  // Find all active blocks that are in the field of view.
  // Note(schmluk): This could potentially also be included in the parallel part
  // but is already almost instantaneous.
  Timer find_timer("tsdf_integration/find_blocks");
  //the block list for each visible submap
  std::unordered_map<int, voxblox::BlockIndexList> block_lists;  // <submap_id, blocklist> 
  // TODO(py): block_lists is actually the key
  if (config_.use_lidar) {
    block_lists = 
      globals_->lidar()->findVisibleBlocks(*submaps, input->T_M_C(),
                                            max_range_in_image_, true); // only active submap
  } else {                                    
    block_lists =
      globals_->camera()->findVisibleBlocks(*submaps, input->T_M_C(),
                                            max_range_in_image_, true);
  }

  if (config_.verbosity > 3) {
    ROS_INFO("Debuging/Found %d visible submaps", block_lists.size());
    ROS_INFO("# Visible submap: %d", block_lists.size());
    for (auto iter = block_lists.begin(); iter != block_lists.end(); ++iter) {
      ROS_INFO("submap_id: %d, # block: %d", iter->first, iter->second.size());
    }
  }

  std::vector<int> id_list;
  id_list.reserve(block_lists.size()); //count of the visible submap
  for (const auto& id_blocklist_pair : block_lists) {
    id_list.emplace_back(id_blocklist_pair.first);
  }
  find_timer.Stop();

  // Integrate in parallel. (for each submap)
  Timer int_timer("tsdf_integration/integration");
  SubmapIndexGetter index_getter(id_list);
  std::vector<std::future<void>> threads;
  for (int i = 0; i < config_.integration_threads; ++i) {
    threads.emplace_back(
        std::async(std::launch::async,
                   [this, &index_getter, &block_lists, submaps, input, i]() {
                     int index;
                     while (index_getter.getNextIndex(&index)) {
                       this->updateSubmap(submaps->getSubmapPtr(index),
                                          interpolators_[i].get(),
                                          block_lists.at(index), *input);

                     }
                   }));
  }

  int raycast_count = 0;
  int projective_count = 0;
  // Join all threads.
  for (auto& thread : threads) {
    thread.get();
  }
  int_timer.Stop();

  // if (config_.verbosity > 2)
  //   ROS_INFO("Adaptive Tsdf integration done: %d submap projective, %d submap raycasting",
  //            projective_count, raycast_count);

  if (config_.verbosity > 3)
    ROS_INFO("Integration done");
}

// for each submap -> determine to use raycasting or projective mapping
void AdaptiveIntegrator::updateSubmap(
    Submap* submap, InterpolatorBase* interpolator,
    const voxblox::BlockIndexList& block_indices,
    const InputData& input) const {
  
  Transformation T_S_C = submap->getT_M_S().inverse() * input.T_M_C();
  Transformation T_C_S = T_S_C.inverse();
  float voxel_size = submap->getConfig().voxel_size;
  float voxel_size_inv =  1.0 / voxel_size;
  float truncation_distance = submap->getConfig().truncation_distance; 
  float voxels_per_side = submap->getConfig().voxels_per_side;
  float voxels_per_side_inv = 1.0 / voxels_per_side;
  float block_size = voxel_size * voxels_per_side;

  Pointcloud submap_points;
  Colors submap_colors;
  Pointcloud submap_normals;
  
  const bool is_free_space_submap =
    submap->getLabel() == PanopticLabel::kFreeSpace;

  // Deal with the freespace submap 
  if (is_free_space_submap) {
    // directly use the input point cloud if it's available
    if (input.has(InputData::InputType::kPointCloud)) {
      submap_points = input.pointCloud();
    } else {
      submap_points = extractSubmapPointCloud(input.vertexMap(), 
                                              input.idImage(), -1);                                         
    } 
    // color is actually not used for free space submap 
    // But we still need it to have the same size as submap_points
    submap_colors.resize(submap_points.size()); 
  } else {
    // TODO(py): speed up
    // in sensor(camera)'s frame
    submap_points = extractSubmapPointCloud(input.vertexMap(), 
                                            input.idImage(), 
                                            submap->getID());
    
    submap_colors = extractSubmapColors(input.colorImage(), 
                                        input.idImage(), 
                                        submap->getID());
  }

  // useProjective()
  int point_count = submap_points.size();
  int voxel_count_ray_cast = point_count * std::ceil(2 * std::sqrt(3) * truncation_distance / voxel_size);
  int block_count = block_indices.size();
  int voxel_count_projective = block_count * voxels_per_side * voxels_per_side * voxels_per_side;
  
  bool use_projective_mapping = false;
  if (voxel_count_ray_cast > voxel_count_projective && !is_free_space_submap)
    use_projective_mapping = true;
  // For free space submap, we directly use raycasting integration
  
  if (use_projective_mapping) {
    if(config_.verbosity > 1)
      ROS_INFO("Use projective mapping for submap %d (%s)", submap->getID(), submap->getName().c_str());
    for (const auto& block_index : block_indices) {
      updateBlock(submap, interpolator, block_index, T_C_S, input);
    }
  } else { // use ray casting
    if(config_.verbosity > 1)
      ROS_INFO("Use ray casting for submap %d (%s)", submap->getID(), submap->getName().c_str());
    const bool normal_reliable = submap->getNormalReliability();
    bool normal_refine_on = false;
    if (config_.apply_normal_refine && 
      normal_reliable && 
      input.has(InputData::InputType::kNormalImage)) {
      int used_id = is_free_space_submap ? -1 : submap->getID();
      submap_normals = extractSubmapNormals(input.normalImage(), 
                                            input.idImage(), 
                                            used_id);
      normal_refine_on = true;      
    }

    // for each point (ray) in the submap
    // for free space submap, it seems to be bit wierd
    for (int i = 0; i < submap_points.size(); i++) {
      const Point& p_C = submap_points[i];
      const Color& color = submap_colors[i];
      Ray n_C;
      if (normal_refine_on) {
        n_C = submap_normals[i];
      }

      bool is_clearing; // TODO(py): all needs clearing
      if (!isPointValid(p_C, false, &is_clearing)) { // Need to be valid point to proceed
        continue;
      }

      const Point origin = T_S_C.getPosition(); // sensor's position in submap's frame
      const Point p_S = T_S_C * p_C;            // target point's position in submap's frame

      bool integrate_full_ray = is_free_space_submap ? 
                                config_.freespace_carving_enabled : 
                                config_.voxel_carving_enabled;

      // for Ray OG
      voxblox::RayCaster ray_caster(origin, p_S, is_clearing,
                                    integrate_full_ray,
                                    config_.max_ray_length_m, 
                                    voxel_size_inv,
                                    truncation_distance);
                

      TsdfBlock::Ptr block = nullptr;
      BlockIndex block_idx;
      GlobalIndex global_voxel_idx;
        
      // for all the voxels along the ray
      while (ray_caster.nextRayIndex(&global_voxel_idx)) {
        const BlockIndex block_idx =
                voxblox::getBlockIndexFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side_inv);

        TsdfBlock::Ptr block =
                submap->getTsdfLayerPtr()->allocateBlockPtrByIndex(block_idx);
        
        const VoxelIndex local_voxel_idx =
                voxblox::getLocalFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side);

        TsdfVoxel* voxel = &(block->getVoxelByVoxelIndex(local_voxel_idx));
        
        if (updateVoxel(voxel, origin, p_C, p_S, n_C, global_voxel_idx, color, 
                        is_free_space_submap, normal_refine_on, T_C_S, 0.0, // no pre-computed weight
                        truncation_distance, voxel_size)) {
          block->setUpdatedAll(); 
        }
      }
    }
  }
  // free the memory
  Pointcloud().swap(submap_points);
  Colors().swap(submap_colors);
  Pointcloud().swap(submap_normals);
}


// if processInput is called in ClassProjectiveIntegrator, then this
// function would not be called, the updateBlock function in ClassProjectiveIntegrator
// would be called instead
void AdaptiveIntegrator::updateBlock(Submap* submap,
                                     InterpolatorBase* interpolator,
                                     const voxblox::BlockIndex& block_index,
                                     const Transformation& T_C_S,
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

  bool normal_reliable = submap->getNormalReliability();
  if (is_free_space_submap && !config_.apply_normal_refine_freespace) {
    normal_reliable = false;
  }
  bool apply_normal_refine = false;
  if (config_.apply_normal_refine && 
      normal_reliable) {
    apply_normal_refine = true;
  }

  // Update all voxels.
  for (size_t i = 0; i < block.num_voxels(); ++i) {
    TsdfVoxel& voxel = block.getVoxelByLinearIndex(i);
    const Point p_C = T_C_S * block.computeCoordinatesFromLinearIndex(
                                  i);  // Voxel center in camera frame.
    if (updateVoxel(interpolator, &voxel, p_C, input, submap_id,
                    is_free_space_submap, apply_normal_refine, T_C_S, 
                    truncation_distance, voxel_size)) {
      was_updated = true;
    }
  }
  if (was_updated) {
    block.setUpdatedAll();
  }
}

// projective mapping for each voxel
// TODO(py): also add the normal refine option
bool AdaptiveIntegrator::updateVoxel(
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

  // Check whether this ray is projected to another submap
  // if so, we discard this voxel
  // TODO(py): not very efficient, consider to put it before signed distance computation
  const bool point_belongs_to_this_submap =
      interpolator->interpolateID(input.idImage()) == submap_id;
  if (!(point_belongs_to_this_submap || config_.foreign_rays_clear ||
        is_free_space_submap)) { 
    //these three statements should all be false to finally return false
    return false;
  }
  
  cv::Vec3f normal;
  Ray n_C, v_C;
  bool update_gradient = false;

  // normal refine
  if (apply_normal_refine && input.has(InputData::InputType::kNormalImage)) {
    // current normal vector
    normal = input.normalImage().at<cv::Vec3f>(v, u);
    n_C = Ray(normal[0], normal[1], normal[2]); //in sensor(camera)'s frame

    float normal_ratio = 1.0f;
    if (voxel->gradient.norm() > kFloatEpsilon) { //use current un-updated normal because the weight is unknown
      v_C = T_C_S.getRotationMatrix() * voxel->gradient; //back to sensor(camera)'s frame
      normal_ratio = std::abs(v_C.dot(p_C)/p_C.norm());
    } else { //gradient not ready yet, use the first (current) normal vector
      if (n_C.norm() > kFloatEpsilon)  //current normal is valid
        normal_ratio = std::abs(n_C.dot(p_C)/p_C.norm());
    }
    sdf *= normal_ratio; //get the non-projective sdf
    update_gradient = true;
  }

  if (sdf < -truncation_distance) {
    return false;
  }

  // Compute the weight of this measurement.
  const float weight = computeVoxelWeight(p_C, voxel_size, truncation_distance, sdf, true);

  // Update voxel gradient
  if (update_gradient) {
    Ray n_S = T_C_S.getRotationMatrix().inverse() * n_C;  //in submap's frame
    if (n_C.norm() > kFloatEpsilon) 
      updateVoxelGradient(voxel, n_S, weight); //direction issue
  }

  
  if (point_belongs_to_this_submap || is_free_space_submap) {
    // Truncate the sdf to the truncation band.
    sdf = std::min(sdf, truncation_distance);
    // Only merge color and classification data near the surface.
    if (!point_belongs_to_this_submap || sdf >= truncation_distance 
      || is_free_space_submap) {
      // far away (truncated part) or not used to be meshed (free submap)
      // , do not interpolate 
      updateVoxelValues(voxel, sdf, weight);
    } else {
      const Color color = interpolator->interpolateColor(input.colorImage());
      updateVoxelValues(voxel, sdf, weight, &color);
    }
  } else { //not free space and not belong to the current submap
    // Voxels that don't belong to the submap are 'cleared' if they are in
    // front of the surface. If the foreign_rays_clear flag is set to false the
    // update step already returned before here.
    if (sdf > 0) { //in front
      updateVoxelValues(voxel, truncation_distance, weight);
    }
  }
  return true;
}

// Used for raycasting
// Updates tsdf_voxel. Thread safe.
bool AdaptiveIntegrator::updateVoxel(TsdfVoxel* voxel, const Point& origin, const Point& p_C, 
                                    const Point& p_S, const Ray& n_C, 
                                    const GlobalIndex& global_voxel_idx,
                                    const Color& color, const bool is_free_space_submap,
                                    const bool apply_normal_refine, const Transformation& T_C_S,
                                    const float init_weight,
                                    const float truncation_distance, const float voxel_size, 
                                    ClassVoxel* class_voxel) const {
  DCHECK(voxel != nullptr);

  const Point voxel_center =
          voxblox::getCenterPointFromGridIndex(global_voxel_idx, voxel_size);

  float sdf = computeSignedDistance(origin, p_S, voxel_center);

  // Lookup the mutex that is responsible for this voxel and lock it
  // std::lock_guard<std::mutex> lock(mutexes_.get(global_voxel_idx));

  Ray v_C;

  if (apply_normal_refine) {
    float normal_ratio = 1.0f;
    if (voxel->gradient.norm() > kFloatEpsilon) { //use current un-updated normal because the weight is unknown
      v_C = T_C_S.getRotationMatrix() * voxel->gradient; //back to sensor(camera)'s frame
      normal_ratio = std::abs(v_C.dot(p_C)/p_C.norm());
    } else { //gradient not ready yet, use the first (current) normal vector
      //NOTE(py): kFloatEpsilon is a safe value in case of numerical rounding error
      if (n_C.norm() > kFloatEpsilon)  //current normal is valid
        normal_ratio = std::abs(n_C.dot(p_C)/p_C.norm());
    }
    sdf *= normal_ratio; //get the non-projective sdf
  }

  if (sdf < -truncation_distance) {
    return false;
  }

  float weight = computeVoxelWeight(p_C, voxel_size, truncation_distance, sdf, false);

  if (apply_normal_refine) {
    Ray n_S = T_C_S.inverse().getRotationMatrix() * n_C;  //in submap's frame
    if (n_C.norm() > kFloatEpsilon) 
      updateVoxelGradient(voxel, n_S, weight); 
  }
  
  // Truncate the sdf to the truncation band.
  sdf = std::min(sdf, truncation_distance);

  // Only merge color and classification data near the surface.
  if (sdf >= truncation_distance || is_free_space_submap) {
    // far away, do not interpolate
    updateVoxelValues(voxel, sdf, weight);
  } else {
    updateVoxelValues(voxel, sdf, weight, &color);
  } 
  return true;   
}

// Used for projective mapping
bool AdaptiveIntegrator::computeSignedDistance(const Point& p_C,
                                               InterpolatorBase* interpolator,
                                               float* sdf, float* u, float* v) const {
  // Skip voxels that are too far or too close.
  
  const float distance_to_voxel = p_C.norm();

  // if (distance_to_voxel < min_range_ ||
  //     distance_to_voxel > max_range_) 
  //   return false;

  // Project the current voxel into the range image, only count points that fall
  // fully into the image.
  if (config_.use_lidar) {
    if (!globals_->lidar()->projectPointToImagePlane(p_C, u, v))
      return false;
  } else { //camera
    if (p_C.z() < 0.0) 
       return false;
  
    if (!globals_->camera()->projectPointToImagePlane(p_C, u, v))
      return false;
  }

  // Set up the interpolator and compute the signed distance.
  interpolator->computeWeights(*u, *v, range_image_); // why not directly use the depth_image from input data

  const float distance_to_surface =
      interpolator->interpolateRange(range_image_);

  *sdf = distance_to_surface - distance_to_voxel; //still the projective distance
  
  return true;
  // projective mapping may not be very efficient here
}

// Used for raycasting integrator
// Figure out whether the voxel is behind or in front of the surface.
// To do this, project the voxel_center onto the ray from origin to point G.
// Then check if the the magnitude of the vector is smaller or greater than
// the original distance...
float AdaptiveIntegrator::computeSignedDistance(const Point& origin,
                                                const Point& point_G,
                                                const Point& voxel_center) const {
  const Point v_voxel_origin = voxel_center - origin;
  const Point v_point_origin = point_G - origin;

  const FloatingPoint dist_G = v_point_origin.norm();
  // projection of a (v_voxel_origin) onto b (v_point_origin)
  const FloatingPoint dist_G_V = v_voxel_origin.dot(v_point_origin) / dist_G;

  const float sdf = static_cast<float>(dist_G - dist_G_V);
  return sdf;
}

// Equ. 1 in the paper
// NOTE(py): very important part, try to improve the current scheme
// Compute the weight for current measurement
// if projective = true, this func would be used for projective mapping 
float AdaptiveIntegrator::computeVoxelWeight(const Point& p_C,
                                             const float voxel_size,
                                             const float truncation_distance,
                                             const float sdf,
                                             const bool projective, 
                                             const bool with_init_weight,
                                             const float init_weight) const {
  
  // Part 1. This approximates the number of rays that would hit this voxel. 
  // Independent of sdf
  // TODO(py): figure out why
  
  float weight = 1.0;
  if (with_init_weight) {
    weight = init_weight;
  } else {
    if (config_.use_lidar) {
      
      float dist = p_C.norm();
      if (projective) {
        weight = std::pow(voxel_size / dist, 2.f) /
                globals_->lidar()->getConfig().res_h_rad /
                globals_->lidar()->getConfig().res_v_rad;
      }
      // Part 2. Weight reduction with distance squared (according to sensor noise models).
      // Also Independent of sdf
      if (config_.weight_reduction) {
        weight /= std::pow(dist, config_.weight_reduction_exp); //ADD(py): weight_reduction_exp, dist^n
      }
    } else {
      if (projective) {
        weight = globals_->camera()->getConfig().fx * 
                globals_->camera()->getConfig().fy *
                std::pow(voxel_size / p_C.z(), 2.f);
      }
      // Weight reduction with distance squared (according to sensor noise models).
      if (config_.weight_reduction) {
        weight /= std::pow(p_C.z(), config_.weight_reduction_exp);
      }
    }
  }

  // Part 3. weight drop-off 
  // NOTE(py): should we use the original projective sdf?
  // NOTE(py): for weight drop-off, sdf should not be too large
  // TODO(py): figure out why there will be the meshing error when enabling this func???
  // Apply weight drop-off if appropriate. //check voxblox's paper
  if (config_.use_weight_dropoff) {
    const float dropoff_epsilon =
        config_.weight_dropoff_epsilon > 0.f
            ? config_.weight_dropoff_epsilon
            : config_.weight_dropoff_epsilon * -voxel_size; 
    //for example, weight_dropoff_epsilon = -1.0 --> dropoff_epsilon = voxel_size
    if (sdf < -dropoff_epsilon) {
      weight *=
          (truncation_distance + sdf) / (truncation_distance - dropoff_epsilon);
      weight = std::max(weight, 0.f); //should >= 0
    }
  }
  return weight;
}

//each voxel has a distance and a weight
//once a new distance and weight is calculated, update it as 
//a kind of weighted average
void AdaptiveIntegrator::updateVoxelValues(TsdfVoxel* voxel, const float sdf,
                                           const float weight,
                                           const Color* color) const {
  
  float new_weight = voxel->weight + weight;
  // it is possible to have weights very close to zero, due to the limited
  // precision of floating points dividing by this small value can cause nans
  if (new_weight < kFloatEpsilon) {
    return;
  }
  
  // Weighted averaging fusion.
  voxel->distance = (voxel->distance * voxel->weight + sdf * weight) /
                    new_weight;
  voxel->weight = std::min(new_weight, config_.max_weight);
  // also take average of the color
  if (color != nullptr) {
    voxel->color =
        Color::blendTwoColors(voxel->color, voxel->weight, *color, weight);
  }
}  

// the input normal should be valid
void AdaptiveIntegrator::updateVoxelGradient(TsdfVoxel* voxel, const Ray normal,
                                              const float weight)
                                              const {
  float new_weight = voxel->weight + weight;
  // it is possible to have weights very close to zero, due to the limited
  // precision of floating points dividing by this small value can cause nans
  if (new_weight < kFloatEpsilon) {
    return;
  }

  if(voxel->gradient.norm() > kFloatEpsilon){ 
    voxel->gradient = ((voxel->gradient * voxel->weight + normal * weight) /
                      new_weight).normalized();
  } else { // 0, newly assigned
    voxel->gradient = normal;
  }
}

void AdaptiveIntegrator::allocateNewBlocks(SubmapCollection* submaps,
                                             const InputData& input) {
  
  Timer alloc_nonfree_timer("tsdf_integration/allocate_blocks/non_free");

  // This method also resets the depth image. (seems to be not very neccessary)
  range_image_.setZero();
  max_range_in_image_ = 0.f;

  // TODO(py): speed up
  // For non-free space submaps
  // Parse through each point to allocate instance + background blocks.
  std::unordered_set<Submap*> touched_submaps; //submaps affected by the current frame
  for (int v = 0; v < input.depthImage().rows; v++) {
    for (int u = 0; u < input.depthImage().cols; u++) {
      const cv::Vec3f& vertex = input.vertexMap().at<cv::Vec3f>(v, u);
      const Point p_C(vertex[0], vertex[1], vertex[2]);

      //NOTE(py): Why not directly use depth from depth image
      const float ray_distance = p_C.norm(); 
      range_image_(v, u) = ray_distance;
      if (ray_distance > max_range_ ||
          ray_distance < min_range_) {
        continue;
      }
      max_range_in_image_ = std::max(max_range_in_image_, ray_distance);
      const int id = input.idImage().at<int>(v, u); //now it should be already the submap id
      
      //seems to be not very efficient
      if (submaps->submapIdExists(id)) {
        //ROS_INFO("%d SubmapID Exists", id); 

        Submap* submap = submaps->getSubmapPtr(id);
        const int voxels_per_side = submap->getConfig().voxels_per_side;
        const Point p_S = submap->getT_S_M() * input.T_M_C() * p_C;

        const voxblox::BlockIndex block_index =
            submap->getTsdfLayer().computeBlockIndexFromCoordinates(p_S);
        const auto block =
            submap->getTsdfLayerPtr()->allocateBlockPtrByIndex(block_index);
        if (submap->hasClassLayer()) {
          // NOTE(schmluk): The projective integrator does not use the class
          // layer but was added here for simplicity.
          submap->getClassLayerPtr()->allocateBlockPtrByIndex(block_index);
        }

        // If required, check whether the point is on the boundary of a block
        // and allocate the neighboring blocks.
        if (config_.allocate_neighboring_blocks) {
          for (float sign : {-1.f, 1.f}) {
            const Point p_neighbor_S =
                submap->getT_S_M() * input.T_M_C() *
                (p_C * (1.f + sign * submap->getConfig().voxel_size / p_C.norm())); // boundary buffer +-1 voxel_size
            const voxblox::BlockIndex neighbor_index =
                submap->getTsdfLayer().computeBlockIndexFromCoordinates(
                    p_neighbor_S);
            if (neighbor_index != block_index) {
              const auto block =
                  submap->getTsdfLayerPtr()->allocateBlockPtrByIndex(
                      neighbor_index);
              if (submap->hasClassLayer()) {
                submap->getClassLayerPtr()->allocateBlockPtrByIndex(
                    neighbor_index);
              }
            }
          }
        }
        touched_submaps.insert(submap);
      }
    }
  }
  max_range_in_image_ = std::min(max_range_in_image_, max_range_);
  alloc_nonfree_timer.Stop();
  // ROS_INFO("Debuging/Allocate blocks of non-free submaps");

  // Update all bounding volumes of non freespace submaps. 
  // This is currently done in every integration
  // step since it's not too expensive and won't do anything if no new block
  // was allocated.
  for (auto& submap : touched_submaps) {
    submap->updateBoundingVolume();
    if (config_.verbosity >= 3) {
      Point center = submap->getBoundingVolumePtr()->getCenter();
      ROS_INFO("Submap [%d] center: (%f,%f,%f), radius: %f, voxel_size: %f",
                submap->getID(),
                center(0), center(1), center(2),
                submap->getBoundingVolumePtr()->getRadius(),
                submap->getTsdfLayerPtr()->voxel_size());
    }
  }

  // TODO(py): very important here
  // Allocate block for free space submap here (we want to integrate freespace TSDF by projective mapping)
  Timer alloc_free_timer("tsdf_integration/allocate_blocks/freespace");

  // Not from the range image
  // Allocate all potential free space blocks. (free space submap's id is 0)
  // TODO(py): It's kind of stupid to allocate the whole space, here the unit is block instead of voxel
  if (submaps->submapIdExists(submaps->getActiveFreeSpaceSubmapID())) {
    Submap* space =
        submaps->getSubmapPtr(submaps->getActiveFreeSpaceSubmapID());
    const float block_size = space->getTsdfLayer().block_size();
    const float block_diag_half = std::sqrt(3.f) * block_size / 2.f;
    const Transformation T_C_S = input.T_M_C().inverse() * space->getT_M_S();
    const Point sensor_S = T_C_S.inverse().getPosition();  // T_S_C //sensor's position
    const int max_steps = std::floor((max_range_in_image_ + block_diag_half) /
                                     space->getTsdfLayer().block_size());
    int max_positive_z_steps = max_steps;
    int max_negative_z_steps = -max_steps;
    
    if (config_.use_lidar){
      max_positive_z_steps = std::floor(max_z_ /
                                     space->getTsdfLayer().block_size());
      max_negative_z_steps = std::floor(min_z_ /
                                     space->getTsdfLayer().block_size());
    }
    
    // TODO(py): allocate less blocks here
    // figure out which block within a range sphere is visible
    // from the sensor's current postion
    for (int x = -max_steps; x <= max_steps; ++x) {
      for (int y = -max_steps; y <= max_steps; ++y) {
        for (int z = max_negative_z_steps; z <= max_positive_z_steps; ++z) {
          const Point offset(x, y, z);
          const Point candidate_S = sensor_S + offset * block_size;

          if (space->getTsdfLayerPtr()->getBlockPtrByCoordinates(candidate_S)) //already allocated
            continue;

          if (config_.use_lidar) {
            if (globals_->lidar()->pointIsInViewFrustum(T_C_S * candidate_S,
                                                        0.0)) {
              space->getTsdfLayerPtr()->allocateBlockPtrByCoordinates(
                candidate_S);
            }
          } else {
            // for free space voxels, we can further check if the height is lower than the ground
            // if so, we can skip the voxel
            if (globals_->camera()->pointIsInViewFrustum(T_C_S * candidate_S,
                                                         block_diag_half)) {
              space->getTsdfLayerPtr()->allocateBlockPtrByCoordinates(
                candidate_S);
            }
          }
        }
      }
    }
    space->getBoundingVolumePtr()->update(); 
  }
  alloc_free_timer.Stop();

}

Pointcloud AdaptiveIntegrator::extractSubmapPointCloud(const cv::Mat& vertex_map,
                                                      const cv::Mat& id_image, int id) const {
  Pointcloud submap_points;
  for (int v = 0; v < id_image.rows; v++) {
    for (int u = 0; u < id_image.cols; u++) {
      // id == -1 means we want to use the whole point cloud 
      if (id_image.at<int>(v, u) == id || id == -1) { 
        cv::Vec3f vertex = vertex_map.at<cv::Vec3f>(v, u);
        Point p_C(vertex[0], vertex[1], vertex[2]);
        submap_points.push_back(p_C);
      }
    }
  }
  return submap_points;
}

Colors AdaptiveIntegrator::extractSubmapColors(const cv::Mat& color_image,
                                              const cv::Mat& id_image, int id) const {
  Colors submap_colors;
  for (int v = 0; v < id_image.rows; v++) {
    for (int u = 0; u < id_image.cols; u++) {
      // id == -1 means we want to use the whole point cloud 
      if (id_image.at<int>(v, u) == id || id == -1) {
        cv::Vec3b color = color_image.at<cv::Vec3b>(v, u); // BGR
        Color c_C(color[2], color[1], color[0]); // RGB
        submap_colors.push_back(c_C);
      }
    }
  }
  return submap_colors;
}

Pointcloud AdaptiveIntegrator::extractSubmapNormals(const cv::Mat& normal_image,
                                                  const cv::Mat& id_image, int id) const {
  Pointcloud submap_normals;
  for (int v = 0; v < id_image.rows; v++) {
    for (int u = 0; u < id_image.cols; u++) {
      // id == -1 means we want to use the whole point cloud 
      if (id_image.at<int>(v, u) == id || id == -1) {
        cv::Vec3f vertex = normal_image.at<cv::Vec3f>(v, u);
        Point n_C(vertex[0], vertex[1], vertex[2]);
        submap_normals.push_back(n_C);
      }
    }
  }
  return submap_normals;
}


}  // namespace panoptic_mapping
