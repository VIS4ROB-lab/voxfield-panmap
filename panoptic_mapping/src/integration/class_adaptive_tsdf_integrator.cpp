#include "panoptic_mapping/integration/class_adaptive_tsdf_integrator.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include <voxblox/integrator/merge_integration.h>

#include "panoptic_mapping/common/index_getter.h"

// TODO(py): try other faster raycasting methods (for the freespace submap),
// such as 'merged' and 'fast' in voxblox

namespace panoptic_mapping {

config_utilities::Factory::RegistrationRos<
    TsdfIntegratorBase, ClassAdaptiveIntegrator, std::shared_ptr<Globals>>
    ClassAdaptiveIntegrator::registration_("class_adaptive");

void ClassAdaptiveIntegrator::Config::checkParams() const {
  checkParamConfig(ai_config);
}

void ClassAdaptiveIntegrator::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("use_binary_classification", &use_binary_classification);
  setupParam("use_instance_classification", &use_instance_classification);
  setupParam("update_only_tracked_submaps", &update_only_tracked_submaps);
  setupParam("adaptive_integrator_config", &ai_config);
}

ClassAdaptiveIntegrator::ClassAdaptiveIntegrator(
    const Config& config, std::shared_ptr<Globals> globals)
    : config_(config.checkValid()),
      AdaptiveIntegrator(config.ai_config, std::move(globals), false) {
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();

  // Store class count.
  if (!config_.use_binary_classification &&
      !config_.use_instance_classification) {
    // The +1 is added because 0 is reserved for the belonging submap.
    num_classes_ = globals_->labelHandler()->numberOfLabels() + 1;
  }
}

void ClassAdaptiveIntegrator::processInput(SubmapCollection* submaps,
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
  AdaptiveIntegrator::processInput(submaps, input);
}

void ClassAdaptiveIntegrator::updateSubmap(
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

  Pointcloud submap_points;
  Colors submap_colors;
  Pointcloud submap_normals;
  
  const bool is_free_space_submap =
    submap->getLabel() == PanopticLabel::kFreeSpace;

  if (is_free_space_submap) {

    if (config_.ai_config.skip_free_space_submap)
      return;

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
  
  // Determine to use which integration method
  int point_count = submap_points.size();
  int voxel_count_ray_cast = point_count * std::ceil(2 * std::sqrt(3) * truncation_distance / voxel_size);
  int block_count = block_indices.size();
  int voxel_count_projective = block_count * voxels_per_side * voxels_per_side * voxels_per_side;
  
  bool use_projective_mapping = false;
  // if (voxel_count_ray_cast > voxel_count_projective || is_free_space_submap)
  if (voxel_count_ray_cast > voxel_count_projective)
    use_projective_mapping = true;
  // For free space submap, we directly use projective mapping
    
  if (use_projective_mapping) {
    if(config_.verbosity > 1)
      ROS_INFO("Use projective mapping for submap %d (%s)", submap->getID(), submap->getName().c_str());
    for (const auto& block_index : block_indices) {
      updateBlock(submap, interpolator, block_index, T_C_S, input);
    }
  } else { // Raycasting integrator
    if(config_.verbosity > 1)
      ROS_INFO("Use ray casting for submap %d (%s)", submap->getID(), submap->getName().c_str());
    const bool normal_reliable = submap->getNormalReliability(); // for free-space submap, default true
    bool normal_refine_on = false;
    bool apply_normal_refine = is_free_space_submap ? 
                               config_.ai_config.apply_normal_refine_freespace :
                               config_.ai_config.apply_normal_refine;
    if (apply_normal_refine && // apply_normal_refine now works for both the freespace and the non-freespace
        normal_reliable && 
        input.has(InputData::InputType::kNormalImage)) {
      
      int used_id = is_free_space_submap ? -1 : submap->getID();
      submap_normals = extractSubmapNormals(input.normalImage(), 
                                            input.idImage(), 
                                            used_id);
      normal_refine_on = true;      
    }

    // integrate pointcloud
    bool submap_valid = true;
    if (config_.ai_config.only_merge_freespace)
      submap_valid = is_free_space_submap;

    if(config_.ai_config.merged_raycasting && submap_valid) {
      integratePointCloudMerged(submap, submap_points, submap_colors, submap_normals,
                                T_S_C, normal_refine_on, is_free_space_submap,
                                voxel_size, voxels_per_side, truncation_distance);
    } else {
      integratePointCloud(submap, submap_points, submap_colors, submap_normals,
                          T_S_C, normal_refine_on, is_free_space_submap,
                          voxel_size, voxels_per_side, truncation_distance);
    }                   
  }
  // free the memory
  Pointcloud().swap(submap_points);
  Colors().swap(submap_colors);
  Pointcloud().swap(submap_normals);
}

//This function is called when ClassProjectiveIntegrator is selected
void ClassAdaptiveIntegrator::updateBlock(
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
  
  bool normal_reliable = submap->getNormalReliability();
  if (is_free_space_submap && 
      !config_.ai_config.apply_normal_refine_freespace) { // apply normal refine for freespace submap or not
    normal_reliable = false;
  }

  bool apply_normal_refine = false;
  if (config_.ai_config.apply_normal_refine && 
      normal_reliable) {
    apply_normal_refine = true;
  }
    
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
    // For LiDAR, some voxels might not be covered since they would lay out of the vertical fov
    // It's better to use bundled raycasting for LiDAR
    if (updateVoxel(interpolator, &voxel, p_C, input, submap_id,
                    is_free_space_submap, apply_normal_refine, T_C_S,
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
bool ClassAdaptiveIntegrator::updateVoxel(
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
  if (config_.ai_config.use_lidar)
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
  if (!(point_belongs_to_this_submap || config_.ai_config.foreign_rays_clear ||
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

  // normal refine
  if (apply_normal_refine && input.has(InputData::InputType::kNormalImage)) {
    //   && !is_free_space_submap) { // now we also apply normal refine on free space submap
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
    sdf *= normal_ratio; //get the non-projective sdf
    update_gradient = true;
  }

  if (sdf < -truncation_distance) {
    return false;
  }

  // Compute the weight of this measurement.
  const float weight = computeVoxelWeight(p_C, voxel_size, 
                                          truncation_distance, sdf, true); // projective mapping

  // Update voxel gradient
  if (update_gradient){
    Ray n_S = T_C_S.getRotationMatrix().inverse() * n_C;  //in submap's frame
    if (n_C.norm() > kFloatEpsilon) 
      updateVoxelGradient(voxel, n_S, weight); //direction issue
  }

  // Truncate the sdf to the truncation band.
  sdf = std::min(sdf, truncation_distance);

  // Only merge color and classification data near the surface.
  if (sdf >= truncation_distance || is_free_space_submap) {
    // far away (truncated part) or not used to be meshed (free submap)
    // , do not interpolate 
    updateVoxelValues(voxel, sdf, weight);
  } 
  else {
    const Color color = interpolator->interpolateColor(input.colorImage());
    updateVoxelValues(voxel, sdf, weight, &color);

    // Update the class voxel. (the category)
    if (class_voxel) {
      updateClassVoxel(interpolator, class_voxel, input, submap_id);
    }
  }
  return true;
}

// Updates tsdf_voxel. Thread safe.
bool ClassAdaptiveIntegrator::updateVoxel(TsdfVoxel* voxel, const Point& origin, const Point& p_C, 
                                         const Point& p_S, const Ray& n_C, 
                                         const GlobalIndex& global_voxel_idx,
                                         const Color& color, const bool is_free_space_submap,
                                         const bool apply_normal_refine, const Transformation& T_C_S,
                                         const float init_weight,  const float truncation_distance, 
                                         const float voxel_size, ClassVoxel* class_voxel) const {
                                         
  DCHECK(voxel != nullptr);

  const Point voxel_center =
          voxblox::getCenterPointFromGridIndex(global_voxel_idx, voxel_size);

  float sdf = computeSignedDistance(origin, p_S, voxel_center);

  // Lookup the mutex that is responsible for this voxel and lock it
  // std::lock_guard<std::mutex> lock(mutexes_.get(global_voxel_idx));

  Ray v_C;

  if (apply_normal_refine) {
    float normal_ratio = 1.0f;
    if (voxel->gradient.norm() > kFloatEpsilon){ //use current un-updated normal because the weight is unknown
      v_C = T_C_S.getRotationMatrix() * voxel->gradient; //back to sensor(camera)'s frame
      normal_ratio = std::abs(v_C.dot(p_C)/p_C.norm());
    } else { //gradient not ready yet, use the first (current) normal vector
      //NOTE(py): kFloatEpsilon is a safe value in case of numerical rounding error
      if (n_C.norm() > kFloatEpsilon) { //current normal is valid
        normal_ratio = std::abs(n_C.dot(p_C)/p_C.norm());
      }
    }
    sdf *= normal_ratio; //get the non-projective sdf
  }

  if (sdf < -truncation_distance) {
    return false;
  }

  bool with_init_weight = false;
  if (init_weight > 0)
    with_init_weight = true;
  float weight = computeVoxelWeight(p_C, voxel_size, truncation_distance, sdf, 
                                    false, with_init_weight, init_weight);
  
  // Truncate the sdf to the truncation band.
  sdf = std::min(sdf, truncation_distance);

  // Only merge color and classification data near the surface.
  if (sdf >= truncation_distance) {
    // far away, do not interpolate color
    updateVoxelValues(voxel, sdf, weight);
  } else { // close to the surface
    if (apply_normal_refine) {
      Ray n_S = T_C_S.inverse().getRotationMatrix() * n_C;  //in submap's frame
      if (n_C.norm() > kFloatEpsilon) 
        updateVoxelGradient(voxel, n_S, weight); 
    }
    if (is_free_space_submap) {
      updateVoxelValues(voxel, sdf, weight);
    } else {
      updateVoxelValues(voxel, sdf, weight, &color);
    }
    // Update the class voxel. (the category)
    if (class_voxel) {
      if(config_.use_binary_classification) {
        class_voxel->incrementCount(0); // TODO(py): the same, check it later
      } else {
        class_voxel->incrementCount(0);
      }
    }
  } 
  return true;   
}

void ClassAdaptiveIntegrator::updateClassVoxel(InterpolatorBase* interpolator,
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

// Simple raycasting integration
void ClassAdaptiveIntegrator::integratePointCloud(Submap* submap, 
                                                  Pointcloud &submap_points,
                                                  Colors &submap_colors,
                                                  Pointcloud &submap_normals,
                                                  Transformation &T_S_C,
                                                  const bool normal_refine_on,
                                                  const bool is_free_space_submap,
                                                  const float voxel_size,
                                                  const int voxels_per_side,
                                                  const float truncation_distance) const {
    float voxel_size_inv = 1.0 / voxel_size;
    float voxels_per_side_inv = 1.0 / voxels_per_side;
    Transformation T_C_S = T_S_C.inverse();
    const Point origin = T_S_C.getPosition(); //sensor's position in submap's frame
    bool integrate_full_ray = is_free_space_submap ? 
                                config_.ai_config.freespace_carving_enabled : 
                                config_.ai_config.voxel_carving_enabled;

    // for each point (ray) in the submap
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

      // no pre-compute weight
      integrateRay(submap, p_C, color, n_C, 0.0, T_S_C,
                    normal_refine_on, is_free_space_submap,
                    is_clearing, integrate_full_ray,
                    voxel_size, voxels_per_side, truncation_distance);
    }
}

// "merged" (bundled) raycasting 
// reference: voxblox
void ClassAdaptiveIntegrator::integratePointCloudMerged(Submap* submap, 
                                                        Pointcloud &submap_points,
                                                        Colors &submap_colors,
                                                        Pointcloud &submap_normals,
                                                        Transformation &T_S_C,
                                                        const bool normal_refine_on,
                                                        const bool is_free_space_submap,
                                                        const float voxel_size,
                                                        const int voxels_per_side,
                                                        const float truncation_distance) const {
  // timing::Timer integrate_timer("integrate/merged");
  CHECK_EQ(submap_points.size(), submap_colors.size());

  // Pre-compute a list of unique voxels to end on.
  // Create a hashmap: VOXEL INDEX -> index in original cloud.
  voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type voxel_map;
  // This is a hash map (same as above) to all the indices that need to be
  // cleared.
  voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type clear_map;

  bool integrate_full_ray = is_free_space_submap ? 
                            config_.ai_config.freespace_carving_enabled : 
                            config_.ai_config.voxel_carving_enabled;


  // bundle rays in each voxel with point inside (isPointValid is judged in it)
  bundleRays(T_S_C, submap_points, 1.0 / voxel_size, &voxel_map, &clear_map);

  voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type::const_iterator it;
  size_t map_size;

  it = voxel_map.begin();
  map_size = voxel_map.size();
  for (size_t i = 0; i < map_size; ++i) { // for each voxel, merge the points inside
    integrateVoxelMerged(submap, submap_points, submap_colors, submap_normals, T_S_C, 
                        *it, false, normal_refine_on, is_free_space_submap, integrate_full_ray,
                        voxel_size, voxels_per_side, truncation_distance);
    ++it;
  }
  
  // The key is here
  // timing::Timer clear_timer("integrate/clear");
  
  // but actually clear is not used
  it = clear_map.begin();
  map_size = clear_map.size(); 
  for (size_t i = 0; i < map_size; ++i) { 
    integrateVoxelMerged(submap, submap_points, submap_colors, submap_normals, T_S_C, 
                        *it, true, normal_refine_on, is_free_space_submap, integrate_full_ray,
                        voxel_size, voxels_per_side, truncation_distance);
    ++it;
  }
  // clear_timer.Stop();
  // integrate_timer.Stop();
}

void ClassAdaptiveIntegrator::bundleRays(const Transformation& T_S_C, 
                                         const Pointcloud& submap_points,
                                         const float voxel_size_inv,
                                         voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type* voxel_map,
                                         voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type* clear_map) const {
    
  DCHECK(voxel_map != nullptr);
  DCHECK(clear_map != nullptr);

  // size_t point_idx;
  for (int i = 0; i < submap_points.size(); i++) {
    const Point& point_C = submap_points[i];
    bool is_clearing;
    if (!isPointValid(point_C, false, &is_clearing)) {
      continue;
    }

    const Point point_S = T_S_C * point_C;

    GlobalIndex voxel_index =
        voxblox::getGridIndexFromPoint<GlobalIndex>(point_S, voxel_size_inv);
    // the voxel contain which points 
    if (is_clearing) {
      (*clear_map)[voxel_index].push_back(i);
    } else {
      (*voxel_map)[voxel_index].push_back(i);
    }
  }
  // VLOG(3) << "Went from " << submap_points.size() << " points to "
  //         << voxel_map->size() << " raycasts  and " << clear_map->size()
  //         << " clear rays.";
}

void ClassAdaptiveIntegrator::integrateVoxelMerged(Submap *submap,
                                                   const Pointcloud &submap_points,
                                                   const Colors &submap_colors,
                                                   const Pointcloud &submap_normals,
                                                   const Transformation& T_S_C,
                                                   const std::pair<GlobalIndex, AlignedVector<size_t>>& kv, 
                                                   const bool clearing_ray, const bool normal_refine_on,
                                                   const bool is_free_space_submap, 
                                                   const bool integrate_full_ray,
                                                   const float voxel_size, const int voxels_per_side,                                              
                                                   const float truncation_distance) const {

  if (kv.second.empty()) {
    return;
  }

  // const Point& origin = T_S_C.getPosition();
  Color merged_color;
  Point merged_point_C = Point::Zero();
  Ray merged_normal_C = Point::Zero();
  FloatingPoint merged_weight = 0.0;

  // merged the point, color and normal in the voxel
  for (const size_t pt_idx : kv.second) { // for each point in the voxel
    
    const Point& point_C = submap_points[pt_idx];
    const Color& color = submap_colors[pt_idx];
    Ray normal_C;
    if (normal_refine_on) {
      normal_C = submap_normals[pt_idx];
    }
    const float point_weight = getVoxelWeight(point_C);
    if (point_weight < kEpsilon) {
      continue;
    }
    merged_point_C = (merged_point_C * merged_weight + point_C * point_weight) /
                     (merged_weight + point_weight);
    merged_color =
        Color::blendTwoColors(merged_color, merged_weight, color, point_weight);
    
    if (normal_refine_on) { // seems to have no problem
      merged_normal_C = merged_normal_C * merged_weight + normal_C * point_weight;
      if (merged_normal_C.norm() > kEpsilon) {
        merged_normal_C.normalize();
      }
    }             
    merged_weight += point_weight;

    // only take first point when clearing
    if (clearing_ray) {
      break;
    }
  }
  
  // also consider the merged weight here
  integrateRay(submap, merged_point_C, merged_color, merged_normal_C, merged_weight, 
               T_S_C, normal_refine_on, is_free_space_submap,
               clearing_ray, integrate_full_ray,
               voxel_size, voxels_per_side, truncation_distance);

}

void ClassAdaptiveIntegrator::integrateRay(Submap* submap, const Point &p_C, const Color &color,
                                           const Ray &n_C, const float init_weight, const Transformation &T_S_C,
                                           const bool normal_refine_on, const bool is_free_space_submap,
                                           const bool is_clearing, const bool integrate_full_ray,
                                           const float voxel_size, const int voxels_per_side,                                              
                                           const float truncation_distance) const {    
                                            
                                                 
  const Point origin = T_S_C.getPosition(); //sensor's position in submap's frame
  const Point p_S = T_S_C * p_C;
  
  // for Ray OG
  voxblox::RayCaster ray_caster(origin, p_S, is_clearing, //is_clearing should be false here
                                integrate_full_ray,
                                config_.ai_config.max_ray_length_m, 
                                1.0 / voxel_size,
                                truncation_distance);
                

  TsdfBlock::Ptr block = nullptr;
  BlockIndex block_idx;
  GlobalIndex global_voxel_idx;
        
  // for all the voxels along the ray
  while (ray_caster.nextRayIndex(&global_voxel_idx)) {
    const BlockIndex block_idx =
        voxblox::getBlockIndexFromGlobalVoxelIndex(global_voxel_idx, 1.0 / voxels_per_side);

    TsdfBlock::Ptr block =
        submap->getTsdfLayerPtr()->allocateBlockPtrByIndex(block_idx);
        
    const VoxelIndex local_voxel_idx =
        voxblox::getLocalFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side);

    TsdfVoxel* voxel = &(block->getVoxelByVoxelIndex(local_voxel_idx));

    ClassVoxel* class_voxel = nullptr;
    if (submap->hasClassLayer() &&
       (!config_.update_only_tracked_submaps || submap->wasTracked())) { 
        //update_only_tracked_submaps: defualt true
        ClassBlock::Ptr class_block =
          submap->getClassLayerPtr()->allocateBlockPtrByIndex(block_idx);
        if (class_block) {
          class_voxel = &class_block->getVoxelByVoxelIndex(local_voxel_idx);
        }
    }

    if (updateVoxel(voxel, origin, p_C, p_S, n_C, global_voxel_idx, color, 
                    is_free_space_submap, normal_refine_on, T_S_C.inverse(), 
                    init_weight, truncation_distance, voxel_size, class_voxel)) {
        block->setUpdatedAll(); 
    } 
  }
}

// Thread safe.
float ClassAdaptiveIntegrator::getVoxelWeight(const Point& point_C) const {
  if (!config_.ai_config.weight_reduction) {
    return 1.0f;
  }
  const FloatingPoint dist_z = std::abs(point_C.z());
  if (dist_z > kEpsilon) {
    return 1.0f / std::pow(dist_z, config_.ai_config.weight_reduction_exp);
  }
  return 0.0f;
}

}  // namespace panoptic_mapping
