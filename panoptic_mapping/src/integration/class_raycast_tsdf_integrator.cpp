#include "panoptic_mapping/integration/class_raycast_tsdf_integrator.h"

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
    TsdfIntegratorBase, ClassRaycastIntegrator, std::shared_ptr<Globals>>
    ClassRaycastIntegrator::registration_("class_raycast");

void ClassRaycastIntegrator::Config::checkParams() const {
  checkParamConfig(ri_config);
}

void ClassRaycastIntegrator::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("use_binary_classification", &use_binary_classification);
  setupParam("use_instance_classification", &use_instance_classification);
  setupParam("update_only_tracked_submaps", &update_only_tracked_submaps);
  setupParam("raycast_integrator_config", &ri_config);
}

ClassRaycastIntegrator::ClassRaycastIntegrator(
    const Config& config, std::shared_ptr<Globals> globals)
    : config_(config.checkValid()),
      RaycastIntegrator(config.ri_config, std::move(globals), false) {
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();

  // Store class count.
  if (!config_.use_binary_classification &&
      !config_.use_instance_classification) {
    // The +1 is added because 0 is reserved for the belonging submap.
    num_classes_ = globals_->labelHandler()->numberOfLabels() + 1;
  }
}

void ClassRaycastIntegrator::processInput(SubmapCollection* submaps,
                                          InputData* input) {
  
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
  RaycastIntegrator::processInput(submaps, input);
}

//TODO (py): problem could be the mutex lock problem, use the same input at the same time
void ClassRaycastIntegrator::updateSubmap(Submap* submap, const InputData& input) {
  
  if(config_.verbosity > 3)
    ROS_INFO("Debugging/Begin to update submap %d", submap->getID());
  
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

  // Deal with the freespace submap 
  if (is_free_space_submap) { // directly skip free space ?
    if (config_.ri_config.skip_free_space_submap)
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

  const bool normal_reliable = submap->getNormalReliability(); // for free-space submap, default true
  bool normal_refine_on = false;
  bool apply_normal_refine = is_free_space_submap ? 
                              config_.ri_config.apply_normal_refine_freespace :
                              config_.ri_config.apply_normal_refine;
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
  // bool submap_valid = !normal_reliable; // only use merge on those normal un-reliable objects (like vegetation or vehicles)
  // if (is_free_space_submap)
  //    submap_valid = true;
  bool submap_valid = true;
  bool multi_threads_on = false;
  if (config_.ri_config.only_merge_freespace)
    submap_valid = is_free_space_submap;
  if (config_.ri_config.free_space_multi_threads)
    multi_threads_on = is_free_space_submap;

  if(config_.ri_config.merged_raycasting && submap_valid) {
    integratePointCloudMerged(submap, submap_points, submap_colors, submap_normals,
                              T_S_C, normal_refine_on, is_free_space_submap, multi_threads_on,
                              voxel_size, voxels_per_side, truncation_distance);
  } else {
    integratePointCloud(submap, submap_points, submap_colors, submap_normals,
                        T_S_C, normal_refine_on, is_free_space_submap,
                        voxel_size, voxels_per_side, truncation_distance);
  }                   

  // free the memory
  Pointcloud().swap(submap_points);
  Colors().swap(submap_colors);
  Pointcloud().swap(submap_normals);
}


// Updates tsdf_voxel. Thread safe.
bool ClassRaycastIntegrator::updateVoxel(TsdfVoxel* voxel, const Point& origin, const Point& p_C, 
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

  bool in_the_reliable_band = true;
  if (sdf > truncation_distance * config_.ri_config.reliable_band_ratio)
    in_the_reliable_band = false;

  Ray v_C;

  if (apply_normal_refine && in_the_reliable_band) {
    float normal_ratio = 1.0f;
    if (voxel->gradient.norm() > kFloatEpsilon){ //use current un-updated normal because the weight is unknown
      v_C = T_C_S.getRotationMatrix() * voxel->gradient; //back to sensor(camera)'s frame
      if (config_.ri_config.curve_assumption && n_C.norm() > kFloatEpsilon) {
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
    } else { //gradient not ready yet, use the first (current) normal vector
      //NOTE(py): kFloatEpsilon is a safe value in case of numerical rounding error
      if (n_C.norm() > kFloatEpsilon) { //current normal is valid
        normal_ratio = std::abs(n_C.dot(p_C)/p_C.norm());
      }
    }
    if (normal_ratio < config_.ri_config.reliable_normal_ratio_thre) // NOTE(py): ruling out extremely large incidence angle
      return false;
    sdf *= normal_ratio; // get the non-projective sdf, if it's still larger than truncation distance, the gradient would not be updated
  }

  if (sdf < -truncation_distance) {
    return false;
  }

  bool with_init_weight = false;
  if (init_weight > 0)
    with_init_weight = true;
  float weight = computeVoxelWeight(p_C, voxel_size, truncation_distance, sdf, 
                                    false, with_init_weight, init_weight);
  

  // Only merge color and classification data near the surface.
  if (sdf > truncation_distance) {
    // far away, do not interpolate color
    updateVoxelValues(voxel, sdf, weight);
    voxel->distance = std::min(truncation_distance, voxel->distance);
  } else { // close to the surface
    if (apply_normal_refine) { // only update the gradient close to the surface
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

// Simple raycasting integration
void ClassRaycastIntegrator::integratePointCloud (Submap* submap, 
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
                                config_.ri_config.freespace_carving_enabled : 
                                config_.ri_config.voxel_carving_enabled;

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
void ClassRaycastIntegrator::integratePointCloudMerged (Submap* submap, 
                                                        Pointcloud &submap_points,
                                                        Colors &submap_colors,
                                                        Pointcloud &submap_normals,
                                                        Transformation &T_S_C,
                                                        const bool normal_refine_on,
                                                        const bool is_free_space_submap,
                                                        const bool multi_threads,
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
                            config_.ri_config.freespace_carving_enabled : 
                            config_.ri_config.voxel_carving_enabled;


  // bundle rays in each voxel with point inside (isPointValid is judged in it)
  bundleRays(T_S_C, submap_points, 1.0 / voxel_size, &voxel_map, &clear_map);

  voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type::const_iterator it;
  size_t map_size;

  if (multi_threads) {
    // multi threads
    std::list<std::thread> integration_threads;
    for (size_t i = 0; i < config_.ri_config.integration_threads; ++i) {
      integration_threads.emplace_back(
            &ClassRaycastIntegrator::integrateVoxelsMerged, this, submap, submap_points, submap_colors, 
                          submap_normals, T_S_C, voxel_map, false, normal_refine_on, is_free_space_submap, integrate_full_ray,
                          voxel_size, voxels_per_side, truncation_distance, i);
    }
    for (std::thread& thread : integration_threads) {
      thread.join();
    }
  } else {
    // single thread
    it = voxel_map.begin();
    map_size = voxel_map.size();

    for (size_t i = 0; i < map_size; ++i) { // for each voxel, merge the points inside
      integrateVoxelMerged(submap, submap_points, submap_colors, submap_normals, T_S_C, 
                          *it, false, normal_refine_on, is_free_space_submap, integrate_full_ray,
                          voxel_size, voxels_per_side, truncation_distance);
      ++it;
    }  
  }

  // timing::Timer clear_timer("integrate/clear");
  
  // but actually clear is not used
  // it = clear_map.begin();
  // map_size = clear_map.size(); 
  // for (size_t i = 0; i < map_size; ++i) { 
  //   integrateVoxelMerged(submap, submap_points, submap_colors, submap_normals, T_S_C, 
  //                       *it, true, normal_refine_on, is_free_space_submap, integrate_full_ray,
  //                       voxel_size, voxels_per_side, truncation_distance);
  //   ++it;
  // }
  // clear_timer.Stop();
  // integrate_timer.Stop();
}

void ClassRaycastIntegrator::bundleRays(const Transformation& T_S_C, 
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

// used for multi-thread merged raycasting
void ClassRaycastIntegrator::integrateVoxelsMerged(Submap *submap,
                                                   const Pointcloud &submap_points,
                                                   const Colors &submap_colors,
                                                   const Pointcloud &submap_normals,
                                                   const Transformation& T_S_C,
                                                   const voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type& voxel_map, 
                                                   const bool clearing_ray, const bool normal_refine_on,
                                                   const bool is_free_space_submap, 
                                                   const bool integrate_full_ray,
                                                   const float voxel_size, const int voxels_per_side,                                              
                                                   const float truncation_distance,
                                                   const int thread_idx) const {
    
  voxblox::LongIndexHashMapType<AlignedVector<size_t>>::type::const_iterator it;
  size_t map_size;
  it = voxel_map.begin();
  map_size = voxel_map.size();

  for (size_t i = 0; i < map_size; ++i) {
    if (((i + thread_idx + 1) % config_.ri_config.integration_threads) == 0) {
      integrateVoxelMerged(submap, submap_points, submap_colors, submap_normals, T_S_C, 
                          *it, clearing_ray, normal_refine_on, is_free_space_submap, integrate_full_ray,
                          voxel_size, voxels_per_side, truncation_distance);
    }
    ++it;
  }
}
                                                   
// for one voxel
void ClassRaycastIntegrator::integrateVoxelMerged(Submap *submap,
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

void ClassRaycastIntegrator::integrateRay(Submap* submap, const Point &p_C, const Color &color,
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
                                config_.ri_config.max_ray_length_m, 
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

// Deprecated: used for the multi-thread processing of a single map
// Will return a pointer to a voxel located at global_voxel_idx in the tsdf
// layer. Thread safe.
// Takes in the last_block_idx and last_block to prevent unneeded map lookups.
// If the block this voxel would be in has not been allocated, a block in
// temp_block_map_ is created/accessed and a voxel from this map is returned
// instead. Unlike the layer, accessing temp_block_map_ is controlled via a
// mutex allowing it to grow during integration.
// These temporary blocks can be merged into the layer later by calling
// updateLayerWithStoredBlocks()
TsdfVoxel* ClassRaycastIntegrator::allocateStorageAndGetVoxelPtr(Submap* submap,
    const GlobalIndex& global_voxel_idx, TsdfBlock::Ptr* last_block,
    BlockIndex* last_block_idx, ClassBlock::Ptr* last_class_block) {
  DCHECK(last_block != nullptr);
  DCHECK(last_block_idx != nullptr);

  float voxel_size = submap->getConfig().voxel_size;
  float voxels_per_side = submap->getConfig().voxels_per_side;
  float voxels_per_side_inv = 1.0 / voxels_per_side;
  float block_size = voxel_size * voxels_per_side;

  const BlockIndex block_idx =
      voxblox::getBlockIndexFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side_inv);

  if ((block_idx != *last_block_idx) || (last_block == nullptr)) {
    *last_block = submap->getTsdfLayerPtr()->getBlockPtrByIndex(block_idx);
    *last_block_idx = block_idx;
  }

  // If no block at this location currently exists, we allocate a temporary
  // voxel that will be merged into the map later
  if (last_block == nullptr) {
    // To allow temp_block_map_ to grow we can only let one thread in at once
    //std::lock_guard<std::mutex> lock(temp_block_mutex_);

    typename TsdfLayer::BlockHashMap::iterator it =
        temp_block_map_.find(block_idx);
    if (it != temp_block_map_.end()) { //found
      *last_block = it->second;
    } else { //not found
      //emplace a new one
      //temp_block_map_: <block_id, TsdfBlock>
      auto insert_status = temp_block_map_.emplace(
          block_idx, std::make_shared<TsdfBlock>(
                         voxels_per_side, voxel_size,
                         voxblox::getOriginPointFromGridIndex(block_idx, block_size)));

      DCHECK(insert_status.second) << "Block already exists when allocating at "
                                   << block_idx.transpose();

      *last_block = insert_status.first->second;
    }
  }

  (*last_block)->setUpdatedAll(); //TODO(py): check the update tag
  
  // Allocate the class block if not yet existent and get it.
  if (submap->hasClassLayer() &&
      (!config_.update_only_tracked_submaps || submap->wasTracked())) { 
        //update_only_tracked_submaps: defualt true
    *last_class_block =
        submap->getClassLayerPtr()->allocateBlockPtrByIndex(block_idx);
  }

  const VoxelIndex local_voxel_idx =
      voxblox::getLocalFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side);

  return &((*last_block)->getVoxelByVoxelIndex(local_voxel_idx));
}

}  // namespace panoptic_mapping
