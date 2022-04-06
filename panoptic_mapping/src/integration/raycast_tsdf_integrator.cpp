#include "panoptic_mapping/integration/raycast_tsdf_integrator.h"

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
    TsdfIntegratorBase, RaycastIntegrator, std::shared_ptr<Globals>>
    RaycastIntegrator::registration_("raycast");

void RaycastIntegrator::Config::checkParams() const {
  checkParamGT(integration_threads, 0, "integration_threads");
  checkParamGT(max_weight, 0.f, "max_weight");
  checkParamGT(weight_reduction_exp, 0.f, "weight_reduction_exp");
  if (use_weight_dropoff) {
    checkParamNE(weight_dropoff_epsilon, 0.f, "weight_dropoff_epsilon");
  }
}

//TODO(py): update these config params
void RaycastIntegrator::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("use_range_image", &use_range_image);
  setupParam("merged_raycasting", &merged_raycasting);
  setupParam("only_merge_freespace", &only_merge_freespace);
  setupParam("free_space_multi_threads", &free_space_multi_threads);
  setupParam("use_weight_dropoff", &use_weight_dropoff);
  setupParam("weight_dropoff_epsilon", &weight_dropoff_epsilon);
  setupParam("weight_reduction", &weight_reduction);
  setupParam("weight_reduction_exp", &weight_reduction_exp);
  setupParam("voxel_carving_enabled", &voxel_carving_enabled);
  setupParam("freespace_carving_enabled", &freespace_carving_enabled);
  setupParam("max_ray_length_m", &max_ray_length_m, "m");
  setupParam("min_ray_length_m", &min_ray_length_m, "m");
  setupParam("foreign_rays_clear", &foreign_rays_clear);
  setupParam("max_weight", &max_weight);
  setupParam("use_longterm_fusion", &use_longterm_fusion);
  setupParam("integration_threads", &integration_threads);
  setupParam("use_lidar", &use_lidar);
  setupParam("apply_normal_refine", &apply_normal_refine);
  setupParam("apply_normal_refine_freespace", &apply_normal_refine_freespace);
  setupParam("curve_assumption", &curve_assumption);
  setupParam("behind_surface_reliable_band", &behind_surface_reliable_band);
  setupParam("reliable_band_ratio", &reliable_band_ratio);
  setupParam("reliable_normal_ratio_thre", &reliable_normal_ratio_thre);
  setupParam("skip_free_space_submap", &skip_free_space_submap);
}

RaycastIntegrator::RaycastIntegrator(const Config& config,
                                     std::shared_ptr<Globals> globals,
                                     bool print_config)
    : config_(config.checkValid()), TsdfIntegratorBase(std::move(globals)) {
  LOG_IF(INFO, config_.verbosity >= 1 && print_config) << "\n"
                                                       << config_.toString();
  // Request all inputs.
  if (config_.use_range_image) {
    addRequiredInputs(
        {InputData::InputType::kColorImage,
         InputData::InputType::kDepthImage,
         InputData::InputType::kVertexMap});
  } else {
    addRequiredInputs(
      {InputData::InputType::kPointCloud,
       InputData::InputType::kPointColor});
  }
  // InputData::InputType::kValidityImage not used
}

// for each submap -> for each point(ray) -> for each voxel along the ray
void RaycastIntegrator::processInput(SubmapCollection* submaps,
                                     InputData* input) {

  CHECK_NOTNULL(submaps);
  CHECK_NOTNULL(input);  
  
  if(config_.verbosity > 3)
    ROS_INFO("Begin to do the raycasting integration");

  //Find visible submaps
  // Timer find_timer("tsdf_integration/find_submaps");
  std::vector<int> submap_id_list = input->submapIDList(); // use a much simple way here
  
  // if (config_.use_lidar) {
    
  //   submap_id_list = 
  //     globals_->lidar()->findVisibleSubmapIDs(*submaps, input->T_M_C(),
  //                                             true, true); // only active submap, include freespace submap
  // } else {   
  //   submap_id_list =                                        
  //     globals_->camera()->findVisibleSubmapIDs(*submaps, input->T_M_C(),
  //                                             true, true);
  // }
  // freespace submap is also included

  if(config_.verbosity > 3)
    ROS_INFO("Debugging/%d visible submaps found", submap_id_list.size());

  // find_timer.Stop();

  //Update the TSDF in each submap
  // Integrate in parallel. (for each submap)
  Timer int_timer("tsdf_integration/integration");

  // Firstly process the free space submap 
  if (config_.free_space_multi_threads) {
    Timer free_init_timer("tsdf_integration/integration/freespace_multi");
    int free_space_submap_id = submaps->getActiveFreeSpaceSubmapID();
    this->updateSubmap(submaps->getSubmapPtr(free_space_submap_id), *input);
    
    submaps->getSubmapPtr(free_space_submap_id)->updateBoundingVolume();
    
    std::vector<int>::iterator position = std::find(submap_id_list.begin(), submap_id_list.end(), free_space_submap_id);
    if (position != submap_id_list.end()) 
      submap_id_list.erase(position);
    free_init_timer.Stop();
  }
                                        
  SubmapIndexGetter index_getter(submap_id_list);
  std::vector<std::future<void>> threads;  
  // NOTE(py): the free space submap would not be covered here since it 
  // does not corresponding to specific part of the IDImage
  // But we do need it for planning issues
  // It's better to switch to the adpative integrator
  for (int i = 0; i < config_.integration_threads; ++i) {
    threads.emplace_back(
        std::async(std::launch::async,
                   [this, &index_getter, submaps, input, i]() {
                     int index;
                     while (index_getter.getNextIndex(&index)) {
                       this->updateSubmap(submaps->getSubmapPtr(index),
                                          *input);   
                     }
                   }));
  }

  // Join all threads.
  for (auto& thread : threads) {
    thread.get();
  }

  for (int i=0; i < submap_id_list.size(); i++) {
    Submap* submap = submaps->getSubmapPtr(submap_id_list[i]);
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
  int_timer.Stop(); 
}

//TODO(py): add class block, voxel
//Single thread
void RaycastIntegrator::updateSubmap(Submap* submap, const InputData& input) {
  
  Transformation T_S_C = submap->getT_M_S().inverse() * input.T_M_C();
  Transformation T_C_S = T_S_C.inverse();

  Pointcloud submap_points;
  Colors submap_colors;

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

  Pointcloud submap_normals;
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

  float voxel_size = submap->getConfig().voxel_size;
  float voxel_size_inv =  1.0 / voxel_size;
  float truncation_distance = submap->getConfig().truncation_distance; 
  float voxels_per_side = submap->getConfig().voxels_per_side;
  float voxels_per_side_inv = 1.0 / voxels_per_side;
  float block_size = voxel_size * voxels_per_side;
  
  // for each point (ray) in the submap
  for(int i = 0; i < submap_points.size(); i++) {
    const Point& p_C = submap_points[i];
    const Color& color = submap_colors[i];
    Ray n_C;
    if (normal_refine_on) {
      n_C = submap_normals[i];
    }

    bool is_clearing; //TODO(py): all needs clearing
    if (!isPointValid(p_C, false, &is_clearing)) { //Need to be valid point to proceed
      continue;
    }

    const Point origin = T_S_C.getPosition(); //sensor's position in submap's frame
    const Point p_S = T_S_C * p_C;            //target point's position in submap's frame

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
                  is_free_space_submap, normal_refine_on, T_C_S, 0.0, 
                  truncation_distance, voxel_size)) {
        block->setUpdatedAll(); 
      }
    }
  }
  // free the memory
  Pointcloud().swap(submap_points);
  Colors().swap(submap_colors);
  Pointcloud().swap(submap_normals);
}

// Updates tsdf_voxel. Thread safe.
bool RaycastIntegrator::updateVoxel(TsdfVoxel* voxel, const Point& origin, const Point& p_C, 
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

// Equ. 1 in the paper
// NOTE(py): very important part, try to improve the current scheme
// Compute the weight for current measurement
// if projective = true, this func would be used for projective mapping 
float RaycastIntegrator::computeVoxelWeight(const Point& p_C,
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
  // Apply weight drop-off if appropriate. // check voxblox's paper
  if (config_.use_weight_dropoff) {
    const float dropoff_epsilon =
        config_.weight_dropoff_epsilon > 0.f
            ? config_.weight_dropoff_epsilon
            : config_.weight_dropoff_epsilon * -voxel_size; 
    //for example, weight_dropoff_epsilon = -1.0 --> dropoff_epsilon = voxel_size
    if (sdf < -dropoff_epsilon) {
      weight *=
          (truncation_distance + sdf) / (truncation_distance - dropoff_epsilon);
      weight = std::max(weight, 0.f); // should >= 0
    }
  }
  return weight;
}

//each voxel has a distance and a weight
//once a new distance and weight is calculated, update it as 
//a kind of weighted average
void RaycastIntegrator::updateVoxelValues(TsdfVoxel* voxel, const float sdf,
                                          const float weight, 
                                          const Color* color) const {
  
  float new_weight = voxel->weight + weight;
  // it is possible to have weights very close to zero, due to the limited
  // precision of floating points dividing by this small value can cause nans
  if (new_weight < kFloatEpsilon) {
    return;
  }
  
  // Weighted averaging fusion.
  voxel->distance = (voxel->distance * voxel->weight + sdf * weight) / new_weight;              

  // float new_sdf = (voxel->distance * voxel->weight + sdf * weight) / new_weight;    
  
  // change it later by adding tunct_dist param
  // voxel->distance = (new_sdf > 0.0) ? 
  //                   std::min(trunc_dist, new_sdf) :
  //                   std::max(-trunc_dist, new_sdf);
  
  voxel->weight = std::min(new_weight, config_.max_weight);
  // also take average of the color
  if (color != nullptr) {
    voxel->color =
        Color::blendTwoColors(voxel->color, voxel->weight, *color, weight);
  }
}  

void RaycastIntegrator::updateVoxelGradient(TsdfVoxel* voxel, const Point normal,
                                            const float weight) const {
                                              
  float new_weight = voxel->weight + weight;
  // it is possible to have weights very close to zero, due to the limited
  // precision of floating points dividing by this small value can cause nans
  if (new_weight < kFloatEpsilon) {
    return;
  }

  if(voxel->gradient.norm() > kFloatEpsilon){ 
    voxel->gradient = ((voxel->gradient * voxel->weight + normal * weight) /
                      new_weight).normalized();
  }
  else { // newly assigned, originally zero vector
    voxel->gradient = normal.normalized();
  }
}

// Thread safe.
// Figure out whether the voxel is behind or in front of the surface.
// To do this, project the voxel_center onto the ray from origin to point G.
// Then check if the the magnitude of the vector is smaller or greater than
// the original distance...
float RaycastIntegrator::computeSignedDistance(const Point& origin,
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

Pointcloud RaycastIntegrator::extractSubmapPointCloud(const cv::Mat& vertex_map,
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

Colors RaycastIntegrator::extractSubmapColors(const cv::Mat& color_image,
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

Pointcloud RaycastIntegrator::extractSubmapNormals(const cv::Mat& normal_image,
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

// Thread safe.
float RaycastIntegrator::getVoxelWeight(const Point& point_C) const {
  if (!config_.weight_reduction) {
    return 1.0f;
  }
  const FloatingPoint dist_z = std::abs(point_C.z());
  if (dist_z > kEpsilon) {
    return 1.0f / std::pow(dist_z, config_.weight_reduction_exp);
  }
  return 0.0f;
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
TsdfVoxel* RaycastIntegrator::allocateStorageAndGetVoxelPtr(Submap* submap,
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
    std::lock_guard<std::mutex> lock(temp_block_mutex_);

    typename TsdfLayer::BlockHashMap::iterator it =
        temp_block_map_.find(block_idx);
    if (it != temp_block_map_.end()) {
      *last_block = it->second;
    } else {
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

  const VoxelIndex local_voxel_idx =
      voxblox::getLocalFromGlobalVoxelIndex(global_voxel_idx, voxels_per_side);

  return &((*last_block)->getVoxelByVoxelIndex(local_voxel_idx));
}

}  // namespace panoptic_mapping
