#include "panoptic_mapping/map_management/map_manager.h"

#include <algorithm>
#include <future>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "panoptic_mapping/common/index_getter.h"

namespace panoptic_mapping {

config_utilities::Factory::RegistrationRos<MapManagerBase, MapManager>
    MapManager::registration_("submaps");

void MapManager::Config::checkParams() const {
  checkParamConfig(activity_manager_config);
  checkParamConfig(tsdf_registrator_config);
  checkParamConfig(layer_manipulator_config);
  checkParamGT(integration_threads, 0, "integration_threads");
}

void MapManager::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("prune_active_blocks_frequency", &prune_active_blocks_frequency);
  setupParam("activity_management_frequency", &activity_management_frequency);
  setupParam("change_detection_frequency", &change_detection_frequency);
  setupParam("merge_deactivated_submaps_if_possible",
             &merge_deactivated_submaps_if_possible);
  setupParam("apply_class_layer_when_deactivating_submaps",
             &apply_class_layer_when_deactivating_submaps);
  setupParam("activity_manager_config", &activity_manager_config,
             "activity_manager");
  setupParam("tsdf_registrator_config", &tsdf_registrator_config,
             "tsdf_registrator");
  setupParam("layer_manipulator_config", &layer_manipulator_config,
             "layer_manipulator");
  setupParam("integration_threads", &integration_threads);
}

MapManager::MapManager(const Config& config) : config_(config.checkValid()) {
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();

  // Setup members.
  activity_manager_ =
      std::make_shared<ActivityManager>(config_.activity_manager_config);
  tsdf_registrator_ =
      std::make_shared<TsdfRegistrator>(config_.tsdf_registrator_config);
  // esdf generation inside
  layer_manipulator_ =
      std::make_shared<LayerManipulator>(config_.layer_manipulator_config);

  // Add all requested tasks. (scheduled by frame interval)
  // Three main functions:
  // 1. pruneActiveBlocks (prune blocks in active submaps, remove empty submaps)
  // 2. manageSubmapActivity (mergeSubmaps)
  // 3. performChangeDetection
  if (config_.prune_active_blocks_frequency > 0) {
    tickers_.emplace_back(
        config_.prune_active_blocks_frequency,
        [this](SubmapCollection* submaps) { pruneActiveBlocks(submaps); });
  }
  if (config_.activity_management_frequency > 0) {
    tickers_.emplace_back(
        config_.activity_management_frequency,
        [this](SubmapCollection* submaps) { manageSubmapActivity(submaps); });
  }
  if (config_.change_detection_frequency > 0) {
    tickers_.emplace_back(
        config_.change_detection_frequency,
        [this](SubmapCollection* submaps) { performChangeDetection(submaps); });
  }
}

void MapManager::tick(SubmapCollection* submaps) {
  // Increment counts for all tickers, which execute the requested actions.
  for (Ticker& ticker : tickers_) {
    ticker.tick(submaps);
  }
}

void MapManager::pruneActiveBlocks(SubmapCollection* submaps) {
  // Process all active instance and background submaps.
  auto t1 = std::chrono::high_resolution_clock::now();
  Timer timer("umap_management/prune_active_blocks");
  //std::stringstream info;
  // TODO(py): try to use multi-thread for each submap here
  
  std::vector<int> submaps_to_check; // id
  std::vector<int> submaps_to_remove; // id

  for (Submap& submap : *submaps) { // for all the active non-freespace submaps
    if (submap.getLabel() == PanopticLabel::kFreeSpace || !submap.isActive()) {
      continue;
    } else {
      submaps_to_check.push_back(submap.getID());
    }
  }

  // NOTE(py): have changed this part to multi-threads
  SubmapIndexGetter index_getter(submaps_to_check);
  std::vector<std::future<void>> threads;  
  for (int i = 0; i < config_.integration_threads; ++i) {
    threads.emplace_back(
        std::async(std::launch::async,
                   [this, &index_getter, submaps, i]() {
                     int index;
                     while (index_getter.getNextIndex(&index)) {
                       this->pruneBlocks(submaps->getSubmapPtr(index));
                     }
                   }));
  }
  // Join all threads.
  for (auto& thread : threads) {
    thread.get();
  }

  for (int i = 0; i < submaps_to_check.size(); i++) {
    Submap* cur_submap = submaps->getSubmapPtr(submaps_to_check[i]);
    if (cur_submap->getTsdfLayer().getNumberOfAllocatedBlocks() == 0) {
      submaps_to_remove.emplace_back(submaps_to_check[i]);
      if (config_.verbosity >= 3) {
        ROS_INFO("Removed submap %d (%s) [pruned]", cur_submap->getID(), cur_submap->getName().c_str());
      }
    }
  }

  // Remove submaps.
  for (int id : submaps_to_remove) {
    submaps->removeSubmap(id);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  timer.Stop();
  if (config_.verbosity >= 4)
    ROS_INFO("Pruned active blocks in %d ms", 
    std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
}

// TODO (py): add timers and understand each func
void MapManager::manageSubmapActivity(SubmapCollection* submaps) {
  Timer timer("umap_management/manage_submap_activity");

  CHECK_NOTNULL(submaps);
  std::unordered_set<int> active_submaps;
  if (config_.merge_deactivated_submaps_if_possible) {
    // Track de-activated submaps if requested.
    for (const Submap& submap : *submaps) {
      if (submap.isActive()) {
        active_submaps.insert(submap.getID()); // current active submaps
      }
    }
  }

  // Perform activity management. 
  // 1. checkRequiredRedetection, if the submap is not tracked consecutive for X frames from the beginning, discard the submap
  // 2. checkMissedDetections, if the submap is not not detected for Y consecutive frames, deactivate the submap, but it should also obey 1
  activity_manager_->processSubmaps(submaps);

  // Process de-activated submaps if requested.
  if (config_.merge_deactivated_submaps_if_possible ||
      config_.apply_class_layer_when_deactivating_submaps) {
    std::unordered_set<int> deactivated_submaps;
    for (Submap& submap : *submaps) {
      if (!submap.isActive() &&
          active_submaps.find(submap.getID()) != active_submaps.end()) { // de-activated at current frame
        deactivated_submaps.insert(submap.getID());
      }
    }

    // Apply the class layer if requested.
    // Removes those non-belonging points from the TSDF and deletes the class layer.
    // Upon deactivating (fron active to inactive), remove non-belonging points
    if (config_.apply_class_layer_when_deactivating_submaps) {
      for (int id : deactivated_submaps) {
        Submap* submap = submaps->getSubmapPtr(id);
        submap->applyClassLayer(*layer_manipulator_); // clear the class layer for this submao
      }
    }

    // Try to merge the submaps. (for each submap that is deactivated in current frame), time-consuming
    if (config_.merge_deactivated_submaps_if_possible) {
      for (int id : deactivated_submaps) { // for all the submaps begin to be inactive at current frame
        int merged_id;
        int current_id = id;
        while (mergeSubmapIfPossible(submaps, current_id, &merged_id)) {
          current_id = merged_id;
        }
        if (current_id == id) { // not merged, deactive the submap without merging
          if (config_.verbosity >= 4)
            ROS_INFO("Submap %d was deactivated without merging", id);
        }
      }
    }
  }

  timer.Stop();
}

void MapManager::performChangeDetection(SubmapCollection* submaps) {
  tsdf_registrator_->checkSubmapCollectionForChange(submaps);
}

void MapManager::finishMapping(SubmapCollection* submaps) {
  // Remove all empty blocks.
  std::stringstream info;
  info << "Finished mapping: ";
  for (Submap& submap : *submaps) { // for all the submaps
    pruneBlocks(&submap);
  }
  LOG_IF(INFO, config_.verbosity >= 3) << info.str();

  // Deactivate last submaps.
  for (Submap& submap : *submaps) {
    if (submap.isActive()) {
      LOG_IF(INFO, config_.verbosity >= 3)
          << "Deactivating submap " << submap.getID();
      submap.finishActivePeriod();
    }
  }
  LOG_IF(INFO, config_.verbosity >= 3) << "Merging Submaps:";

  // Merge what is possible.
  bool merged_something = true;
  while (merged_something) {
    for (Submap& submap : *submaps) {
      merged_something = mergeSubmapIfPossible(submaps, submap.getID());
      if (merged_something) {
        break;
      }
    }
  }

  // Finish submaps.
  if (config_.apply_class_layer_when_deactivating_submaps) {
    LOG_IF(INFO, config_.verbosity >= 3) << "Applying class layers:";
    std::vector<int> empty_submaps;
    for (Submap& submap : *submaps) {
      if (submap.hasClassLayer()) {
        if (!submap.applyClassLayer(*layer_manipulator_)) {
          empty_submaps.emplace_back(submap.getID());
        }
      }
    }
    for (const int id : empty_submaps) {
      submaps->removeSubmap(id);
      LOG_IF(INFO, config_.verbosity >= 3)
          << "Removed submap " << id << " which was empty.";
    }
  }
}

bool MapManager::mergeSubmapIfPossible(SubmapCollection* submaps, int submap_id,
                                       int* merged_id) {
  // Used on deactivated submaps of current frame, 
  // checks for possible matches with other inactive submaps.
 
  if (!submaps->submapIdExists(submap_id)) {
    return false;
  }

  // Setup.
  Submap* submap = submaps->getSubmapPtr(submap_id);
  if (submap->isActive()) {
    // Current query submap for merging should be inactive, if it's still active, first de-activated it.
    submap->finishActivePeriod();
  } else if (submap->getChangeState() == ChangeState::kAbsent) {
    return false;
  }

  // merge condition: inactive, same classID, boundingbox has some intersections
  // Find all potential matches.
  for (Submap& other : *submaps) {
    if (other.isActive() || other.getClassID() != submap->getClassID() ||
        other.getID() == submap->getID() ||
        !submap->getBoundingVolume().intersects(other.getBoundingVolume())) {
      continue;
    }
    // "other" become a matching candidate for current submap
    bool submaps_match;
    // check if these two submaps can be merged by tsdf registration
    if (!tsdf_registrator_->submapsConflict(*submap, other, &submaps_match)) {
      if (submaps_match) {
        // It's a match, merge the submap into the candidate.

        // Make sure both maps have or don't have class layers.
        if (!(submap->hasClassLayer() && other.hasClassLayer())) {
          submap->applyClassLayer(*layer_manipulator_);
          other.applyClassLayer(*layer_manipulator_);
        }
        layer_manipulator_->mergeSubmapAintoB(*submap, &other);
        if (config_.verbosity >= 2) {
          ROS_INFO("Merge submap %d (%s) into submap %d (%s) !!!", 
          submap->getID(), submap->getName().c_str(), 
          other.getID(), other.getName().c_str());
        }
        other.setChangeState(ChangeState::kPersistent);
        submaps->removeSubmap(submap_id); // current submap is merged to a older one and discarded
        if (merged_id) {
          *merged_id = other.getID(); // the merged submap would be the older submap
        }
        return true;
      }
    }
  }
  return false;
}

// Prune the blocks in one submap
void MapManager::pruneBlocks(Submap* submap) const {
  auto t1 = std::chrono::high_resolution_clock::now();
  // Setup.
  ClassLayer* class_layer = nullptr;
  if (submap->hasClassLayer()) {
    class_layer = submap->getClassLayerPtr().get();
  }
  TsdfLayer* tsdf_layer = submap->getTsdfLayerPtr().get();
  MeshLayer* mesh_layer = submap->getMeshLayerPtr().get();
  const int voxel_indices = std::pow(submap->getConfig().voxels_per_side, 3);
  int count = 0;

  // Remove all blocks that don't have any belonging voxels.
  voxblox::BlockIndexList block_indices;
  tsdf_layer->getAllAllocatedBlocks(&block_indices);
  // For each block, do the checking
  for (const auto& block_index : block_indices) {
    ClassBlock::Ptr class_block;
    if (class_layer) {
      if (class_layer->hasBlock(block_index)) {
        class_block = class_layer->getBlockPtrByIndex(block_index);
      }
    }
    const TsdfBlock& tsdf_block = tsdf_layer->getBlockByIndex(block_index);
    bool has_beloning_voxels = false;

    // Check all voxels.
    for (int voxel_index = 0; voxel_index < voxel_indices; ++voxel_index) {
      if (tsdf_block.getVoxelByLinearIndex(voxel_index).weight >= 
          config_.tsdf_registrator_config.min_voxel_weight) {
        if (class_block) {
          if (class_block->getVoxelByLinearIndex(voxel_index)
                  .belongsToSubmap()) {
            has_beloning_voxels = true;
            break;
          }
        } else {
          has_beloning_voxels = true;
          break;
        }
      }
    }

    // Prune blocks.
    if (!has_beloning_voxels) {
      if (class_layer) {
        class_layer->removeBlock(block_index);
      }
      tsdf_layer->removeBlock(block_index);
      mesh_layer->removeMesh(block_index);
      count++;
    }
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  //std::stringstream ss;
  if (count > 0 && config_.verbosity >= 3) {
    ROS_INFO("Pruned %d blocks from submap %d (%s) in %d ms", 
      count, submap->getID(), submap->getName().c_str(),
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
  }
}

// a kind of timer scheduler
void MapManager::Ticker::tick(SubmapCollection* submaps) {
  // Perform 'action' every 'max_ticks' ticks.
  current_tick_++;
  if (current_tick_ >= max_ticks_) {
    action_(submaps);
    current_tick_ = 0;
  }
}

}  // namespace panoptic_mapping
