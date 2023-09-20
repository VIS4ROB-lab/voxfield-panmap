#include "panoptic_mapping/map/submap_collection.h"

#include <sys/stat.h>

#include <memory>
#include <string>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>

#include "panoptic_mapping/SubmapCollection.pb.h"

namespace panoptic_mapping {

Submap* SubmapCollection::createSubmap(const Submap::Config& config) {
  submaps_.emplace_back(std::make_unique<Submap>(config, &submap_id_manager_,
                                                 &instance_id_manager_)); //add a new one
  Submap* new_submap = submaps_.back().get(); //get the new one
  id_to_index_[new_submap->getID()] = submaps_.size() - 1; //submap index
  return new_submap;
}

bool SubmapCollection::removeSubmap(int id) {
  auto it = id_to_index_.find(id);
  if (it == id_to_index_.end()) {
    // Submap does not exist.
    return false;
  }
  size_t previous_index = it->second;
  submaps_.erase(submaps_.begin() + it->second);
  id_to_index_.erase(it);
  // correct the index table
  for (auto& id_index_pair : id_to_index_) {
    if (id_index_pair.second > previous_index) {
      id_index_pair.second -= 1;
    }
  }
  return true;
}

bool SubmapCollection::submapIdExists(int id) const {
  return id_to_index_.find(id) != id_to_index_.end();
}

const Submap& SubmapCollection::getSubmap(int id) const {
  auto it = id_to_index_.find(id);
  if (it == id_to_index_.end()) {
    LOG(FATAL) << "Tried to get inexistent submap " << id << ".";
  }
  // This assumes we checked that the id exists.
  return *submaps_[id_to_index_.at(id)];
}

Submap* SubmapCollection::getSubmapPtr(int id) {
  // This assumes we checked that the id exists.
  return submaps_[id_to_index_.at(id)].get();
}

void SubmapCollection::clear() {
  submaps_.clear();
  id_to_index_.clear();
  instance_id_manager_ = InstanceIDManager();
  submap_id_manager_ = SubmapIDManager();
  instance_to_submap_ids_.clear();
  active_freespace_submap_id_ = -1;
}

void SubmapCollection::updateIDList(const std::vector<int>& id_list,
                                    std::vector<int>* new_ids,
                                    std::vector<int>* deleted_ids) const {
  CHECK_NOTNULL(new_ids);
  CHECK_NOTNULL(deleted_ids);
  // Find all deleted submaps.
  for (const int& id : id_list) {
    if (!submapIdExists(id)) {
      deleted_ids->emplace_back(id);
    }
  }
  // Find all new submaps.
  for (const auto& id_submap_pair : id_to_index_) {
    auto it = std::find(id_list.begin(), id_list.end(), id_submap_pair.first);
    if (it == id_list.end()) {
      new_ids->emplace_back(id_submap_pair.first);
    }
  }
}

void SubmapCollection::updateInstanceToSubmapIDTable() {
  // Clear the table and add all new elements.
  instance_to_submap_ids_ = std::unordered_map<int, std::unordered_set<int>>();
  for (const Submap& submap : *this) {
    instance_to_submap_ids_[submap.getInstanceID()].emplace(submap.getID());
  }
}

// Save load functionality was heavily adapted from cblox.
bool SubmapCollection::saveToFile(const std::string& file_path) const {
  CHECK(!file_path.empty());

  // Check for proper extensions.
  const std::string file_name = checkMapFileExtension(file_path);

  // Open the ouput file.
  std::fstream outfile;
  outfile.open(file_name, std::fstream::out | std::fstream::binary);
  if (!outfile.is_open()) {
    LOG(ERROR) << "Could not open file '" << file_name
               << "' to save the submap collection.";
    return false;
  }

  // Saving the submap collection header object.
  SubmapCollectionProto submap_collection_proto;
  submap_collection_proto.set_num_submaps(submaps_.size());
  if (!voxblox::utils::writeProtoMsgToStream(submap_collection_proto,
                                             &outfile)) {
    LOG(ERROR) << "Could not write submap collection header message.";
    outfile.close();
    return false;
  }

  // Saving the submaps.
  for (const auto& submap : submaps_) {
    if (!submap->saveToStream(&outfile)) {
      LOG(WARNING) << "Failed to save submap with ID '" << submap->getID()
                   << "'.";
      outfile.close();
      return false;
    } 
    // else {
    //   LOG(INFO) << "Save submap with ID '" << submap->getID() << "' (" 
    //             << submap->getName().c_str() << ").";        
    // }
  }
  outfile.close();
  return true;
}

bool SubmapCollection::loadFromFile(const std::string& file_path,
                                    bool recompute_data) {
  CHECK(!file_path.empty());
  const std::string file_name = checkMapFileExtension(file_path);

  // Check the file exists.
  struct stat buffer;
  if (stat(file_name.c_str(), &buffer) != 0) {
    LOG(ERROR) << "Target file '" << file_name << "' does not exist.";
    return false;
  }

  // Clear the current maps.
  submaps_.clear();

  // Open and check the file.
  std::ifstream proto_file;
  proto_file.open(file_name, std::fstream::in);
  if (!proto_file.is_open()) {
    LOG(ERROR) << "Could not open protobuf file '" << file_name << "'.";
    return false;
  }

  // Unused byte offset result.
  uint64_t tmp_byte_offset = 0u;
  SubmapCollectionProto submap_collection_proto;
  if (!voxblox::utils::readProtoMsgFromStream(
          &proto_file, &submap_collection_proto, &tmp_byte_offset)) {
    LOG(ERROR) << "Could not read the protobuf message.";
    return false;
  }

  // Loading each of the submaps.
  for (size_t sub_map_index = 0u;
       sub_map_index < submap_collection_proto.num_submaps(); ++sub_map_index) {
    std::unique_ptr<Submap> submap_ptr =
        Submap::loadFromStream(&proto_file, &tmp_byte_offset,
                               &submap_id_manager_, &instance_id_manager_);
    if (submap_ptr == nullptr) {
      LOG(ERROR) << "Failed to load submap '" << sub_map_index
                 << "' from stream.";
      proto_file.close();
      return false;
    }

    // Add to the collection.
    id_to_index_[submap_ptr->getID()] = submaps_.size();
    submaps_.emplace_back(std::move(submap_ptr));
  }
  proto_file.close();

  // Recompute data that is not stored with the submap (bounding volume and mesh)
  if (recompute_data) {
    for (Submap& submap : *this) {
      submap.updateEverything();
    }
  }
  return true;
}

std::string SubmapCollection::checkMapFileExtension(const std::string& file) {
  const std::string extension = ".panmap";
  if (!(file.size() >= extension.size() &&
        file.compare(file.size() - extension.size(), extension.size(),
                     extension) == 0)) {
    return file + extension;
  }
  return file;
}

bool SubmapCollection::saveMeshToFile(const std::string& folder_path) const {
  CHECK(!folder_path.empty());

  if (!boost::filesystem::exists(folder_path.c_str())) {
    if(!boost::filesystem::create_directory(folder_path.c_str()))
      return false;
  }

  // TODO(py): consider make it run with multi-thread to speed up
  for (const Submap& submap : *this) {
    std::string id_pad;
    std::ostringstream oss;
		oss << std::setfill('0') << std::setw(5) << submap.getID() << "_" << submap.getName() << ".ply";
    std::string file_path = folder_path + "/" + oss.str();
    voxblox::outputMeshLayerAsPly(file_path,
                                  submap.getMeshLayer());
  }

  return true;
}


std::unique_ptr<SubmapCollection> SubmapCollection::clone() const {
  std::unique_ptr<SubmapCollection> result =
      std::make_unique<SubmapCollection>();

  // Copy all the meta data.
  result->submap_id_manager_ = submap_id_manager_;
  result->instance_id_manager_ = instance_id_manager_;
  result->id_to_index_ = id_to_index_;
  result->instance_to_submap_ids_ = instance_to_submap_ids_;
  result->active_freespace_submap_id_ = active_freespace_submap_id_;

  // Deep copy all the submaps to the new managers.
  for (const Submap& submap : *this) {
    result->submaps_.emplace_back(submap.clone(&result->submap_id_manager_,
                                               &result->instance_id_manager_));
  }

  return result;
}

}  // namespace panoptic_mapping
