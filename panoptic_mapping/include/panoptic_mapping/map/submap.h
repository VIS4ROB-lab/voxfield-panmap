#ifndef PANOPTIC_MAPPING_MAP_SUBMAP_H_
#define PANOPTIC_MAPPING_MAP_SUBMAP_H_

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <voxblox/core/layer.h>
#include <voxblox/mesh/mesh_layer.h>
#include <voxblox/integrator/occupancy_tsdf_integrator.h>
#include <voxblox/integrator/esdf_occ_fiesta_integrator.h> // fiesta
#include <voxblox/integrator/esdf_voxfield_integrator.h> // voxfield
#include <voxblox/integrator/esdf_occ_edt_integrator.h>  // edt
#include <voxblox/integrator/esdf_integrator.h>  // voxblox esdf mapping

#include "panoptic_mapping/3rd_party/config_utilities.hpp"
#include "panoptic_mapping/Submap.pb.h"
#include "panoptic_mapping/common/common.h"
#include "panoptic_mapping/integration/mesh_integrator.h"
#include "panoptic_mapping/map/classification/class_block.h"
#include "panoptic_mapping/map/classification/class_layer.h"
#include "panoptic_mapping/map/classification/class_voxel.h"
#include "panoptic_mapping/map/instance_id.h"
#include "panoptic_mapping/map/submap_bounding_volume.h"
#include "panoptic_mapping/map/submap_id.h"


namespace panoptic_mapping {

class LayerManipulator;

class Submap {
 public:
  // Config.
  // Different submaps have different voxel size (resolution)
  struct Config : public config_utilities::Config<Config> {
    float voxel_size = 0.1;           // m
    float truncation_distance = 0.2;  // m, negative values = #vs
    int voxels_per_side = 16;         // Needs to be a multiple of 2.
    bool use_class_layer = false;

    MeshIntegrator::Config mesh_config;

    // Config of the classification voxels to be used. Leave the config
    // uninitialized (not setting the 'type' param) can be used to not use any
    // classification.
    config_utilities::VariableConfig<ClassLayer> classification;

    voxblox::OccTsdfIntegrator::Config occ_tsdf_integrator_config;
    
    // ESDF integrator options
    voxblox::EsdfVoxfieldIntegrator::Config esdf_voxfield_integrator_config;
    voxblox::EsdfIntegrator::Config esdf_voxblox_integrator_config;
    voxblox::EsdfOccEdtIntegrator::Config esdf_edt_integrator_config;
    voxblox::EsdfOccFiestaIntegrator::Config esdf_fiesta_integrator_config;
    
    // ESDF mapping
    std::string esdf_integrator_name = "voxfield";
    bool esdf_patch_on = true;
    bool esdf_early_break = false;
    bool finer_esdf_on = true;
    bool fixed_band_esdf_on = false;
    float tsdf_min_weight = 1e-6;
    float occ_voxel_size_ratio = 0.865;
    float fixed_band_ratio = 1.0;
    int esdf_local_offset_x = 20;
    int esdf_local_offset_y = 20;
    int esdf_local_offset_z = 5;
    float esdf_max_distance_m = 5.0f;
    float gradient_sign = -1.0f;
    int num_buckets = 50;
    
    Config() { setConfigName("Submap"); }

    // Utility tool that checks whether a classification layer was specified.
    bool useClassLayer() const;

   protected:
    void setupParamsAndPrinting() override;
    void checkParams() const override;
    void initializeDependentVariableDefaults() override;
  };

  // Construction.
  explicit Submap(
      const Config& config,
      SubmapIDManager* submap_id_manager = SubmapIDManager::getGlobalInstance(),
      InstanceIDManager* instance_id_manager =
          InstanceIDManager::getGlobalInstance());
  // set id_manager
  
  virtual ~Submap() = default;

  // Const accessors.
  const Config& getConfig() const { return config_; }
  int getID() const { return id_; }
  int getInstanceID() const { return instance_id_; }
  int getClassID() const { return class_id_; }
  PanopticLabel getLabel() const { return label_; }
  const std::string& getName() const { return name_; }
  const std::string& getFrameName() const { return frame_name_; }
  const TsdfLayer& getTsdfLayer() const { return *tsdf_layer_; }
  const ClassLayer& getClassLayer() const { return *class_layer_; }
  const EsdfLayer& getEsdfLayer() const { return *esdf_layer_; }
  const OccLayer& getOccLayer() const { return *occ_layer_; }
  const voxblox::MeshLayer& getMeshLayer() const { return *mesh_layer_; }
  const Transformation& getT_M_S() const { return T_M_S_; }
  const Transformation& getT_S_M() const { return T_M_S_inv_; }
  bool isActive() const { return is_active_; }
  bool wasTracked() const { return was_tracked_; }
  bool hasClassLayer() const { return has_class_layer_; }
  bool getNormalReliability() const {return normal_reliable_;}
  const std::vector<IsoSurfacePoint>& getIsoSurfacePoints() const {
    return iso_surface_points_;
  }
  ChangeState getChangeState() const { return change_state_; }
  const SubmapBoundingVolume& getBoundingVolume() const {
    return bounding_volume_;
  }

  // Modifying accessors.
  std::shared_ptr<TsdfLayer>& getTsdfLayerPtr() { return tsdf_layer_; }
  std::shared_ptr<ClassLayer>& getClassLayerPtr() { return class_layer_; }
  std::shared_ptr<EsdfLayer>& getEsdfLayerPtr() { return esdf_layer_; }
  std::shared_ptr<OccLayer>& getOccLayerPtr() { return occ_layer_; }
  std::shared_ptr<voxblox::MeshLayer>& getMeshLayerPtr() { return mesh_layer_; }
  std::vector<IsoSurfacePoint>* getIsoSurfacePointsPtr() {
    return &iso_surface_points_;
  }
  SubmapBoundingVolume* getBoundingVolumePtr() { return &bounding_volume_; }

  // Setters.
  void setT_M_S(const Transformation& T_M_S);
  void setInstanceID(int id) { instance_id_ = id; }
  void setClassID(int id) { class_id_ = id; }
  void setLabel(PanopticLabel label) { label_ = label; }
  void setName(const std::string& name) { name_ = name; }
  void setFrameName(const std::string& name) { frame_name_ = name; }
  void setChangeState(ChangeState state) { change_state_ = state; }
  void setIsActive(bool is_active) { is_active_ = is_active; }
  void setWasTracked(bool was_tracked) { was_tracked_ = was_tracked; }
  void setNormalReliability(bool normal_reliable) {normal_reliable_ = normal_reliable;}
  void setHasEsdf(bool has_esdf) {has_esdf_layer_ = has_esdf;}
  // Processing.
  /**
   * @brief Set the submap status to inactive and update its status accordingly.
   */
  void finishActivePeriod(bool update_all = true);

  /**
   * @brief Update all dynamically computable quantities.
   *
   * @param only_updated_blocks If false, recompute all quantities from scratch.
   * If true, recompute based on what is flagged updated.
   */
  void updateEverything(bool only_updated_blocks = true);

  /**
   * @brief Update the bounding volume based on all allocated blocks.
   */
  void updateBoundingVolume();

  /**
   * @brief Update the mesh based on the current tsdf blocks. Set
   * only_updated_blocks true for incremental mesh updates, false for a full
   * re-computation.
   *
   * @param only_updated_blocks If false, recompute the mesh from scratch. If
   * true, update based on the updated(kMesh) flag of the TSDF layer.
   * @param use_class_layer Set to true to use the class layer if it is
   * available.
   */
  void updateMesh(bool only_updated_blocks = true, bool use_class_layer = true);

  void saveMesh(std::string folder_path);

  void updateOccFromTsdf(bool clear_updated_flag_occ = true, bool in_batch = false) const;

  void updateEsdfFromOcc(bool clear_updated_flag_esdf = true) const;

  void updateEsdfFromTsdf(bool clear_updated_flag_esdf = true) const;

  /**
   * @brief Compute the iso-surface points of the submap based on its current
   * mesh. Currently all surface points are computed from scratch every time,
   * but since they are currently only computed when a submap is finished it
   * should be fine. This function utilizes the stored mesh so make sure
   * updateMesh is called earlier.
   */
  void computeIsoSurfacePoints();

  /**
   * @brief Removes non-belonging points from the TSDF and deletes the class
   * layer. Uses the provided manipulator to perform the class layer
   * integration.
   *
   * @param manipulator Manipulator used to carry out the application
   * of the class layer.
   * @param clear_class_layer True: erase the class layer. False: keep the class
   * layer for lookups, but no further manipulations.
   * @return True if any blocks remain, false if the TSDF map was cleared.
   */
  bool applyClassLayer(const LayerManipulator& manipulator,
                       bool clear_class_layer = true);

  /**
   * @brief Create a deep copy of the submap. Notice that new submapID and
   * instanceID managers need to be provided to not corrupt the ID counts. ID
   * counts will not be double checked so use with care.
   *
   * @param submap_id_manager Pointer to the new submapID manager that holds
   * this submap.
   * @param instance_id_manager Pointer to the new instanceID manager that holds
   * this submap.
   * @return Unique pointer holding the copy.
   */
  std::unique_ptr<Submap> clone(SubmapIDManager* submap_id_manager,
                                InstanceIDManager* instance_id_manager) const;

 private:
  friend class SubmapCollection;
  const Config config_;

  // This constructor is intended to allow deep copies of the submap collection,
  // moving the id to the new id managers.
  Submap(const Config& config, SubmapIDManager* submap_id_manager,
         InstanceIDManager* instance_id_manager, int submap_id);

  // Setup.
  void initialize();

  // IO.
  /**
   * @brief Serialize the submap to protobuf.
   *
   * @param proto The output protobuf object.
   */
  void getProto(SubmapProto* proto) const;

  /**
   * @brief Save the submap to file.
   *
   * @param outfile_ptr The file to write the protobuf data to.
   * @return Success of the saving operation.
   */
  bool saveToStream(std::fstream* outfile_ptr) const;
  
  /**
   * @brief Load the submap from file.
   *
   * @param proto_file_ptr File from where to read the protobuf data. // defined in proto
   * @param tmp_byte_offset_ptr Byte offset result, used to keep track where we
   * are in the file if necessary. NOTE(schmluk): Mostly unused, initialize to
   * 0.
   * @param id_manager Submap ID manager of the collection to laod the submap
   * into.
   * @param instance_manager Instance ID manager of the collection to laod the
   * submap into.
   * @return Unique pointer to the loaded submap.
   */
  static std::unique_ptr<Submap> loadFromStream(
      std::istream* proto_file_ptr, uint64_t* tmp_byte_offset_ptr,
      SubmapIDManager* id_manager = SubmapIDManager::getGlobalInstance(),
      InstanceIDManager* instance_manager =
          InstanceIDManager::getGlobalInstance());

  // Labels.
  const SubmapID id_;       // UUID
  InstanceID instance_id_;  // Per default sets up a new unique ID.
  int class_id_ = -1;       // semantic class id
  PanopticLabel label_ = PanopticLabel::kUnknown;
  std::string name_ = "Unknown";
  bool normal_reliable_ = true;
  int plant_id_ = -1;       // to which plant does it belong (only used for PhenoRob stuffs)

  bool countable_;  // can have instance id or not

  // State.
  bool is_active_ = true;
  bool was_tracked_ = true;  // Set to true by the id tracker if matched.
  bool has_class_layer_ = false;
  bool has_esdf_layer_ = false; // at present, only free space submap has the esdf layer
  ChangeState change_state_ = ChangeState::kNew;

  // Transformations.
  std::string frame_name_;
  Transformation T_M_S_;  // Transformation from world(mission) to submap, currently this is set to be identity transformation. 
  Transformation T_M_S_inv_;

  // Map.
  std::shared_ptr<TsdfLayer> tsdf_layer_;
  std::shared_ptr<ClassLayer> class_layer_;
  std::shared_ptr<MeshLayer> mesh_layer_;
  std::shared_ptr<OccLayer> occ_layer_;
  std::shared_ptr<EsdfLayer> esdf_layer_;
  std::vector<IsoSurfacePoint> iso_surface_points_;
  SubmapBoundingVolume bounding_volume_;

  // Processing Integrators.
  std::unique_ptr<MeshIntegrator> mesh_integrator_;
  std::unique_ptr<voxblox::OccTsdfIntegrator> occ_integrator_;
  std::unique_ptr<voxblox::EsdfVoxfieldIntegrator> esdf_voxfield_integrator_;
  std::unique_ptr<voxblox::EsdfIntegrator> esdf_voxblox_integrator_;
  std::unique_ptr<voxblox::EsdfOccEdtIntegrator> esdf_edt_integrator_;
  std::unique_ptr<voxblox::EsdfOccFiestaIntegrator> esdf_fiesta_integrator_;
};

}  // namespace panoptic_mapping

#endif  // PANOPTIC_MAPPING_MAP_SUBMAP_H_
