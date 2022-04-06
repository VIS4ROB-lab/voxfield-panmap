#include "panoptic_mapping_ros/panoptic_mapper.h"

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <panoptic_mapping/common/common.h>
#include <panoptic_mapping/common/camera.h>
#include <panoptic_mapping/labels/label_handler_base.h>
#include <panoptic_mapping/submap_allocation/freespace_allocator_base.h>
#include <panoptic_mapping/submap_allocation/submap_allocator_base.h>
#include <voxblox/io/layer_io_inl.h>


namespace panoptic_mapping {

// Modules that don't have a default type will be required to be explicitly set.
// Entries: <key, <ros_namespace, default type parameter>.
// All the keys needed to be set
const std::map<std::string, std::pair<std::string, std::string>>
    PanopticMapper::default_names_and_types_ = { // TODO(py): update
        {"lidar", {"lidar", ""}},
        {"camera", {"camera", ""}},
        {"label_handler", {"labels", "null"}},
        {"submap_allocator", {"submap_allocator", "null"}},
        {"freespace_allocator", {"freespace_allocator", "null"}},
        {"id_tracker", {"id_tracker", ""}},
        {"tsdf_integrator", {"tsdf_integrator", ""}},
        {"map_management", {"map_management", "null"}},
        {"vis_submaps", {"visualization/submaps", "submaps"}},
        {"vis_tracking", {"visualization/tracking", ""}},
        {"vis_planning", {"visualization/planning", ""}},
        {"data_writer", {"data_writer", "null"}}};
        // {"map_evaluator", {"map_evaluator", ""}}};

void PanopticMapper::Config::checkParams() const {
  checkParamCond(!global_frame_name.empty(),
                 "'global_frame_name' may not be empty.");
  checkParamGT(ros_spinner_threads, 1, "ros_spinner_threads");
  checkParamGT(check_input_interval, 0.f, "check_input_interval");
}

void PanopticMapper::Config::setupParamsAndPrinting() {
  setupParam("verbosity", &verbosity);
  setupParam("global_frame_name", &global_frame_name);
  setupParam("robot_mesh_file", &robot_mesh_file);
  setupParam("robot_mesh_scale", &robot_mesh_scale);
  setupParam("robot_frame_name", &robot_frame_name);
  setupParam("visualization_interval", &visualization_interval, "s");
  setupParam("data_logging_interval", &data_logging_interval, "s");
  setupParam("print_timing_interval", &print_timing_interval, "s");
  setupParam("esdf_update_interval", &esdf_update_interval, "s");
  setupParam("robot_mesh_interval", &robot_mesh_interval, "s");
  setupParam("use_threadsafe_submap_collection",
             &use_threadsafe_submap_collection);
  setupParam("ros_spinner_threads", &ros_spinner_threads);
  setupParam("check_input_interval", &check_input_interval, "s");
  setupParam("load_submaps_conservative", &load_submaps_conservative);
  setupParam("shutdown_when_finished", &shutdown_when_finished);
  setupParam("save_map_path_when_finished", &save_map_path_when_finished);
  setupParam("display_config_units", &display_config_units);
  setupParam("indicate_default_values", &indicate_default_values);
  setupParam("input_from_bag", &input_from_bag);
  setupParam("use_lidar", &use_lidar);
  setupParam("use_range_image", &use_range_image);
  setupParam("estimate_normal", &estimate_normal);
  setupParam("filter_moving_objects", &filter_moving_objects);
  setupParam("use_panoptic_color", &use_panoptic_color);
  setupParam("msg_latency_s", &msg_latency_s);
}

PanopticMapper::PanopticMapper(const ros::NodeHandle& nh,
                               const ros::NodeHandle& nh_private)
    : nh_(nh),
      nh_private_(nh_private),
      config_(
          config_utilities::getConfigFromRos<PanopticMapper::Config>(nh_private)
              .checkValid()) {
  // Setup printing of configs.
  // NOTE(schmluk): These settings are global so multiple panoptic mappers in
  // the same process might interfere.
  config_utilities::GlobalSettings().indicate_default_values =
      config_.indicate_default_values;
  config_utilities::GlobalSettings().indicate_units =
      config_.display_config_units;
  LOG_IF(INFO, config_.verbosity >= 1) << "\n" << config_.toString();

  // Setup all components of the panoptic mapper.
  setupMembers();
  setupRos();

}

void PanopticMapper::setupMembers() {
  // Map.
  submaps_ = std::make_shared<SubmapCollection>();

  // Threadsafe wrapper for the map.
  thread_safe_submaps_ = std::make_shared<ThreadSafeSubmapCollection>(submaps_);

  // Depth Camera. 
  auto camera = std::make_shared<Camera>(
      config_utilities::getConfigFromRos<Camera::Config>(defaultNh("camera")));
  
  // Lidar.
  auto lidar = std::make_shared<Lidar>(
      config_utilities::getConfigFromRos<Lidar::Config>(defaultNh("lidar")));

  // Label Handler.
  std::shared_ptr<LabelHandlerBase> label_handler =
      config_utilities::FactoryRos::create<LabelHandlerBase>(
          defaultNh("label_handler"));

  // Globals.
  if(config_.use_lidar)
    globals_ = std::make_shared<Globals>(lidar, label_handler); //lidar version
  else
    globals_ = std::make_shared<Globals>(camera, label_handler); //depth cam version
  
  // Submap Allocation.
  std::shared_ptr<SubmapAllocatorBase> submap_allocator =
      config_utilities::FactoryRos::create<SubmapAllocatorBase>(
          defaultNh("submap_allocator"));
  std::shared_ptr<FreespaceAllocatorBase> freespace_allocator =
      config_utilities::FactoryRos::create<FreespaceAllocatorBase>(
          defaultNh("freespace_allocator"));
  
  // Transformer.
  transformer_ = std::make_shared<Transformer>(nh_, nh_private_);
     
  // ID Tracking.
  id_tracker_ = config_utilities::FactoryRos::create<IDTrackerBase>(
      defaultNh("id_tracker"), globals_);
  id_tracker_->setSubmapAllocator(submap_allocator);
  id_tracker_->setFreespaceAllocator(freespace_allocator);

  // Tsdf Integrator.
  tsdf_integrator_ = config_utilities::FactoryRos::create<TsdfIntegratorBase>(
      defaultNh("tsdf_integrator"), globals_);

  // Map Manager.
  map_manager_ = config_utilities::FactoryRos::create<MapManagerBase>(
      defaultNh("map_management"));

  // Visualization.
  ros::NodeHandle visualization_nh(nh_private_, "visualization");

  // Submaps.
  submap_visualizer_ = config_utilities::FactoryRos::create<SubmapVisualizer>(
      defaultNh("vis_submaps"), globals_);
  submap_visualizer_->setGlobalFrameName(config_.global_frame_name);

  // Tracking.
  tracking_visualizer_ = std::make_unique<TrackingVisualizer>(
      config_utilities::getConfigFromRos<TrackingVisualizer::Config>(
          defaultNh("vis_tracking")));
  tracking_visualizer_->registerIDTracker(id_tracker_.get());

  // Planning. TODO(py):
  setupCollectionDependentMembers();

  // Data Logging.
  data_logger_ = config_utilities::FactoryRos::create<DataWriterBase>(
      defaultNh("data_writer"));

  // Evaluator.
  // map_evaluator_ = config_utilities::FactoryRos::create<MapEvaluator>(
  //     defaultNh("map_evaluator"));

  robot_mesh_pub_ = 
       nh_.advertise<visualization_msgs::Marker>("Robot_mesh", 100);

  // Input subscriber (from rosbag)
  if (config_.input_from_bag) {
    int pointcloud_queue_size_ = 20; //2s // problem could be here
    pointcloud_sub_ = nh_.subscribe("pointcloud", pointcloud_queue_size_,
                                    &PanopticMapper::insertPointcloud, this);



    // int pose_queue_size_ = 200;
    // transform_sub_ = nh_.subscribe("pose", pose_queue_size_,
    //                                 &PanopticMapper::publishRobotMesh, this);                               
  }
  
  // Setup all requested inputs from all modules.
  InputData::InputTypes requested_inputs; //set
  std::vector<InputDataUser*> input_data_users = {
      id_tracker_.get(), tsdf_integrator_.get(), submap_allocator.get(),
      freespace_allocator.get()}; //class that use the input data

  for (const InputDataUser* input_data_user : input_data_users) {
    requested_inputs.insert(input_data_user->getRequiredInputs().begin(),
                            input_data_user->getRequiredInputs().end());
  }
  
  compute_vertex_map_ = // kVertexMap is requested
      requested_inputs.find(InputData::InputType::kVertexMap) !=
      requested_inputs.end();
  compute_validity_image_ = // kValidityImage is requested
      requested_inputs.find(InputData::InputType::kValidityImage) !=
      requested_inputs.end();

  // Setup the input synchronizer.
  input_synchronizer_ = std::make_unique<InputSynchronizer>(
      config_utilities::getConfigFromRos<InputSynchronizer::Config>(
          nh_private_),  nh_);
  input_synchronizer_->requestInputs(requested_inputs);
}

void PanopticMapper::setupRos() {
  // Setup all input topics.
  input_synchronizer_->advertiseInputTopics();

  // Services.
  save_map_srv_ = nh_private_.advertiseService(
      "save_map", &PanopticMapper::saveMapCallback, this);
  load_map_srv_ = nh_private_.advertiseService(
      "load_map", &PanopticMapper::loadMapCallback, this);
  set_visualization_mode_srv_ = nh_private_.advertiseService(
      "set_visualization_mode", &PanopticMapper::setVisualizationModeCallback,
      this); //change the visualization mode
  print_timings_srv_ = nh_private_.advertiseService(
      "print_timings", &PanopticMapper::printTimingsCallback, this);
  finish_mapping_srv_ = nh_private_.advertiseService(
      "finish_mapping", &PanopticMapper::finishMappingCallback, this);
  save_mesh_srv_ = nh_private_.advertiseService(
      "save_mesh", &PanopticMapper::saveMeshCallback, this);
  save_free_esdf_srv_ = nh_private_.advertiseService(
      "save_esdf_map", &PanopticMapper::saveFreeEsdfCallback, this);
  // evaluate_map_srv_ = nh_private_.advertiseService(
  //     "evaluate_map", &MapEvaluator::evaluate, this);

  // set rainbow colormap
  color_map_.reset(new RainbowColorMap());
 
  // Timers.
  if (config_.visualization_interval > 0.0) { // <0, then the timer would be based on frame instead of second
    visualization_timer_ = nh_private_.createTimer(
        ros::Duration(config_.visualization_interval),
        &PanopticMapper::publishVisualizationCallback, this); //Visualization update
  }
  if (config_.data_logging_interval > 0.0) {
    data_logging_timer_ =
        nh_private_.createTimer(ros::Duration(config_.data_logging_interval),
                                &PanopticMapper::dataLoggingCallback, this);
  }
  if (config_.print_timing_interval > 0.0) {
    print_timing_timer_ =
        nh_private_.createTimer(ros::Duration(config_.print_timing_interval),
                                &PanopticMapper::printTimingsCallback, this);
  }
  if (config_.esdf_update_interval > 0.0) {
    update_esdf_timer_ =
        nh_private_.createTimer(ros::Duration(config_.esdf_update_interval),
                                &PanopticMapper::updateFreeEsdfCallback, this);
  }
  if (config_.robot_mesh_interval > 0.0) { //
    robot_mesh_timer_ = nh_private_.createTimer(
        ros::Duration(config_.robot_mesh_interval),
        &PanopticMapper::publishRobotMeshCallback, this);
  }

  //input data not from rosbag (mainly for rgbd data)
  if (!config_.input_from_bag)
    input_timer_ =
        nh_private_.createTimer(ros::Duration(config_.check_input_interval),
                                &PanopticMapper::inputCallback, this);
}

// Import data from data folder instead of the rosbag
void PanopticMapper::inputCallback(const ros::TimerEvent&) {
  if (input_synchronizer_->hasInputData()) {
    std::shared_ptr<InputData> data = input_synchronizer_->getInputData();
    if (data) {
      processInput(data.get());
      if (config_.shutdown_when_finished) {
        last_input_ = ros::Time::now();
        got_a_frame_ = true;
      }
    }
  } else {
    if (config_.shutdown_when_finished && got_a_frame_ &&
        (ros::Time::now() - last_input_).toSec() >= 3.0) {
      // No more frames, finish up.
      // TODO(py): take a close look at the finishMapping function
      LOG_IF(INFO, config_.verbosity >= 1)
          << "No more frames received for 3 seconds, shutting down.";
      finishMapping();
      if (!config_.save_map_path_when_finished.empty()) {
        saveMap(config_.save_map_path_when_finished);
      }
      LOG_IF(INFO, config_.verbosity >= 1) << "Finished.";
      ros::shutdown();
    }
  }
}

// Import point cloud from rosbag
void PanopticMapper::insertPointcloud(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg_in) {
  if (pointcloud_msg_in->header.stamp - last_msg_time_ptcloud_ >
      min_time_between_msgs_) {
    last_msg_time_ptcloud_ = pointcloud_msg_in->header.stamp;
    // So we have to process the queue anyway... Push this back.
    pointcloud_queue_.push(pointcloud_msg_in);
  }

  Transformation T_G_C;
  sensor_msgs::PointCloud2::Ptr pointcloud_msg;
  bool processed_any = false;
  // Get input data and process the data once the data is ready
  while (
      getNextPointcloudFromQueue(&pointcloud_queue_, &pointcloud_msg, &T_G_C)) {
    constexpr bool is_freespace_pointcloud = false;
    processPointCloudMessageAndInsert(pointcloud_msg, T_G_C,
                                      is_freespace_pointcloud);
    processed_any = true;
  }

  if (!processed_any) {
    return;
  }
}

// Checks if we can get the next message from queue.
bool PanopticMapper::getNextPointcloudFromQueue(
    std::queue<sensor_msgs::PointCloud2::Ptr>* queue,
    sensor_msgs::PointCloud2::Ptr* pointcloud_msg, Transformation* T_G_C) {
  const size_t kMaxQueueSize = 10;
  if (queue->empty()) {
    return false;
  }
  *pointcloud_msg = queue->front();
  if (transformer_->lookupTransform((*pointcloud_msg)->header.frame_id,
                                     config_.global_frame_name,      
                                     (*pointcloud_msg)->header.stamp,       
                                     T_G_C)) {                                         
    queue->pop();
    return true;
  } else {
    if (queue->size() >= kMaxQueueSize) {
      ROS_ERROR_THROTTLE(60,
                         "Input pointcloud queue getting too long! Dropping "
                         "some pointclouds. Either unable to look up transform "
                         "timestamps or the processing is taking too long.");
      while (queue->size() >= kMaxQueueSize) {
        queue->pop();
      }
    }
  }
  return false;
}

// Preprocess the imported point cloud 
void PanopticMapper::processPointCloudMessageAndInsert(
    const sensor_msgs::PointCloud2::Ptr& pointcloud_msg,
    const Transformation& T_G_C, const bool is_freespace_pointcloud) {
  // Convert the PCL pointcloud into our awesome format.

  // Horrible hack fix to fix color parsing colors in PCL.
  bool color_pointcloud = false;
  bool has_intensity = false;
  bool has_label = false;  
  for (size_t d = 0; d < pointcloud_msg->fields.size(); ++d) {
    if (pointcloud_msg->fields[d].name == std::string("rgb")) {
      pointcloud_msg->fields[d].datatype = sensor_msgs::PointField::FLOAT32;
      color_pointcloud = true;
    } else if (pointcloud_msg->fields[d].name == std::string("intensity")) {
      has_intensity = true;
    } else if (pointcloud_msg->fields[d].name == std::string("label")) { 
      has_label = true;
      if (config_.verbosity >= 3) 
        ROS_INFO("Found semantic/instance label in the point cloud");
    }
  }

  Timer pre_timer("preprocess");
  Timer lidar_pre_timer("preprocess/point_cloud");

  Pointcloud points_C;
  Colors colors;
  Labels labels;

  // We need the panoptic labels for the panoptic mapping
  if (has_label) {
    pcl::PointCloud<pcl::PointXYZRGBL>::Ptr 
        pointcloud_pcl(new pcl::PointCloud<pcl::PointXYZRGBL>());
    // pointcloud_pcl is modified below:
    pcl::fromROSMsg(*pointcloud_msg, *pointcloud_pcl);
    convertPointcloud(*pointcloud_pcl, color_map_, &points_C, &colors, &labels, 
                      config_.filter_moving_objects,
                      config_.use_panoptic_color, 19);
    pointcloud_pcl.reset(new pcl::PointCloud<pcl::PointXYZRGBL>());
    // record labels to the LabelHandler 
    globals_->labelHandler()->assignLabelsSemanticKITTI(labels);
  } // should also allow the case when no label is available
  else if (color_pointcloud) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr 
        pointcloud_pcl(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(*pointcloud_msg, *pointcloud_pcl);
    convertPointcloud(*pointcloud_pcl, color_map_, &points_C, &colors);
  } 
  else if (has_intensity) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr 
        pointcloud_pcl(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg(*pointcloud_msg, *pointcloud_pcl);
    convertPointcloud(*pointcloud_pcl, color_map_, &points_C, &colors);
  } 
  else {
    pcl::PointCloud<pcl::PointXYZ>::Ptr 
        pointcloud_pcl(new pcl::PointCloud<pcl::PointXYZ>());
    // pointcloud_pcl is modified below:
    pcl::fromROSMsg(*pointcloud_msg, *pointcloud_pcl);
    convertPointcloud(*pointcloud_pcl, color_map_, &points_C, &colors);
  }

  lidar_pre_timer.Stop();

  // Set input data
  std::shared_ptr<InputData> data (new InputData);
  
  // Mandatory information: pose and point cloud
  data->setT_M_C(T_G_C);

  if (!config_.use_range_image) {
    data->setPointCloud(points_C);
    data->setPointColor(colors);
  } else {
    Timer range_pre_timer("preprocess/range_image");
    // Preprocess the point cloud
    // convert to range images
    cv::Mat vertex_map;
    if (config_.use_lidar) {
      vertex_map = cv::Mat::zeros(globals_->lidar()->getConfig().height, 
                                  globals_->lidar()->getConfig().width, 
                                  CV_32FC3);
    } else {
      vertex_map = cv::Mat::zeros(globals_->camera()->getConfig().height, 
                                  globals_->camera()->getConfig().width, 
                                  CV_32FC3);
    }
    cv::Mat depth_image(vertex_map.size(), CV_32FC1, -1.0);

    cv::Mat id_image(vertex_map.size(), CV_32SC1, -1); //-1: none-measurement
                              
    cv::Mat color_image = cv::Mat::zeros(vertex_map.size(), CV_8UC3);

    cv::Mat normal_image;

    if (config_.use_lidar) {                    
      globals_->lidar()->projectPointCloudToImagePlane(points_C, colors, labels,
                                                       vertex_map, depth_image,  
                                                       color_image, id_image);     
      if (config_.estimate_normal)                                                                                                  
        normal_image = globals_->lidar()->computeNormalImage(vertex_map, depth_image);       
    } else {
      globals_->camera()->projectPointCloudToImagePlane(points_C, colors, labels,
                                                        vertex_map, depth_image,  
                                                        color_image, id_image);     
      if (config_.estimate_normal)                                                                                                  
        normal_image = globals_->camera()->computeNormalImage(vertex_map, depth_image);    
    }

    // Additional information
    data->setVertexMap(vertex_map);
    data->setDepthImage(depth_image);
    data->setColorImage(color_image);
    data->setIdImage(id_image);
    if (config_.estimate_normal)  
      data->setNormalImage(normal_image);

    range_pre_timer.Stop();

    // TODO(py): Densify and refine, flood filling 
    // floodfill reference: semantic_suma/src/shader/floodfill.frag 
  }

  pre_timer.Stop();

  // Main entrance of each frame !!!
  processInput(data.get());

  // free the memory
  //Timer post_timer("postprocess");
  data.reset(new InputData);
  Pointcloud().swap(points_C);
  Colors().swap(colors);
  Labels().swap(labels);

  // vertex_map.release();
  // depth_image.release();
  // id_image.release();
  // color_image.release();
  // normal_image.release();
  // post_timer.Stop();
}

void PanopticMapper::setupCollectionDependentMembers() {
  // Planning Interface.
  planning_interface_ = std::make_shared<PlanningInterface>(submaps_);

  // Planning Visualizer.
  planning_visualizer_ = std::make_unique<PlanningVisualizer>(
      config_utilities::getConfigFromRos<PlanningVisualizer::Config>(
          defaultNh("vis_planning")),
      planning_interface_);
  planning_visualizer_->setGlobalFrameName(config_.global_frame_name);
}

// main entrance for panoptic mapping processing 
void PanopticMapper::processInput(InputData* input) {
  
  CHECK_NOTNULL(input);
  frame_count_ ++;

  if (config_.verbosity >= 2)
    ROS_INFO("Process frame %d", frame_count_);

  Timer timer("process");
  frame_timer_ = std::make_unique<Timer>("frame");

  ros::WallTime t00 = ros::WallTime::now();
  // Compute and store the validity image.
  if (config_.use_range_image) {
    if (compute_validity_image_) {
      //Timer validity_timer("process/compute_validity_image");
      if(config_.use_lidar) {
        input->setValidityImage(
          globals_->lidar()->computeValidityImage(input->depthImage()));
      }
      else {
        input->setValidityImage(
          globals_->camera()->computeValidityImage(input->depthImage()));
      }
      //validity_timer.Stop()
    }
    // Compute and store the vertex map.
    if (compute_vertex_map_) { 
      if (!config_.use_lidar && !config_.input_from_bag) {
        Timer vertex_timer("process/compute_vertex_map");
        input->setVertexMap(
            globals_->camera()->computeVertexMap(input->depthImage()));
        vertex_timer.Stop();

        if (input->idImage().rows == 0) { // IdImage not available
          cv::Mat id_image(input->depthImage().size(), CV_32SC1, -1); // -1: none-measurement, default no label
          input->setIdImage(id_image);
        }

        if (config_.estimate_normal) {
          // compute the normal image (could take a long time)
          Timer normal_timer("process/compute_normal_image");
          input->setNormalImage(
              globals_->camera()->computeNormalImage(input->vertexMap(),
                                                    input->depthImage()));
          normal_timer.Stop();
        }
      }
    }
  }

  ros::WallTime t0 = ros::WallTime::now();

  // Track the segmentation images and allocate new submaps.
  // On each range image, each pixel is assigned with a submap id
  // each submap has its unique id and its category id
  // Upon visualization, we can choose to render with the semantic label or the 
  // submap id 

  Timer id_timer("process/id_tracking");
  id_tracker_->processInput(submaps_.get(), input); //input of this frame
  ros::WallTime t1 = ros::WallTime::now();
  id_timer.Stop();
  // NOTE(py): image visualization time also involved in process/id_tracking

  // Integrate the images.
  Timer tsdf_timer("process/tsdf_integration");
  tsdf_integrator_->processInput(submaps_.get(), input);
  ros::WallTime t2 = ros::WallTime::now();
  tsdf_timer.Stop();

  // Perform all requested map management actions.
  // NOTE(py): use umap in order to have the correct order for time log output
  Timer management_timer("process/umap_management"); 
  map_manager_->tick(submaps_.get());
  ros::WallTime t3 = ros::WallTime::now();
  management_timer.Stop();

  // Update freespace ESDF map for planning applications.
  if (config_.esdf_update_interval < 0.f 
      && frame_count_ % int(config_.esdf_update_interval) == 0) {
    Timer esdf_timer("process/esdf_mapping");
    updateFreeEsdfCallback(ros::TimerEvent());
    esdf_timer.Stop();
  }

  // If requested perform visualization and logging.
  if (config_.visualization_interval < 0.f 
      && frame_count_ % int(config_.visualization_interval) == 0) {
    Timer vis_timer("process/visualization");
    publishVisualizationCallback(ros::TimerEvent());
    vis_timer.Stop();
  }
  if (config_.data_logging_interval < 0.f) {
    dataLoggingCallback(ros::TimerEvent());
  }

  ros::WallTime t4 = ros::WallTime::now();

  // If requested update the thread_safe_submaps.
  // TODO(py): figure out the memory leakage problem
  if (config_.use_threadsafe_submap_collection) {
    thread_safe_submaps_->update(); //too time consuming here
  }

  // Logging.
  timer.Stop();
  //some time are spent on the validity image and vertex map computation
  std::stringstream info;
  if (config_.verbosity >= 3) {
    info << "Processed input data.";
    info << "\n(preprocessing: " << int((t0 - t00).toSec() * 1000)
         << " + tracking: " << int((t1 - t0).toSec() * 1000)
         << " + integration: " << int((t2 - t1).toSec() * 1000)
         << " + management: " << int((t3 - t2).toSec() * 1000);
    if (config_.visualization_interval <= 0.f) {
      info << " + visualization: " << int((t4 - t3).toSec() * 1000);
    }
    info << " = " << int((t4 - t00).toSec() * 1000) << " ms, frame: "
         << static_cast<int>(
                (ros::WallTime::now() - previous_frame_time_).toSec() * 1000)
         << "ms)";
  }
  previous_frame_time_ = ros::WallTime::now();
  LOG_IF(INFO, config_.verbosity >= 2) << info.str();
  
  //ROS_INFO_STREAM("Timings: " << std::endl << Timing::Print());
  LOG_IF(INFO, config_.print_timing_interval < 0.0) << "\n" << Timing::Print();

}

void PanopticMapper::finishMapping() {
  LOG_IF(INFO, config_.verbosity >= 1) << "Finished mapping.";
  map_manager_->finishMapping(submaps_.get());
  submap_visualizer_->visualizeAll(submaps_.get());
}

void PanopticMapper::publishRobotMesh() {
  Transformation robot_pose;
  ros::Time cur_time = ros::Time::now();
  int ns_in_s = 1000000000;
  int msg_latency_ns = (int)(config_.msg_latency_s * ns_in_s);
  if (cur_time.nsec > msg_latency_ns)
    cur_time.nsec -= msg_latency_ns;
  else {
    cur_time.sec -= 1;
    cur_time.nsec += (ns_in_s - msg_latency_ns); 
  }

  transformer_->lookupTransform(config_.robot_frame_name,
                                config_.global_frame_name,      
                                cur_time,
                                &robot_pose, true);

  // publish the robot model with the pose
  visualization_msgs::Marker robot_mesh;
  robot_mesh.header.frame_id = config_.global_frame_name;
  robot_mesh.header.stamp = ros::Time();
  robot_mesh.mesh_resource = "file://" + config_.robot_mesh_file;
  robot_mesh.mesh_use_embedded_materials = true;
  robot_mesh.scale.x = robot_mesh.scale.y = robot_mesh.scale.z = config_.robot_mesh_scale;
  robot_mesh.lifetime = ros::Duration();
  robot_mesh.action = visualization_msgs::Marker::MODIFY;
  robot_mesh.color.a = robot_mesh.color.r = robot_mesh.color.g = robot_mesh.color.b = 1.;
  robot_mesh.type =  visualization_msgs::Marker::MESH_RESOURCE;
  
  Eigen::Quaternionf quatrot = robot_pose.getEigenQuaternion();
  // Eigen::Quaternionf quat45(0.924, 0.0, 0.0, 0.383); // rotation by 45 degree around z axis
  // quatrot = quatrot * quat45;
  
  Point quat_vec = quatrot.vec();
  robot_mesh.pose.orientation.x = quat_vec(0);
  robot_mesh.pose.orientation.y = quat_vec(1);
  robot_mesh.pose.orientation.z = quat_vec(2);
  robot_mesh.pose.orientation.w = quatrot.w();
  
  Point translation = robot_pose.getPosition();
  robot_mesh.pose.position.x = translation(0);
  robot_mesh.pose.position.y = translation(1);
  robot_mesh.pose.position.z = translation(2);

  //LOG(INFO) << "drone position: (" << translation(0) << "," << translation(1) << "," << translation(2) << ")";
  robot_mesh_pub_.publish(robot_mesh);
}

void PanopticMapper::publishVisualization() {
  Timer timer("visualization");

  submap_visualizer_->visualizeAll(submaps_.get());
  planning_visualizer_->visualizeAll();
}

void PanopticMapper::updateFreeEsdfCallback(const ros::TimerEvent& /*event*/) {
  //ROS_INFO("Update FreeEsdf");
  updateFreeEsdfFromTsdf();
}

// incrementally update ESDF map from the updated TSDF map
void PanopticMapper::updateFreeEsdfFromTsdf() {
  const int free_space_id = submaps_->getActiveFreeSpaceSubmapID();
  if (submaps_->submapIdExists(free_space_id)) {
    Submap* free_space_submap = submaps_->getSubmapPtr(free_space_id);
    if(free_space_submap->getTsdfLayer().getNumberOfAllocatedBlocks() > 0){
      if (config_.verbosity > 3)
        ROS_INFO("Update Esdf layer from Tsdf layer");
      free_space_submap->updateEsdfFromTsdf();
      free_space_submap->setHasEsdf(true);
    }
  }
}

// incrementally update occupancy map from the updated TSDF map
void PanopticMapper::updateFreeOccFromTsdf() {
  const int free_space_id = submaps_->getActiveFreeSpaceSubmapID();
  if (submaps_->submapIdExists(free_space_id)) {
    if(submaps_->getSubmap(free_space_id).getTsdfLayer().getNumberOfAllocatedBlocks() > 0){
      if (config_.verbosity > 3)
        ROS_INFO("Update Occ layer from Tsdf layer");
      submaps_->getSubmap(free_space_id).updateOccFromTsdf();
    }
  }
}

// incrementally update occupancy map from the updated TSDF map
void PanopticMapper::updateNonFreeOccFromTsdf() {
  Timer timer("visualization/generate_occ");
  if (config_.verbosity > 3)
    ROS_INFO("Update Occ layers from Tsdf layers");
  for (const auto& submap : *submaps_) {
    if (submap.getLabel() == PanopticLabel::kFreeSpace)
      continue;
    if (submap.getTsdfLayer().getNumberOfAllocatedBlocks() > 0)
      submap.updateOccFromTsdf();
  }
  timer.Stop();
}

// // Incrementally update Esdf from occupancy map via FIESTA
// void PanopticMapper::updateFreeEsdfFromOcc() {
//   const int free_space_id = submaps_->getActiveFreeSpaceSubmapID();
//   if (submaps_->submapIdExists(free_space_id)) {
//     if(submaps_->getSubmap(free_space_id).getOccLayer().getNumberOfAllocatedBlocks() > 0){
//       if (config_.verbosity > 2)
//         ROS_INFO("Update Esdf layer from Occ layer");
//       submaps_->getSubmap(free_space_id).updateEsdfFromOcc();
//     }
//   }
// }

bool PanopticMapper::saveMap(const std::string& file_path) {
  bool success = submaps_->saveToFile(file_path);
  LOG_IF(INFO, success) << "Successfully saved " << submaps_->size()
                        << " submaps to '" << file_path << "'.";
  return success;
}

bool PanopticMapper::loadMap(const std::string& file_path) {
  auto loaded_map = std::make_shared<SubmapCollection>();

  // Load the map.
  if (!loaded_map->loadFromFile(file_path, true)) {
    return false;
  }

  // Loaded submaps are 'from the past' so set them to inactive.
  for (Submap& submap : *loaded_map) {
    submap.finishActivePeriod();
    if (config_.load_submaps_conservative) {
      submap.setChangeState(ChangeState::kUnobserved);
    } else {
      submap.setChangeState(ChangeState::kPersistent);
    }
  }

  // Set the map.
  submaps_ = loaded_map;

  // Setup the interfaces that use the new collection.
  setupCollectionDependentMembers();

  // Reproduce the mesh and visualization.
  submap_visualizer_->clearMesh();
  submap_visualizer_->reset();
  submap_visualizer_->visualizeAll(submaps_.get());

  LOG_IF(INFO, config_.verbosity >= 1)
      << "Successfully loaded " << submaps_->size() << " submaps.";
  return true;
}

bool PanopticMapper::saveMesh(const std::string& folder_path) {
  bool success = submaps_->saveMeshToFile(folder_path);
  LOG_IF(INFO, success) << "Successfully saved " << submaps_->size()
                        << " submaps mesh to folder '" << folder_path << "'.";
  return success;
}

bool PanopticMapper::saveFreeEsdf(const std::string& file_path) {
  Submap* free_space_submap = submaps_->getSubmapPtr(submaps_->getActiveFreeSpaceSubmapID());
  
  bool success = voxblox::io::SaveLayer(free_space_submap->getEsdfLayer(), file_path, false);

  LOG_IF(INFO, success) << "Successfully saved the free space Esdf to file '" << file_path << "'.";
  return success;
}

void PanopticMapper::dataLoggingCallback(const ros::TimerEvent&) {
  data_logger_->writeData(ros::Time::now().toSec(), *submaps_);
}


void PanopticMapper::publishVisualizationCallback(const ros::TimerEvent&) {
  // updateNonFreeOccFromTsdf();
  publishVisualization();
}

void PanopticMapper::publishRobotMeshCallback(const ros::TimerEvent&) {
  publishRobotMesh();
}

//update the visualization mode
bool PanopticMapper::setVisualizationModeCallback(
    panoptic_mapping_msgs::SetVisualizationMode::Request& request,
    panoptic_mapping_msgs::SetVisualizationMode::Response& response) {
  response.visualization_mode_set = false;
  response.color_mode_set = false;
  bool success = true;

  // Set the visualization mode if requested.
  if (!request.visualization_mode.empty()) {
    SubmapVisualizer::VisualizationMode visualization_mode =
        SubmapVisualizer::visualizationModeFromString(
            request.visualization_mode);
    submap_visualizer_->setVisualizationMode(visualization_mode);
    std::string visualization_mode_is =
        SubmapVisualizer::visualizationModeToString(visualization_mode);
    LOG_IF(INFO, config_.verbosity >= 2)
        << "Set visualization mode to '" << visualization_mode_is << "'.";
    response.visualization_mode_set =
        visualization_mode_is == request.visualization_mode;
    if (!response.visualization_mode_set) {
      success = false;
    }
  }

  // Set the color mode if requested.
  if (!request.color_mode.empty()) {
    SubmapVisualizer::ColorMode color_mode =
        SubmapVisualizer::colorModeFromString(request.color_mode);
    submap_visualizer_->setColorMode(color_mode);
    std::string color_mode_is = SubmapVisualizer::colorModeToString(color_mode);
    LOG_IF(INFO, config_.verbosity >= 2)
        << "Set color mode to '" << color_mode_is << "'.";
    response.color_mode_set = color_mode_is == request.color_mode;
    if (!response.color_mode_set) {
      success = false;
    }
  }

  // Republish the visualization.
  submap_visualizer_->visualizeAll(submaps_.get());
  return success;
}

bool PanopticMapper::saveMapCallback(
    panoptic_mapping_msgs::SaveLoadMap::Request& request,
    panoptic_mapping_msgs::SaveLoadMap::Response& response) {
  response.success = saveMap(request.file_path);
  return response.success;
}

bool PanopticMapper::loadMapCallback(
    panoptic_mapping_msgs::SaveLoadMap::Request& request,
    panoptic_mapping_msgs::SaveLoadMap::Response& response) {
  response.success = loadMap(request.file_path);
  return response.success;
}

bool PanopticMapper::saveMeshCallback(
    panoptic_mapping_msgs::SaveLoadMap::Request& request,
    panoptic_mapping_msgs::SaveLoadMap::Response& response) {
  response.success = saveMesh(request.file_path);
  return response.success;
}

bool PanopticMapper::saveFreeEsdfCallback(
    panoptic_mapping_msgs::SaveLoadMap::Request& request,
    panoptic_mapping_msgs::SaveLoadMap::Response& response) {
  response.success = saveFreeEsdf(request.file_path);
  return response.success;
}

bool PanopticMapper::printTimingsCallback(std_srvs::Empty::Request& request,
                                          std_srvs::Empty::Response& response) {
  printTimings();
  return true;
}

void PanopticMapper::printTimingsCallback(const ros::TimerEvent&) {
  printTimings();
}

void PanopticMapper::printTimings() const { LOG(INFO) << Timing::Print(); }

bool PanopticMapper::finishMappingCallback(
    std_srvs::Empty::Request& request, std_srvs::Empty::Response& response) {
  finishMapping();
  return true;
}

ros::NodeHandle PanopticMapper::defaultNh(const std::string& key) const {
  // Essentially just read the default namespaces list and type params.
  // NOTE(schmluk): Since these lookups are quasi-static we don't check for
  // correct usage here.
  const std::pair<std::string, std::string>& ns_and_type =
      default_names_and_types_.at(key);
  ros::NodeHandle nh_out(nh_private_, ns_and_type.first);
  if (!ns_and_type.second.empty() && !nh_out.hasParam("type")) {
    nh_out.setParam("type", ns_and_type.second);
  }
  return nh_out;
}

}  // namespace panoptic_mapping
