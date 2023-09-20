
base_path=/media/yuepan/DATA/1_data/CKA/CKA_2022/SweetPepper
rosbag_name=trolley_2022-09-05-13-05-43_sp_row1_before
rosbag_in=${base_path}/${rosbag_name}.bag
rosbag_out=${base_path}/${rosbag_name}_filtered.bag

rosbag filter ${rosbag_in} ${rosbag_out} "topic == '/trollomatic/camera2/aligned_depth_to_color/image_raw' or topic == '/trollomatic/camera2/color/image_raw' or topic == '/trollomatic/camera2/color/camera_info' or topic == '/trollomatic/camera2/aligned_depth_to_color/camera_info' or topic == '/trollomatic/camera2/extrinsics/depth_to_color' or topic == '/tf' or topic == '/tf_static' or topic == '/clock'"