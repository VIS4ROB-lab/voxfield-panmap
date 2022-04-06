#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# common
mpl.rcParams['font.family'] = ['serif']
mpl.rcParams['font.serif'] = ['Arial']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

cc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
dpi_for_show = 400
fig_width = 8
fig_height = 6
line_width = 5
marker_size = 15
legend_font = 20
x_tick_font = 20
y_tick_font = 20
x_label_font = 25
y_label_font = 25
title_font = 28
y_tick_log_font = 16
y_label_log_font = 23.5
alpha = 0.8


# Flat dataset experiment results
flat_tsdf_ours = np.array([0.010583, 0.024113, 0.036537, 0.051442, 0.072433]) * 100.0
flat_tsdf_voxblox = np.array([0.014940, 0.032751, 0.045754, 0.063610, 0.088687]) * 100.0

flat_tsdf_acc_improvement = (flat_tsdf_voxblox - flat_tsdf_ours) / flat_tsdf_voxblox
print('Flat dataset')
print('Improvment of TSDF accuracy (reduction of the TSDF error)')
print(flat_tsdf_acc_improvement)
print('In average (%):')
print(100.0 * np.mean(flat_tsdf_acc_improvement)) # in %

flat_mesh_ours = np.array([0.007675, 0.012193, 0.019522, 0.023570, 0.040639]) * 100.0
flat_mesh_voxblox = np.array([0.008339, 0.013382, 0.019278, 0.024175, 0.040856]) * 100.0

flat_chamfer_ours = np.array([0.024175, 0.047508, 0.071888, 0.098162, 0.120169]) * 100.0
flat_chamfer_voxblox = np.array([0.023192, 0.048239, 0.074779, 0.102823, 0.122713]) * 100.0

flat_chamfer_l1_ours = np.array([0.016502, 0.032583, 0.050695, 0.068167, 0.090792]) * 100.0
flat_chamfer_l1_voxblox = np.array([0.017392, 0.034043, 0.053118, 0.071788, 0.090413]) * 100.0

flat_coverage_ours = np.array([77.02, 82.08, 84.15, 87.46, 90.49])
flat_coverage_voxblox = np.array([75.58, 80.41, 82.85, 85.31, 88.23])

flat_coverage_improvement = (flat_coverage_ours - flat_coverage_voxblox) / flat_coverage_voxblox
print('Improvment of coverage')
print(flat_coverage_improvement)
print('In average (%):')
print(100.0 * np.mean(flat_coverage_improvement)) # in %

flat_esdf_occ_ours = np.array([0.000264, 0.001312, 0.002830, 0.003715, 0.004013]) * 100.0
flat_esdf_occ_ours_2 = np.array([0.023808, 0.044681, 0.059845, 0.071851, 0.100507]) * 100.0
flat_esdf_occ_voxblox = np.array([0.028618, 0.049641, 0.060068, 0.073047, 0.122605]) * 100.0
flat_esdf_occ_fiesta = np.array([0.000784, 0.002006, 0.002205, 0.005456, 0.005863]) * 100.0
flat_esdf_occ_edt = np.array([0.003646, 0.007013, 0.009817, 0.011729, 0.008791]) * 100.0

flat_esdf_gt_ours = np.array([0.013758, 0.026618, 0.039706, 0.051136, 0.065019]) * 100.0
flat_esdf_gt_ours_2 = np.array([0.022674, 0.042732, 0.055756, 0.070484, 0.105568]) * 100.0
flat_esdf_gt_voxblox = np.array([0.035935, 0.059597, 0.064879, 0.083333, 0.115038]) * 100.0
flat_esdf_gt_fiesta = np.array([0.021043, 0.040452, 0.047320, 0.058019, 0.096814]) * 100.0
flat_esdf_gt_edt = np.array([0.020891, 0.042050, 0.048431, 0.060731, 0.098208]) * 100.0

flat_esdf_gt_sota_mean = (flat_esdf_gt_voxblox + flat_esdf_gt_fiesta + flat_esdf_gt_edt) / 3.0
flat_esdf_acc_improvement = (flat_esdf_gt_sota_mean - flat_esdf_gt_ours) / flat_esdf_gt_sota_mean
print('Improvment of ESDF accuracy (reduction of the ESDF error) over the mean of the SOTAs')
print(flat_esdf_acc_improvement)
print('In average (%):')
print(100.0 * np.mean(flat_esdf_acc_improvement)) # in %

flat_esdf_time_ours = np.array([86.7, 13.7, 6.0, 2.4, 1.5])
flat_esdf_time_voxblox = np.array([347.9, 44.2, 14.7, 8.1, 4.6])
flat_esdf_time_fiesta = np.array([109.2, 17.5, 7.0, 2.8, 2.1])
flat_esdf_time_edt = np.array([127.9, 19.6, 7.8, 3.4, 2.6])

flat_esdf_time_sota_mean = (flat_esdf_time_voxblox + flat_esdf_time_fiesta + flat_esdf_time_edt) / 3.0
flat_esdf_time_improvement = (flat_esdf_time_sota_mean - flat_esdf_time_ours) / flat_esdf_time_sota_mean
print('Improvment of ESDF efficiency (reduction of the ESDF computing time) over the mean of the SOTAs')
print(flat_esdf_time_improvement)
print('In average (%):')
print(100.0 * np.mean(flat_esdf_time_improvement)) # in %

flat_voxel_size = [5, 10, 15, 20, 25]

# v4 without title
# v3 with title

# fig_flat = plt.figure(figsize=(fig_width, fig_height))
# flat_dataset = mpimg.imread('/home/yuepan/Pictures/thesis_experiments/flat_a.png') 
# plt.axis('off')
# plt.imshow(flat_dataset)
# plt.title('Flat synthetic RGB-D dataset', fontsize=title_font)
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/flat_c.png", dpi=120, bbox_inches = 'tight')
# plt.show()

# # tsdf error (flat dataset)
# fig1 = plt.figure(figsize=(fig_width, fig_height))
# flat_voxblox_tsdf_acc = plt.plot(flat_voxel_size, flat_tsdf_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox', alpha=alpha, color=cc[1])
# flat_ours_tsdf_acc = plt.plot(flat_voxel_size, flat_tsdf_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours', alpha=alpha, color=cc[0])
# # l1_tsdf_acc = plt.plot(flat_voxel_size, flat_tsdf_ours, '-o', linewidth = 5, markersize = 20, label='With projective TSDF correction (Ours)')
# # l2_tsdf_acc = plt.plot(flat_voxel_size, flat_tsdf_voxblox, '-^', linewidth = 5, markersize = 20, label='Without projective TSDF correction (Voxblox)')
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(5, 30, step=5), fontsize=x_tick_font)
# plt.yticks(np.arange(2, 12, step=2), fontsize=y_tick_font)
# plt.ylim([0.5, 10.5])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('TSDF error (cm)', fontsize=y_label_font)
# plt.legend(loc='upper left')
# # plt.title('Flat synthetic RGB-D dataset', fontsize=x_label_font)
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/flat_tsdf_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # chamfer distance (flat dataset)
# fig2 = plt.figure(figsize=(fig_width, fig_height))
# flat_voxblox_chamfer_acc = plt.plot(flat_voxel_size, flat_chamfer_l1_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox', alpha=alpha, color=cc[1])
# flat_ours_chamfer_acc = plt.plot(flat_voxel_size, flat_chamfer_l1_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours', alpha=alpha, color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(5, 30, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_font)
# plt.ylim([0.5, 10.5])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('Chamfer distance (cm)', fontsize=y_label_font)
# plt.legend(loc='upper left')
# # plt.title('Flat synthetic RGB-D dataset', fontsize=x_label_font)
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/flat_chamfer_l1_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # reconstruction coverage (flat dataset)
# fig3 = plt.figure(figsize=(fig_width, fig_height))
# flat_voxblox_cov = plt.plot(flat_voxel_size, flat_coverage_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox', alpha=alpha, color=cc[1])
# flat_ours_cov = plt.plot(flat_voxel_size, flat_coverage_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours',  alpha=alpha, color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(5, 30, step=5), fontsize=x_tick_font)
# plt.yticks(np.arange(75, 95, step=5), fontsize=y_tick_font)
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('Coverage (%)', fontsize=y_label_font)
# plt.legend(loc='upper left')
# # plt.title('Flat synthetic RGB-D dataset', fontsize=x_label_font)
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/flat_coverage_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # esdf error gt reference (flat dataset)
# fig4 = plt.figure(figsize=(fig_width, fig_height))
# flat_voxblox_esdf_acc_gt = plt.plot(flat_voxel_size, flat_esdf_gt_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox',  alpha=alpha, zorder = 1, color=cc[1])
# flat_fiesta_esdf_acc_gt = plt.plot(flat_voxel_size, flat_esdf_gt_fiesta, '-v', linewidth = line_width, markersize = marker_size, label='FIESTA',  alpha=alpha, zorder = 2, color=cc[2])
# flat_edt_esdf_acc_gt = plt.plot(flat_voxel_size, flat_esdf_gt_edt, '-X', linewidth = line_width, markersize = marker_size, label='EDT',  alpha=alpha, zorder = 1, color=cc[3])
# flat_ours_esdf_acc_gt = plt.plot(flat_voxel_size, flat_esdf_gt_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours', alpha=alpha, zorder = 3, color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(5, 30, step=5), fontsize=x_tick_font)
# plt.yticks(np.arange(2, 12, step=2), fontsize=y_tick_font)
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('ESDF error (cm)', fontsize=y_label_font)
# plt.legend(loc='upper left')
# # plt.title('Flat synthetic RGB-D dataset', fontsize=title_font)
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/flat_esdf_gt_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # # esdf error occupied voxel centers reference (flat dataset)
# fig5 = plt.figure(figsize=(fig_width, fig_height))
# flat_voxblox_esdf_acc_occ = plt.plot(flat_voxel_size, flat_esdf_occ_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox', alpha=alpha, color=cc[1])
# flat_fiesta_esdf_acc_occ = plt.plot(flat_voxel_size, flat_esdf_occ_fiesta, '-v', linewidth = line_width, markersize = marker_size, label='FIESTA', alpha=alpha, color=cc[2])
# flat_edt_esdf_acc_occ = plt.plot(flat_voxel_size, flat_esdf_occ_edt, '-X', linewidth = line_width, markersize = marker_size, label='EDT', alpha=alpha, color=cc[3])
# flat_ours_esdf_acc_occ = plt.plot(flat_voxel_size, flat_esdf_occ_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours',  alpha=alpha, color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(5, 30, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_log_font)
# plt.yscale('log')
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('ESDF error (cm)', fontsize=y_label_log_font)
# plt.legend(loc='upper left')
# # plt.title('Flat synthetic RGB-D dataset', fontsize=x_label_font)
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/flat_esdf_occ_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # esdf computation time (flat dataset)
# fig6 = plt.figure(figsize=(fig_width, fig_height))
# flat_voxblox_esdf_time = plt.plot(flat_voxel_size, flat_esdf_time_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox',  alpha=alpha, zorder = 1, color=cc[1])
# flat_fiesta_esdf_time = plt.plot(flat_voxel_size, flat_esdf_time_fiesta, '-v', linewidth = line_width, markersize = marker_size, label='FIESTA',  alpha=alpha, zorder = 2, color=cc[2])
# flat_edt_esdf_time = plt.plot(flat_voxel_size, flat_esdf_time_edt, '-X', linewidth = line_width, markersize = marker_size, label='EDT',  alpha=alpha, zorder = 1, color=cc[3])
# flat_ours_esdf_time = plt.plot(flat_voxel_size, flat_esdf_time_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours',  alpha=alpha, zorder = 3, color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(5, 30, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_log_font)
# plt.yscale('log')
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('ESDF update time (ms)', fontsize=y_label_log_font)
# plt.legend(loc='upper right')
# # plt.title('Flat synthetic RGB-D dataset', fontsize=x_label_font)
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/flat_esdftime_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

print('-----------------------------------------------------------')

# MaiCity Dataset experiment results
mai_tsdf_ours = np.array([0.014968, 0.027714, 0.040541, 0.064733, 0.074458, 0.094721, 0.104881]) * 100.0
mai_tsdf_voxblox = np.array([0.043657, 0.078265, 0.118020, 0.164887, 0.192402, 0.209249, 0.223521]) * 100.0

mai_tsdf_acc_improvement = (mai_tsdf_voxblox - mai_tsdf_ours) / mai_tsdf_voxblox
print('Maicity dataset')
print('Improvment of TSDF accuracy (decreasing of the TSDF error)')
print(mai_tsdf_acc_improvement)
print('In average (%):')
print(100.0 * np.mean(mai_tsdf_acc_improvement)) # in %

mai_mesh_ours = np.array([0.036759, 0.058151, 0.068234, 0.101228, 0.114352, 0.124643, 0.140532]) * 100.0
mai_mesh_voxblox = np.array([0.039103, 0.061638, 0.078188, 0.109485, 0.123240, 0.133580, 0.150718]) * 100.0

mai_chamfer_ours = np.array([0.048204, 0.075392, 0.094810, 0.130635, 0.150917, 0.175598, 0.190937]) * 100.0
mai_chamfer_voxblox = np.array([0.051746, 0.078703, 0.100711, 0.133263, 0.153123, 0.177588, 0.196262]) * 100.0

mai_chamfer_l1_ours = np.array([0.040522, 0.063484, 0.079526, 0.113675, 0.131905, 0.144468, 0.162446]) * 100.0
mai_chamfer_l1_voxblox = np.array([0.043747, 0.064929, 0.085147, 0.111982, 0.128996, 0.150020, 0.163024]) * 100.0

mai_coverage_ours = np.array([98.89, 99.04, 98.94, 99.16, 99.24, 99.52, 99.76]) 
mai_coverage_voxblox = np.array([98.42, 98.61, 98.79, 98.92, 99.01, 99.34, 99.65])

mai_coverage_improvement = (mai_coverage_ours - mai_coverage_voxblox) / mai_coverage_voxblox
print('Improvment of coverage')
print(mai_coverage_improvement)
print('In average (%):')
print(100.0 * np.mean(mai_coverage_improvement)) # in %

mai_esdf_occ_ours = np.array([0.007548, 0.011165, 0.012473, 0.021607, 0.020906, 0.022850]) * 100.0
mai_esdf_occ_ours_2 = np.array([0.048313, 0.079564, 0.086715, 0.101150, 0.140885, 0.129498]) * 100.0
mai_esdf_occ_voxblox = np.array([0.105129, 0.135906, 0.159407, 0.190374, 0.215760, 0.233799]) * 100.0
mai_esdf_occ_fiesta = np.array([0.009990, 0.010689, 0.020770, 0.020486, 0.023124, 0.019748]) * 100.0
mai_esdf_occ_edt = np.array([0.009259, 0.021329, 0.028463, 0.026515, 0.033678, 0.030134]) * 100.0

mai_esdf_gt_ours = np.array([0.061703, 0.092117, 0.117943, 0.140956, 0.162598, 0.182063]) * 100.0
mai_esdf_gt_ours_2 = np.array([0.063367, 0.091992, 0.131013, 0.152049, 0.182574, 0.185861]) * 100.0
mai_esdf_gt_voxblox = np.array([0.118326, 0.175636, 0.217876, 0.261014, 0.283866, 0.300789]) * 100.0
mai_esdf_gt_fiesta = np.array([0.055197, 0.104627, 0.141282, 0.177998, 0.185972, 0.189154]) * 100.0
mai_esdf_gt_edt = np.array([0.057547, 0.109508, 0.144468, 0.180567, 0.191044, 0.197996]) * 100.0

mai_esdf_gt_sota_mean = (mai_esdf_gt_voxblox + mai_esdf_gt_fiesta + mai_esdf_gt_edt) / 3.0
mai_esdf_acc_improvement = (mai_esdf_gt_sota_mean - mai_esdf_gt_ours) / mai_esdf_gt_sota_mean
print('Improvment of ESDF accuracy (reduction of the ESDF error) over the mean of the SOTAs')
print(mai_esdf_acc_improvement)
print('In average (%):')
print(100.0 * np.mean(mai_esdf_acc_improvement)) # in %

mai_esdf_time_ours = np.array([1295.5, 619.4, 351.6, 206.3, 160.3, 101.7])
mai_esdf_time_voxblox = np.array([2793.1, 1254.9, 685.0, 423.2, 332.8, 283.6])
mai_esdf_time_fiesta = np.array([1910.3, 671.4, 422.1, 284.7, 202.7, 137.9])
mai_esdf_time_edt = np.array([1502.3, 901.7, 449.3, 261.8, 195.7, 121.6])

mai_esdf_time_sota_mean = (mai_esdf_time_voxblox + mai_esdf_time_fiesta + mai_esdf_time_edt) / 3.0
mai_esdf_time_improvement = (mai_esdf_time_sota_mean - mai_esdf_time_ours) / mai_esdf_time_sota_mean
print('Improvment of ESDF efficiency (reduction of the ESDF computing time) over the mean of the SOTAs')
print(mai_esdf_time_improvement)
print('In average (%):')
print(100.0 * np.mean(mai_esdf_time_improvement)) # in %

mai_voxel_size = [10, 15, 20, 25, 30, 35, 40]
mai_voxel_size_esdf = [15, 20, 25, 30, 35, 40] # voxel size = 10 is too computational heavily for esdf mapping

# fig_mai = plt.figure(figsize=(fig_width, fig_height))
# mai_dataset = mpimg.imread('/home/yuepan/Pictures/thesis_experiments/mai_a.png') 
# plt.axis('off')
# plt.imshow(mai_dataset)
# plt.title('MaiCity synthetic LiDAR dataset', fontsize=title_font)
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/mai_c.png", dpi=150, bbox_inches = 'tight')
# plt.show()

# # tsdf error (maicity dataset)
# fig7 = plt.figure(figsize=(fig_width, fig_height))
# mai_voxblox_tsdf_acc = plt.plot(mai_voxel_size, mai_tsdf_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox',  alpha=alpha, color=cc[1])
# mai_ours_tsdf_acc = plt.plot(mai_voxel_size, mai_tsdf_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours',  alpha=alpha, color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(10, 45, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_font)
# # plt.ylim([0.5, 10.5])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('TSDF error (cm)', fontsize=y_label_font)
# # plt.title('MaiCity synthetic LiDAR dataset', fontsize=x_label_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/mai_tsdf_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # chamfer distance (maicity dataset)
# fig8 = plt.figure(figsize=(fig_width, fig_height))
# mai_voxblox_chamfer_acc = plt.plot(mai_voxel_size, mai_chamfer_l1_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox', zorder = 1,  alpha=alpha, color=cc[1])
# mai_ours_chamfer_acc = plt.plot(mai_voxel_size, mai_chamfer_l1_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours', zorder = 2,  alpha=alpha, color=cc[0])
# # mai_surfel_chamfer_acc = plt.axhline(y=11.0, linestyle='--', color='c', linewidth = line_width, label='Surfels',  alpha=alpha) 
# # mai_puma_chamfer_acc = plt.axhline(y=5.0, linestyle='--', color='m', linewidth = line_width, label='Puma',  alpha=alpha) 
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(10, 45, step=5), fontsize=x_tick_font)
# plt.yticks(np.arange(5, 21, step=4), fontsize=y_tick_font)
# plt.ylim([3, 18])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('Chamfer distance (cm)', fontsize=y_label_font)
# # plt.title('MaiCity synthetic LiDAR dataset', fontsize=x_label_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/mai_chamfer_l1_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # reconstruction coverage (maicity dataset)
# fig9 = plt.figure(figsize=(fig_width, fig_height))
# mai_voxblox_cov = plt.plot(mai_voxel_size, mai_coverage_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox',  alpha=alpha, color=cc[1])
# mai_ours_cov = plt.plot(mai_voxel_size, mai_coverage_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours',  alpha=alpha, color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(10, 45, step=5), fontsize=x_tick_font)
# plt.yticks(np.arange(98, 100, step=1), fontsize=y_tick_font)
# plt.ylim([97.5, 100])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('Coverage (%)', fontsize=y_label_font)
# # plt.title('MaiCity synthetic LiDAR dataset', fontsize=x_label_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/mai_coverage_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # esdf error gt reference (maicity dataset)
# fig10 = plt.figure(figsize=(fig_width, fig_height))
# mai_voxblox_esdf_acc_gt = plt.plot(mai_voxel_size_esdf, mai_esdf_gt_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox',  alpha=alpha, zorder = 1, color=cc[1])
# mai_fiesta_esdf_acc_gt = plt.plot(mai_voxel_size_esdf, mai_esdf_gt_fiesta, '-v', linewidth = line_width, markersize = marker_size, label='FIESTA',  alpha=alpha, zorder = 2, color=cc[2])
# mai_edt_esdf_acc_gt = plt.plot(mai_voxel_size_esdf, mai_esdf_gt_edt, '-X', linewidth = line_width, markersize = marker_size, label='EDT',  alpha=alpha, zorder = 1, color=cc[3])
# mai_ours_esdf_acc_gt = plt.plot(mai_voxel_size_esdf, mai_esdf_gt_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours',  alpha=alpha, zorder = 3, color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(15, 45, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_font)
# plt.ylim([3, 31])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('ESDF error (cm)', fontsize=y_label_font)
# # plt.title('MaiCity synthetic LiDAR dataset', fontsize=title_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/mai_esdf_gt_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # esdf error occupied voxel centers reference (maicity dataset)
# fig11 = plt.figure(figsize=(fig_width, fig_height))
# mai_voxblox_esdf_acc_occ = plt.plot(mai_voxel_size_esdf, mai_esdf_occ_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox',  alpha=alpha, color=cc[1])
# mai_fiesta_esdf_acc_occ = plt.plot(mai_voxel_size_esdf, mai_esdf_occ_fiesta, '-v', linewidth = line_width, markersize = marker_size, label='FIESTA',  alpha=alpha, color=cc[2])
# mai_edt_esdf_acc_occ = plt.plot(mai_voxel_size_esdf, mai_esdf_occ_edt, '-X', linewidth = line_width, markersize = marker_size, label='EDT',  alpha=alpha, color=cc[3])
# mai_ours_esdf_acc_occ = plt.plot(mai_voxel_size_esdf, mai_esdf_occ_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours',  alpha=alpha, color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(15, 45, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_font)
# plt.yscale('log')
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('ESDF error (cm)', fontsize=y_label_font)
# # plt.title('MaiCity synthetic LiDAR dataset', fontsize=x_label_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/mai_esdf_occ_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # esdf computation time (maicity dataset)
# fig12 = plt.figure(figsize=(fig_width, fig_height))
# mai_voxblox_esdf_time = plt.plot(mai_voxel_size_esdf, mai_esdf_time_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox',  alpha=alpha, zorder = 1, color=cc[1])
# mai_fiesta_esdf_time = plt.plot(mai_voxel_size_esdf, mai_esdf_time_fiesta, '-v', linewidth = line_width, markersize = marker_size, label='FIESTA',  alpha=alpha, zorder = 2, color=cc[2])
# mai_edt_esdf_time = plt.plot(mai_voxel_size_esdf, mai_esdf_time_edt, '-X', linewidth = line_width, markersize = marker_size, label='EDT',  alpha=alpha, zorder = 1, color=cc[3])
# mai_ours_esdf_time = plt.plot(mai_voxel_size_esdf, mai_esdf_time_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours',  alpha=alpha, zorder = 3, color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(15, 45, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_log_font)
# plt.yscale('log')
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('ESDF update time (ms)', fontsize=y_label_log_font)
# #plt.title('MaiCity synthetic LiDAR dataset', fontsize=x_label_font)
# plt.legend(loc='upper right')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/mai_esdftime_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

print('-----------------------------------------------------------')

# Cow & Lady Dataset experiment results
cow_tsdf_ours = np.array([0.021730, 0.041152, 0.061421, 0.076994, 0.089392, 0.100308]) * 100.0
cow_tsdf_voxblox = np.array([0.025528, 0.047204, 0.070043, 0.088750, 0.102293, 0.117120]) * 100.0

cow_tsdf_acc_improvement = (cow_tsdf_voxblox - cow_tsdf_ours) / cow_tsdf_voxblox
print('Cow&Lady dataset')
print('Improvment of TSDF accuracy (decreasing of the TSDF error)')
print(cow_tsdf_acc_improvement)
print('In average (%):')
print(100.0 * np.mean(cow_tsdf_acc_improvement)) # in %

cow_mesh_ours = np.array([0.019080, 0.041004, 0.077135, 0.103229, 0.126100, 0.142602]) * 100.0
cow_mesh_voxblox = np.array([0.019285, 0.043385, 0.078242, 0.101553, 0.125004, 0.141065]) * 100.0

cow_chamfer_ours = np.array([0.020410, 0.048524, 0.087024, 0.118372, 0.146667, 0.178275]) * 100.0
cow_chamfer_voxblox = np.array([0.020872, 0.049593, 0.089401, 0.117131, 0.146998, 0.177641]) * 100.0

cow_chamfer_l1_ours = np.array([0.018159, 0.041359, 0.074050, 0.098170, 0.122406, 0.145597]) * 100.0
cow_chamfer_l1_voxblox = np.array([0.018233, 0.042285, 0.075274, 0.097772, 0.121455, 0.144901]) * 100.0

cow_coverage_ours = np.array([74.74, 77.41, 78.19, 79.63, 79.80, 80.87]) 
cow_coverage_voxblox = np.array([73.51, 76.67, 77.57, 78.84, 79.35, 80.08])

cow_coverage_improvement = (cow_coverage_ours - cow_coverage_voxblox) / cow_coverage_voxblox
print('Improvment of coverage')
print(cow_coverage_improvement)
print('In average (%):')
print(100.0 * np.mean(cow_coverage_improvement)) # in %

cow_esdf_occ_ours = np.array([0.007396, 0.004621, 0.002608, 0.001956, 0.001365]) * 100.0
cow_esdf_occ_ours_2 = np.array([0.021373, 0.042698, 0.062937, 0.078136, 0.112796]) * 100.0
cow_esdf_occ_voxblox = np.array([0.032315, 0.072092, 0.069326, 0.086641, 0.104482]) * 100.0
cow_esdf_occ_fiesta = np.array([0.007478, 0.005826, 0.002122, 0.002948, 0.006688]) * 100.0
cow_esdf_occ_edt = np.array([0.007280, 0.007692, 0.004191, 0.002198, 0.004563]) * 100.0

cow_esdf_gt_ours = np.array([0.035570, 0.068852, 0.090651, 0.114565, 0.135114]) * 100.0
cow_esdf_gt_ours_2 = np.array([0.040691, 0.077728, 0.104099, 0.134671, 0.161160]) * 100.0
cow_esdf_gt_voxblox = np.array([0.042938, 0.085359, 0.108825, 0.134526, 0.153991]) * 100.0
cow_esdf_gt_fiesta = np.array([0.039401, 0.072888, 0.097224, 0.119858, 0.140690]) * 100.0
cow_esdf_gt_edt = np.array([0.039633, 0.074739, 0.096936, 0.123012, 0.142823]) * 100.0

cow_esdf_gt_sota_mean = (cow_esdf_gt_voxblox + cow_esdf_gt_fiesta + cow_esdf_gt_edt) / 3.0
cow_esdf_acc_improvement = (cow_esdf_gt_sota_mean - cow_esdf_gt_ours) / cow_esdf_gt_sota_mean
print('Improvment of ESDF accuracy (reduction of the ESDF error) over the mean of the SOTAs')
print(cow_esdf_acc_improvement)
print('In average (%):')
print(100.0 * np.mean(cow_esdf_acc_improvement)) # in %

cow_esdf_time_ours = np.array([64.9, 10.8, 4.5, 2.1, 1.2])
cow_esdf_time_voxblox = np.array([198.5, 31.4, 11.9, 5.9, 3.3])
cow_esdf_time_fiesta = np.array([84.0, 11.2, 4.8, 2.3, 1.3])
cow_esdf_time_edt = np.array([125.9, 14.8, 7.1, 2.7, 1.7])

cow_esdf_time_sota_mean = (cow_esdf_time_voxblox + cow_esdf_time_fiesta + cow_esdf_time_edt) / 3.0
cow_esdf_time_improvement = (cow_esdf_time_sota_mean - cow_esdf_time_ours) / cow_esdf_time_sota_mean
print('Improvment of ESDF efficiency (reduction of the ESDF computing time) over the mean of the SOTAs')
print(cow_esdf_time_improvement)
print('In average (%):')
print(100.0 * np.mean(cow_esdf_time_improvement)) # in %


cow_voxel_size = [2, 5, 10, 15, 20, 25]
cow_voxel_size_esdf = [5, 10, 15, 20, 25] # voxel size = 2 is too computational heavily for esdf mapping

# fig_cow = plt.figure(figsize=(fig_width, fig_height))
# cow_dataset = mpimg.imread('/home/yuepan/Pictures/thesis_experiments/cow_a.png') 
# plt.axis('off')
# plt.imshow(cow_dataset)
# plt.title('Cow&Lady real-world RGB-D dataset', fontsize=title_font)
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/cow_c.png", dpi=150, bbox_inches = 'tight')
# plt.show()

# # tsdf error (cow dataset)
# fig13 = plt.figure(figsize=(fig_width, fig_height))
# cow_voxblox_tsdf_acc = plt.plot(cow_voxel_size, cow_tsdf_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox',  alpha=alpha,color=cc[1])
# cow_ours_tsdf_acc = plt.plot(cow_voxel_size, cow_tsdf_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours',  alpha=alpha ,color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(5, 30, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_font)
# plt.ylim([1, 13])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('TSDF error (cm)', fontsize=y_label_font)
# # plt.title('Cow&Lady real-world RGB-D dataset', fontsize=x_label_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/cow_tsdf_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # chamfer distance (cow dataset)
# fig14 = plt.figure(figsize=(fig_width, fig_height))
# cow_voxblox_chamfer_acc = plt.plot(cow_voxel_size, cow_chamfer_l1_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox', zorder = 1,  alpha=alpha,color=cc[1])
# cow_ours_chamfer_acc = plt.plot(cow_voxel_size, cow_chamfer_l1_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours', zorder = 2,  alpha=alpha,color=cc[0] )
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(5, 30, step=5), fontsize=x_tick_font)
# plt.yticks(np.arange(1, 17, step=2),fontsize=y_tick_font)
# plt.ylim([1, 16])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('Chamfer distance (cm)', fontsize=y_label_font)
# # plt.title('Cow&Lady real-world RGB-D dataset', fontsize=x_label_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/cow_chamfer_l1_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # reconstruction coverage (cow dataset)
# fig15 = plt.figure(figsize=(fig_width, fig_height))
# cow_voxblox_cov = plt.plot(cow_voxel_size, cow_coverage_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox',  alpha=alpha,color=cc[1])
# cow_ours_cov = plt.plot(cow_voxel_size, cow_coverage_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours',  alpha=alpha,color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(5, 30, step=5), fontsize=x_tick_font)
# plt.yticks(np.arange(74, 82, step=2), fontsize=y_tick_font)
# plt.ylim([72.5, 81.5])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('Coverage (%)', fontsize=y_label_font)
# #plt.title('Cow&Lady real-world RGB-D dataset', fontsize=x_label_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/cow_coverage_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # esdf error gt reference (cow dataset)
# fig16 = plt.figure(figsize=(fig_width, fig_height))
# cow_voxblox_esdf_acc_gt = plt.plot(cow_voxel_size_esdf, cow_esdf_gt_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox',  alpha=alpha, zorder = 1,color=cc[1])
# cow_fiesta_esdf_acc_gt = plt.plot(cow_voxel_size_esdf, cow_esdf_gt_fiesta, '-v', linewidth = line_width, markersize = marker_size, label='FIESTA',  alpha=alpha, zorder = 2,color=cc[2])
# cow_edt_esdf_acc_gt = plt.plot(cow_voxel_size_esdf, cow_esdf_gt_edt, '-X', linewidth = line_width, markersize = marker_size, label='EDT',  alpha=alpha, zorder = 1,color=cc[3])
# cow_ours_esdf_acc_gt = plt.plot(cow_voxel_size_esdf, cow_esdf_gt_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours',  alpha=alpha, zorder = 3,color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(5, 30, step=5), fontsize=x_tick_font)
# plt.yticks(np.arange(4, 19, step=3), fontsize=y_tick_font)
# plt.ylim([3, 17])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('ESDF error (cm)', fontsize=y_label_font)
# # plt.title('Cow&Lady real-world RGB-D dataset', fontsize=title_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/cow_esdf_gt_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # esdf error occupied voxel centers reference (cow dataset)
# fig17 = plt.figure(figsize=(fig_width, fig_height))
# cow_voxblox_esdf_acc_occ = plt.plot(cow_voxel_size_esdf, cow_esdf_occ_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox',  alpha=alpha,color=cc[1])
# cow_fiesta_esdf_acc_occ = plt.plot(cow_voxel_size_esdf, cow_esdf_occ_fiesta, '-v', linewidth = line_width, markersize = marker_size, label='FIESTA',  alpha=alpha,color=cc[2])
# cow_edt_esdf_acc_occ = plt.plot(cow_voxel_size_esdf, cow_esdf_occ_edt, '-X', linewidth = line_width, markersize = marker_size, label='EDT',  alpha=alpha,color=cc[3])
# cow_ours_esdf_acc_occ = plt.plot(cow_voxel_size_esdf, cow_esdf_occ_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours',  alpha=alpha,color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(5, 30, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_font)
# plt.yscale('log')
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('ESDF error (cm)', fontsize=y_label_font)
# # plt.title('Cow&Lady real-world RGB-D dataset', fontsize=x_label_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/cow_esdf_occ_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # esdf computation time (cow dataset)
# fig18 = plt.figure(figsize=(fig_width, fig_height))
# cow_voxblox_esdf_time = plt.plot(cow_voxel_size_esdf, cow_esdf_time_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox',  alpha=alpha, zorder = 1,color=cc[1])
# cow_fiesta_esdf_time = plt.plot(cow_voxel_size_esdf, cow_esdf_time_fiesta, '-v', linewidth = line_width, markersize = marker_size, label='FIESTA',  alpha=alpha, zorder = 2,color=cc[2])
# cow_edt_esdf_time = plt.plot(cow_voxel_size_esdf, cow_esdf_time_edt, '-X', linewidth = line_width, markersize = marker_size, label='EDT',  alpha=alpha, zorder = 1,color=cc[3])
# cow_ours_esdf_time = plt.plot(cow_voxel_size_esdf, cow_esdf_time_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours',  alpha=alpha, zorder = 3,color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(5, 30, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_log_font)
# plt.yscale('log')
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('ESDF update time (ms)', fontsize=y_label_log_font)
# # plt.title('Cow&Lady real-world RGB-D dataset', fontsize=x_label_font)
# plt.legend(loc='upper right')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/cow_esdftime_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

print('-----------------------------------------------------------')

# KITTI Dataset experiment results
kitti_tsdf_ours = np.array([0.054706, 0.076064, 0.091253, 0.117357, 0.140246, 0.163294, 0.184160, 0.207537]) * 100.0
kitti_tsdf_voxblox = np.array([0.076666, 0.111605, 0.142605, 0.172805, 0.198182, 0.229126, 0.261363, 0.292414]) * 100.0

kitti_tsdf_acc_improvement = (kitti_tsdf_voxblox - kitti_tsdf_ours) / kitti_tsdf_voxblox
print('KITTI dataset')
print('Improvment of TSDF accuracy (decreasing of the TSDF error)')
print(kitti_tsdf_acc_improvement)
print('In average (%):')
print(100.0 * np.mean(kitti_tsdf_acc_improvement)) # in %

kitti_mesh_ours = np.array([0.053939, 0.071321, 0.100758, 0.110055, 0.136142, 0.153278, 0.187616, 0.205459]) * 100.0
kitti_mesh_voxblox = np.array([0.052804, 0.072688, 0.095083, 0.112696, 0.123240, 0.149492, 0.169810, 0.191830]) * 100.0

kitti_chamfer_ours = np.array([0.070184, 0.096353, 0.127042, 0.153013, 0.185023, 0.217662, 0.248017, 0.277064]) * 100.0
kitti_chamfer_voxblox = np.array([0.070230, 0.098754, 0.126423, 0.155122, 0.183367, 0.214261, 0.244005, 0.274732]) * 100.0

kitti_chamfer_l1_ours = np.array([0.061305, 0.080357, 0.109573, 0.128464, 0.148603, 0.175990, 0.196009, 0.222668]) * 100.0
kitti_chamfer_l1_voxblox = np.array([0.063471, 0.083412, 0.110612, 0.126414, 0.149271, 0.171398, 0.195541, 0.218660]) * 100.0

kitti_coverage_ours = np.array([93.28, 95.87, 96.86, 97.62, 97.88, 98.17, 98.40, 98.61]) 
kitti_coverage_voxblox = np.array([92.00, 94.27, 95.88, 96.89, 96.86, 97.44, 97.70, 97.79])

kitti_coverage_improvement = (kitti_coverage_ours - kitti_coverage_voxblox) / kitti_coverage_voxblox
print('Improvment of coverage')
print(kitti_coverage_improvement)
print('In average (%):')
print(100.0 * np.mean(kitti_coverage_improvement)) # in %

kitti_esdf_occ_ours = np.array([0.004703, 0.009169, 0.012170, 0.004613, 0.005155]) * 100.0
kitti_esdf_occ_ours_2 = np.array([0.083964, 0.108268, 0.120864, 0.133990, 0.161840]) * 100.0
kitti_esdf_occ_voxblox = np.array([0.170741, 0.199926, 0.229767, 0.259356, 0.287793]) * 100.0
kitti_esdf_occ_fiesta = np.array([0.007711, 0.020564, 0.013083, 0.004115, 0.003534]) * 100.0
kitti_esdf_occ_edt = np.array([0.011328, 0.030120, 0.013641, 0.014424, 0.016813]) * 100.0

kitti_esdf_gt_ours = np.array([0.150135, 0.175931, 0.202164, 0.225743, 0.251263]) * 100.0
kitti_esdf_gt_ours_2 = np.array([0.154645, 0.182860, 0.209611, 0.231595, 0.262381]) * 100.0
kitti_esdf_gt_voxblox = np.array([0.243014, 0.285154, 0.327336, 0.368115, 0.409706]) * 100.0
kitti_esdf_gt_fiesta = np.array([0.170639, 0.200180, 0.226982, 0.255516, 0.284789]) * 100.0
kitti_esdf_gt_edt = np.array([0.172190, 0.202405, 0.230046, 0.259588, 0.288858]) * 100.0

kitti_esdf_gt_sota_mean = (kitti_esdf_gt_voxblox + kitti_esdf_gt_fiesta + kitti_esdf_gt_edt) / 3.0
kitti_esdf_acc_improvement = (kitti_esdf_gt_sota_mean - kitti_esdf_gt_ours) / kitti_esdf_gt_sota_mean
print('Improvment of ESDF accuracy (reduction of the ESDF error) over the mean of the SOTAs')
print(kitti_esdf_acc_improvement)
print('In average (%):')
print(100.0 * np.mean(kitti_esdf_acc_improvement)) # in %

kitti_esdf_time_ours = np.array([425.7, 227.8, 144.6, 96.4, 74.7])
kitti_esdf_time_voxblox = np.array([ 877.4, 518.3, 384.4, 256.3, 186.7])
kitti_esdf_time_fiesta = np.array([473.1, 250.0, 166.2, 112.3, 86.9])
kitti_esdf_time_edt = np.array([468.0, 258.3, 176.0, 121.6, 94.9])

kitti_esdf_time_sota_mean = (kitti_esdf_time_voxblox + kitti_esdf_time_fiesta + kitti_esdf_time_edt) / 3.0
kitti_esdf_time_improvement = (kitti_esdf_time_sota_mean - kitti_esdf_time_ours) / kitti_esdf_time_sota_mean
print('Improvment of ESDF efficiency (reduction of the ESDF computing time) over the mean of the SOTAs')
print(kitti_esdf_time_improvement)
print('In average (%):')
print(100.0 * np.mean(kitti_esdf_time_improvement)) # in %

kitti_voxel_size = [10, 15, 20, 25, 30, 35, 40, 45]
kitti_voxel_size_esdf = [25, 30, 35, 40, 45] # voxel size = 10, 15, 20 is too computational heavily for esdf mapping

# fig_kitti = plt.figure(figsize=(fig_width, fig_height))
# kitti_dataset = mpimg.imread('/home/yuepan/Pictures/thesis_experiments/kitti_a.png') 
# plt.axis('off')
# plt.imshow(kitti_dataset)
# plt.title('KITTI real-world LiDAR dataset', fontsize=title_font)
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/kitti_c.png", format='pdf', dpi=150, bbox_inches = 'tight')
# plt.show()

# # tsdf error (kitti dataset)
# fig19 = plt.figure(figsize=(fig_width, fig_height))
# kitti_voxblox_tsdf_acc = plt.plot(kitti_voxel_size, kitti_tsdf_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox', alpha=alpha,color=cc[1])
# kitti_ours_tsdf_acc = plt.plot(kitti_voxel_size, kitti_tsdf_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours', alpha=alpha,color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(10, 50, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_font)
# plt.ylim([3, 31])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('TSDF error (cm)', fontsize=y_label_font)
# #plt.title('KITTI real-world LiDAR dataset', fontsize=x_label_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/kitti_tsdf_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # chamfer distance (kitti dataset)
# fig20 = plt.figure(figsize=(fig_width, fig_height))
# kitti_voxblox_chamfer_acc = plt.plot(kitti_voxel_size, kitti_chamfer_l1_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox', alpha = alpha, zorder = 1,color=cc[1])
# kitti_ours_chamfer_acc = plt.plot(kitti_voxel_size, kitti_chamfer_l1_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours', alpha = alpha, zorder = 2,color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(10, 50, step=5), fontsize=x_tick_font)
# plt.yticks(np.arange(5, 25, step=5), fontsize=y_tick_font)
# # plt.ylim([3, 21])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('Chamfer distance (cm)', fontsize=y_label_font)
# #plt.title('KITTI real-world LiDAR dataset', fontsize=x_label_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/kitti_chamfer_l1_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # reconstruction coverage (kitti dataset)
# fig21 = plt.figure(figsize=(fig_width, fig_height))
# cow_voxblox_cov = plt.plot(kitti_voxel_size, kitti_coverage_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox', alpha=alpha,color=cc[1])
# cow_ours_cov = plt.plot(kitti_voxel_size, kitti_coverage_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours', alpha=alpha,color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(10, 50, step=5), fontsize=x_tick_font)
# plt.yticks(np.arange(92, 100, step=2), fontsize=y_tick_font)
# plt.ylim([91.5, 99.5])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('Coverage (%)', fontsize=y_label_font)
# #plt.title('KITTI real-world LiDAR dataset', fontsize=x_label_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/kitti_coverage_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # esdf error gt reference (kitti dataset)
# fig22 = plt.figure(figsize=(fig_width, fig_height))

# kitti_voxblox_esdf_acc_gt = plt.plot(kitti_voxel_size_esdf, kitti_esdf_gt_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox', alpha=alpha, zorder = 1,color=cc[1])
# kitti_fiesta_esdf_acc_gt = plt.plot(kitti_voxel_size_esdf, kitti_esdf_gt_fiesta, '-v', linewidth = line_width, markersize = marker_size, label='FIESTA', alpha=alpha, zorder = 2,color=cc[2])
# kitti_edt_esdf_acc_gt = plt.plot(kitti_voxel_size_esdf, kitti_esdf_gt_edt, '-X', linewidth = line_width, markersize = marker_size, label='EDT', alpha=alpha, zorder = 1,color=cc[3])
# kitti_ours_esdf_acc_gt = plt.plot(kitti_voxel_size_esdf, kitti_esdf_gt_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours', alpha=alpha, zorder = 3,color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(25, 50, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_font)
# # plt.ylim([3, 31])
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('ESDF error (cm)', fontsize=y_label_font)
# #plt.title('KITTI real-world LiDAR dataset', fontsize=title_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/kitti_esdf_gt_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # esdf error occupied voxel centers reference (kitti dataset)
# fig23 = plt.figure(figsize=(fig_width, fig_height))
# kitti_voxblox_esdf_acc_occ = plt.plot(kitti_voxel_size_esdf, kitti_esdf_occ_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox', alpha=alpha,color=cc[1])
# kitti_fiesta_esdf_acc_occ = plt.plot(kitti_voxel_size_esdf, kitti_esdf_occ_fiesta, '-v', linewidth = line_width, markersize = marker_size, label='FIESTA', alpha=alpha,color=cc[2])
# kitti_edt_esdf_acc_occ = plt.plot(kitti_voxel_size_esdf, kitti_esdf_occ_edt, '-X', linewidth = line_width, markersize = marker_size, label='EDT', alpha=alpha,color=cc[3])
# kitti_ours_esdf_acc_occ = plt.plot(kitti_voxel_size_esdf, kitti_esdf_occ_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours', alpha=alpha,color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(25, 50, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_font)
# plt.yscale('log')
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('ESDF error (cm)', fontsize=y_label_font)
# #plt.title('KITTI real-world LiDAR dataset', fontsize=x_label_font)
# plt.legend(loc='upper left')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/kitti_esdf_occ_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()

# # esdf computation time (cowcity dataset)
# fig24 = plt.figure(figsize=(fig_width, fig_height))
# kitti_voxblox_esdf_time = plt.plot(kitti_voxel_size_esdf, kitti_esdf_time_voxblox, '-o', linewidth = line_width, markersize = marker_size, label='Voxblox', alpha=alpha, zorder = 1,color=cc[1])
# kitti_fiesta_esdf_time = plt.plot(kitti_voxel_size_esdf, kitti_esdf_time_fiesta, '-v', linewidth = line_width, markersize = marker_size, label='FIESTA', alpha=alpha, zorder = 2,color=cc[2])
# kitti_edt_esdf_time = plt.plot(kitti_voxel_size_esdf, kitti_esdf_time_edt, '-X', linewidth = line_width, markersize = marker_size, label='EDT', alpha=alpha, zorder = 1,color=cc[3])
# kitti_ours_esdf_time = plt.plot(kitti_voxel_size_esdf, kitti_esdf_time_ours, '-^', linewidth = line_width, markersize = marker_size, label='Ours', alpha=alpha, zorder = 3,color=cc[0])
# plt.rcParams.update({'font.size': legend_font})
# plt.xticks(np.arange(25, 50, step=5), fontsize=x_tick_font)
# plt.yticks(fontsize=y_tick_log_font)
# plt.ylim([60, 1100])
# plt.yscale('log')
# plt.xlabel('Voxel size (cm)', fontsize=x_label_font)
# plt.ylabel('ESDF update time (ms)', fontsize=y_label_log_font)
# #plt.title('KITTI real-world LiDAR dataset', fontsize=x_label_font)
# plt.legend(loc='upper right')
# plt.savefig("/home/yuepan/Pictures/thesis_experiments/kitti_esdftime_v5.png", dpi=dpi_for_show, bbox_inches = 'tight')
# plt.show()


# Panmap related experiments

# On semantic KITTI dataset

# Panmap GT 3-5-12-12 cm
kitti_panmap_gt_3_12_normal_tsdf = 0.025441
kitti_panmap_gt_3_12_normal_mesh = 0.032008
kitti_panmap_gt_3_12_normal_chamfer = 0.043204
kitti_panmap_gt_3_12_normal_chamfer_l2 = 0.051676
kitti_panmap_gt_3_12_normal_cov = 98.86
kitti_panmap_gt_3_12_normal_time = 127.8

kitti_panmap_gt_3_12_tsdf = 0.039189
kitti_panmap_gt_3_12_mesh = 0.035573
kitti_panmap_gt_3_12_chamfer = 0.048701
kitti_panmap_gt_3_12_chamfer_l2 = 0.058143
kitti_panmap_gt_3_12_cov = 98.57
kitti_panmap_gt_3_12_time = 123.6

# Panmap Rangenet 3-5-12-12 cm
kitti_panmap_nn_3_12_normal_tsdf = 0.027610
kitti_panmap_nn_3_12_normal_mesh = 0.035121
kitti_panmap_nn_3_12_normal_chamfer = 0.049350
kitti_panmap_nn_3_12_normal_chamfer_l2 = 0.055820
kitti_panmap_nn_3_12_normal_cov = 98.75
kitti_panmap_nn_3_12_normal_time = 153.4

kitti_panmap_nn_3_12_tsdf = 0.040040
kitti_panmap_nn_3_12_mesh = 0.036210
kitti_panmap_nn_3_12_chamfer = 0.054281
kitti_panmap_nn_3_12_chamfer_l2 = 0.061241
kitti_panmap_nn_3_12_cov = 98.30
kitti_panmap_nn_3_12_time = 148.8

# Panmap GT 5-10-15-15 cm
kitti_panmap_gt_5_15_normal_tsdf = 0.029673
kitti_panmap_gt_5_15_normal_mesh = 0.047089
kitti_panmap_gt_5_15_normal_chamfer = 0.061025
kitti_panmap_gt_5_15_normal_chamfer_l2 = 0.067733
kitti_panmap_gt_5_15_normal_cov = 99.22
kitti_panmap_gt_5_15_normal_time = 92.5

kitti_panmap_gt_5_15_tsdf = 0.049275
kitti_panmap_gt_5_15_mesh = 0.049521
kitti_panmap_gt_5_15_chamfer = 0.068129
kitti_panmap_gt_5_15_chamfer_l2 = 0.074305
kitti_panmap_gt_5_15_cov = 98.81
kitti_panmap_gt_5_15_time = 88.1

# Panmap Rangenet 5-10-15-15 cm
kitti_panmap_nn_5_15_normal_tsdf = 0.029837
kitti_panmap_nn_5_15_normal_mesh = 0.048265
kitti_panmap_nn_5_15_normal_chamfer = 0.067180
kitti_panmap_nn_5_15_normal_chamfer_l2 = 0.071307
kitti_panmap_nn_5_15_normal_cov = 99.12
kitti_panmap_nn_5_15_normal_time = 96.1

kitti_panmap_nn_5_15_tsdf = 0.049047
kitti_panmap_nn_5_15_mesh = 0.049635
kitti_panmap_nn_5_15_chamfer = 0.074031
kitti_panmap_nn_5_15_chamfer_l2 = 0.078851
kitti_panmap_nn_5_15_cov = 98.92
kitti_panmap_nn_5_15_time = 92.3

# Panmap GT 5-10-25-25 cm
kitti_panmap_gt_5_25_normal_tsdf = 0.041324
kitti_panmap_gt_5_25_normal_mesh = 0.059106
kitti_panmap_gt_5_25_normal_chamfer = 0.082128
kitti_panmap_gt_5_25_normal_chamfer_l2 = 0.103771
kitti_panmap_gt_5_25_normal_cov = 99.33
kitti_panmap_gt_5_25_normal_time = 56.1

kitti_panmap_gt_5_25_tsdf = 0.076445
kitti_panmap_gt_5_25_mesh = 0.064080
kitti_panmap_gt_5_25_chamfer = 0.091128
kitti_panmap_gt_5_25_chamfer_l2 = 0.116551
kitti_panmap_gt_5_25_cov = 98.93
kitti_panmap_gt_5_25_time = 53.2

# Panmap Rangenet 5-10-25-25 cm
kitti_panmap_nn_5_25_normal_tsdf = 0.041711
kitti_panmap_nn_5_25_normal_mesh = 0.066756
kitti_panmap_nn_5_25_normal_chamfer = 0.090437
kitti_panmap_nn_5_25_normal_chamfer_l2 = 0.112207
kitti_panmap_nn_5_25_normal_cov = 99.27
kitti_panmap_nn_5_25_normal_time = 66.7

kitti_panmap_nn_5_25_tsdf = 0.078117
kitti_panmap_nn_5_25_mesh = 0.066834
kitti_panmap_nn_5_25_chamfer = 0.097135
kitti_panmap_nn_5_25_chamfer_l2 = 0.120492
kitti_panmap_nn_5_25_cov = 98.90
kitti_panmap_nn_5_25_time = 64.6


# Panmap GT 5-15-30-30 cm
kitti_panmap_gt_5_30_normal_tsdf = 0.047728
kitti_panmap_gt_5_30_normal_mesh = 0.073682
kitti_panmap_gt_5_30_normal_chamfer = 0.097753
kitti_panmap_gt_5_30_normal_chamfer_l2 = 0.123112
kitti_panmap_gt_5_30_normal_cov = 99.38
kitti_panmap_gt_5_30_normal_time = 45.3

kitti_panmap_gt_5_30_tsdf = 0.082164
kitti_panmap_gt_5_30_mesh = 0.076760
kitti_panmap_gt_5_30_chamfer = 0.106484
kitti_panmap_gt_5_30_chamfer_l2 = 0.136638
kitti_panmap_gt_5_30_cov = 98.91
kitti_panmap_gt_5_30_time = 43.2

# Panmap Rangenet 5-15-30-30 cm
kitti_panmap_nn_5_30_normal_tsdf = 0.049494
kitti_panmap_nn_5_30_normal_mesh = 0.084231
kitti_panmap_nn_5_30_normal_chamfer = 0.105793
kitti_panmap_nn_5_30_normal_chamfer_l2 = 0.135201
kitti_panmap_nn_5_30_normal_cov = 99.34
kitti_panmap_nn_5_30_normal_time = 50.7

kitti_panmap_nn_5_30_tsdf = 0.084581
kitti_panmap_nn_5_30_mesh = 0.085664
kitti_panmap_nn_5_30_chamfer = 0.113184
kitti_panmap_nn_5_30_chamfer_l2 = 0.144015
kitti_panmap_nn_5_30_cov = 98.88
kitti_panmap_nn_5_30_time = 48.3


# Panmap GT 10-15-30-30 cm
kitti_panmap_gt_10_30_normal_tsdf = 0.048672
kitti_panmap_gt_10_30_normal_mesh = 0.077897
kitti_panmap_gt_10_30_normal_chamfer = 0.101211
kitti_panmap_gt_10_30_normal_chamfer_l2 = 0.129826
kitti_panmap_gt_10_30_normal_cov = 99.33
kitti_panmap_gt_10_30_normal_time = 40.6

kitti_panmap_gt_10_30_tsdf = 0.082794
kitti_panmap_gt_10_30_mesh = 0.081446
kitti_panmap_gt_10_30_chamfer = 0.109001
kitti_panmap_gt_10_30_chamfer_l2 = 0.142227
kitti_panmap_gt_10_30_cov = 98.75
kitti_panmap_gt_10_30_time = 38.1

# Panmap Rangenet 10-15-30-30 cm
kitti_panmap_nn_10_30_normal_tsdf = 0.049438
kitti_panmap_nn_10_30_normal_mesh = 0.088738
kitti_panmap_nn_10_30_normal_chamfer = 0.110728
kitti_panmap_nn_10_30_normal_chamfer_l2 = 0.139743
kitti_panmap_nn_10_30_normal_cov = 99.28
kitti_panmap_nn_10_30_normal_time = 47.9

kitti_panmap_nn_10_30_tsdf = 0.084472
kitti_panmap_nn_10_30_mesh = 0.092015
kitti_panmap_nn_10_30_chamfer = 0.119734
kitti_panmap_nn_10_30_chamfer_l2 = 0.150291
kitti_panmap_nn_10_30_cov = 98.68
kitti_panmap_nn_10_30_time = 45.3

# Panmap GT 10-20-40-40 cm
kitti_panmap_gt_10_40_normal_tsdf = 0.062719
kitti_panmap_gt_10_40_normal_mesh = 0.101403
kitti_panmap_gt_10_40_normal_chamfer = 0.132846
kitti_panmap_gt_10_40_normal_chamfer_l2 = 0.175044
kitti_panmap_gt_10_40_normal_cov = 99.31
kitti_panmap_gt_10_40_normal_time = 33.5

kitti_panmap_gt_10_40_tsdf = 0.105634
kitti_panmap_gt_10_40_mesh = 0.105154
kitti_panmap_gt_10_40_chamfer = 0.142429
kitti_panmap_gt_10_40_chamfer_l2 = 0.187198
kitti_panmap_gt_10_40_cov = 98.80
kitti_panmap_gt_10_40_time = 31.8

# Panmap Rangenet 10-20-40-40 cm
kitti_panmap_nn_10_40_normal_tsdf = 0.064324
kitti_panmap_nn_10_40_normal_mesh = 0.121589
kitti_panmap_nn_10_40_normal_chamfer = 0.149718
kitti_panmap_nn_10_40_normal_chamfer_l2 = 0.190226
kitti_panmap_nn_10_40_normal_cov = 99.25
kitti_panmap_nn_10_40_normal_time = 39.4

kitti_panmap_nn_10_40_tsdf = 0.109667
kitti_panmap_nn_10_40_mesh = 0.123705
kitti_panmap_nn_10_40_chamfer = 0.158041
kitti_panmap_nn_10_40_chamfer_l2 = 0.199888
kitti_panmap_nn_10_40_cov = 98.72
kitti_panmap_nn_10_40_time = 37.7


# voxblox
kitti_voxblox_10_normal_tsdf = 0.054706
kitti_voxblox_10_normal_mesh = 0.053939
kitti_voxblox_10_normal_chamfer = 0.061305
kitti_voxblox_10_normal_chamfer_l2 = 0.070184
kitti_voxblox_10_normal_cov = 93.28
kitti_voxblox_10_normal_time = 284.3

kitti_voxblox_10_tsdf = 0.076666
kitti_voxblox_10_mesh = 0.052804
kitti_voxblox_10_chamfer = 0.063471
kitti_voxblox_10_chamfer_l2 = 0.070230
kitti_voxblox_10_cov = 92.00
kitti_voxblox_10_time = 261.5

kitti_voxblox_15_normal_tsdf = 0.076064
kitti_voxblox_15_normal_mesh = 0.071321
kitti_voxblox_15_normal_chamfer = 0.080357
kitti_voxblox_15_normal_chamfer_l2 = 0.096353
kitti_voxblox_15_normal_cov = 95.87
kitti_voxblox_15_normal_time = 152.3

kitti_voxblox_15_tsdf = 0.111605
kitti_voxblox_15_mesh = 0.072688
kitti_voxblox_15_chamfer = 0.083412
kitti_voxblox_15_chamfer_l2 = 0.098754
kitti_voxblox_15_cov = 94.27
kitti_voxblox_15_time = 140.9

kitti_voxblox_normal_20_tsdf = 0.091253
kitti_voxblox_normal_20_mesh = 0.100758
kitti_voxblox_normal_20_chamfer = 0.109573
kitti_voxblox_normal_20_chamfer_l2 = 0.127042
kitti_voxblox_normal_20_cov = 96.86
kitti_voxblox_normal_20_time = 105.7

kitti_voxblox_20_tsdf = 0.142605
kitti_voxblox_20_mesh = 0.095083
kitti_voxblox_20_chamfer = 0.110612
kitti_voxblox_20_chamfer_l2 = 0.126423
kitti_voxblox_20_cov = 95.88
kitti_voxblox_20_time = 98.4

kitti_voxblox_normal_25_tsdf = 0.117357
kitti_voxblox_normal_25_mesh = 0.110055
kitti_voxblox_normal_25_chamfer = 0.128464
kitti_voxblox_normal_25_chamfer_l2 = 0.153013
kitti_voxblox_normal_25_cov = 97.62
kitti_voxblox_normal_25_time = 75.5

kitti_voxblox_25_tsdf = 0.172805
kitti_voxblox_25_mesh = 0.112696
kitti_voxblox_25_chamfer = 0.126414
kitti_voxblox_25_chamfer_l2 = 0.155122
kitti_voxblox_25_cov = 96.89
kitti_voxblox_25_time = 68.1

kitti_voxblox_30_normal_tsdf = 0.140246
kitti_voxblox_30_normal_mesh = 0.136142
kitti_voxblox_30_normal_chamfer = 0.148603
kitti_voxblox_30_normal_chamfer_l2 = 0.185023
kitti_voxblox_30_normal_cov = 97.88
kitti_voxblox_30_normal_time = 53.8

kitti_voxblox_30_tsdf = 0.198182
kitti_voxblox_30_mesh = 0.129695
kitti_voxblox_30_chamfer = 0.149271
kitti_voxblox_30_chamfer_l2 = 0.183367
kitti_voxblox_30_cov = 96.86
kitti_voxblox_30_time = 48.5

kitti_voxblox_35_normal_tsdf = 0.163294
kitti_voxblox_35_normal_mesh = 0.153278
kitti_voxblox_35_normal_chamfer = 0.175990
kitti_voxblox_35_normal_chamfer_l2 = 0.214261
kitti_voxblox_35_normal_cov = 98.17
kitti_voxblox_35_normal_time = 39.3

kitti_voxblox_35_tsdf = 0.229126
kitti_voxblox_35_mesh = 0.149492
kitti_voxblox_35_chamfer = 0.171398
kitti_voxblox_35_chamfer_l2 = 0.214261
kitti_voxblox_35_cov = 97.44
kitti_voxblox_35_time = 34.4

kitti_voxblox_40_normal_tsdf = 0.184160
kitti_voxblox_40_normal_mesh = 0.187616
kitti_voxblox_40_normal_chamfer = 0.196009
kitti_voxblox_40_normal_chamfer_l2 = 0.255842
kitti_voxblox_40_normal_cov = 98.40
kitti_voxblox_40_normal_time = 30.5

kitti_voxblox_40_tsdf = 0.261363
kitti_voxblox_40_mesh = 0.169810
kitti_voxblox_40_chamfer = 0.195541
kitti_voxblox_40_chamfer_l2 = 0.244005
kitti_voxblox_40_cov = 97.70
kitti_voxblox_40_time = 26.2

# cblox
kitti_cblox_7_tsdf = 0.108217
kitti_cblox_7_chamfer = 0.065143
kitti_cblox_7_chamfer_l2 = 0.074180
kitti_cblox_7_time = 185.5

kitti_cblox_10_tsdf = 0.150266
kitti_cblox_10_chamfer = 0.083488
kitti_cblox_10_time = 113.5

kitti_cblox_15_tsdf = 0.194661
kitti_cblox_15_chamfer = 0.125392
kitti_cblox_15_time = 56.1

kitti_cblox_20_tsdf = 0.226104
kitti_cblox_20_chamfer = 0.167096
kitti_cblox_20_time = 39.4

kitti_cblox_25_tsdf = 0.250282
kitti_cblox_25_chamfer = 0.203289
kitti_cblox_25_time = 32.7

kitti_cblox_30_tsdf = 0.271832
kitti_cblox_30_chamfer = 0.243917
kitti_cblox_30_time = 26.6

kitti_cblox_35_tsdf = 0.310542
kitti_cblox_35_chamfer = 0.279082
kitti_cblox_35_time = 22.4

kitti_cblox_40_tsdf = 0.367881
kitti_cblox_40_chamfer = 0.304851
kitti_cblox_40_time = 17.3

chamfer_panmap_gt_normal = np.array([kitti_panmap_gt_3_12_normal_chamfer, kitti_panmap_gt_5_15_normal_chamfer, kitti_panmap_gt_5_25_normal_chamfer, kitti_panmap_gt_5_30_normal_chamfer, kitti_panmap_gt_10_30_normal_chamfer, kitti_panmap_gt_10_40_normal_chamfer]) * 100.0 
chamfer_panmap_gt = np.array([kitti_panmap_gt_3_12_chamfer, kitti_panmap_gt_5_15_chamfer, kitti_panmap_gt_5_25_chamfer, kitti_panmap_gt_5_30_chamfer, kitti_panmap_gt_10_30_chamfer, kitti_panmap_gt_10_40_chamfer]) * 100.0 
chamfer_panmap_nn_normal = np.array([kitti_panmap_nn_3_12_normal_chamfer, kitti_panmap_nn_5_15_normal_chamfer, kitti_panmap_nn_5_25_normal_chamfer, kitti_panmap_nn_5_30_normal_chamfer, kitti_panmap_nn_10_30_normal_chamfer, kitti_panmap_nn_10_40_normal_chamfer]) * 100.0 
chamfer_panmap_nn = np.array([kitti_panmap_nn_3_12_chamfer, kitti_panmap_nn_5_15_chamfer, kitti_panmap_nn_5_25_chamfer, kitti_panmap_nn_5_30_chamfer, kitti_panmap_nn_10_30_chamfer, kitti_panmap_nn_10_40_chamfer]) * 100.0 
# chamfer_voxblox_normal =
chamfer_voxblox = np.array([kitti_voxblox_10_chamfer, kitti_voxblox_15_chamfer, kitti_voxblox_20_chamfer, kitti_voxblox_25_chamfer, kitti_voxblox_30_chamfer, kitti_voxblox_35_chamfer, kitti_voxblox_40_chamfer]) * 100.0 
chamfer_cblox = np.array([kitti_cblox_7_chamfer, kitti_cblox_10_chamfer, kitti_cblox_15_chamfer, kitti_cblox_20_chamfer, kitti_cblox_25_chamfer, kitti_cblox_30_chamfer, kitti_cblox_35_chamfer, kitti_cblox_40_chamfer]) * 100.0 

time_panmap_gt_normal = np.array([kitti_panmap_gt_3_12_normal_time, kitti_panmap_gt_5_15_normal_time, kitti_panmap_gt_5_25_normal_time, kitti_panmap_gt_5_30_normal_time, kitti_panmap_gt_10_30_normal_time, kitti_panmap_gt_10_40_normal_time])
time_panmap_gt = np.array([kitti_panmap_gt_3_12_time, kitti_panmap_gt_5_15_time, kitti_panmap_gt_5_25_time, kitti_panmap_gt_5_30_time, kitti_panmap_gt_10_30_time, kitti_panmap_gt_10_40_time])
time_panmap_nn_normal = np.array([kitti_panmap_nn_3_12_normal_time, kitti_panmap_nn_5_15_normal_time, kitti_panmap_nn_5_25_normal_time, kitti_panmap_nn_5_30_normal_time, kitti_panmap_nn_10_30_normal_time, kitti_panmap_nn_10_40_normal_time])
time_panmap_nn = np.array([kitti_panmap_nn_3_12_time, kitti_panmap_nn_5_15_time, kitti_panmap_nn_5_25_time, kitti_panmap_nn_5_30_time, kitti_panmap_nn_10_30_time, kitti_panmap_nn_10_40_time])
time_voxblox = np.array([kitti_voxblox_10_time, kitti_voxblox_15_time, kitti_voxblox_20_time, kitti_voxblox_25_time, kitti_voxblox_30_time, kitti_voxblox_35_time, kitti_voxblox_40_time])
time_cblox = np.array([kitti_cblox_7_time, kitti_cblox_10_time, kitti_cblox_15_time, kitti_cblox_20_time, kitti_cblox_25_time, kitti_cblox_30_time, kitti_cblox_35_time, kitti_cblox_40_time])


fig_width_2 = 12
fig_height_2 = 8

# Plot the results
fig101 = plt.figure(figsize=(fig_width_2, fig_height_2))
plt.rcParams.update({'font.size': legend_font})

line_voxblox = plt.plot(time_voxblox,  chamfer_voxblox, '-o', linewidth=line_width, markersize=marker_size, label='Voxblox',  alpha=alpha, color = cc[1])
#line_cblox = plt.plot(time_cblox, chamfer_cblox, '-X', linewidth=line_width, markersize=marker_size, label='C-blox',  alpha=alpha, color = cc[2])
line_panmap_gt = plt.plot(time_panmap_gt,  chamfer_panmap_gt, '-^', linewidth=line_width, markersize=marker_size, label='Panmap (ground truth)',  alpha=alpha, color = cc[3])
line_panmap_gt_normal = plt.plot(time_panmap_gt_normal,  chamfer_panmap_gt_normal, '-^', linewidth=line_width, markersize=marker_size, label='Voxfield Panmap (ground truth)',  alpha=alpha, color = cc[0])
line_panmap_nn = plt.plot(time_panmap_nn,  chamfer_panmap_nn, '--v', linewidth=line_width, markersize=marker_size, label='Panmap (predictions)',  alpha=alpha, color = cc[6])
line_panmap_nn_normal = plt.plot(time_panmap_nn_normal,  chamfer_panmap_nn_normal, '--v', linewidth=line_width, markersize=marker_size, label='Voxfield Panmap (predictions)',  alpha=alpha, color = cc[9])

line_realtime = plt.plot([100, 100], [0, 38], '--k', linewidth=1, label='Real time (10Hz)')

#plt.annotate('40cm', (time_cblox[-1], chamfer_cblox[-1]), xytext=(time_cblox[-1]+2, chamfer_cblox[-1]+0.5), color=cc[2])
#plt.annotate('30cm', (time_cblox[-3], chamfer_cblox[-3]), xytext=(time_cblox[-3]+2, chamfer_cblox[-3]+0.5), color=cc[2])
plt.annotate('40cm', (time_voxblox[-1], chamfer_voxblox[-1]), xytext=(time_voxblox[-1]-7, chamfer_voxblox[-1]+0.5), color=cc[1])

#plt.annotate('10cm', (time_cblox[1], chamfer_cblox[1]), xytext=(time_cblox[1]+1, chamfer_cblox[1]+0.2), color=cc[2])
plt.annotate('15cm', (time_voxblox[1], chamfer_voxblox[1]), xytext=(time_voxblox[1]+2, chamfer_voxblox[1]+0.5), color=cc[1])

plt.annotate('3-12cm', (time_panmap_gt_normal[0], chamfer_panmap_gt_normal[0]), xytext=(time_panmap_gt_normal[0]+3, chamfer_panmap_gt_normal[0]-0.2), color=cc[0])
plt.annotate('5-25cm', (time_panmap_gt_normal[2], chamfer_panmap_gt_normal[2]), xytext=(time_panmap_gt_normal[2]-2, chamfer_panmap_gt_normal[2]-1.5), color=cc[0])
plt.annotate('10-40cm', (time_panmap_gt_normal[-1], chamfer_panmap_gt_normal[-1]),  xytext=(time_panmap_gt_normal[-1]-15, chamfer_panmap_gt_normal[-1]+1.5), color=cc[0])

# plt.arrow(50, -2, 5, 0) 

plt.xticks(fontsize=x_tick_font)
plt.yticks(fontsize=y_tick_font)

# plt.xscale('log')
plt.xlim((15, 170))
plt.ylim((3, 21))

plt.xlabel('Time per frame (ms)', fontsize=x_label_font)
plt.ylabel('Reconstruction Chamfer distance (cm)', fontsize=y_label_font)
# plt.title('KITTI real-world LiDAR dataset', fontsize=x_label_font+5)
plt.legend(loc='upper right', fontsize=legend_font, handlelength=3.0)
plt.savefig("/home/yuepan/Pictures/thesis_experiments/kitti_panmap_compare_text_v6.png", dpi=dpi_for_show, bbox_inches = 'tight')
plt.show()


# labels = ['Flat', 'KITTI']
# without_normal_mae_mean = [0.91, 6.80]
# # without_normal_mae_std = [0.02, 0.07]
# with_normal_mae_mean = [0.76, 4.29]
# # with_normal_mae_std = [0.01, 0.05]

# without_normal_coverage_mean = [66.62, 94.35]
# # without_normal_coverage_std = []
# with_normal_coverage_mean = [76.11, 96.82]
# # with_normal_coverage_std = []

# x = np.arange(len(labels))  # the label locations
# width = 0.25  # the width of the bars

# plt.rcParams.update({'font.size': 24})

# fig_2, ax_2 = plt.subplots()
# rects1 = ax_2.bar(x - width/2, without_normal_mae_mean, width, label='Without correction')
# rects2 = ax_2.bar(x + width/2, with_normal_mae_mean, width, label='With correction (Ours)')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax_2.set_ylabel('Mean reconstruction error (cm)')
# ax_2.set_xticks(x)
# ax_2.set_xticklabels(labels)

# ax_2.legend()

# fig_2.set_size_inches(9.5, 10.5)
# fig_2.tight_layout()


# fig_3, ax_3 = plt.subplots()
# rects1 = ax_3.bar(x - width/2, without_normal_coverage_mean, width, label='Without correction')
# rects2 = ax_3.bar(x + width/2, with_normal_coverage_mean, width, label='With correction (Ours)')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax_3.set_ylabel('Coverage (%)')
# ax_3.set_xticks(x)
# ax_3.set_xticklabels(labels)

# ax_3.legend()

# fig_3.set_size_inches(9.5, 10.5)
# fig_3.tight_layout()

# plt.show()