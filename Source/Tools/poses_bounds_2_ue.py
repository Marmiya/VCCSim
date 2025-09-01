# -*- coding: utf-8 -*-
"""
将 Mip-NeRF 360 / LLFF 的 poses_bounds.npy 转换并输出为PLY文件
同时输出UE格式的文本文件

两步转换法：
1. LLFF [right, up, back] -> 标准右手系 [forward(X), right(Y), up(Z)]
2. 标准右手系 -> UE左手系 (翻转Y轴，米转厘米)

注意：UE相机默认朝向X轴正方向（Forward方向）
"""

import os
import math
import numpy as np

INPUT_NPY  = r"D:\Data\360_v2\garden\poses_bounds.npy"
OUTPUT_DIR = r"C:\UEProjects\VCCSimDev\Saved"
CAMERAS_RH_PLY = os.path.join(OUTPUT_DIR, "llff_cameras_rh.ply")  # 右手系
CAMERAS_LH_PLY = os.path.join(OUTPUT_DIR, "llff_cameras_lh.ply")  # 左手系(UE)
POSES_TXT = os.path.join(OUTPUT_DIR, "poses_original.txt")        # UE格式文本


def load_poses_bounds(npy_path):
    data = np.load(npy_path)  # shape (N, 17)
    if data.ndim != 2 or data.shape[1] < 15:
        raise ValueError("Invalid poses_bounds.npy format: need shape (N, >=17) with first 15 as poses.")
    return data


def write_ply(path: str,
              xyz: np.ndarray,
              rgb: np.ndarray = None,
              normals: np.ndarray = None):
    """写入PLY文件，参考colmap_2_ply.py的实现"""
    n = int(xyz.shape[0])
    has_rgb = rgb is not None and rgb.shape[0] == n
    has_n = normals is not None and normals.shape[0] == n
    with open(path, 'w', encoding='utf-8') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_n:
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
        if has_rgb:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            row = [f"{xyz[i,0]:.6f}", f"{xyz[i,1]:.6f}", f"{xyz[i,2]:.6f}"]
            if has_n:
                row += [f"{normals[i,0]:.6f}", f"{normals[i,1]:.6f}", f"{normals[i,2]:.6f}"]
            if has_rgb:
                row += [str(int(rgb[i,0])), str(int(rgb[i,1])), str(int(rgb[i,2]))]
            f.write(" ".join(row) + "\n")


def llff_to_standard_rh(c2w_llff):
    """第一步：LLFF -> 标准右手系"""
    R_llff = c2w_llff[:, :3]
    t_llff = c2w_llff[:, 3]
    
    right = R_llff[:, 0]
    up = R_llff[:, 1]  
    back = R_llff[:, 2]
    forward = -back
    
    R_std_rh = np.column_stack([forward, right, up])
    t_std_rh = t_llff.copy()
    
    return R_std_rh, t_std_rh


def standard_rh_to_ue(R_std_rh, t_std_rh):
    """第二步：标准右手系 -> UE左手系"""
    coord_transform = np.diag([1.0, -1.0, 1.0])
    
    R_ue = coord_transform @ R_std_rh
    t_ue = coord_transform @ t_std_rh * 100.0
    
    return R_ue, t_ue


def llff_to_ue_transform_two_step(c2w_llff):
    """两步转换：LLFF -> 标准右手系 -> UE左手系"""
    R_std_rh, t_std_rh = llff_to_standard_rh(c2w_llff)
    R_ue, t_ue = standard_rh_to_ue(R_std_rh, t_std_rh)
    return R_ue, t_ue


def rotation_matrix_to_euler_degrees(R):
    """旋转矩阵转欧拉角（度），UE的FRotator格式"""
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    
    if not singular:
        pitch = math.atan2(-R[2,0], sy)
        yaw = math.atan2(R[1,0], R[0,0]) 
        roll = math.atan2(R[2,1], R[2,2])
    else:
        pitch = math.atan2(-R[2,0], sy)
        yaw = math.atan2(-R[0,1], R[1,1])
        roll = 0
    
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def convert_single_pose(row15):
    """转换单个LLFF姿态为UE格式"""
    pose_3x5 = row15.reshape(3, 5)
    c2w = pose_3x5[:, :4]
    
    R_ue, t_ue = llff_to_ue_transform_two_step(c2w)
    pitch, yaw, roll = rotation_matrix_to_euler_degrees(R_ue)
    X, Y, Z = t_ue.tolist()
    
    return X, Y, Z, pitch, yaw, roll


def main():
    if not os.path.isfile(INPUT_NPY):
        raise FileNotFoundError(f"Input file not found: {INPUT_NPY}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_poses_bounds(INPUT_NPY)
    N = data.shape[0]
    
    print(f"[info] Loaded {N} camera poses")

    # 处理所有帧，同时生成右手系和左手系数据
    camera_positions_rh = []
    camera_directions_rh = []
    camera_positions_lh = []
    camera_directions_lh = []
    ue_poses = []  # 存储UE格式的姿态
    
    for i in range(N):
        row = data[i, :15]
        pose_3x5 = row.reshape(3, 5)
        c2w = pose_3x5[:, :4]
        
        # 第一步：转换到标准右手系
        R_std_rh, t_std_rh = llff_to_standard_rh(c2w)
        cam_pos_rh = t_std_rh
        # UE相机默认朝向X轴正方向（Forward）
        cam_dir_rh = R_std_rh[:, 0]  # X轴是UE相机的forward方向
        
        camera_positions_rh.append(cam_pos_rh)
        camera_directions_rh.append(cam_dir_rh)
        
        # 第二步：转换到UE左手系
        R_ue, t_ue = standard_rh_to_ue(R_std_rh, t_std_rh)
        cam_pos_lh = t_ue
        # UE相机默认朝向X轴正方向（Forward）
        cam_dir_lh = R_ue[:, 0]  # X轴是UE相机的forward方向
        
        camera_positions_lh.append(cam_pos_lh)
        camera_directions_lh.append(cam_dir_lh)
        
        # 计算UE格式的姿态（位置+欧拉角）
        pitch, yaw, roll = rotation_matrix_to_euler_degrees(R_ue)
        X, Y, Z = t_ue.tolist()
        ue_poses.append(f"{X:.6f} {Y:.6f} {Z:.6f} {pitch:.6f} {yaw:.6f} {roll:.6f}")
    
    # 转换为numpy数组
    cam_xyz_rh = np.array(camera_positions_rh)
    cam_dirs_rh = np.array(camera_directions_rh)
    cam_xyz_lh = np.array(camera_positions_lh)
    cam_dirs_lh = np.array(camera_directions_lh)
    
    # 归一化方向向量
    norms_rh = np.linalg.norm(cam_dirs_rh, axis=1, keepdims=True)
    cam_dirs_rh = cam_dirs_rh / np.maximum(norms_rh, 1e-12)
    
    norms_lh = np.linalg.norm(cam_dirs_lh, axis=1, keepdims=True)
    cam_dirs_lh = cam_dirs_lh / np.maximum(norms_lh, 1e-12)
    
    # 设置不同颜色区分
    cam_rgb_rh = np.tile(np.array([0, 255, 0], dtype=np.uint8), (N, 1))  # 绿色 - 右手系
    cam_rgb_lh = np.tile(np.array([255, 0, 0], dtype=np.uint8), (N, 1))  # 红色 - 左手系(UE)
    
    # 写入右手系PLY文件
    print(f"[info] Writing right-hand PLY file: {CAMERAS_RH_PLY}")
    write_ply(CAMERAS_RH_PLY, cam_xyz_rh, rgb=cam_rgb_rh, normals=cam_dirs_rh)
    
    # 写入左手系PLY文件
    print(f"[info] Writing left-hand PLY file: {CAMERAS_LH_PLY}")
    write_ply(CAMERAS_LH_PLY, cam_xyz_lh, rgb=cam_rgb_lh, normals=cam_dirs_lh)
    
    # 写入UE格式文本文件
    print(f"[info] Writing UE format text file: {POSES_TXT}")
    with open(POSES_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(ue_poses) + "\n")
    
    print(f"[done] Output files:")
    print(f"  Right-hand PLY (green, meters): {CAMERAS_RH_PLY}")
    print(f"  Left-hand PLY (red, centimeters): {CAMERAS_LH_PLY}")
    print(f"  UE format text (X Y Z Pitch Yaw Roll): {POSES_TXT}")
    print(f"Use MeshLab to view PLY files: green arrows = right-hand, red arrows = left-hand (UE)")


if __name__ == "__main__":
    main()