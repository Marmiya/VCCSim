import numpy as np
import os
import random
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import itertools

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """四元数转旋转矩阵"""
    qw, qx, qy, qz = q / np.linalg.norm(q)
    return np.array([
        [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)]
    ], dtype=float)

def read_camera_centers_from_images_txt(filepath: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    从images.txt读取相机中心坐标
    COLMAP定义: x_cam = R * X_world + t
    相机中心: C = -R^T * t
    """
    centers = []
    image_names = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith('#'):
            i += 1
            continue
        
        parts = line.split()
        if len(parts) >= 10:
            # 读取姿态参数
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            image_name = parts[9]  # 图像名称
            
            # 计算相机中心
            q = np.array([qw, qx, qy, qz])
            t = np.array([tx, ty, tz])
            R = quaternion_to_rotation_matrix(q)
            
            # C = -R^T * t
            center = -R.T @ t
            
            centers.append(center)
            image_names.append(image_name)
        
        i += 2  # 跳过下一行（特征点行）
    
    return centers, image_names

def find_common_cameras(centers1: List[np.ndarray], names1: List[str], 
                       centers2: List[np.ndarray], names2: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """找到共同的相机并返回对应的中心坐标"""
    name_to_center1 = {name: center for name, center in zip(names1, centers1)}
    name_to_center2 = {name: center for name, center in zip(names2, centers2)}
    
    common_names = set(names1) & set(names2)
    print(f"找到 {len(common_names)} 个共同图像")
    
    if len(common_names) < 3:
        raise ValueError(f"共同图像数量不足: {len(common_names)} < 3")
    
    # 提取共同相机的中心坐标
    common_centers1 = []
    common_centers2 = []
    
    for name in common_names:
        common_centers1.append(name_to_center1[name])
        common_centers2.append(name_to_center2[name])
    
    return np.array(common_centers1), np.array(common_centers2)

def estimate_scale_robust_median(centers_rc: np.ndarray, centers_base: np.ndarray, 
                                max_pairs: int = 5000) -> Tuple[float, Dict]:
    """
    方法1: 稳健中位数法估计尺度
    计算所有相机对的距离比值，取中位数
    """
    n_cameras = len(centers_rc)
    
    # 生成所有可能的相机对
    all_pairs = list(itertools.combinations(range(n_cameras), 2))
    
    # 如果相机对太多，随机采样
    if len(all_pairs) > max_pairs:
        pairs_to_use = random.sample(all_pairs, max_pairs)
        print(f"随机采样 {max_pairs} 对相机（总共 {len(all_pairs)} 对）")
    else:
        pairs_to_use = all_pairs
        print(f"使用全部 {len(all_pairs)} 对相机")
    
    ratios = []
    valid_pairs = []
    
    for i, j in pairs_to_use:
        # 计算RC重建中的距离
        dist_rc = np.linalg.norm(centers_rc[i] - centers_rc[j])
        # 计算BASE重建中的距离
        dist_base = np.linalg.norm(centers_base[i] - centers_base[j])
        
        # 避免除零
        if dist_rc > 1e-6 and dist_base > 1e-6:
            ratio = dist_base / dist_rc
            ratios.append(ratio)
            valid_pairs.append((i, j))
    
    ratios = np.array(ratios)
    
    # 计算中位数作为尺度因子
    scale_median = np.median(ratios)
    
    # 计算MAD (Median Absolute Deviation) 用于异常值检测
    mad = np.median(np.abs(ratios - scale_median))
    
    # 过滤异常值 (超过3*MAD的被认为是异常值)
    mad_threshold = 3.0
    valid_mask = np.abs(ratios - scale_median) <= mad_threshold * mad
    ratios_filtered = ratios[valid_mask]
    
    scale_robust = np.median(ratios_filtered)
    
    stats = {
        'n_pairs_total': len(all_pairs),
        'n_pairs_used': len(valid_pairs),
        'n_pairs_after_filter': len(ratios_filtered),
        'scale_median': scale_median,
        'scale_robust': scale_robust,
        'mad': mad,
        'ratios_mean': np.mean(ratios),
        'ratios_std': np.std(ratios),
        'ratios_min': np.min(ratios),
        'ratios_max': np.max(ratios),
        'outlier_ratio': 1 - len(ratios_filtered) / len(ratios)
    }
    
    return scale_robust, stats

def estimate_scale_procrustes(centers_rc: np.ndarray, centers_base: np.ndarray) -> Tuple[float, Dict]:
    """
    方法2: Procrustes分析估计尺度
    在允许旋转和平移的情况下找到最优尺度
    """
    # 去中心化
    centroid_rc = np.mean(centers_rc, axis=0)
    centroid_base = np.mean(centers_base, axis=0)
    
    centered_rc = centers_rc - centroid_rc
    centered_base = centers_base - centroid_base
    
    # 计算协方差矩阵
    H = centered_rc.T @ centered_base
    
    # SVD分解
    U, sigma, Vt = np.linalg.svd(H)
    
    # 最优尺度: s = trace(Sigma) / ||X_RC_centered||_F^2
    trace_sigma = np.sum(sigma)
    frobenius_norm_squared = np.sum(centered_rc ** 2)
    
    scale_procrustes = trace_sigma / frobenius_norm_squared
    
    # 计算最优旋转矩阵（用于统计）
    R_optimal = Vt.T @ U.T
    
    # 确保旋转矩阵的行列式为正
    if np.linalg.det(R_optimal) < 0:
        Vt[-1, :] *= -1
        R_optimal = Vt.T @ U.T
    
    # 计算对齐后的残差
    aligned_rc = scale_procrustes * (R_optimal @ centered_rc.T).T + centroid_base
    residuals = np.linalg.norm(aligned_rc - centers_base, axis=1)
    
    stats = {
        'scale_procrustes': scale_procrustes,
        'trace_sigma': trace_sigma,
        'frobenius_norm_squared': frobenius_norm_squared,
        'mean_residual': np.mean(residuals),
        'max_residual': np.max(residuals),
        'rms_residual': np.sqrt(np.mean(residuals**2))
    }
    
    return scale_procrustes, stats

def visualize_distance_ratios(ratios: np.ndarray, scale_estimate: float, title: str, output_path: str):
    """可视化距离比值分布"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(ratios, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(scale_estimate, color='red', linestyle='--', linewidth=2, label=f'估计尺度: {scale_estimate:.4f}')
    plt.xlabel('距离比值')
    plt.ylabel('频次')
    plt.title('距离比值分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.boxplot(ratios)
    plt.ylabel('距离比值')
    plt.title('距离比值箱线图')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(sorted(ratios), marker='o', markersize=2, alpha=0.6)
    plt.axhline(scale_estimate, color='red', linestyle='--', linewidth=2, label=f'估计尺度: {scale_estimate:.4f}')
    plt.xlabel('排序索引')
    plt.ylabel('距离比值')
    plt.title('距离比值排序图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"距离比值可视化保存到: {output_path}")
    plt.close()

def apply_scale_to_colmap_model(input_dir: str, output_dir: str, scale: float):
    """对COLMAP模型应用尺度变换"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制cameras.txt
    import shutil
    shutil.copy(os.path.join(input_dir, 'cameras.txt'), 
                os.path.join(output_dir, 'cameras.txt'))
    
    # 处理images.txt - 只缩放平移向量
    with open(os.path.join(input_dir, 'images.txt'), 'r') as f:
        lines = f.readlines()
    
    with open(os.path.join(output_dir, 'images.txt'), 'w') as f:
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith('#'):
                f.write(line)
                i += 1
                continue
            
            parts = line.strip().split()
            if len(parts) >= 10:
                # 缩放平移向量
                tx, ty, tz = map(float, parts[5:8])
                tx_scaled = tx * scale
                ty_scaled = ty * scale  
                tz_scaled = tz * scale
                
                # 重写这一行
                new_line = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]} {parts[4]} "
                new_line += f"{tx_scaled:.8f} {ty_scaled:.8f} {tz_scaled:.8f} {parts[8]} {parts[9]}\n"
                f.write(new_line)
                
                # 复制下一行（特征点行）
                i += 1
                if i < len(lines):
                    f.write(lines[i])
            i += 1
    
    # 处理points3D.txt - 缩放所有3D点
    with open(os.path.join(input_dir, 'points3D.txt'), 'r') as f:
        lines = f.readlines()
    
    with open(os.path.join(output_dir, 'points3D.txt'), 'w') as f:
        for line in lines:
            if line.startswith('#'):
                f.write(line)
                continue
            
            parts = line.strip().split()
            if len(parts) >= 8:
                # 缩放3D点坐标
                x, y, z = map(float, parts[1:4])
                x_scaled = x * scale
                y_scaled = y * scale
                z_scaled = z * scale
                
                # 重写这一行
                new_line = f"{parts[0]} {x_scaled:.8f} {y_scaled:.8f} {z_scaled:.8f}"
                for part in parts[4:]:
                    new_line += f" {part}"
                new_line += "\n"
                f.write(new_line)

def main():
    # 配置路径
    rc_images_path = r"D:\Data\360_v2\garden\mesh\rc_colmap\images.txt"
    base_images_path = r"D:\Data\360_v2\garden\sparse\0\images.txt"
    output_dir = r"D:\Data\360_v2\garden\mesh\rc_scale_aligned"
    
    print("=== 基于相机中心距离的尺度估计 ===")
    
    # 检查并转换文件格式
    def ensure_txt_format(model_dir, images_path):
        if not os.path.exists(images_path):
            print(f"转换 {model_dir} 为TXT格式...")
            txt_dir = model_dir + "_txt"
            os.system(f'colmap model_converter --input_path "{model_dir}" --output_path "{txt_dir}" --output_type TXT')
            return os.path.join(txt_dir, 'images.txt')
        return images_path
    
    rc_images_path = ensure_txt_format(r"D:\Data\360_v2\garden\mesh\rc_colmap", rc_images_path)
    base_images_path = ensure_txt_format(r"D:\Data\360_v2\garden\sparse\0", base_images_path)
    
    # 读取相机中心
    print("读取RC重建的相机中心...")
    centers_rc, names_rc = read_camera_centers_from_images_txt(rc_images_path)
    
    print("读取BASE重建的相机中心...")
    centers_base, names_base = read_camera_centers_from_images_txt(base_images_path)
    
    print(f"RC重建: {len(centers_rc)} 个相机")
    print(f"BASE重建: {len(centers_base)} 个相机")
    
    # 找到共同的相机
    common_centers_rc, common_centers_base = find_common_cameras(
        centers_rc, names_rc, centers_base, names_base
    )
    
    print(f"\n=== 方法1: 稳健中位数法 ===")
    scale_median, stats_median = estimate_scale_robust_median(common_centers_rc, common_centers_base)
    
    print(f"尺度估计结果:")
    print(f"  稳健尺度因子: {scale_median:.6f}")
    print(f"  使用的相机对数: {stats_median['n_pairs_used']}")
    print(f"  过滤后的相机对数: {stats_median['n_pairs_after_filter']}")
    print(f"  异常值比例: {stats_median['outlier_ratio']:.2%}")
    print(f"  比值统计: 均值={stats_median['ratios_mean']:.6f}, 标准差={stats_median['ratios_std']:.6f}")
    print(f"  比值范围: [{stats_median['ratios_min']:.6f}, {stats_median['ratios_max']:.6f}]")
    
    print(f"\n=== 方法2: Procrustes分析 ===")
    scale_procrustes, stats_procrustes = estimate_scale_procrustes(common_centers_rc, common_centers_base)
    
    print(f"Procrustes尺度估计:")
    print(f"  尺度因子: {scale_procrustes:.6f}")
    print(f"  平均残差: {stats_procrustes['mean_residual']:.6f}")
    print(f"  最大残差: {stats_procrustes['max_residual']:.6f}")
    print(f"  RMS残差: {stats_procrustes['rms_residual']:.6f}")
    
    print(f"\n=== 方法对比 ===")
    print(f"中位数法尺度: {scale_median:.6f}")
    print(f"Procrustes尺度: {scale_procrustes:.6f}")
    print(f"相对差异: {abs(scale_median - scale_procrustes) / scale_median * 100:.2f}%")
    
    # 选择最终尺度（推荐使用中位数法，更稳健）
    final_scale = scale_median
    print(f"\n推荐使用稳健中位数法的结果: {final_scale:.6f}")
    
    # 可视化距离比值分布
    # 重新计算所有比值用于可视化
    all_pairs = list(itertools.combinations(range(len(common_centers_rc)), 2))
    all_ratios = []
    for i, j in all_pairs:
        dist_rc = np.linalg.norm(common_centers_rc[i] - common_centers_rc[j])
        dist_base = np.linalg.norm(common_centers_base[i] - common_centers_base[j])
        if dist_rc > 1e-6 and dist_base > 1e-6:
            all_ratios.append(dist_base / dist_rc)
    
    visualize_distance_ratios(
        np.array(all_ratios), 
        final_scale,
        "距离比值分布", 
        os.path.join(os.path.dirname(output_dir), "scale_estimation_analysis.png")
    )
    
    # 应用尺度变换
    print(f"\n应用尺度变换到RC模型...")
    rc_model_dir = os.path.dirname(rc_images_path)
    apply_scale_to_colmap_model(rc_model_dir, output_dir, final_scale)
    
    # 转换为二进制格式
    print("转换为二进制格式...")
    os.system(f'colmap model_converter --input_path "{output_dir}" --output_path "{output_dir}_bin" --output_type BIN')
    
    print(f"\n=== 完成! ===")
    print(f"缩放后的模型保存在: {output_dir}")
    print(f"二进制格式: {output_dir}_bin")
    print(f"最终尺度因子: {final_scale:.6f}")
    
    # 验证结果
    print(f"\n验证: RC模型的坐标将乘以 {final_scale:.6f} 来匹配BASE模型的尺度")

if __name__ == "__main__":
    main()