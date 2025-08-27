import os

# 修改成你的两个目录路径
folder1 = r"D:\Data\L7\Imgs\25"
folder2 = r"E:\03-Polytech\images"

# 获取文件名集合（只保留文件名，不含路径）
files1 = {f for f in os.listdir(folder1) if os.path.isfile(os.path.join(folder1, f))}
files2 = {f for f in os.listdir(folder2) if os.path.isfile(os.path.join(folder2, f))}

# 找差异
only_in_1 = sorted(files1 - files2)
only_in_2 = sorted(files2 - files1)

print(f"文件夹1独有 ({len(only_in_1)}):")
for f in only_in_1:
    print("  ", f)

print(f"\n文件夹2独有 ({len(only_in_2)}):")
for f in only_in_2:
    print("  ", f)
