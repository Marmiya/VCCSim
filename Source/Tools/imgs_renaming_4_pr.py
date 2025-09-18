import os

folder_path = r"C:\UEProjects\VCCSimDev\Saved\RuntimeLogs\20250918_002016\BP_PreciseDrone_1C_1D_L_FP_C_2\Normal"

png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

png_files.sort()

for index, old_name in enumerate(png_files):
    new_name = f"image_{index + 1:06d}.png" 
    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)
    
    os.rename(old_path, new_path)
    print(f"Renamed: {old_name} -> {new_name}")

print("Renaming complete!")
