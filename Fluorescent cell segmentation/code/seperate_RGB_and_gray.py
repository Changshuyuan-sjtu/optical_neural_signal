import os
import shutil
from PIL import Image

# ================== 用户配置区 ==================
source_folder = r"data_20250817/jiekang/24"  # 修改为你的图像所在文件夹路径
target_folder = r"data_20250817_only_RGB/jiekang/24"  # 修改为你想保存彩色图像的新文件夹路径
# =============================================

# 支持的图像格式
image_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp')

# 创建目标文件夹
os.makedirs(target_folder, exist_ok=True)

# 遍历源文件夹
for filename in os.listdir(source_folder):
    file_path = os.path.join(source_folder, filename)

    # 检查是否是文件且是图像格式
    if os.path.isfile(file_path) and filename.lower().endswith(image_extensions):
        # 如果文件名不以 ORG 结尾，则认为是彩色图像
        if not filename.upper().endswith('_ORG.TIF') and \
                not filename.upper().endswith('_ORG.TIFF') and \
                not filename.upper().endswith('_ORG.PNG') and \
                not filename.upper().endswith('_ORG.JPG') and \
                not filename.upper().endswith('_ORG.JPEG') and \
                not filename.upper().endswith('_ORG.BMP'):
            # 复制文件到目标文件夹
            shutil.copy2(file_path, target_folder)
            print(f"已复制: {filename}")

print("✅ 彩色图像分离完成！")