import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 设置图像文件夹路径
image_folder = 'data_20250817/jidong/Image 6_label'  # 修改为您图像文件夹的路径
output_folder = 'data_20250817/jidong/cell_crop'  # 修改为输出文件夹的路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 设置红色区域的阈值（假设红色为主要标注颜色）
lower_red = np.array([150, 0, 0])  # 红色的最低阈值
upper_red = np.array([255, 100, 100])  # 红色的最高阈值

# 保存已检测的矩形框坐标
rectangles = []

# 处理第一帧图像，检测并保存矩形框
first_image_path = os.path.join(image_folder, 'Image 6_t000.jpg')  # 修改为实际文件名
first_image = Image.open(first_image_path)
first_image_np = np.array(first_image)

# 创建掩膜来获取红色区域
mask = cv2.inRange(first_image_np, lower_red, upper_red)
mask = cv2.dilate(mask, None, iterations=2)  # 扩张掩膜以去除噪声

# 查找掩膜中的轮廓
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 提取矩形框坐标并保存
for contour in contours:
    if cv2.contourArea(contour) > 100:  # 过滤掉小的轮廓（噪声）
        # 获取最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # 使用 np.int32 替代 np.int0

        # 获取矩形的中心
        center = (int(rect[0][0]), int(rect[0][1]))

        # 计算正方形的边长，基于椭圆的最大宽度或高度
        width, height = rect[1]
        square_size = int(max(width, height) * 1.1)  # 比矩形稍大的正方形

        # 计算正方形的左上角和右下角坐标
        top_left = (center[0] - square_size // 2, center[1] - square_size // 2)
        bottom_right = (center[0] + square_size // 2, center[1] + square_size // 2)

        # 将矩形框坐标添加到列表
        rectangles.append((top_left, bottom_right))

# 对每个图像帧进行裁剪
for filename in os.listdir(image_folder):
    # 检查文件是否是彩色图像，排除ORG后缀的原图
    if filename.endswith('.jpg') and 'ORG' not in filename:
        # 构建图像路径
        image_path = os.path.join(image_folder, filename)

        # 加载彩色图像
        image = Image.open(image_path)
        image_np = np.array(image)

        # 对每个矩形框进行裁剪
        for i, (top_left, bottom_right) in enumerate(rectangles):
            # 裁剪细胞区域
            cropped_cell = image_np[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            # 确定输出文件夹，按位置（即矩形框编号）创建子文件夹
            position_folder = os.path.join(output_folder, f'cell_position_{i + 1}')
            if not os.path.exists(position_folder):
                os.makedirs(position_folder)

            # 确定输出路径，按帧编号和位置保存
            output_filename = f'{os.path.splitext(filename)[0]}.jpg'
            output_path = os.path.join(position_folder, output_filename)

            # 保存裁剪后的细胞图像
            cv2.imwrite(output_path, cropped_cell)

        print(f'Processed: {filename}')

print("Batch processing complete!")
