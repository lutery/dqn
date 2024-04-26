import cv2
import numpy as np

# 设置图片的高度和宽度
height = 768
width = 512

# 生成随机颜色函数
def random_color():
    return tuple(np.random.randint(0, 256, size=3).tolist())

# 生成随机颜色，每个颜色通道一个随机值
color = random_color()  # 生成随机颜色（B, G, R）

# 创建一张具有随机颜色的图片
image = np.zeros((height, width, 3), dtype=np.uint8)
image[:] = color  # 将随机颜色赋值给整张图片

# 正方体的大小
cube_size = 64

# 正方体在图片中的位置（底部中央）
bottom_center = (width // 2, height // 2)

# 定义正方体的8个顶点
# 正方体正面4个顶点
p1 = (bottom_center[0] - cube_size // 2, bottom_center[1] - cube_size // 2)
p2 = (p1[0] + cube_size, p1[1])
p3 = (p2[0], p2[1] + cube_size)
p4 = (p1[0], p1[1] + cube_size)

# 正方体被面4个顶点（简单3D视角）
offset = cube_size // 4  # 视角偏移
p5 = (p1[0] + offset, p1[1] - offset)
p6 = (p2[0] + offset, p2[1] - offset)
p7 = (p3[0] + offset, p3[1] - offset)
p8 = (p4[0] + offset, p4[1] - offset)

cube_color = random_color()
# 绘制正方体的面和边，使用随机颜色填充面，边框为黑色
# cv2.fillPoly(image, [np.array([p1, p2, p6, p5])], cube_color)  # 侧面
cv2.fillPoly(image, [np.array([p2, p3, p7, p6])], cube_color)  # 侧面
cv2.fillPoly(image, [np.array([p1, p2, p6, p5])], cube_color)  # 侧面
# cv2.fillPoly(image, [np.array([p4, p1, p5, p8])], cube_color)  # 侧面
# cv2.fillPoly(image, [np.array([p5, p6, p7, p8])], cube_color)  # 顶面
cv2.fillPoly(image, [np.array([p1, p2, p3, p4])], cube_color)  # 底面

# 用黑色绘制正方体的所有边
cv2.polylines(image, [np.array([p1, p2, p3, p4, p1])], True, (0, 0, 0), 1)  # 底边框
cv2.polylines(image, [np.array([p2, p6, p5, p1])], True, (0, 0, 0), 1)  # 顶边框
# cv2.polylines(image, [np.array([p3, p7, p6])], True, (0, 0, 0), 1)  # 顶边框
# cv2.line(image, p1, p5, (0, 0, 0), 1)  # 侧边
cv2.line(image, p3, p7, (0, 0, 0), 1)  # 侧边
cv2.line(image, p7, p6, (0, 0, 0), 1)  # 侧边
# cv2.line(image, p4, p8, (0, 0, 0), 1)  # 侧边

# 保存图片到文件
cv2.imwrite('random_color_image_with_cube.png', image)

# 可选：显示图片
cv2.imshow('Random Color Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
