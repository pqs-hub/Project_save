import pya
import csv
from rtree import index
import numpy as np
from New_Data_Structure import *
from PIL import Image, ImageDraw
import os
import shutil
import matplotlib.pyplot as plt
def get_square_corners(center, width, height):
    half_width = width / 2
    half_height = height / 2
    return [(center[0] - half_width, center[1] - half_height),
            (center[0] + half_width, center[1] - half_height),
            (center[0] + half_width, center[1] + half_height),
            (center[0] - half_width, center[1] + half_height)]

def get_min_max(rect):
    min_x, min_y = np.min(rect, axis=0)
    max_x, max_y = np.max(rect, axis=0)
    return min_x, min_y, max_x, max_y

def get_rectangles_in_bounds(marker_bounds, rect_index, rectangles):
    min_x, min_y, max_x, max_y = marker_bounds
    return [rectangles[i] for i in rect_index.intersection((min_x, min_y, max_x, max_y))]



from PIL import Image, ImageDraw

def save_marker_image(marker, image_path, image_size=(200, 200)):
    # 创建一个黑色背景的图像
    img = Image.new("RGB", image_size, "black")
    draw = ImageDraw.Draw(img)

    # 获取Marker的宽度和高度
    marker_width = marker.width
    marker_height = marker.height

    # 计算缩放比例，保持宽高比
    scale_x = image_size[0] / marker_width
    scale_y = image_size[1] / marker_height
    scale = min(scale_x, scale_y)  # 保持比例一致

    # 对所有多边形进行坐标缩放和平移，使其适应图像
    for polygon in marker.childPolygon:
        # 获取多边形的坐标并进行缩放
        coords = [(int((pt[0] - marker.centerX + marker_width / 2) * scale),
                   int((pt[1] - marker.centerY + marker_height / 2) * scale)) 
                  for pt in polygon.to_contour()]

        # 绘制多边形
        draw.polygon(coords, outline="white", fill="white")

    # 保存图像
    img.save(image_path)


def save_marker_image1(marker, image_path):
    # 创建一个黑色背景的图像
    fig_size = (6, 6)
    dpi = 100
    fig, ax = plt.subplots(figsize=fig_size)
    ax.clear()
    ax.set_facecolor('black')
    for poly in marker.childPolygon:
        for polygon in poly.polygons:
            ax.fill(*zip(*polygon), color=(1, 1, 1))
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(marker.centerX - marker.width / 2, marker.centerX + marker.width / 2)
    ax.set_ylim(marker.centerY - marker.height / 2,marker.centerY + marker.height / 2)
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0, facecolor='black', dpi=dpi)
    plt.close()

def clip_generate(oas_path, marker_width, marker_height, layer_indices, x_offset=400, y_offset=400, marker_extend=False):
    
    save_path = os.path.splitext(oas_path)[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    layout = pya.Layout()
    layout.read(oas_path)
    cell_index = layout.cell(0)
    
    clip_area = marker_width * marker_height
    # 存储output坐标
    for layer in layer_indices:
        print(f"Processing layer {layer}")
        # 创建层对应的文件夹
        # 初始化 R-tree 索引
        rect_index = index.Index()
        
        rectangles = []
        clip_cells = []
        layer_folder = os.path.join(save_path, f"layer_{layer}")
        
        if os.path.exists(layer_folder):
            shutil.rmtree(layer_folder)
        os.makedirs(layer_folder, exist_ok=True)

        output_layer_index = layout.layer(layer[0], layer[1])  # 获取预定义层的索引

        layer_max_x, layer_max_y = 0, 0
        layer_min_x, layer_min_y = np.inf, np.inf

        rectangles = []  # 用于存储矩形
        rect_index = index.Index()  # 用于快速查询
        get_poly = 0
        # 遍历所有单元格
        for cell in layout.each_cell():
            print(f"Cell name: {cell.name}, Cell index: {cell.cell_index()}")

            # 创建展平后的cell view
            flat_shapes = cell.flatten(output_layer_index)
            output_shapes = cell.shapes(output_layer_index)
            if output_shapes.is_empty():
                print(f"No shapes in this cell for layer {layer[0]}, datatype {layer[1]}")
            else:
                print(f"Shapes found in this cell for layer {layer[0]}, datatype {layer[1]}")

                # 遍历并存储所有形状
                for idx, shape in enumerate(output_shapes.each(), start=1):
                    if shape.is_polygon() or shape.is_box():
                        get_poly = 1
                        for rec in shape.polygon.decompose_convex():  # 对复杂形状分解成简单多边形
                            ll = rec.bbox().p1
                            ru = rec.bbox().p2
                            min_x, min_y = ll.x, ll.y
                            max_x, max_y = ru.x, ru.y
                            rect_index.insert(len(rectangles), (min_x, min_y, max_x, max_y))
                            rectangles.append(((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)))

                            # 更新全局的最大最小值
                            if max_x > layer_max_x: layer_max_x = max_x
                            if max_y > layer_max_y: layer_max_y = max_y
                            if min_x < layer_min_x: layer_min_x = min_x
                            if min_y < layer_min_y: layer_min_y = min_y

        # 循环生成clip并处理图像
        clip_cells = []  # 用于存储生成的Marker
        id = 0

        # 遍历生成clip的网格
        print("Generating clips...")
        if get_poly == 1:
            for x in range(0, int(layer_max_x - layer_min_x), x_offset):
                for y in range(0, int(layer_max_y - layer_min_y), y_offset):
                    snot = 0
                    id += 1
                    marker_width_tmp = marker_width
                    marker_height_tmp = marker_height
                    center = (x, y)
                    center_x, center_y = center

                    if marker_extend:
                        # 扩展Marker的边界
                        marker_width_tmp += ru.x - ll.x
                        marker_height_tmp += ru.y - ll.y
                        marker_corners = get_square_corners(center, marker_width_tmp, marker_height_tmp)
                        marker_bounds = get_min_max(marker_corners)

                        # 获取扩展后的矩形边界
                        marker_corners_in = get_square_corners(center, marker_width, marker_height)
                        marker_bounds_in = get_min_max(marker_corners_in)
                        candidate_rectangles = get_rectangles_in_bounds(marker_bounds, rect_index, rectangles)
                    else:
                        # 仅获取Marker内的矩形
                        marker_corners_in = get_square_corners(center, marker_width, marker_height)
                        marker_bounds_in = get_min_max(marker_corners_in)
                        candidate_rectangles = get_rectangles_in_bounds(marker_bounds_in, rect_index, rectangles)
                    
                    # 初始化Marker
                    marker_tmp = Marker(marker_width, marker_height, center_x, center_y, id)
                                            
                    # 遍历候选的矩形，插入到Marker中
                    for rect in candidate_rectangles:
                        rect_min_x, rect_min_y, rect_max_x, rect_max_y = get_min_max(rect)

                        # 计算矩形和Marker的交集
                        intersect_xmin = max(marker_bounds_in[0], rect_min_x)
                        intersect_ymin = max(marker_bounds_in[1], rect_min_y)
                        intersect_xmax = min(marker_bounds_in[2], rect_max_x)
                        intersect_ymax = min(marker_bounds_in[3], rect_max_y)
                        
                        # 如果有交集，插入Polygon
                        if intersect_xmax > intersect_xmin and intersect_ymax > intersect_ymin:
                            marker_tmp.insertPolygon(intersect_xmin, intersect_xmax, intersect_ymin, intersect_ymax)
                        
                        # 如果允许扩展，处理子区域
                        if marker_extend:
                            sub_bounds = {
                                'left': (marker_bounds[0], marker_bounds_in[1], marker_bounds_in[0], marker_bounds_in[3]),
                                'right': (marker_bounds_in[2], marker_bounds_in[1], marker_bounds[2], marker_bounds_in[3]),
                                'top': (marker_bounds_in[0], marker_bounds_in[3], marker_bounds_in[2], marker_bounds[3]),
                                'bottom': (marker_bounds_in[0], marker_bounds[1], marker_bounds_in[2], marker_bounds_in[1]),
                                'top_left': (marker_bounds[0], marker_bounds_in[3], marker_bounds_in[0], marker_bounds[3]),
                                'top_right': (marker_bounds_in[2], marker_bounds_in[3], marker_bounds[2], marker_bounds[3]),
                                'bottom_left': (marker_bounds[0], marker_bounds[1], marker_bounds_in[0], marker_bounds_in[1]),
                                'bottom_right': (marker_bounds_in[2], marker_bounds[1], marker_bounds[2], marker_bounds_in[1])
                            }

                            # 处理每个子区域
                            for key, bounds in sub_bounds.items():
                                intersect_xmin = max(bounds[0], rect_min_x)
                                intersect_ymin = max(bounds[1], rect_min_y)
                                intersect_xmax = min(bounds[2], rect_max_x)
                                intersect_ymax = min(bounds[3], rect_max_y)
                                if intersect_xmax > intersect_xmin and intersect_ymax > intersect_ymin:
                                    marker_tmp.insert_subPolygon(intersect_xmin, intersect_xmax, intersect_ymin, intersect_ymax, key)

                    # 保存图像
                    if marker_tmp.areaPolygon()/clip_area > 0.05: 
                        clip_cells.append(marker_tmp)

        # 返回生成的Marker列表
        return clip_cells

   
if __name__ == "__main__":
    oas_path = r"E:\博士生涯\oas_read\ICCAD16-N7M2EUV\mkLanaiCPU.gds"
    layer_indices = [(66, 20), (67, 20), (68, 20)]  # 预定义的层集合
    clip_cells = clip_generate(oas_path, marker_width = 3000, marker_height = 3000, layer_indices=layer_indices, x_offset=2000, y_offset=2000, marker_extend=False)