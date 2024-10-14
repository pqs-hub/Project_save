import pya
import csv
from rtree import index
import numpy as np
from New_Data_Structure import *
from PIL import Image, ImageDraw
import os
import shutil

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



def save_marker_image(marker, image_path, image_size=(200, 200)):
    # 创建一个白色背景的图像
    img = Image.new("RGB", image_size, "black")
    draw = ImageDraw.Draw(img)

    for polygon in marker.childPolygon:
        # 获取多边形的坐标
        coords = [(pt[0], pt[1]) for pt in polygon.to_contour()]  # 修改此行
        # 绘制多边形
        draw.polygon(coords, outline="white", fill="white")

    # 保存图像
    img.save(image_path)



def clip_generate(oas_path, marker_width, marker_height, marker_extend = False):
    
    save_path = os.path.splitext(oas_path)[0]
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    print(save_path)
    layout = pya.Layout()
    layout.read(oas_path)
    cell_index = layout.cell(0)
    marker_layer_index = layout.layer(10000, 0)  # 获取第 10000 层的层索引
    output_layer_index = layout.layer(1000, 0)    # 获取第 1000 层的层索引
    marker_shapes = cell_index.shapes(marker_layer_index)
    output_shapes = cell_index.shapes(output_layer_index)
    
    # 初始化 R-tree 索引
    rect_index = index.Index()
    
    rectangles = []
    clip_cells = []
    # 存储output坐标
    for idx, shape in enumerate(output_shapes.each(), start=1):
        # 打印 shape 对象的所有成员，并区分属性和方法
        # 遍历 shape 对象的所有属性和方法，并显示其具体值

        
        for rec in shape.polygon.decompose_convex():
            ll = rec.bbox().p1
            ru = rec.bbox().p2
            min_x, min_y = ll.x, ll.y
            max_x, max_y = ru.x, ru.y
            rect_index.insert(len(rectangles), (min_x, min_y, max_x, max_y))
            rectangles.append(((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)))
        
    for idx, shape in enumerate(marker_shapes.each(), start=1):
        marker_width_tmp = marker_width
        marker_height_tmp = marker_height
        ll = shape.box_p1
        ru = shape.box_p2
        center_x, center_y = (ru.x + ll.x)/2, (ru.y + ll.y)/2
        center = (center_x, center_y)
        if marker_extend:
            x0, y0, x1, y1 = shape
            marker_width_tmp += x1 - x0 
            marker_height_tmp += y1 - y0
            marker_corners = get_square_corners(center, marker_width_tmp, marker_height_tmp)
            marker_bounds = get_min_max(marker_corners)

            marker_corners_in = get_square_corners(center, marker_width, marker_height)
            marker_bounds_in = get_min_max(marker_corners_in)
            candidate_rectangles = get_rectangles_in_bounds(marker_bounds, rect_index, rectangles)

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
        else:
            marker_corners_in = get_square_corners(center, marker_width, marker_height)
            marker_bounds_in = get_min_max(marker_corners_in)
            candidate_rectangles = get_rectangles_in_bounds(marker_bounds_in, rect_index, rectangles)
        marker_tmp = Marker(marker_width, marker_height, center_x, center_y, idx)
                                
        for rect in candidate_rectangles:
            # 计算矩形的边界
            # print(rect)
            rect_min_x, rect_min_y, rect_max_x, rect_max_y = get_min_max(rect)
            # print(rect_min_x, rect_min_y, rect_max_x, rect_max_y)
            # print()
            # 计算交集
            intersect_xmin = max(marker_bounds_in[0], rect_min_x)
            intersect_ymin = max(marker_bounds_in[1], rect_min_y)
            intersect_xmax = min(marker_bounds_in[2], rect_max_x)
            intersect_ymax = min(marker_bounds_in[3], rect_max_y)
            
            # 如果有交集
            if intersect_xmax > intersect_xmin and intersect_ymax > intersect_ymin:
                # 添加交集到 marker
                marker_tmp.insertPolygon(intersect_xmin, intersect_xmax, intersect_ymin, intersect_ymax)
                
                # print("存在交集") 
            # 分别计算各个部分的相交矩形
            if marker_extend:
                for key, bounds in sub_bounds.items():
                    intersect_xmin = max(bounds[0], rect_min_x)
                    intersect_ymin = max(bounds[1], rect_min_y)
                    intersect_xmax = min(bounds[2], rect_max_x)
                    intersect_ymax = min(bounds[3], rect_max_y)
                    if intersect_xmax > intersect_xmin and intersect_ymax > intersect_ymin:
                    # 添加交集到 marker
                        # print("存在交集")
                        marker_tmp.insert_subPolygon(intersect_xmin, intersect_xmax, intersect_ymin, intersect_ymax, key) 
            
        clip_cells.append(marker_tmp)

    return clip_cells
if __name__ == "__main__":
    oas_path = r"E:\data\ICCAD2016C\Extend_case3\Extend_case3.gds"
    clip_cells = clip_generate(oas_path, marker_width=200, marker_height=200, marker_extend=False)
