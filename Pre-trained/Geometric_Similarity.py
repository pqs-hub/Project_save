from New_Data_Structure import *
from itertools import product
import copy


def calculate_intersection_area(polygon1, polygon2):
    x_left = max(polygon1.xLeft, polygon2.xLeft)
    x_right = min(polygon1.xRight, polygon2.xRight)
    y_down = max(polygon1.yDown, polygon2.yDown)
    y_up = min(polygon1.yUp, polygon2.yUp)
    if x_left < x_right and y_down < y_up:
        return (x_right - x_left) * (y_up - y_down)
    return 0

def ACC(A, B,marker_width, a):
    Intersection = 0
    for i in range(A.countPolygon()):
        for j in range(B.countPolygon()):
            poly_A = A.returnPolygon(i)
            poly_B = B.returnPolygon(j)
            Intersection += calculate_intersection_area(poly_A, poly_B)
    
    XOR = abs(float(A.areaPolygon()) + float(B.areaPolygon()) - 2 * Intersection)
    # print(XOR /(marker_width*marker_width))
    if (XOR /(marker_width*marker_width)) <= 1-a:
        return True
    else:
        return False

def ACC_value(A, B,marker_width):
    Intersection = 0
    for i in range(A.countPolygon()):
        for j in range(B.countPolygon()):
            poly_A = A.returnPolygon(i)
            poly_B = B.returnPolygon(j)
            Intersection += calculate_intersection_area(poly_A, poly_B)  
    XOR = abs(float(A.areaPolygon()) + float(B.areaPolygon()) - 2 * Intersection)
    return XOR /(marker_width*marker_width)


def inverse_transform(marker, transformation_log):
    new_marker = Marker(marker.width, marker.height, marker.centerX, marker.centerY, marker.ID)
    
    for poly in marker.childPolygon:
        x0, x1, y0, y1 = poly.xLeft, poly.xRight, poly.yDown, poly.yUp
            
        # Apply inverse flips
        if transformation_log[1]:
            y0, y1 = marker.height - y1, marker.height - y0
        if transformation_log[0]:
            x0, x1 = marker.width - x1, marker.width - x0
            
        new_marker.insertrePolygon(x0, x1, y0, y1)
        
    return new_marker

def transform(marker, transformation_log):
    new_marker = Marker(marker.width, marker.height, marker.centerX, marker.centerY, marker.ID)
    
    for poly in marker.childPolygon:
        x0, x1, y0, y1 = poly.xLeft, poly.xRight, poly.yDown, poly.yUp
            
        # Apply flips
        
        if transformation_log[0]:
            x0, x1 = marker.width - x1, marker.width - x0
        if transformation_log[1]:
            y0, y1 = marker.height - y1, marker.height - y0

        new_marker.insertrePolygon(x0, x1, y0, y1)
        
    return new_marker

def generate_transformations(marker):
    transformations = []
    transformations_log = []
    for flip_x, flip_y in product([False, True], repeat=2):
        new_marker = Marker(marker.width, marker.height, marker.centerX, marker.centerY, marker.ID)
        for poly in marker.childPolygon:
            x0, x1, y0, y1 = poly.xLeft, poly.xRight, poly.yDown, poly.yUp
            
            if flip_x:
                x0, x1 = marker.width - x1, marker.width - x0
            if flip_y:
                y0, y1 = marker.height - y1, marker.height - y0
            
            new_marker.insertrePolygon(x0, x1, y0, y1)
        transformations_log.append([flip_x, flip_y])
    # Add the original marker to the transformations list
        transformations.append(new_marker)
    return transformations, transformations_log

def ACC_RepUpdate(current_clip, markers, marker_width, a):
    new_markers = []
    anomaly = []
    bias = np.zeros((len(markers), current_clip.countPolygon(),  4))
    for idx, marker in enumerate(markers):

        assert current_clip.countPolygon() == marker.countPolygon(), ("The number of polygons in current_clip and marker should be the same.")       

        current_clip_all, transformations_log = generate_transformations(current_clip)
        np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
        bias_buff = np.zeros((current_clip.countPolygon(),  4))  
        flage = 0
        for ind, j in enumerate(current_clip_all):  
            if ACC(j, marker, marker_width, a ):
                marker_invers = inverse_transform(marker, transformations_log[ind])
                for i in range(current_clip.countPolygon()):
                    x0_bias = -current_clip.returnPolygon(i).xLeft + marker_invers.returnPolygon(i).xLeft
                    x1_bias = -current_clip.returnPolygon(i).xRight + marker_invers.returnPolygon(i).xRight
                    y0_bias = -current_clip.returnPolygon(i).yDown + marker_invers.returnPolygon(i).yDown
                    y1_bias = -current_clip.returnPolygon(i).yUp + marker_invers.returnPolygon(i).yUp

                    bias_buff[i] = [x0_bias, x1_bias, y0_bias, y1_bias]             
                bias[idx] = bias_buff

                flage = 1
                new_markers.append(marker_invers)#直接存储已经翻转对齐过的图像
                break
        if flage == 0:
            anomaly.append(marker)

    bias = np.transpose(bias, axes=(1, 2, 0))
    print(bias)
    return bias, new_markers, anomaly

def ACCe_RepUpdate(current_clip, markers, marker_width, a):
    new_markers = []
    anomaly = []
    bias = np.zeros((len(markers), current_clip.countPolygon(),  4))
    for idx, marker in enumerate(markers):

        assert current_clip.countPolygon() == marker.countPolygon(), ("The number of polygons in current_clip and marker should be the same.")       

        current_clip_all, transformations_log = generate_transformations(current_clip)
        np.set_printoptions(suppress=True, formatter={'float_kind':'{:f}'.format})
        bias_buff = np.zeros((current_clip.countPolygon(),  4))  
        flage = 0
        for ind, j in enumerate(current_clip_all):  
            if ACC(j, marker, marker_width, a ):
                weight = []
                weight_poly = []
                marker_invers = inverse_transform(marker, transformations_log[ind])
                for i in range(current_clip.countPolygon()):
                    weight_poly = [marker_invers.returnPolygon(i).xRight - marker_invers.returnPolygon(i).xLeft, marker_invers.returnPolygon(i).yUp - marker_invers.returnPolygon(i).yDown]/np.power(marker_width, 2)
                    x0_bias = -current_clip.returnPolygon(i).xLeft + marker_invers.returnPolygon(i).xLeft
                    x1_bias = -current_clip.returnPolygon(i).xRight + marker_invers.returnPolygon(i).xRight
                    y0_bias = -current_clip.returnPolygon(i).yDown + marker_invers.returnPolygon(i).yDown
                    y1_bias = -current_clip.returnPolygon(i).yUp + marker_invers.returnPolygon(i).yUp
                    weight.append(list(weight_poly))
                    bias_buff[i] = [x0_bias, x1_bias, y0_bias, y1_bias]             
                bias[idx] = bias_buff
                
                flage = 1
                new_markers.append(marker_invers)#直接存储已经翻转对齐过的图像
                break
        if flage == 0:
            anomaly.append(marker)

    bias = np.transpose(bias, axes=(1, 2, 0))
    # print(bias)
    return bias, new_markers, anomaly, weight


def ECC(markerA, markerB, constraint):
    linesH = []
    linesV = []
    linesB = []
    polygons = []

    # 插入markerA的多边形边
    for i in range(markerA.countPolygon()):
        current_polygon = markerA.returnPolygon(i)
        linesH.append(Line(current_polygon.xLeft, current_polygon.yDown, current_polygon.xRight, current_polygon.yDown))
        linesH.append(Line(current_polygon.xLeft, current_polygon.yUp, current_polygon.xRight, current_polygon.yUp))
        linesV.append(Line(current_polygon.xLeft, current_polygon.yDown, current_polygon.xLeft, current_polygon.yUp))
        linesV.append(Line(current_polygon.xRight, current_polygon.yDown, current_polygon.xRight, current_polygon.yUp))

    # 插入markerB的多边形边
    for i in range(markerB.countPolygon()):
        current_polygon = markerB.returnPolygon(i)
        linesB.append(Line(current_polygon.xLeft, current_polygon.yDown, current_polygon.xRight, current_polygon.yDown))
        linesB.append(Line(current_polygon.xLeft, current_polygon.yUp, current_polygon.xRight, current_polygon.yUp))
        linesB.append(Line(current_polygon.xLeft, current_polygon.yDown, current_polygon.xLeft, current_polygon.yUp))
        linesB.append(Line(current_polygon.xRight, current_polygon.yDown, current_polygon.xRight, current_polygon.yUp))

    # 生成新的多边形
    for line in linesB:
        polygons.append(Polygon(line.x0 - constraint, line.x1 + constraint, line.y0 - constraint, line.y1 + constraint))

    # 计算边缘溢出
    for polygon in polygons:
        new_linesH = []
        for line in linesH:
            if polygon.yDown <= line.y0 <= polygon.yUp:
                if polygon.xLeft < line.x0 and polygon.xRight > line.x1:
                    continue
                elif polygon.xLeft > line.x0 and polygon.xRight < line.x1:
                    new_linesH.append(Line(line.x0, line.y0, polygon.xLeft, line.y0))
                    new_linesH.append(Line(polygon.xRight, line.y0, line.x1, line.y0))
                elif polygon.xLeft > line.x0 and polygon.xRight > line.x1 and polygon.xLeft < line.x1:
                    new_linesH.append(Line(line.x0, line.y0, polygon.xLeft, line.y0))
                elif polygon.xLeft < line.x0 and polygon.xRight > line.x0 and polygon.xRight < line.x1:
                    new_linesH.append(Line(polygon.xRight, line.y0, line.x1, line.y0))
                else:
                    new_linesH.append(line)
            else:
                new_linesH.append(line)
        linesH = new_linesH

        new_linesV = []
        for line in linesV:
            if polygon.xLeft <= line.x0 <= polygon.xRight:
                if polygon.yDown < line.y0 and polygon.yUp > line.y1:
                    continue
                elif polygon.yDown > line.y0 and polygon.yUp < line.y1:
                    new_linesV.append(Line(line.x0, line.y0, line.x0, polygon.yDown))
                    new_linesV.append(Line(line.x0, polygon.yUp, line.x0, line.y1))
                elif polygon.yDown > line.y0 and polygon.yUp > line.y1 and polygon.yDown < line.y1:
                    new_linesV.append(Line(line.x0, line.y0, line.x0, polygon.yDown))
                elif polygon.yDown < line.y0 and polygon.yUp > line.y0 and polygon.yUp < line.y1:
                    new_linesV.append(Line(line.x0, polygon.yUp, line.x0, line.y1))
                else:
                    new_linesV.append(line)
            else:
                new_linesV.append(line)
        linesV = new_linesV

    linesV = [line for line in linesV if not (line.x0 == line.x1 and line.y0 == line.y1)]
    linesH = [line for line in linesH if not (line.x0 == line.x1 and line.y0 == line.y1)]

    return len(linesH) == 0 and len(linesV) == 0

