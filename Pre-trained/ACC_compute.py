from oas2class_nomarkers import clip_generate, get_rectangles_in_bounds, get_square_corners, get_min_max
import numpy as np
from Geometric_Similarity import generate_transformations, transform, ACC_value
from New_Data_Structure import Marker
def get_min_max(rect):
    min_x, min_y = np.min(rect, axis=0)
    max_x, max_y = np.max(rect, axis=0)
    return min_x, min_y, max_x, max_y
def calculate_shifted_clip(shifted_clip, rects, bound):
    for rect in rects:

        rect_min_x, rect_min_y, rect_max_x, rect_max_y = get_min_max(rect)

        intersect_xmin = max(bound[0], rect_min_x)
        intersect_ymin = max(bound[1], rect_min_y)
        intersect_xmax = min(bound[2], rect_max_x)
        intersect_ymax = min(bound[3], rect_max_y)

        if intersect_xmax > intersect_xmin and intersect_ymax > intersect_ymin:

            shifted_clip.insertPolygon(intersect_xmin, intersect_xmax, intersect_ymin, intersect_ymax)
    return shifted_clip

def shift_check(clipA, clipB, marker_width, marker_height, rectangles, rel_index):
    if abs(clipA.areaPolygon() - clipB.areaPolygon()) > 0.25*marker_width*marker_height:
        return 0
    elif abs(clipA.count_nonedgePolygon() - clipB.count_nonedgePolygon()) > 1:
        return 1 
    # if abs(clipA.area_center() - clipB.area_center()) > 10:
    #     return 1
    best_ACC_v = 1
    best_directions = []
    best_transformation_log = []
    bias = []
    current_clip_all, transformations_log = generate_transformations(clipA)

    for j,t in zip(current_clip_all, transformations_log):

        direction = [clipB.polygon_centroid_shift()[0] - j.polygon_centroid_shift()[0], clipB.polygon_centroid_shift()[1] - j.polygon_centroid_shift()[1]]

        best_directions.append(direction)
        bias.append(abs(direction[0])+abs(direction[1]))
        best_transformation_log.append(t)

    sorted_indices = sorted(range(len(bias)), key=lambda k: bias[k])
    best_transformation_log = [best_transformation_log[i] for i in sorted_indices]
    best_directions = [best_directions[i] for i in sorted_indices]

    for log, best_direction in zip(best_transformation_log, best_directions):
        # print(best_direction, log)

        # if abs(best_direction[0]) > 3*j.w_bias and abs(best_direction[1]) > 3*j.h_bias:
        #     return 0
        if best_direction[0] > 0:
            best_direction[0] = min(best_direction[0], clipA.w_bias + clipB.w_bias)
        else:
            best_direction[0] = max(best_direction[0], -clipA.w_bias - clipB.w_bias)

        if best_direction[1] > 0:
            best_direction[1] = min(best_direction[1], clipA.h_bias + clipB.h_bias)
        else:
            best_direction[1] = max(best_direction[1], -clipA.h_bias - clipB.h_bias)
        if log[0]:
            best_direction[0] = -best_direction[0]
        if log[1]:
            best_direction[1] = -best_direction[1]

        best_direction[1] = -best_direction[1]
        best_direction[0] = -best_direction[0]
        
        A_shift =  clipA.shift_list([max(best_direction[0] - clipB.w_bias, -clipA.w_bias), min(best_direction[0] + clipB.w_bias, clipA.w_bias)], [max(best_direction[1] - clipB.h_bias, -clipA.h_bias), min(best_direction[1] + clipB.h_bias, clipA.h_bias)])
        B_shift = clipA.shift_list([max(best_direction[0] - clipA.w_bias, -clipB.w_bias), min(best_direction[0] + clipA.w_bias, clipB.w_bias)], [max(best_direction[1] - clipA.h_bias, -clipB.h_bias), min(best_direction[1] + clipA.h_bias, clipB.h_bias)])
        combin_shift = set()
        for A in A_shift:
            combin_shift.add((A,(best_direction[0] - A[0], best_direction[1] - A[1])))
        for B in B_shift:
            combin_shift.add(((best_direction[0] - B[0], best_direction[1] - B[1]), B))
        for i in combin_shift:

            clip_A_centerX = clipA.centerX + i[0][0]
            clip_A_centerY = clipA.centerY + i[0][1]
            clip_shifted_A = Marker(clipA.width, clipA.height, clip_A_centerX, clip_A_centerY, clipA.ID)
            marker_bounds_A = (clipA.centerX-clipA.width/2, clipA.centerY-clipA.height/2, clipA.centerX+clipA.width/2, clipA.centerY+clipA.height/2)
            candidate_rectangles_A = get_rectangles_in_bounds(marker_bounds_A, rel_index, rectangles)
            calculate_shifted_clip(clip_shifted_A, candidate_rectangles_A, marker_bounds_A)

            if log[0] or log[1]:
                clip_shifted_A = transform(clip_shifted_A, log)

            clip_B_centerX = clipB.centerX + i[1][0]
            clip_B_centerY = clipB.centerY + i[1][1]
            clip_shifted_B = Marker(clipB.width, clipB.height, clip_B_centerX, clip_B_centerY, clipB.ID)
            marker_bounds_B = (clipB.centerX-clipB.width/2, clipB.centerY-clipB.height/2, clipB.centerX+clipB.width/2, clipB.centerY+clipB.height/2)
            candidate_rectangles_B = get_rectangles_in_bounds(marker_bounds_B, rel_index, rectangles)
            calculate_shifted_clip(clip_shifted_B, candidate_rectangles_B, marker_bounds_B)
            
            ACC_v = ACC_value(clip_shifted_A, clip_shifted_B, marker_width, marker_height)
            if best_ACC_v > ACC_v:
                best_ACC_v = ACC_v
    return best_ACC_v