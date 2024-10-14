
import numpy as np
from PIL import Image, ImageDraw
class Polygon:
    def __init__(self, x0, x1, y0, y1):
        self.xLeft = x0
        self.xRight = x1
        self.yDown = y0
        self.yUp = y1

    def __str__(self):
        return f"Polygon(xLeft={self.xLeft}, xRight={self.xRight}, yDown={self.yDown}, yUp={self.yUp})"

    def __repr__(self):
        return f"Polygon({self.xLeft}, {self.xRight}, {self.yDown}, {self.yUp})"

    def get_area(self):
        return (self.xRight - self.xLeft) * (self.yUp - self.yDown)

    def get_perimeter(self):
        return 2 * ((self.xRight - self.xLeft) + (self.yUp - self.yDown))

    def get_centroid(self):
        cx = (self.xLeft + self.xRight) / 2
        cy = (self.yDown + self.yUp) / 2
        return (cx, cy)
    def to_contour(self):
        return np.array([[self.xLeft, self.yDown], [self.xRight, self.yDown],
                         [self.xRight, self.yUp], [self.xLeft, self.yUp]], dtype=np.int32)


class Line:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

class Marker:
    def __init__(self, w=0, h=0, X=0, Y=0, I=0):
        self.width = w
        self.height = h
        self.centerX = X
        self.centerY = Y
        self.ID = I
        self.childPolygon = []
        self.exMarker = []
        self.w_bias = 0
        self.h_bias = 0
    def insertPolygon(self, x0, x1, y0, y1):
        if x0 <= (self.centerX - (self.width / 2)):
            x0 = 0
        else:
            x0 -= (self.centerX - (self.width / 2))
        if y0 <= (self.centerY - (self.height / 2)):
            y0 = 0
        else:
            y0 -= (self.centerY - (self.height / 2))
        if x1 >= (self.centerX + (self.width / 2)):
            x1 = self.width
        else:
            x1 -= (self.centerX - (self.width / 2))
        if y1 >= (self.centerY + (self.height / 2)):
            y1 = self.height
        else:
            y1 -= (self.centerY - (self.height / 2))
        flag = 0
        for ind, p in enumerate(self.childPolygon):
            if x0 ==  p.xLeft and x1 == p.xRight:
                if y0 == p.yUp:
                    self.childPolygon[ind].yUp = y1
                    flag = 1
                    break
                if y1 == p.yDown:
                    self.childPolygon[ind].yDown = y0
                    flag = 1
                    break
            if y0 ==  p.yDown and y1 == p.yUp:
                if x0 == p.xRight:
                    self.childPolygon[ind].xRight = x1
                    flag = 1
                    break
                if x1 == p.xLeft:
                    self.childPolygon[ind].xLeft = x0
                    flag = 1
                    break

        if flag == 0:
            self.childPolygon.append(Polygon(x0, x1, y0, y1))

    def insert_exPolygon(self, x0, x1, y0, y1):
        x0 -= (self.centerX - (self.width / 2))
        y0 -= (self.centerY - (self.height / 2))
        x1 -= (self.centerX - (self.width / 2))
        y1 -= (self.centerY - (self.height / 2))
        flag = 0
        for ind, p in enumerate(self.exMarker):
            if x0 ==  p.xLeft and x1 == p.xRight:
                if y0 == p.yUp:
                    self.exMarker[ind].yUp = y1
                    flag = 1
                    break
                if y1 == p.yDown:
                    self.exMarker[ind].yDown = y0
                    flag = 1
                    break
            if y0 ==  p.yDown and y1 == p.yUp:
                if x0 == p.xRight:
                    self.exMarker[ind].xRight = x1
                    flag = 1
                    break
                if x1 == p.xLeft:
                    self.exMarker[ind].xLeft = x0
                    flag = 1
                    break
        if flag == 0:
            self.exMarker.append(Polygon(x0, x1, y0, y1))

    def insertrePolygon(self, x0, x1, y0, y1):
        self.childPolygon.append(Polygon(x0, x1, y0, y1))

    def countPolygon(self):
        return len(self.childPolygon)

    def returnPolygon(self, i):
        return self.childPolygon[i]

    def count_nonedgePolygon(self, edge = 27):

        return len([i for i in self.childPolygon if self.width - i.xLeft > edge and i.xRight > edge and self.height - i.yDown > edge and i.yUp > edge and i.get_area() > 0.05*self.areaPolygon()])

    def areaPolygon(self):
        result = 0
        for i in range(self.countPolygon()):
            polygon = self.childPolygon[i]
            result += (polygon.xRight - polygon.xLeft) * (polygon.yUp - polygon.yDown)
        return result
  
    def to_grayscale_image(self, img_size=None, image_size=(200, 200)):
        # 创建一个白色背景的图像
        img = Image.new("L", image_size, "black")
        draw = ImageDraw.Draw(img)

        for polygon in self.childPolygon:
            # 获取多边形的坐标
            coords = [(pt[0], pt[1]) for pt in polygon.to_contour()]  # 修改此行
            # 绘制多边形
            draw.polygon(coords, outline="white", fill="white")
        
        return img