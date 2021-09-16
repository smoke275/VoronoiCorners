import csv
import time
import visilibity as vis
import math
import color
import numpy as np
import stddraw
from math import cos, sin
import shapely
from shapely.geometry import LineString
import random

# Used to plot the example
import matplotlib.pylab as p

ITERATIONS = 100
SHOW_VORONOI = True
JUST_POLYGON = False
DELAY = 1000
file_name = 'resources/sites_poly2.csv'


def test():
    x = []
    y = []
    poly = []

    with open(file_name) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            x_val = float(row[0])
            y_val = float(row[1])
            x.append(x_val)
            y.append(y_val)
            poly.append(vis.Point(x_val, y_val))

    print(max(x), min(x), max(y), min(y))

    x_range = max(x) - min(x)
    y_range = max(y) - min(y)
    print(x_range, y_range)

    x = [j - min(x) - x_range / 2 for j in x]

    y = [j + max(y) + y_range / 2 for j in y]

    for i in range(len(x)):
        print(x[i], ',', y[i])


def point_to_xy_vectors(points):
    x = []
    y = []
    for i in points:
        x.append(i[0])
        y.append(i[1])
    return x, y


def run():
    stddraw.setCanvasSize(700, 700)

    stddraw.setXscale(-500.0, 500.0)
    stddraw.setYscale(-500.0, 500.0)

    x = []
    y = []

    alpha = 0.2  # ve/vp

    poly = []

    with open(file_name) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            x_val = float(row[0])
            y_val = float(row[1])
            x.append(x_val)
            y.append(y_val)
            poly.append(vis.Point(x_val, y_val))

    epsilon = 0.0000001

    observer = vis.Point(50, -5)
    evader = vis.Point(-100, -5)

    new_evader = None

    for k in range(ITERATIONS):
        if not JUST_POLYGON:
            walls = vis.Polygon([ele for ele in reversed(poly)])
            env = vis.Environment([walls])

            observer.snap_to_boundary_of(env, epsilon)
            observer.snap_to_vertices_of(env, epsilon)

            vispol = vis.Visibility_Polygon(observer, env, epsilon)
            visibility_polygon = VisibilityPolygon(vispol)
            visibility_polygon.set_original_polygon(poly)
            visibility_polygon.set_observer(observer)
            visibility_polygon.orient()

            corner_set = CornerSet(visibility_polygon, alpha)
            corner_set.process()

            features = []

            for i in corner_set.corners:
                features.append(i)
                if i.segment2 is not None:
                    free = FreeEdges(i.segment2, i)
                    features.append(free)

            x_min = math.ceil(visibility_polygon.object.bbox().x_min)
            x_max = math.floor(visibility_polygon.object.bbox().x_max)
            y_min = math.ceil(visibility_polygon.object.bbox().y_min)
            y_max = math.floor(visibility_polygon.object.bbox().y_max)

            inside = []

            for i in range(x_min, x_max, 2):
                for j in range(y_min, y_max, 2):
                    pt = vis.Point(i, j)
                    if pt._in(visibility_polygon.object, visibility_polygon.epsilon):
                        inside.append(pt)

        '''
        Draw
        '''

        # randomize the position of the evader
        new_evader = vis.Point(evader.x(), evader.y())
        del_x = random.random() - 0.5
        del_y = random.random() - 0.5
        del_y *= 3
        del_x *= 3

        test_pt = vis.Point(new_evader.x() + del_x, new_evader.y() + del_y)

        if test_pt._in(visibility_polygon.object, visibility_polygon.epsilon):
            new_evader.set_x(new_evader.x() + del_x)
            new_evader.set_y(new_evader.y() + del_y)

        closest_feature = -1
        d = float('inf')
        for m in range(len(features)):
            d1 = features[m].distance(new_evader)
            if d1 < d:
                d = d1
                closest_feature = m

        evader_delta = distance(evader, new_evader)
        vec = None
        feature = features[closest_feature]

        if type(feature) is Corner:
            pt = visibility_polygon.points[feature.visibility_index].coord
            qt = None
            if feature.orientation == 'R':
                qt = visibility_polygon.original_polygon[(feature.original_index - 1) %
                                                         len(visibility_polygon.original_polygon)]

            else:
                qt = visibility_polygon.original_polygon[(feature.original_index + 1) %
                                                         len(visibility_polygon.original_polygon)]

            vec = np.array([pt.x() - qt.x(), pt.y() - qt.y()])
            vec = vec / np.linalg.norm(vec)
            vec2 = vec
            vec = 1000 * vec
            end = vis.Point(vec[0] + pt.x(), vec[1] + pt.y())
            d = minimum_distance(pt, end, observer)

            d1 = distance(observer, pt)

            if d < d1:
                vec = np.array([pt.x() - observer.x(), pt.y() - observer.y()])
            else:
                vec = np.array([pt.x() - observer.x(), pt.y() - observer.y()])

            vec = vec / np.linalg.norm(vec)
            vec = vec * evader_delta * (1 / alpha)
            test_pt = vis.Point(observer.x() + vec[0], observer.y() + vec[1])
            if test_pt._in(visibility_polygon.object, visibility_polygon.epsilon):
                new_observer = vis.Point(observer.x() + vec[0], observer.y() + vec[1])

        elif type(feature) is FreeEdges:
            corner = feature.corner
            if feature.orientation == 'R':
                angle_mult = -1

            else:
                angle_mult = 1
            pt = visibility_polygon.points[corner.visibility_index].coord
            theta = angle_mult * evader_delta * (1 / alpha) / distance(observer, pt)  # d = theta * r

            rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
            vec = np.array([observer.x() - pt.x(), observer.y() - pt.y()])

            vec2 = np.dot(rot, vec2)
            test_pt = vis.Point(pt.x() + vec2[0], pt.y() + vec2[1])
            if test_pt._in(visibility_polygon.object, visibility_polygon.epsilon):
                new_observer = vis.Point(pt.x() + vec2[0], pt.y() + vec2[1])

        if not JUST_POLYGON:
            if SHOW_VORONOI:
                for i in inside:
                    j = -1
                    d = float('inf')
                    for m in range(len(features)):
                        d1 = features[m].distance(i)
                        if d1 < d:
                            d = d1
                            j = m
                    stddraw.setPenRadius(0.004)
                    stddraw.setPenColor(get_color(j))
                    stddraw.point(i.x(), i.y())

            stddraw.setPenRadius(0.001)
            stddraw.setPenColor(stddraw.BLUE)
            stddraw.polygon(visibility_polygon.x, visibility_polygon.y)
            # stddraw.filledPolygon(visibility_polygon.x, visibility_polygon.y)

            for i in range(visibility_polygon.n):
                if visibility_polygon.points[i].mapping is None:
                    stddraw.setPenRadius(0.003)
                    x0 = visibility_polygon.points[i].coord.x()
                    y0 = visibility_polygon.points[i].coord.y()
                    x1 = x0
                    y1 = y0
                    if visibility_polygon.points[i].link[0] == 'R':
                        stddraw.setPenColor(stddraw.YELLOW)
                        x1 = visibility_polygon.points[(i + 1) % visibility_polygon.n].coord.x()
                        y1 = visibility_polygon.points[(i + 1) % visibility_polygon.n].coord.y()
                    else:
                        stddraw.setPenColor(stddraw.GREEN)
                        x1 = visibility_polygon.points[(i - 1) % visibility_polygon.n].coord.x()
                        y1 = visibility_polygon.points[(i - 1) % visibility_polygon.n].coord.y()
                    # stddraw.line(x0, y0, x1, y1)

            stddraw.setPenRadius(0.01)
            stddraw.setPenColor(stddraw.BLACK)
            stddraw.point(evader.x(), evader.y())

            stddraw.setPenRadius(0.005)
            stddraw.setPenColor(stddraw.BLACK)
            stddraw.point(observer.x(), observer.y())

            for i in corner_set.corners:
                stddraw.setPenRadius(0.001)
                stddraw.setPenColor(stddraw.PINK)
                stddraw.circle(i.center.x(), i.center.y(), i.radius)
                if i.segment1 is not None:
                    stddraw.line(i.segment1.endpoint_a.x(), i.segment1.endpoint_a.y(),
                                 i.segment1.endpoint_b.x(), i.segment1.endpoint_b.y())
                if i.segment2 is not None:
                    stddraw.line(i.segment2.endpoint_a.x(), i.segment2.endpoint_a.y(),
                                 i.segment2.endpoint_b.x(), i.segment2.endpoint_b.y())

        stddraw.setPenRadius(0.001)
        stddraw.setPenColor(stddraw.RED)
        stddraw.polygon(x, y)

        evader.set_x(new_evader.x())
        evader.set_y(new_evader.y())

        observer.set_x(new_observer.x())
        observer.set_y(new_observer.y())

        stddraw.show(DELAY)
        stddraw.clear()


class VisibilityPolygon:
    def __init__(self, vispol: vis.Visibility_Polygon):
        self.x, self.y = poly_to_points(vispol)
        self.original_polygon = None
        self.object = vispol
        self.points = []
        for i in range(vispol.n()):
            self.points.append(Point(vispol[i]))

        self.x.reverse()
        self.y.reverse()
        self.points.reverse()
        self.n = self.n()
        self.observer = None
        self.epsilon = 0.0000001

    def set_observer(self, observer: vis.Point):
        self.observer = observer

    def set_original_polygon(self, polygon):
        self.original_polygon = polygon

    def n(self):
        return len(self.points)

    def orient(self):
        if self.original_polygon is not None:
            free_vertices = []
            for i in range(self.n):
                flag = False
                for j in range(len(self.original_polygon)):
                    if distance(self.points[i].coord, self.original_polygon[j]) < 0.001:
                        flag = True
                        self.points[i].mapping = j
                if not flag:
                    free_vertices.append(i)
                    stddraw.setPenRadius(0.005)

            for i in range(self.n):
                if self.points[i].mapping is None:
                    if points_collinear(self.observer, self.points[i].coord, self.points[(i + 1) % self.n].coord):
                        self.points[i].link = ('R', (i + 1) % self.n)
                        self.points[(i + 1) % self.n].corner = True
                        self.points[(i + 1) % self.n].corner_direction = 'R'
                    elif points_collinear(self.observer, self.points[i].coord, self.points[(i - 1) % self.n].coord):
                        self.points[i].link = ('L', (i - 1) % self.n)
                        self.points[(i - 1) % self.n].corner = True
                        self.points[(i - 1) % self.n].corner_direction = 'L'
                else:
                    None
                    # make sure the points are mapped
                    # stddraw.setPenRadius(0.005)
                    # stddraw.setPenColor(stddraw.BLUE)
                    # stddraw.point(self.points[i].coord.x(), self.points[i].coord.y())

            # stddraw.setPenRadius(0.005)
            # stddraw.setPenColor(stddraw.BLUE)
            # id = 11
            # stddraw.point(self.points[id].coord.x(), self.points[id].coord.y())
            # mapping = self.points[id].mapping
            # stddraw.point(self.original_polygon[mapping].x(), self.original_polygon[mapping].y())
        else:
            print("Not possible")


def points_collinear(p1: vis.Point, p2: vis.Point, p3: vis.Point):
    a = -1 * (p1.x() * (p2.y() - p3.y()) + p2.x() * (p3.y() - p1.y()) + p3.x() * (p1.y() - p2.y()))
    if a < 0.0001:
        return True
    else:
        return False


class LineSegment:
    def __init__(self, endpoint1: vis.Point, endpoint2: vis.Point):
        self.endpoint_a = endpoint1
        self.endpoint_b = endpoint2


class Point:
    def __init__(self, pt: vis.Point, mapping=None):
        self.coord = pt
        self.mapping = mapping
        self.link = None
        self.corner = False
        self.corner_direction = None


class CornerSet:
    def __init__(self, visibility_polygon: VisibilityPolygon, alpha):
        self.corners = []
        self.visibility_polygon = visibility_polygon
        self.alpha = alpha

    def process(self):
        for i in range(self.visibility_polygon.n):
            if self.visibility_polygon.points[i].mapping is None:  # free edges
                stddraw.setPenColor(stddraw.BLACK)
                index = self.visibility_polygon.points[i].link[1]
                d = 0
                pt = self.visibility_polygon.points[index].coord
                fe = self.visibility_polygon.points[i].coord
                q = None
                angle = 0
                r = None
                mapping = self.visibility_polygon.points[index].mapping
                vertex = pt
                orientation = self.visibility_polygon.points[i].link[0]

                if orientation == 'R':
                    q = self.visibility_polygon.original_polygon[(mapping - 1) %
                                                                 len(self.visibility_polygon.original_polygon)]
                    angle = -90

                    r = self.visibility_polygon.points[(index - 1) % self.visibility_polygon.n].coord

                    stddraw.setPenColor(stddraw.YELLOW)

                elif orientation == 'L':
                    q = self.visibility_polygon.original_polygon[(mapping + 1) %
                                                                 len(self.visibility_polygon.original_polygon)]
                    angle = 90

                    r = self.visibility_polygon.points[(index + 1) % self.visibility_polygon.n].coord

                    stddraw.setPenColor(stddraw.GREEN)

                else:
                    print('Error')

                vec = np.array([pt.x() - q.x(), pt.y() - q.y()])
                vec = vec / np.linalg.norm(vec)
                vec2 = vec
                vec = 1000 * vec
                end = vis.Point(vec[0] + pt.x(), vec[1] + pt.y())

                theta = np.deg2rad(angle)
                rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

                stddraw.setPenRadius(0.001)
                d = minimum_distance(pt, end, self.visibility_polygon.observer)
                radius = d * self.alpha
                # stddraw.circle(vertex.x(), vertex.y(), radius)
                corner = Corner(vertex, radius)
                corner.set_orientation(orientation)
                corner.set_original_index(mapping)
                corner.set_visibility_index(index)

                stddraw.setPenRadius(0.005)
                stddraw.setPenColor(stddraw.BLUE)
                # stddraw.point(q.x(), q.y())

                stddraw.setPenColor(stddraw.PINK)
                vec2 = 1.01 * radius * vec2
                vec2 = np.dot(rot, vec2)
                end2 = vis.Point(vec2[0] + pt.x(), vec2[1] + pt.y())
                stddraw.line(pt.x(), pt.y(), end2.x(), end2.y())

                vec3 = np.dot(rot, vec2)
                vec3 = 1000 * vec3 / np.linalg.norm(vec3)
                end3 = vis.Point(vec3[0] + end2.x(), vec3[1] + end2.y())
                # stddraw.line(end2.x(), end2.y(), end3.x(), end3.y())

                try:

                    vec = np.array([r.x() - pt.x(), r.y() - pt.y()])
                    vec = vec / np.linalg.norm(vec)
                    vec = 1000 * vec
                    end4 = vis.Point(vec[0] + pt.x(), vec[1] + pt.y())

                    A = shapely.geometry.Point(pt.x(), pt.y())
                    B = shapely.geometry.Point(end4.x(), end4.y())
                    C = shapely.geometry.Point(end2.x(), end2.y())
                    D = shapely.geometry.Point(end3.x(), end3.y())

                    line1 = LineString([A, B])
                    line2 = LineString([C, D])
                    int_pt = line1.intersection(line2)

                    stddraw.setPenRadius(0.05)
                    stddraw.setPenColor(stddraw.BOOK_RED)

                    point1 = vis.Point(end2.x(), end2.y())
                    point2 = vis.Point(int_pt.x, int_pt.y)

                    stddraw.line(point1.x(), point1.y(), point2.x(), point2.y())
                    corner.set_segment1(LineSegment(point1, point2))

                    d1 = get_shapely_point(point2).distance(get_shapely_point(r))
                    d2 = get_shapely_point(point2).distance(get_shapely_point(point1))
                    d3 = get_shapely_point(point1).distance(get_shapely_point(r))

                    if point2._in(self.visibility_polygon.object, self.visibility_polygon.epsilon):
                        corner.set_segment2(LineSegment(point2, r))
                except AttributeError:
                    None

                self.corners.append(corner)

        return self.corners


def get_shapely_point(point: vis.Point):
    return shapely.geometry.Point(point.x(), point.y())


class Corner:
    def __init__(self, point: vis.Point, radius: float):
        self.center = point
        self.radius = radius
        self.segment1 = None
        self.segment2 = None
        self.orientation = None
        self.original_index = None
        self.visibility_index = None

    def set_segment1(self, segment: LineSegment):
        self.segment1 = segment

    def set_segment2(self, segment: LineSegment):
        self.segment2 = segment

    def set_orientation(self, orientation: str):
        self.orientation = orientation

    def set_original_index(self, original_index: int):
        self.original_index = original_index

    def set_visibility_index(self, visibility_index: int):
        self.visibility_index = visibility_index

    def distance(self, point: vis.Point):
        d1 = float('inf')
        if self.segment1 is not None:
            d1 = minimum_distance(self.segment1.endpoint_a, self.segment1.endpoint_b, point)
        d2 = get_shapely_point(self.center).distance(get_shapely_point(point)) - self.radius

        d = min(d1, d2)
        return d


class FreeEdges:
    def __init__(self, segment: LineSegment, corner: Corner):
        self.segment = segment
        self.corner = corner

    def distance(self, point: vis.Point):
        return minimum_distance(self.segment.endpoint_a, self.segment.endpoint_b, point)


def distance(p1, p2):
    return math.sqrt((p1.x() - p2.x()) ** 2 +
                     (p1.y() - p2.y()) ** 2)


def minimum_distance(pt_a: vis.Point, pt_b: vis.Point, pt_e: vis.Point):
    # vector AB
    AB = np.array([pt_b.x() - pt_a.x(), pt_b.y() - pt_a.y()])

    # vector BP
    BE = np.array([pt_e.x() - pt_b.x(), pt_e.y() - pt_b.y()])

    # vector AP
    AE = np.array([pt_e.x() - pt_a.x(), pt_e.y() - pt_a.y()])

    # Variables to store dot product

    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]

    # Minimum distance from
    # point E to the line segment
    reqAns = 0

    # Case 1
    if AB_BE > 0:

        # Finding the magnitude
        y = pt_e.y() - pt_b.y()
        x = pt_e.x() - pt_b.x()
        req_ans = math.sqrt(x * x + y * y)

    # Case 2
    elif AB_AE < 0:
        y = pt_e.y() - pt_a.y()
        x = pt_e.x() - pt_a.x()
        req_ans = math.sqrt(x * x + y * y)

    # Case 3
    else:

        # Finding the perpendicular distance
        x1 = AB[0]
        y1 = AB[1]
        x2 = AE[0]
        y2 = AE[1]
        mod = math.sqrt(x1 * x1 + y1 * y1)
        req_ans = abs(x1 * y2 - y1 * x2) / mod

    return req_ans


def poly_to_points(polygon):
    end_pos_x = []
    end_pos_y = []
    for i in range(polygon.n()):
        x = polygon[i].x()
        y = polygon[i].y()

        end_pos_x.append(x)
        end_pos_y.append(y)

    return end_pos_x, end_pos_y


def get_color(v):
    x = v % 8
    if x == 0:
        return stddraw.GREEN
    elif x == 1:
        return stddraw.RED
    elif x == 2:
        return stddraw.BLUE
    elif x == 3:
        return stddraw.ORANGE
    elif x == 4:
        return stddraw.YELLOW
    elif x == 5:
        return stddraw.BOOK_RED
    elif x == 6:
        return stddraw.BOOK_BLUE
    else:
        return stddraw.DARK_GRAY
