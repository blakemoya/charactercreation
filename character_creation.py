import tkinter as tk
from os import path
from os import listdir
import glob
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy import stats
import math
import dlib
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class CharacterCreation(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand=True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        frame = ControlPanel(container, self)
        self.frames[ControlPanel] = frame
        frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(ControlPanel)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


def ratios_to_int(arr, num):
    assert isinstance(num, int)
    arr = arr * num
    arr = np.round(arr).astype(int)
    return arr


def standings():
    dist_ind_tuple = tree.query(curr_point, k=NUM_NEIGHBORS)
    dist_z = stats.zscore(dist_ind_tuple[0])
    dist_z = 1 / (EXAG_FACTOR ** dist_z)
    dist_z_sum = np.sum(dist_z)

    dist_z_norm = dist_z / dist_z_sum
    dist_z_ints = ratios_to_int(dist_z_norm, NUM_IMAGES)

    ids = []
    parts = []
    for i in range(len(dist_ind_tuple[0])):
        ids.append(df.index[dist_ind_tuple[1][i]])
        parts.append(dist_z_ints[i])
    standing = np.array([ids, parts])
    return standing


def paths_from_standings(standings):
    path_list = []
    for i in range(len(standings[0])):
        sample = np.random.choice(listdir(VGG_TRAIN_PATH + standings[0][i] + "/"), size=int(standings[1][i]))
        if len(sample) > 0:
            for s in sample:
                path_list.append(VGG_TRAIN_PATH + standings[0][i] + "/" + s)

    return path_list


class ControlPanel(tk.Frame):

    def __init__(self, parent, controller):
        panel_width = 8
        panel_height = 16
        system_button_face = np.array([61680, 61680, 61680]) / 65535
        tk.Frame.__init__(self, parent)
        for row in range(panel_height):
            for col in range(panel_width):
                button = tk.Button(self, text="{}".format((row * panel_width + col)), command=lambda axis=int(row * panel_width + col): self.click(axis))
                button.grid(row=row, column=col, sticky="nsew")

        self.update_idletasks()

        self.root = parent
        self.standing = standings()

        # Length should be width of frame. find out what parent.winfo_width() returns and make is a screen unit
        self.scale = tk.Scale(self, from_=-1, to=1, resolution=0.01, orient="horizontal", length=button.winfo_height() * 24)
        self.scale.configure(command=self.update_scale)
        self.scale.grid(row=panel_height, column=0, columnspan=panel_width + panel_height)
        self._scaling = None

        self.active_label = tk.Label(self, text="The active axis is: {}".format(active_axis))
        self.active_label.grid(row=panel_height + 1, column=0, columnspan=panel_height + panel_height, sticky="s")

        self.fig = Figure(figsize=(4, 4), dpi=100, facecolor=system_button_face)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('off')
        # set initial data to whatever, later this will link to the image at mean_point preloaded
        # self.data = np.zeros((8, 8))
        self.data = generate_img(paths_from_standings(self.standing), "test.jpg")
        # for i in range(8):
        #     self.data[i] = curr_point[8 * i:8 * i + 8]
        self.ax.imshow(self.data)

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=panel_width, columnspan=panel_height, rowspan=panel_height, sticky="nsew")

        # Add histogram of active axis

    def click(self, axis):
        assert 0 <= axis < 128
        global active_axis
        active_axis = axis
        self.active_label.configure(text="The active axis is: {}".format(active_axis))
        self.scale.set(curr_point[axis])
        # Maybe reconfigure scale here to increment based on a multiple of that axis' variance

    def update_scale(self, value):
        if self._scaling:
            self.root.after_cancel(self._scaling)
        # Don't forget to reevaluate this time delay after seeing how it works w images
        self._scaling = self.root.after(500, lambda: self.set_axis(value))

    def set_axis(self, value):
        if active_axis is not None:
            global curr_point
            curr_point[active_axis] = value
            curr_point /= np.linalg.norm(curr_point, axis=0)

            self.scale.set(curr_point[active_axis])

            # if new standing, refresh img
            s = standings()
            if not np.array_equal(s, self.standing):
                self.standing = s
                print(self.standing)
                self.data = generate_img(paths_from_standings(self.standing), "test.jpg")
                self.ax.imshow(self.data)
                self.canvas.draw()


def read_points(path_list):
    drop_list = []
    points_array = []

    for file_path in path_list:
        if file_path.endswith(".jpg"):
            points = []

            img = dlib.load_rgb_image(file_path)
            detected_faces = detector(img, 1)

            if len(detected_faces) > 0:
                shape = predictor(img, detected_faces[0])
                for i in range(68):
                    points.append((int(shape.part(i).x), int(shape.part(i).y)))
                points_array.append(points)
            else:
                drop_list.append(file_path)

    return points_array, drop_list


def read_images(path_list, drop_list):
    imgs_array = []

    for file_path in path_list:
        if file_path.endswith(".jpg") and file_path not in drop_list:
            img = cv2.imread(file_path)[:, :, ::-1]
            img = np.float32(img) / 255.0
            imgs_array.append(img)

    return imgs_array


def similarity_transform(in_points, out_points):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    inPts = np.copy(in_points).tolist()
    outPts = np.copy(out_points).tolist()

    xin = c60 * (inPts[0][0] - inPts[1][0]) - s60 * (inPts[0][1] - inPts[1][1]) + inPts[1][0]
    yin = s60 * (inPts[0][0] - inPts[1][0]) + c60 * (inPts[0][1] - inPts[1][1]) + inPts[1][1]

    inPts.append([np.int(xin), np.int(yin)])

    xout = c60 * (outPts[0][0] - outPts[1][0]) - s60 * (outPts[0][1] - outPts[1][1]) + outPts[1][0]
    yout = s60 * (outPts[0][0] - outPts[1][0]) + c60 * (outPts[0][1] - outPts[1][1]) + outPts[1][1]

    outPts.append([np.int(xout), np.int(yout)])

    transform = cv2.estimateAffinePartial2D(np.array([inPts]), np.array([outPts]))

    return transform[0]


def rectangle_contains(rectangle, point):
    if point[0] < rectangle[0] or point[1] < rectangle[1] or point[0] > rectangle[2] or point[1] > rectangle[3]:
        return False
    return True


def calculate_delaunay_triangles(rectangle, points):
    subdiv = cv2.Subdiv2D(rectangle)

    for p in points:
        subdiv.insert((p[0], p[1]))

    triangle_list = subdiv.getTriangleList()

    delaunay_tri = []

    for t in triangle_list:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectangle_contains(rectangle, pt1) and rectangle_contains(rectangle, pt2) and rectangle_contains(rectangle,
                                                                                                            pt3):
            ind = []
            for j in range(3):
                for k in range(len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                        ind.append(k)
            if len(ind) == 3:
                delaunay_tri.append((ind[0], ind[1], ind[2]))

    return delaunay_tri


def constrain_point(p, w, h):
    p = min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1)

    return p


def apply_affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def warp_triangle(img1, img2, t1, t2):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])
    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)
    img2_rect = img2_rect * mask

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect


def generate_img(path_list, save_path):
    w = 256
    h = 256

    all_points, drops = read_points(path_list)
    images = read_images(path_list, drops)
    images_norm = []
    points_norm = []

    eye_corner_dst = [(np.int(0.3 * w), np.int(h / 3)), (np.int(0.7 * w), np.int(h / 3))]
    boundary_points = np.array(
        [(0, 0), (w / 2, 0), (w - 1, 0), (w - 1, h / 2), (w - 1, h - 1), (w / 2, h - 1), (0, h - 1), (0, h / 2)])
    points_avg = np.array([(0, 0)] * (len(all_points[0]) + len(boundary_points)), np.float32())
    n = len(all_points[0])
    num_images = len(images)

    for i in range(num_images):
        points1 = all_points[i]
        eye_corner_src = [all_points[i][36], all_points[i][45]]
        transform = similarity_transform(eye_corner_src, eye_corner_dst)
        img = cv2.warpAffine(images[i], transform, (w, h))

        points2 = np.reshape(np.array(points1), (68, 1, 2))
        points = cv2.transform(points2, transform)
        points = np.float32(np.reshape(points, (68, 2)))

        points = np.append(points, boundary_points, axis=0)

        points_avg = points_avg + points / num_images

        points_norm.append(points)
        images_norm.append(img)

    rectangle = (0, 0, w, h)
    dt = calculate_delaunay_triangles(rectangle, np.array(points_avg))

    output = np.zeros((h, w, 3), np.float32())

    for i in range(len(images_norm)):
        img = np.zeros((h, w, 3), np.float32())

        for j in range(len(dt)):
            t_in = []
            t_out = []

            for k in range(3):
                p_in = points_norm[i][dt[j][k]]
                p_in = constrain_point(p_in, w, h)

                p_out = points_avg[dt[j][k]]
                p_out = constrain_point(p_out, w, h)

                t_in.append(p_in)
                t_out.append(p_out)

            warp_triangle(images_norm[i], img, t_in, t_out)

        output = output + img

    output = output / num_images

    return output


if __name__ == "__main__":

    CSV_PATH = "test_master.csv"
    VGG_TRAIN_PATH = "C:\\Users\Blake\Desktop\\face/"
    NUM_NEIGHBORS = 10
    EXAG_FACTOR = 4
    NUM_IMAGES = 10
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


    # Add startpage to link to master.csv and train folder of VGGFace2, for now assert
    assert (path.exists(CSV_PATH))
    df = pd.read_csv(CSV_PATH, index_col=0)
    curr_point = df.mean().values / np.linalg.norm(df.mean().values, axis=0)
    mean_point_norm = curr_point
    active_axis = None
    tree = KDTree(df)

    app = CharacterCreation()
    app.mainloop()
