import tkinter as tk
from os import path
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy import stats
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
        self.standing = np.empty(0)

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
        self.data = np.zeros((8, 8))
        for i in range(8):
            self.data[i] = curr_point[8 * i:8 * i + 8]
        self.ax.imshow(self.data, interpolation='nearest')

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
        self._scaling = self.root.after(100, lambda: self.set_axis(value))

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
                for i in range(8):
                    self.data[i] = curr_point[8 * i:8 * i + 8]
                self.ax.imshow(self.data, interpolation='nearest')
                self.canvas.draw()


if __name__ == "__main__":

    CSV_PATH = "master.csv"
    VGG_TRAIN_PATH = ""
    NUM_NEIGHBORS = 10
    EXAG_FACTOR = 4
    NUM_IMAGES = 10

    # Add startpage to link to master.csv and train folder of VGGFace2, for now assert
    assert (path.exists(CSV_PATH))
    df = pd.read_csv("master.csv", index_col=0)
    curr_point = df.mean().values / np.linalg.norm(df.mean().values, axis=0)
    mean_point_norm = curr_point
    active_axis = None
    tree = KDTree(df)

    app = CharacterCreation()
    app.mainloop()
