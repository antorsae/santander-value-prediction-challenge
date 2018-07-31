import itertools
import functools
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('classic')

from matplotlib.patches import Rectangle
import pickle
import numpy as np
import matplotlib.cm as cm
import pandas as pd
import cv2
from tqdm import tqdm

df = pd.read_feather('train.csv.pd')
df = df.drop(["ID", "target"], axis=1)
giba_rows = pickle.load(open('giba_rows.pk', 'rb'))
df = df.loc[giba_rows]

DRAW = True


class SelectArea(object):
    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0, 0), 0, 0, fill=False)

        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.on_release)

    def on_press(self, event):
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()

        area = np.argwhere((Y[:, 0] < self.x1) & (Y[:, 0] > self.x0) &
                           (Y[:, 1] > self.y1) & (Y[:, 1] < self.y0))
        cols = INV_COLMAP_ARRAY[area.flatten()]
        print(cols)
        if DRAW:

            def find_nearest_column(col0, seen):
                x = df[[col0]].values.flatten()
                max_sum = -1
                max_non_zero = -1
                max_sum_col = None
                for col in cols:
                    if col == col0:
                        continue
                    if col in seen:
                        continue
                    y = df[[col]].values.flatten()
                    #if y.sum() == 0:
                    #    continue
                    sum = np.sum(x[0:-1] == y[1:])
                    non_zero = len(y[1:][y[1:] > 0])
                    if max_sum == -1 or max_sum < sum:
                        max_sum = sum
                        max_sum_col = col
                    elif max_sum == sum and max_non_zero < non_zero:
                        max_non_zero = non_zero
                        max_sum_col = col

                y = df[[max_sum_col]].values.flatten()
                return max_sum_col

            def distance(a, b):
                x = df[[a]].values.flatten()
                y = df[[b]].values.flatten()
                sum = 1. / (np.sum(x[0:-1] == y[1:]) + 1e-10)
                #non_zero = len(y[1:][y[1:] > 0])
                return sum

            #pairs = list(itertools.combinations(cols, 2))
            #pairs = sorted(pairs, key=lambda x: distance(x[0], x[1]))
            #print(pairs)

            min_loss = -1
            for i in tqdm(range(len(cols))):
                #cols_sorted = sorted(
                #    list(cols), key=lambda x: distance(x, cols[i]))
                cols_sorted = [cols[i]]
                col0 = cols[i]
                for j in range(len(cols) - 1):
                    col0 = find_nearest_column(col0, cols_sorted)
                    cols_sorted.append(col0)
                cols_sorted = np.array(cols_sorted)

                m = df[cols_sorted].values
                loss = np.sqrt(
                    np.square(np.roll(np.roll(m, -1, 0), -1, 1) - m).sum())
                if min_loss == -1 or min_loss > loss:
                    best_sort = cols_sorted
                    min_loss = loss
            cols = best_sort
            print(cols)
            img = np.log1p(df[cols].values) * 11
            img = np.array(np.stack((img, ) * 3, -1), dtype=np.uint8).copy()
            img = cv2.resize(
                img, (img.shape[1] * 10, img.shape[0] * 10),
                interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Sorted Columns %d' % len(cols), img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            pass


a = SelectArea()
Y = np.load('Y.npy')
plt.scatter(Y[:, 0], Y[:, 1])

COLMAP = pickle.load(open('colmap.pk', 'rb'))
INV_COLMAP = {v: k for k, v in COLMAP.items()}
INV_COLMAP_ARRAY = []
for key in sorted(INV_COLMAP.keys()):
    INV_COLMAP_ARRAY.append(INV_COLMAP[key])
INV_COLMAP_ARRAY = np.array(INV_COLMAP_ARRAY)

column_groups = np.array(pickle.load(open('groups40.pk', 'rb')))
column_groups = np.delete(column_groups, [8, 9, 14, 37, 42, 45], axis=0)

colors = cm.rainbow(np.linspace(0, 1, column_groups.shape[0]))

for i, group in enumerate(column_groups):
    for col in group:
        j = COLMAP[col]
        plt.scatter([Y[j, 0]], [Y[j, 1]], color=colors[i])

plt.show()
