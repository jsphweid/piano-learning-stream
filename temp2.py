from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

data = [(60, 5, 121), (61, 5, 103), (62, 5, 14.8), (63, 5, 48.5), (64, 5, 57.5), (65, 5, 75.7), (66, 5, 89.6), (67, 5, 55.3), (68, 5, 63.3), (69, 5, 118), (70, 5, 128), (71, 5, 105), (72, 5, 115), (73, 5, 104), (74, 5, 134), (75, 5, 123), (76, 5, 66.3), (77, 5, 132), (78, 5, 145), (79, 5, 115), (80, 5, 38.2), (81, 5, 10.4), (82, 5, 18.4), (83, 5, 87), (84, 5, 86.7), (85, 5, 78.9), (86, 5, 89.9), (87, 5, 108), (88, 5, 57.1), (89, 5, 51.1), (90, 5, 69.1), (91, 5, 59.8), (60, 6, 48.9), (61, 6, 33.3), (62, 6, -19.2), (63, 6, -17.5), (64, 6, -6.5), (65, 6, 75.7), (66, 6, 89.6), (67, 6, 55.3), (68, 6, 99.8), (69, 6, 156), (70, 6, 141), (71, 6, 54.1), (72, 6, 66.1), (73, 6, 98.9), (74, 6, 155), (75, 6, 146), (76, 6, 111), (77, 6, 132), (78, 6, 145), (79, 6, 97.3), (80, 6, 101), (81, 6, 59.4), (82, 6, 70.4), (83, 6, 142), (84, 6, 145), (85, 6, 140), (86, 6, 56.9), (87, 6, 77.8), (88, 6, 21.1), (89, 6, 27.1), (90, 6, 48.1), (91, 6, 41.8)]
x, y, z = zip(*data)
grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.Spectral)
plt.show()