"""
home ex. Claps.
sound positioning based on Time Difference Of Arrival in a 2d plane.

Written by Ofri Gabso
"""
# =================================
# Imports
# =================================
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from numba import njit

# =================================
# Parameters
# =================================
sensors_geometry = np.array(
    [[0, 0], [0, 20], [30, 0]])  # (0,0) is the mic, the other two are the reflection on the other side of the walls
delta_r = 0.1  # space resolution
v = 1  # m/s
misfit_matrix = np.zeros((int(15 / delta_r), int(10 / delta_r)))
tdoa_matrix = np.zeros((int(15 / delta_r), int(10 / delta_r), sensors_geometry.shape[0]))
# test times
test_t1 = 16.74922424
test_t2 = 25.8702747
test_t3 = 36.82250459
test_toas = (test_t1, test_t2, test_t3)


# =================================
# Functions
# =================================


@njit
def distance_2_points(pt1, pt2):
    dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2 + (pt1[2] - pt2[2]) ** 2)
    return dist


def find_min(array):
    return np.where(array == array.min())


def create_cmap():
    c = list()
    c.append([1.0, 1.0, 1.0, 1.0])
    c.append([1.0, 1.0, 0.0, 1.0])
    c.append([1.0, 0.0, 0.0, 1.0])
    c.append([1.0, 0.3, 1.0, 1.0])
    c.append([0.0, 0.0, 1.0, 1.0])
    c.append([0.0, 0.0, 0.0, 1.0])
    c = np.array(c)
    vec = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

    r = np.interp(np.linspace(0, 1, num=100), vec, c[:, 0])
    g = np.interp(np.linspace(0, 1, num=100), vec, c[:, 1])
    b = np.interp(np.linspace(0, 1, num=100), vec, c[:, 2])
    cmap_1 = ListedColormap(np.array([r, g, b]).T)

    return cmap_1


def calc_tdoa_misfit(tdoa_mat, misfit_mat, toa_list=test_toas):
    """ calcs the misfit matrix, a matrix that represents how much each point in space is "close" to the claps point,
    TDOA wise

    :param tdoa_mat: zeroes matrix in the shape of [room_x, room_y, num of sensors]
    :param misfit_mat: zeroes matrix in the shape of the room
    :param toa_list: list of toas
    :return: the TDOA misfit matrix
    """
    event_dt = np.array([[toa_list[1] - toa_list[0]], [toa_list[2] - toa_list[1]], [toa_list[2] - toa_list[0]]])
    for i in range(tdoa_mat.shape[0]):
        for j in range(tdoa_mat.shape[1]):
            toa = np.sqrt((i * delta_r - sensors_geometry[:, 0]) ** 2 + (
                    j * delta_r - sensors_geometry[:, 1]) ** 2) / v  # time of arrival for each "mic" in sec
            tdoa_mat[i, j, :] = [toa[1] - toa[0], toa[2] - toa[1], toa[2] - toa[0]]
            misfit_mat[i, j] = distance_2_points(tdoa_mat[i, j, :], event_dt.squeeze())
    return misfit_mat


def plot_misfit(misfit, cmap, clap_x, clap_y):
    plt.imshow(misfit.T, cmap=cmap, origin='lower')
    plt.suptitle('Claps coordinates, visualisation')
    plt.title(f"Claps coordinates [cm]: ({int(clap_x)},{int(clap_y)})")
    plt.xlabel('x axis [cm]')
    plt.ylabel('y axis [cm]')
    plt.show()


# =================================
# Main
# =================================

if __name__ == '__main__':
    misfit_matrix = calc_tdoa_misfit(tdoa_matrix, misfit_matrix)
    clap_x, clap_y = find_min(misfit_matrix)
    print(f"Claps coordinates [cm]: ({int(clap_x)},{int(clap_y)})")

    plot_misfit(misfit=misfit_matrix, cmap=create_cmap(), clap_x=clap_x, clap_y=clap_y)
