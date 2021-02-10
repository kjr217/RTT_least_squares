import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler



class DeviceHolder:
    """
    :title:
    Holder function for an APs attributes

    :attr:
    name - e.g. AP_1
    x - x coordinate according to the custom coordinate system
    y - y coordinate according to the custom coordinate system
    z - z coordinate according to the custom coordinate system
    mac - mac address of the device, initially set to unknown
    distance_to_device - distance of an AP to the device in questionm initially set as run determine_distances
    df - holder for the dataframe associated with the ap, initially set as uninitialised
    data_list - holder for the array version of the means of the data, initially set as uninitialised
    data_list_df - holder for the dataframe version of the data_list, initially set as uninitialised

    """

    def __init__(self, ap, mac="unknown"):
        self.name = ap[0]
        self.x = ap[1]
        self.y = ap[2]
        self.z = ap[3]
        self.mac = mac
        self.distance_to_device = "run determine_distances"
        self.df = "uninitialised"
        self.data_list = "uninitialised"
        self.data_list_df = "uninitialised"


class LSProcessor:
    """
    :title:
    Runs all processing for this specific least squares algorithm

    :attr:
    self.aps - list of aps needs to be consist of lists with: [AP name, x, y]
    self.device - mobile device
    self.name - trial designation
    self.position_2d - array containing calculated position from ls_over_df_length_2d, presents empty array if not run.
    self.position_3d - array containing calculated position from ls_over_df_length_3d, presents empty array if not run.

    :methods:
    determine_distances - calculates the distance between the mobile device and the APs, then sets this to
    the device holder ls_per_epoch - runs least squares over a single epoch - returns the x_plus value

    ls_over_df_length_2d - finds the shortest dataframe height (pseudo_range_data) and runs ls_per_epoch_2d over each
    iteration, designed for 2d variations

    ls_over_df_length_3d - finds the shortest dataframe height (pseudo_range_data) and runs ls_per_epoch_3d over each
    iteration, designed for 3d variations

    ls_per_epoch_2d - performs least squares on a specified epoch and returns the expected position of the device,
    designed for 2d

    ls_per_epoch_3d - performs least squares on a specified epoch and returns the expected position of the device,
    designed for 3d

    dbscan_2d - implements the dbscan with just the x and y values, parameters include the eps (distance for a point
    to be clustered) and min_samples the minimum number of points that must be in range of a point for it to be
    considered a main point

    dbsscan_3d - implements the dbscan with the x y and z values, parameters include the eps (distance for a point
    to be clustered) and min_samples the minimum number of points that must be in range of a point for it to be
    considered a main point

    plot_2d - plot the x and y data of the positions as a scatter

    plot_3d - plot the x y and z data of the positions as a scatter

    """

    def __init__(self, name, aps, device):
        self.name = name
        ap_list = []
        for ap in aps:
            ap_list.append(DeviceHolder(ap))
        self.aps = ap_list
        self.device = DeviceHolder(device)
        self.position_3d = []
        self.position_2d = []

    def determine_distances(self):
        for ap in self.aps:
            hoz_d = ap.x - self.device.x
            width_d = ap.y - self.device.y
            vert_d = ap.z - self.device.z
            ap.distance_to_device = round(math.sqrt(hoz_d ** 2 + width_d ** 2 + vert_d ** 2))

    def ls_over_df_length_2d(self):
        df_length_list = []
        for ap in self.aps:
            df_length_list.append(len(ap.df.index))
        df_len = min(df_length_list)
        for n in range(0, df_len):
            coords = self.ls_per_epoch_2d(n)
            self.position_2d.append(coords)
            print("Position at epoch " + str(n) +
                  ":\n" + "x: " + str(coords[0]) +
                  "\ny: " + str(coords[1]))

    def ls_over_df_length_3d(self):
        df_length_list = []
        for ap in self.aps:
            df_length_list.append(len(ap.df.index))
        df_len = min(df_length_list)
        for n in range(0, df_len):
            coords = self.ls_per_epoch_3d(n)
            self.position_3d.append(coords)
            print("Position at epoch " + str(n) +
                  ":\n" + "x: " + str(coords[0]) +
                  "\ny: " + str(coords[1]) +
                  "\nz: " + str(coords[2]))

    def ls_per_epoch_2d(self, pseudo_range_index):
        self.determine_distances()
        a = np.array([0, 0])
        b = np.array([0])
        for ap in self.aps[:-1]:
            a_line = np.array([(self.aps[-1].x - ap.x), (self.aps[-1].y - ap.y)])
            a = np.vstack([a, a_line])
            df = float(ap.df['<Est. Range(m)>'][pseudo_range_index])
            b_line = np.array([((ap.df['<Est. Range(m)>'][pseudo_range_index]*1000) ** 2 -
                                (self.aps[-1].df['<Est. Range(m)>'][pseudo_range_index]*1000) ** 2 +
                                self.aps[-1].x ** 2 +
                                self.aps[-1].y ** 2 -
                                ap.x ** 2 -
                                ap.y ** 2)
                               / 2])
            b = np.vstack([b, b_line])
        a = np.delete(a, 0, axis=0)
        b = np.delete(b, 0, axis=0)
        a_trans = np.transpose(a)
        a_trans_x_a = np.dot(a_trans, a)
        try:
            a_trans_x_a_inv = np.linalg.inv(a_trans_x_a)
        except np.linalg.LinAlgError:
            print("Determinant of matrix is 0 so has no inverse, pushing to pseudo inverse to force an inverse output")
            a_trans_x_a_inv = np.linalg.pinv(a_trans_x_a)
        a_other = np.dot(a_trans, b)
        x_plus = np.dot(a_trans_x_a_inv, a_other)

        return x_plus

    def ls_per_epoch_3d(self, pseudo_range_index):
        self.determine_distances()
        a = np.array([0, 0, 0])
        b = np.array([0])
        for ap in self.aps[:-1]:
            a_line = np.array([(self.aps[-1].x - ap.x), (self.aps[-1].y - ap.y), (self.aps[-1].z - ap.z)])
            a = np.vstack([a, a_line])
            b_line = np.array([((ap.df['<Est. Range(m)>'][pseudo_range_index]*1000) ** 2 -
                                (self.aps[-1].df['<Est. Range(m)>'][pseudo_range_index]*1000) ** 2 +
                                self.aps[-1].x ** 2 +
                                self.aps[-1].y ** 2 +
                                self.aps[-1].z ** 2 -
                                ap.x ** 2 -
                                ap.y ** 2 -
                                ap.z ** 2)
                               / 2])
            b = np.vstack([b, b_line])
        a = np.delete(a, 0, axis=0)
        b = np.delete(b, 0, axis=0)
        a_trans = np.transpose(a)
        a_trans_x_a = np.matmul(a_trans, a)
        try:
            a_trans_x_a_inv = np.linalg.inv(a_trans_x_a)
        except np.linalg.LinAlgError:
            print("Determinant of matrix is 0 so has no inverse, pushing to pseudo inverse to force an inverse output")
            a_trans_x_a_inv = np.linalg.pinv(a_trans_x_a)
        a_other = np.matmul(a_trans, b)
        x_plus = np.matmul(a_trans_x_a_inv, a_other)

        return x_plus

    def dbscan_3d(self, eps, min_samples):
        metric = 'euclidean'
        data = []
        for pos in self.position_3d:
            data.append([pos[0][0], pos[1][0], pos[2][0]])
        db = DBSCAN(eps, min_samples=min_samples, metric=metric).fit_predict(data)
        print(db)

    def dbscan_2d(self, eps, min_samples):
        metric = 'euclidean'
        data = []
        for pos in self.position_3d:
            data.append([pos[0][0], pos[1][0]])



        db = DBSCAN(eps, min_samples=min_samples, metric=metric).fit(data)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = data[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=2)

            xy = data[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()

    def plot_3d(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        pos_x = []
        pos_y = []
        pos_z = []
        for pos in self.position_3d:
            pos_x.append(pos[0])
            pos_y.append(pos[1])
            pos_z.append(pos[2])

        ax.scatter3D(pos_x, pos_y, pos_z)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    def plot_2d(self):
        fig = plt.figure()
        ax = plt.axes()
        pos_x = []
        pos_y = []
        for pos in self.position_2d:
            pos_x.append(pos[0])
            pos_y.append(pos[1])

        ax.scatter(pos_x, pos_y)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()

    # def ls_per_epoch(self, pseudo_range_index, rea):
    #     self.determine_distances()
    #     h = np.array([0, 0, 0, 0])
    #     dz = np.array([0])
    #     for ap in self.aps:
    #         rej = np.array([[ap.x], [ap.y], [ap.z]])
    #         raj = float(ap.distance_to_device)
    #         u = (rej - rea)/raj
    #         ux = -u[0][0]
    #         uy = -u[1][0]
    #         uz = -u[2][0]
    #         h_line = np.array([ux, uy, uz])
    #         h = np.vstack([h, h_line])
    #         df = float(ap.df['<Est. Range(m)>'][pseudo_range_index])
    #         # dz is a one column array
    #         dz_line = np.array([df - raj - 0])
    #         dz = np.vstack([dz, dz_line])
    #
    #     h = np.delete(h, 0, axis=0)
    #     dz = np.delete(dz, 0, axis=0)
    #     x = np.array([[rea[0][0]], [rea[1][0]], [rea[2][0]], [0]])
    #     h_trans = np.transpose(h)
    #     h_trans_x_h = np.dot(h_trans, h)
    #     try:
    #         h_trans_x_h_inv = np.linalg.inv(h_trans_x_h)
    #     except np.linalg.LinAlgError:
    #         print("Determinant of matrix is 0 so has no inverse, pushing to pseudo inverse to force an inverse output")
    #         h_trans_x_h_inv = np.linalg.pinv(h_trans_x_h)
    #     h_other = np.dot(h_trans, dz)
    #     x_plus = x + np.dot(h_trans_x_h_inv, h_other)
    #     print(x_plus)
    #
    #     return x_plus
