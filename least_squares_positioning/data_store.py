import pandas as pd
import numpy as np
import math


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


class DataStore:
    """
    :title:
    Storage container for all data pertaining to one trial run (i.e. one session of positioning)

    :attr:
    self.aps - list of aps needs to be consist of lists with: [AP name, x, y]
    self.device - mobile device
    self.name - trial designation

    :methods:
    determine_distances - calculates the distance between the mobile device and the APs, then sets this to
    the device holder ls_per_epoch - runs least squares over a single epoch - returns the x_plus value
    ls_over_df_length - finds the shortest df (pseudo_range_data) and runs ls_per_epoch over each iteration,

    """

    def __init__(self, name, aps, device):
        self.name = name
        ap_list = []
        for ap in aps:
            ap_list.append(DeviceHolder(ap))
        self.aps = ap_list
        self.device = DeviceHolder(device)

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
            print("Position at epoch " + str(n) +
                  ":\n" + "x: " + str(coords[0]) +
                  "\ny: " + str(coords[1]) +
                  "\nz: " + str(coords[2]))

    # X = (AtA)-1Atb
    def ls_per_epoch_2d(self, pseudo_range_index):
        self.determine_distances()
        a = np.array([0, 0])
        b = np.array([0])
        for ap in self.aps[:-1]:
            a_line = np.array([(self.aps[-1].x - ap.x), (self.aps[-1].y - ap.y)])
            a = np.vstack([a, a_line])
            df = float(ap.df['<Est. Range(m)>'][pseudo_range_index])
            b_line = np.array([(ap.df['<Est. Range(m)>'][pseudo_range_index] ** 2 -
                                self.aps[-1].df['<Est. Range(m)>'][pseudo_range_index] ** 2 +
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

    # X = (AtA)-1Atb
    def ls_per_epoch_3d(self, pseudo_range_index):
        self.determine_distances()
        a = np.array([0, 0, 0])
        b = np.array([0])
        for ap in reversed(self.aps[:-1]):
            a_line = np.array([(self.aps[-1].x - ap.x), (self.aps[-1].y - ap.y), (self.aps[-1].z - ap.z)])
            a = np.vstack([a, a_line])
            b_line = np.array([(ap.df['<Est. Range(m)>'][pseudo_range_index] ** 2 -
                                self.aps[-1].df['<Est. Range(m)>'][pseudo_range_index] ** 2 +
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
        a_trans_x_a = np.dot(a_trans, a)
        try:
            a_trans_x_a_inv = np.linalg.inv(a_trans_x_a)
        except np.linalg.LinAlgError:
            print("Determinant of matrix is 0 so has no inverse, pushing to pseudo inverse to force an inverse output")
            a_trans_x_a_inv = np.linalg.pinv(a_trans_x_a)
        a_other = np.dot(a_trans, b)
        x_plus = np.dot(a_trans_x_a_inv, a_other)

        return x_plus

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
