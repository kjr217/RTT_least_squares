import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from least_squares_positioning.LSProcessor import LSProcessor


def main(file_path):
    def dataframe_maker(path, ap):
        df_all = pd.DataFrame(
            columns=['#<Time(ms)>', '<True Range(m)>', '<Est. Range(m)>', '<Std dev(m)>', '<Successes#>', '<Burst#>',
                     '<RSSI(dBm)>', '<Ch-MHz>', '<AP-MAC>'])
        data_list = []
        df_list = []

        for file in glob.iglob(path):
            df = pd.read_csv(file)
            data_list.append([df['<True Range(m)>'].mean(),
                              df['<Est. Range(m)>'].mean(),
                              df['<Est. Range(m)>'].std(),
                              df['<RSSI(dBm)>'].mean(),
                              ap])
            df['AP'] = ap

        data_list_pd = pd.DataFrame(data_list, columns=['<True Range(m)>',
                                                        '<Est. Range(m)>',
                                                        '<Std dev(m)>',
                                                        '<RSSI>',
                                                        '<AP>'])
        return df, data_list, data_list_pd

    """
    SETUP
    """

    # set basic attributes for the datastore class, set z to 0 for 2d

    name = 'test'
    ap_1_params = ["AP_1", 2500, 8900, 760]
    ap_2_params = ["AP_2", 2500, 6350, 760]
    ap_3_params = ["AP_3", 7325, 3600, 850]
    ap_4_params = ["AP_4", 8550, 3250, 600]
    ap_5_params = ["AP_5", 9460, 6375, 750]
    ap_6_params = ["AP_6", 7000, 9750, 1290]
    ap_params = [ap_1_params, ap_2_params, ap_3_params, ap_4_params, ap_5_params, ap_6_params]

    device = ["device", 5150, 8220, 850]

    # declare datastore object
    data_store = LSProcessor(name, ap_params, device)
    # calculate distances
    data_store.determine_distances()

    for n, file in enumerate(file_path):
        data_store.aps[n].df, data_store.aps[n].data_list, data_store.aps[n].data_list_df = \
            dataframe_maker(file, ap_params[n][0])

    # for ap in data_store.aps:
    #     print(ap.df)

    data_store.ls_over_df_length_3d()
    # data_store.plot_3d()
    data_store.dbscan_2d(0.2, 10)
    # show distances
    # print(data_store.ls_per_epoch_3d(3))


    for ap in data_store.aps:
        print(ap.name + " real distance to device: " + str(ap.distance_to_device))

    print("Device position: " + "\nx: " + str(data_store.device.x) + "\ny: " + str(data_store.device.y))
    # print(data_store.position)

if __name__ == "__main__":
    # input all file paths in a list
    main(["JanDataTrial3/AP_1/*.csv",
          "JanDataTrial3/AP_2/*.csv",
          "JanDataTrial3/AP_3/*.csv",
          "JanDataTrial3/AP_4/*.csv",
          "JanDataTrial3/AP_5/*.csv",
          "JanDataTrial3/AP_6/*.csv"])
