#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 13:18, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
from pandas import read_csv

filenames = ["ec2_cpu_utilization_5f5533", "elb_request_count_8c0756", "iio_us-east-1_i-a2eb1cd9_NetworkIn",
             "exchange-3_cpc_results", "exchange-3_cpm_results",
             "occupancy_t4013", "speed_7578", "TravelTime_451"]
cols = [[0, 1], [0, 1], [0, 3], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
names = ["value", "value", "MB", "value", "value", "value", "value", "value"]

for it in range(len(filenames)):
    df = read_csv(filenames[it] + ".csv", usecols=cols[it], header=0)

    mean_val = df[names[it]].mean(skipna=True)

    temp = (df[names[it]] == 0).sum(axis=0)
    print(temp)

    df[names[it]] = df[names[it]].mask(df[names[it]] == 0, mean_val)

    df.columns = ["time", "value"]

    df.to_csv("f_" + filenames[it] + ".csv", index=False)


    df = df.set_index('time')

    plt.figure(figsize=(6, 4))
    df.plot()
    plt.savefig("img_" + filenames[it] + ".png")
    plt.show()

    tsaplots.plot_acf(df['value'], lags=72)
    plt.savefig("acf_" + filenames[it] + ".png")
    plt.show()