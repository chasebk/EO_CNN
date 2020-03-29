#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 18:18, 26/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from pandas import read_csv

filenames = ["ec2_cpu_utilization_5f5533.csv", "elb_request_count_8c0756.csv", "iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
             "exchange-3_cpc_results.csv", "exchange-3_cpm_results.csv",
             "occupancy_t4013.csv", "speed_7578.csv", "TravelTime_451.csv"]
cols = [[0, 1], [0, 1], [0, 3], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
names = ["value", "value", "MB", "value", "value", "value", "value", "value"]

for it in range(len(filenames)):

    df = read_csv(filenames[it], usecols=cols[it], header=0)

    mean_val = df[names[it]].mean(skipna=True)

    temp = (df[names[it]] == 0).sum(axis=0)
    print(temp)

    df[names[it]] = df[names[it]].mask(df[names[it]] == 0, mean_val)

    df.columns = ["time", "value"]

    df.to_csv("f_" + filenames[it], index=False)


