#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:43, 09/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

import matplotlib.pyplot as plt


def _draw_predict_with_error__(data=None, error=None, filename=None, pathsave=None):
    plt.plot(data[0])       # True
    plt.plot(data[1])       # Prediction
    plt.ylabel('Value')
    plt.xlabel('Time Step (5 minutes)')
    plt.legend(['Actual y... RMSE= ' + str(error[0]), 'Predict y... MAE= ' + str(error[1])], loc='upper right')
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None


def draw_predict(fig_id=None, y_test=None, y_pred=None, filename=None, pathsave=None):
    plt.figure(fig_id)
    plt.plot(y_test)
    plt.plot(y_pred)
    plt.ylabel('CPU')
    plt.xlabel('Timestamp')
    plt.legend(['Actual', 'Predict'], loc='upper right')
    plt.savefig(pathsave + filename + ".png")
    plt.close()
    return None


def draw_raw_time_series_data(data=None, label=None, title=None, filename=None, pathsave=None):
    plt.plot(data)
    plt.xlabel(label["y"])
    plt.ylabel(label["x"])
    plt.title(title, fontsize=8)
    plt.savefig(pathsave + filename + ".pdf")
    plt.close()
    return None


def draw_raw_time_series_data_and_show(data=None, label=None, title=None):
    plt.plot(data)
    plt.xlabel(label["y"])
    plt.ylabel(label["x"])
    plt.title(title, fontsize=8)
    plt.show()
    return None
