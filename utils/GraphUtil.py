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


def __create_time_steps__(length):
    return list(range(-length, 0))

def _plot_history_true_future_prediciton__(plot_data, delta, title):
    """
    :param plot_data: 2D-numpy array
    :param delta:
    :param title:
    :return:
    """
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = __create_time_steps__(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 5) * 2])
    plt.xlabel('Time-Step')
    plt.show()
    return plt


def _plot_train_history__(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()
    return 0

