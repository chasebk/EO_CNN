#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:44, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import concatenate, savetxt, array
from csv import DictWriter
from os import getcwd, path, makedirs
from pandas import read_csv


def _save_results_to_csv__(item=None, filename=None, pathsave=None):
	check_directory = getcwd() + "/" + pathsave
	if not path.exists(check_directory):
		makedirs(check_directory)
	with open(pathsave + filename + ".csv", 'a') as file:
		w = DictWriter(file, delimiter=',', lineterminator='\n', fieldnames=item.keys())
		if file.tell() == 0:
			w.writeheader()
		w.writerow(item)

def _save_prediction_to_csv__(y_true=None, y_pred=None, y_true_scaled=None, y_pred_scaled=None, filename=None, pathsave=None):
	check_directory = getcwd() + "/" + pathsave
	if not path.exists(check_directory):
		makedirs(check_directory)
	temp = concatenate((y_true, y_pred, y_true_scaled, y_pred_scaled), axis=1)
	savetxt(pathsave + filename + ".csv", X=temp, delimiter=",")
	return None

def _save_loss_train_to_csv__(error=None, filename=None, pathsave=None):
	savetxt(pathsave + filename + ".csv", array(error), delimiter=",")
	return None

def _load_dataset__(path_to_data=None, cols=None):
	"""
	:param path_to_data:
	:type path_to_data:
	:param features_selected:  example -> ["bytes"], ["degree", "wind", ...]
	:type features_selected:
	:param features_index:  example -> ["date"], ["time"], ....
	:type features_index:
	:return:
	:rtype:
	"""
	df = read_csv(path_to_data + ".csv", usecols=cols)
	return df.values





