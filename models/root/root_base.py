#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 14:20, 25/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from utils.PreprocessingUtil import TimeSeries
from utils.MeasureUtil import RegressionMetrics
from utils.IOUtil import _save_results_to_csv__, _save_prediction_to_csv__, _save_loss_train_to_csv__
from utils.GraphUtil import _draw_predict_with_error__


class RootBase:
    """ This is the root of all networks """

    def __init__(self, root_base_paras=None):
        self.data_original = root_base_paras["data_original"]
        self.train_split = root_base_paras["train_split"]
        self.data_window = root_base_paras["data_window"]
        self.sliding = len(self.data_window)
        self.scaling = root_base_paras["scaling"]
        self.feature_size = root_base_paras["feature_size"]
        self.network_type = root_base_paras["network_type"]

        self.path_save_result = root_base_paras["path_save_result"]
        self.log_filename = root_base_paras["log_filename"]
        self.draw = root_base_paras["draw"]
        self.log = root_base_paras["log"]

        self.model, self.solution, self.loss_train, self.filename = None, None, [], None
        self.X_train, self.y_train, self.X_test, self.y_test = None, None, None, None
        self.time_total_train, self.time_epoch, self.time_predict, self.time_system = None, None, None, None
        self.time_series = None

    def _processing__(self):
        ## standardize the dataset using the mean and standard deviation of the training data.
        self.time_series = TimeSeries(data=self.data_original, train_split=self.train_split)
        data_new = self.time_series._scaling__(self.scaling, False)
        self.X_train, self.y_train = self.time_series._univariate_data__(data_new, self.data_window, 0, self.time_series.train_split, self.network_type)
        self.X_test, self.y_test = self.time_series._univariate_data__(data_new, self.data_window, self.time_series.train_split, None, self.network_type)

    def _save_results__(self, y_true=None, y_pred=None, y_true_scaled=None, y_pred_scaled=None, loss_train=None):
        # For this project only
        y_pred[y_pred < 0] = 0
        measure_unscaled = RegressionMetrics(y_true, y_pred, None, number_rounding=4)
        mm = measure_unscaled._fit__()
        item = {'model_name': self.filename, 'total_time_train': self.time_total_train, 'time_epoch': self.time_epoch,
                'time_predict': self.time_predict, 'time_system': self.time_system,
                'EVS': mm["evs"], 'ME': mm["me"], 'MAE': mm["mae"], "MSE": mm["mse"], "RMSE": mm["rmse"], "MSLE": mm["msle"], "MedAE": mm["medae"],
                "R2": mm["r2"], "MRE": mm["mre"], "MAPE": mm["mape"], "SMAPE": mm["smape"], "MAAPE": mm["maape"], "MASE": mm["mase"],
                "NSE": mm["nse"], "Willmott_Index": mm["wi"], "R": mm["r"], "Confidence": mm["c"]}

        _save_prediction_to_csv__(y_true, y_pred, y_true_scaled, y_pred_scaled, self.filename, self.path_save_result)
        _save_loss_train_to_csv__(loss_train, self.filename, self.path_save_result + "Error-")
        if self.draw:
            _draw_predict_with_error__([y_true, y_pred], [mm["r"], mm["c"]], self.filename, self.path_save_result)
        if self.log:
            print('Predict DONE - RMSE: %f, MAE: %f' % (mm["rmse"], mm["mae"]))
        _save_results_to_csv__(item, self.log_filename, self.path_save_result)

    def _forecasting__(self):
        pass

    def _training__(self):
        pass

    def _running__(self):
        pass
