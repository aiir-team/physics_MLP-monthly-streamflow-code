#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 22:18, 17/05/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
#-------------------------------------------------------------------------------------------------------%

from numpy import ndarray, array, max, round, sqrt, abs, where, mean, dot, divide, arctan, sum, any
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_log_error, median_absolute_error, r2_score


class RegressionMetrics:
    """
        https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
    """
    def __init__(self, y_true, y_pred, multi_output="raw_values", number_rounding=3):
        """
        :param y_true:
        :param y_pred:
        :param multi_output:    string in [‘raw_values’, ‘uniform_average’, ‘variance_weighted’] or array-like of shape (n_outputs)
        :param number_rounding:
        """
        self.onedim = False
        if type(y_true) is ndarray and type(y_pred) is ndarray:
            self.y_true = y_true
            self.y_pred = y_pred
            if y_true.ndim == 1 and y_pred.ndim == 1:
                self.onedim = True
                self.y_true_clear = self.y_true[self.y_true != 0]
                self.y_pred_clear = self.y_pred[self.y_true != 0]
            else:
                self.y_true_clear = self.y_true[~any(self.y_true == 0, axis=1)]
                self.y_pred_clear = self.y_pred[~any(self.y_true == 0, axis=1)]
        else:
            print("=====Failed on y_true/y_pred ndarray=======")
            exit(0)
        self.multi_output = multi_output
        self.number_rounding = number_rounding

    def evs_function(self):         # explained_variance_score
        """
            EV < 1. Best possible score is 1.0, lower values are worse.
        """
        if self.onedim:
            return round(explained_variance_score(self.y_true, self.y_pred), self.number_rounding)
        else:
            return round(explained_variance_score(self.y_true, self.y_pred, multioutput=self.multi_output), self.number_rounding)

    def me_function(self):            # max_error
        """
            Smaller is better
        """
        if self.onedim:
            return round(max_error(self.y_true, self.y_pred), self.number_rounding)
        else:
            if self.multi_output == "raw_values":
                absolute = abs(self.y_true - self.y_pred)
                residual = max(absolute, axis=0)
            else:
                absolute = abs(self.y_true - self.y_pred)
                residual = max(absolute)
            return residual

    def mae_function(self):     # mean_absolute_error functions
        """
            Smaller is better
        """
        if self.onedim:
            return round(mean_absolute_error(self.y_true, self.y_pred), self.number_rounding)
        else:
            return round(mean_absolute_error(self.y_true, self.y_pred, multioutput=self.multi_output), self.number_rounding)

    def mse_function(self):  # mean_squared_error function
        """
            Smaller is better
        """
        if self.onedim:
            return round(mean_squared_error(self.y_true, self.y_pred), self.number_rounding)
        else:
            return round(mean_squared_error(self.y_true, self.y_pred, multioutput=self.multi_output), self.number_rounding)

    def rmse_function(self):  # root_mean_squared_error
        """
            Smaller is better
        """
        if self.onedim:
            return round(sqrt(mean_squared_error(self.y_true, self.y_pred)), self.number_rounding)
        else:
            return round(sqrt(mean_squared_error(self.y_true, self.y_pred, multioutput=self.multi_output)), self.number_rounding)

    def msle_function(self):    # mean_squared_log_error function
        """
            Smaller is better
        """
        y_true = where(self.y_true < 0, 0, self.y_true)
        y_pred = where(self.y_pred < 0, 0, self.y_pred)
        if self.onedim:
            return round(mean_squared_log_error(y_true, y_pred), self.number_rounding)
        else:
            return round(mean_squared_log_error(y_true, y_pred, multioutput=self.multi_output), self.number_rounding)

    def medae_function(self):     # median_absolute_error
        """
            Smaller is better
        """
        if self.onedim:
            return round(median_absolute_error(self.y_true, self.y_pred), self.number_rounding)
        else:
            return round(median_absolute_error(self.y_true, self.y_pred, multioutput=self.multi_output), self.number_rounding)

    def r2_function(self):  # r2_score
        """
            R2 < 1. Larger is better
            R^2 (coefficient of determination) regression score function. Best possible score is 1.0 and it can be negative
        """
        if self.onedim:
            return round(r2_score(self.y_true, self.y_pred), self.number_rounding)
        else:
            return round(r2_score(self.y_true, self.y_pred, multioutput=self.multi_output), self.number_rounding)

    def mre_function(self):     # Mean relative error
        """
            Good if mre < 40%. Smaller is better
        """
        if self.onedim:
            return round(mean(divide(abs(self.y_true_clear - self.y_pred_clear), self.y_true_clear)), self.number_rounding)
        else:
            temp = mean(divide(abs(self.y_true_clear - self.y_pred_clear), self.y_true_clear), axis=0)
            if self.multi_output is None:
                return round(mean(temp), self.number_rounding)
            elif isinstance(self.multi_output, (tuple, list, set)):
                weights = array(self.multi_output)
                if self.y_true_clear.ndim != len(weights):
                    print("==========Failed because different dimension==============")
                    exit(0)
                return round(dot(temp, weights), self.number_rounding)
            elif self.multi_output == "raw_values":
                return round(temp, self.number_rounding)
            else:
                print("=========Not supported===================")
                exit(0)

    def mape_function(self):  # mean_absolute_percentage_error
        """
            Good if mape < 30%
        """
        if self.onedim:
            return round(mean(divide(abs(self.y_true_clear - self.y_pred_clear), abs(self.y_true_clear))) * 100, self.number_rounding)
        else:
            temp = mean(divide(abs(self.y_true_clear - self.y_pred_clear), abs(self.y_true_clear)), axis=0) * 100
            if self.multi_output is None:
                return round(mean(temp), self.number_rounding)
            elif isinstance(self.multi_output, (tuple, list, set)):
                weights = array(self.multi_output)
                if self.y_true_clear.ndim != len(weights):
                    print("==========Failed because different dimension==============")
                    exit(0)
                return round(dot(temp, weights), self.number_rounding)
            elif self.multi_output == "raw_values":
                return round(temp, self.number_rounding)
            else:
                print("=========Not supported===================")
                exit(0)

    def smape_function(self):   # symmetric_mean_absolute_percentage_error
        if self.onedim:
            return round(mean(2 * abs(self.y_pred_clear - self.y_true_clear) / (abs(self.y_true_clear) + abs(self.y_pred_clear))) * 100, self.number_rounding)
        else:
            temp = mean(2 * abs(self.y_pred_clear - self.y_true_clear) / (abs(self.y_true_clear) + abs(self.y_pred_clear)), axis=0) * 100
            if self.multi_output is None:
                return round(mean(temp), self.number_rounding)
            elif isinstance(self.multi_output, (tuple, list, set)):
                weights = array(self.multi_output)
                if self.y_true_clear.ndim != len(weights):
                    print("==========Failed because different dimension==============")
                    exit(0)
                return round(dot(temp, weights), self.number_rounding)
            elif self.multi_output == "raw_values":
                return round(temp, self.number_rounding)
            else:
                print("=========Not supported===================")
                exit(0)

    def maape_function(self):   # Mean Arctangent Absolute Percentage Error (output: radian values)
        """
            https://support.numxl.com/hc/en-us/articles/115001223463-MAAPE-Mean-Arctangent-Absolute-Percentage-Error
        """
        if self.onedim:
            return round(mean(arctan(divide(abs(self.y_true_clear - self.y_pred_clear), self.y_true_clear))), self.number_rounding)
        else:
            temp = mean(arctan(divide(abs(self.y_true_clear - self.y_pred_clear), self.y_true_clear)), axis=0)
            if self.multi_output is None:
                return round(mean(temp), self.number_rounding)
            elif isinstance(self.multi_output, (tuple, list, set)):
                weights = array(self.multi_output)
                if self.y_true_clear.ndim != len(weights):
                    print("==========Failed because different dimension==============")
                    exit(0)
                return round(dot(temp, weights), self.number_rounding)
            elif self.multi_output == "raw_values":
                return round(temp, self.number_rounding)
            else:
                print("=========Not supported===================")
                exit(0)

    def mase_function(self, m=1):       # Mean absolute scaled error
        """
            https://en.wikipedia.org/wiki/Mean_absolute_scaled_error
            m = 1 for non-seasonal data, m > 1 for seasonal data
        """
        m = 1
        if self.onedim:
            return round(mean(abs(self.y_true_clear - self.y_pred_clear)) / mean(abs(self.y_true_clear[m:] - self.y_true_clear[:-m])), self.number_rounding)
        else:
            temp = mean(abs(self.y_true_clear - self.y_pred_clear), axis=0) / mean(abs(self.y_true_clear[m:] - self.y_true_clear[:-m]), axis=0)
            if self.multi_output is None:
                return round(mean(temp), self.number_rounding)
            elif isinstance(self.multi_output, (tuple, list, set)):
                weights = array(self.multi_output)
                if self.y_true_clear.ndim != len(weights):
                    print("==========Failed because different dimension==============")
                    exit(0)
                return round(dot(temp, weights), self.number_rounding)
            elif self.multi_output == "raw_values":
                return round(temp, self.number_rounding)
            else:
                print("=========Not supported===================")
                exit(0)

    def nse_function(self):  # Nash-Sutcliffe efficiency coefficient
        """
            https://agrimetsoft.com/calculators/Nash%20Sutcliffe%20model%20Efficiency%20coefficient
            -unlimited < NSE < 1.   Larger is better
        """
        if self.onedim:
            return round(1 - sum((self.y_true - self.y_pred) ** 2) / sum((self.y_true - mean(self.y_true)) ** 2), self.number_rounding)
        else:
            temp = 1 - sum((self.y_true - self.y_pred) ** 2, axis=0) / sum((self.y_true - mean(self.y_true, axis=0)) ** 2, axis=0)
            if self.multi_output is None:
                return round(mean(temp), self.number_rounding)
            elif isinstance(self.multi_output, (tuple, list, set)):
                weights = array(self.multi_output)
                if self.y_true.ndim != len(weights):
                    print("==========Failed because different dimension==============")
                    exit(0)
                return round(dot(temp, weights), self.number_rounding)
            elif self.multi_output == "raw_values":
                return round(temp, self.number_rounding)
            else:
                print("=========Not supported===================")
                exit(0)

    def wi_function(self):  # Willmott Index (WI) (Willmott, 1984):
        """
        https://www.researchgate.net/publication/319699360_Reference_evapotranspiration_for_Londrina_Parana_Brazil_performance_of_different_estimation_methods
            Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods
            0 < WI < 1. Larger is better
        """
        if self.onedim:
            m1 = mean(self.y_true)
            return round(1 - sum((self.y_pred - self.y_true) ** 2) / sum((abs(self.y_pred - m1) + abs(self.y_true - m1)) ** 2), self.number_rounding)
        else:
            m1 = mean(self.y_true, axis=0)
            temp = 1 - sum((self.y_pred - self.y_true) ** 2, axis=0) / sum((abs(self.y_pred - m1) + abs(self.y_true - m1)) ** 2, axis=0)
            if self.multi_output is None:
                return round(mean(temp), self.number_rounding)
            elif isinstance(self.multi_output, (tuple, list, set)):
                weights = array(self.multi_output)
                if self.y_true.ndim != len(weights):
                    print("==========Failed because different dimension==============")
                    exit(0)
                return round(dot(temp, weights), self.number_rounding)
            elif self.multi_output == "raw_values":
                return round(temp, self.number_rounding)
            else:
                print("=========Not supported===================")
                exit(0)

    def r_function(self):  # Pearson’s correlation index (Willmott, 1984):
        """
            Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods
            https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
            -1 < R < 1. Larger is better
        """
        if self.onedim:
            m1, m2 = mean(self.y_true), mean(self.y_pred)
            temp = sum((abs(self.y_true - m1) * abs(self.y_pred - m2))) / (sqrt(sum((self.y_true - m1)**2)) * sqrt(sum((self.y_pred - m2)**2)))
            return round(temp, self.number_rounding)
        else:
            m1, m2 = mean(self.y_true, axis=0), mean(self.y_pred, axis=0)
            t1 = sqrt(sum((self.y_true - m1) ** 2, axis=0))
            t2 = sqrt(sum((self.y_pred - m2) ** 2, axis=0))
            t3 = sum((abs(self.y_true - m1) * abs(self.y_pred - m2)), axis=0)
            temp = t3 / (t1 * t2)
            if self.multi_output is None:
                return round(mean(temp), self.number_rounding)
            elif isinstance(self.multi_output, (tuple, list, set)):
                weights = array(self.multi_output)
                if self.y_true.ndim != len(weights):
                    print("==========Failed because different dimension==============")
                    exit(0)
                return round(dot(temp, weights), self.number_rounding)
            elif self.multi_output == "raw_values":
                return round(temp, self.number_rounding)
            else:
                print("=========Not supported===================")
                exit(0)

    def confidence_function(self):  # confidence or performance index (c)
        """
        https://www.researchgate.net/publication/319699360_Reference_evapotranspiration_for_Londrina_Parana_Brazil_performance_of_different_estimation_methods
        Reference evapotranspiration for Londrina, Paraná, Brazil: performance of different estimation methods
            > 0.85          Excellent
            0.76-0.85       Very good
            0.66-0.75       Good
            0.61-0.65       Satisfactory
            0.51-0.60       Poor
            0.41-0.50       Bad
            ≤ 0.40          Very bad
        """
        r = self.r_function()
        d = self.wi_function()
        if self.multi_output is None:
            return round(mean(r*d), self.number_rounding)
        elif isinstance(self.multi_output, (tuple, list, set)):
            weights = array(self.multi_output)
            if self.y_true.ndim != len(weights):
                print("==========Failed because different dimension==============")
                exit(0)
            return round(dot(r*d, weights), self.number_rounding)
        elif self.multi_output == "raw_values":
            return round(r*d, self.number_rounding)
        else:
            print("=========Not supported===================")
            exit(0)


    def _fit__(self):
        evs = self.evs_function()
        me = self.me_function()
        mae = self.mae_function()
        mse = self.mse_function()
        rmse = self.rmse_function()
        msle = self.msle_function()
        medae = self.medae_function()
        r2 = self.r2_function()
        mre = self.mre_function()
        mape = self.mape_function()
        smape = self.smape_function()
        maape = self.maape_function()
        mase = self.mase_function()
        nse = self.nse_function()
        wi = self.wi_function()
        r = self.r_function()
        c = self.confidence_function()
        return {"evs":evs, "me":me, "mae":mae, "mse":mse, "rmse":rmse, "msle":msle, "medae":medae,
                "r2":r2, "mre":mre, "mape":mape, "smape":smape, "maape":maape, "mase":mase, "nse": nse, "wi": wi, "r": r, "c": c}
