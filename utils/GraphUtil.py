#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 21:43, 09/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%
from numpy import array, arange
from matplotlib import pyplot as plt
import seaborn as sns
from pandas import DataFrame
import platform

if platform.system() == "Linux":  # Linux: "Linux", Mac: "Darwin", Windows: "Windows"
    import matplotlib
    matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.

# def _draw_predict_with_error__(data=None, error=None, filename=None, pathsave=None):
#     plt.plot(data[0])       # True
#     plt.plot(data[1])       # Prediction
#     plt.ylabel('Value')
#     plt.xlabel('Time Step')
#     plt.legend(['True Value (RMSE= ' + str(error[0]) + ')', 'Predicted Value (MAE= ' + str(error[1]) + ')'], loc='upper right')
#     plt.savefig(pathsave + filename + ".png", bbox_inches='tight')
#     plt.close()
#     return None


def _draw_predict_with_error__(data=None, error=None, filename=None, pathsave=None):
    # Import Data
    df = DataFrame({'y_true': data[0][:, 0], 'y_pred': data[1][:, 0]})
    list_data = [df.loc[:, "y_true"], df.loc[:, "y_pred"]]
    list_data[0].rename("Observed streamflow", inplace=True)
    list_data[1].rename("Predicted streamflow", inplace=True)

    # Draw Plot
    plt.rcParams['figure.figsize'] = 10, 3.5

    # sns.set(color_codes=True)
    my_fig = plt.figure(constrained_layout=True)
    gs = my_fig.add_gridspec(nrows=1, ncols=5)

    ax1 = my_fig.add_subplot(gs[0, :3])
    sns.lineplot(data=list_data, ax=ax1)
    ax1.set(xlabel='Months', ylabel=r'Streamflow ($m^3/s$)', title='Performance Prediction: C=' + str(round(error[1], 2)))
    # ax1.set(xlabel='Months', ylabel=r'Streamflow ($m^3/sec$)')

    ax2 = my_fig.add_subplot(gs[0:, 3:])
    sns.regplot(x="y_true", y="y_pred", data=df, ax=ax2)
    ax2.set(xlabel=r'Observed ($m^3/s$)', ylabel=r'Predicted ($m^3/s$)', title='Linear Relationship: R=' + str(round(error[0], 3)))
    # ax2.set(xlabel=r'Observed ($m^3/s$)', ylabel=r'Predicted ($m^3/s$)')
    ax2.legend(['Fit'])

    plt.savefig(pathsave + filename + ".png", bbox_inches='tight')
    # plt.show()
    plt.close()
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


def multi_step_plot(history, true_future, prediction, num_steps):
    plt.figure(figsize=(12, 6))
    num_in = __create_time_steps__(len(history))
    num_out = len(true_future)

    plt.plot(num_in, array(history[:, 1]), label='History')
    plt.plot(arange(num_out) / num_steps, array(true_future), 'bo', label='True Future')
    if prediction.any():
        plt.plot(arange(num_out) / num_steps, array(prediction), 'ro', label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()
