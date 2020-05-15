import warnings

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import History


def smooth_curve(points: np.array,
                 factor: float = 0.8):
    """Applies exponential moving average smoothing to the series

    Args:
        points (np.array): array of metric_values
        factor (float): smoothing factor
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points


def plot_1_metric(metric_values: np.array,
                  title: str,
                  metric_style: str = 'b',
                  x_label: str = 'Epochs',
                  y_label: str = None,
                  smooth_factor: float = 0.):
    """
     Plots a single metric
     :param metric_values: array of metric values
     :param title: plot title
     :param metric_style: color and style of the curve plot
     :param x_label: horizontal axis label
     :param y_label: vertical axis label
     :param smooth_factor: if > 0, applies smoothing to the curve
    """
    epochs = range(1, len(metric_values) + 1)
    plt.clf()
    if smooth_factor > 0.:
        plt.plot(epochs, smooth_curve(metric_values, smooth_factor), metric_style)
    else:
        plt.plot(epochs, metric_values, metric_style)

    if x_label:
        plt.xlabel(x_label)

    if y_label:
        plt.ylabel(y_label)

    plt.title(title)
    plt.grid()
    plt.show()


def plot_1_metric_history(history: History,
                          metric: str,
                          title: str,
                          metric_style: str = 'b',
                          x_label: str = 'Epochs',
                          y_label: str = None,
                          smooth_factor: float = 0.):
    """
    Plots a single metric based on a History object
    :param history: training history object, containing the training metrics series
    :param metric: metric key of the series to be plotted
    :param title: plot title
    :param metric_style: color and style of the curve plot
    :param x_label: horizontal axis label
    :param y_label: vertical axis label
    :param smooth_factor: if > 0, applies smoothing to the curve
    """
    if history is None:
        raise ValueError('No history object passed, trying to plot history metric')

    if metric not in history.history:
        raise RuntimeError('Metric {} not in history object, trying to plot history metric'.format(metric))

    plot_1_metric(metric_values=history.history[metric],
                  title=title,
                  metric_style=metric_style,
                  x_label=x_label,
                  y_label=y_label,
                  smooth_factor=smooth_factor)


def plot_2_metrics(metric1_values: np.array,
                   metric2_values: np.array,
                   title: str,
                   metric1_label: str,
                   metric2_label: str,
                   metric1_style: str = 'b',
                   metric2_style: str = 'r',
                   metric1_smooth_factor: float = 0.,
                   metric2_smooth_factor: float = 0.,
                   x_label: str = 'Epochs',
                   y_label: str = '',
                   clear: bool = True):
    """
    Plots two metrics (y1 and y2) sharing the same horizontal axis
    :param metric1_values: first array of metric values
    :param metric2_values: second array of metric values
    :param title: plot title
    :param metric1_label: first metric legend label
    :param metric2_label: second metric legend label
    :param metric1_style: plot style of the first metric
    :param metric2_style: plot style of the second metric
    :param metric1_smooth_factor: if > 0, apply exponential smoothing to the first metric
    :param metric2_smooth_factor: if > 0, apply exponential smoothing to the second metric
    :param x_label: horizontal axis label
    :param y_label: vertical axis label
    :param clear: clear last plot before current plot
    """
    if metric1_values is None:
        warnings.warn('Metric 1 ({}) array is empty: cannot be plot in graphic'.format(metric1_label))

    if metric2_values is None:
        warnings.warn('Metric 2 ({}) array is empty: cannot be plot in graphic'.format(metric2_label))

    len1 = 0
    if metric1_values is not None:
        len1 = len(metric1_values)

    len2 = 0
    if metric2_values is not None:
        len2 = len(metric2_values)

    if len1 > 0 and len2 > 0:
        if len1 != len2:
            raise ValueError('Metric 1 array is size {} and metric 2 array is size {}'.format(len1, len2))

    epochs = range(1, len1 + 1)
    if clear:
        plt.clf()

    if metric1_values is not None:
        if metric1_smooth_factor > 0.:
            plt.plot(epochs, smooth_curve(metric1_values, metric1_smooth_factor), metric1_style, label=metric1_label)
        else:
            plt.plot(epochs, metric1_values, metric1_style, label=metric1_label)

    if metric2_values is not None:
        if metric2_smooth_factor > 0.:
            plt.plot(epochs, smooth_curve(metric2_values, metric2_smooth_factor), metric2_style, label=metric2_label)
        else:
            plt.plot(epochs, metric2_values, metric2_style, label=metric2_label)

    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.show()


def plot_metrics_list(metric_values_list: list,
                      metric_labels_list: list,
                      metric_style_list: list,
                      title: str = None,
                      x_label: str = 'Epochs',
                      y_label: str = None,
                      metrics_smooth_factor: float = 0.):
    """Plots a list of metrics in a single plot

    Args:
        metric_values_list (list): list of metric series
        metric_labels_list (list): list of metric labels
        metric_style_list (list): list of metric plot styles
        title (str): plot title
        x_label (str): label of the X axis
        y_label (str): label of the Y axis
        metrics_smooth_factor (float): exponential smooth factor for the curves
    """
    if metric_values_list is None:
        raise RuntimeError('No metric values list to plot')

    epochs = range(1, 0)
    if np.shape(metric_values_list)[0] > 0:
        epochs = range(1, np.shape(metric_values_list)[1] + 1)

    plt.clf()
    for i in range(0, len(metric_values_list)):
        if metrics_smooth_factor > 0.:
            metric_values = metric_values_list[i]
        else:
            metric_values = smooth_curve(metric_values_list[i], metrics_smooth_factor)

        metric_style = ''
        if len(metric_style_list) > i:
            metric_style = metric_style_list[i]

        metric_label = ''
        if len(metric_labels_list) > i:
            metric_label = metric_labels_list[i]

        plt.plot(epochs, metric_values, metric_style, label=metric_label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    plt.show()


def plot_2_metrics_dict(history_metrics: dict,
                        metric1: str,
                        metric2: str,
                        title: str,
                        metric1_label: str,
                        metric2_label: str,
                        metric1_style: str = 'b',
                        metric2_style: str = 'bo',
                        metric1_smooth_factor: float = 0.,
                        metric2_smooth_factor: float = 0.,
                        x_label: str = 'Epochs',
                        y_label: str = '',
                        clear: bool = False):
    """ Plots two metrics
    """
    metric1_values = None
    metric2_values = None

    if metric1 in history_metrics:
        metric1_values = history_metrics[metric1]

    if metric2 in history_metrics:
        metric2_values = history_metrics[metric2]

    plot_2_metrics(metric1_values=metric1_values,
                   metric2_values=metric2_values,
                   title=title,
                   metric1_label=metric1_label,
                   metric2_label=metric2_label,
                   metric1_style=metric1_style,
                   metric2_style=metric2_style,
                   metric1_smooth_factor=metric1_smooth_factor,
                   metric2_smooth_factor=metric2_smooth_factor,
                   x_label=x_label,
                   y_label=y_label,
                   clear=clear)


def plot_2_metrics_history(history: History,
                           metric1: str,
                           metric2: str,
                           title: str,
                           metric1_label: str,
                           metric2_label: str,
                           metric1_style: str = 'b',
                           metric2_style: str = 'r',
                           metric1_smooth_factor: float = 0.,
                           metric2_smooth_factor: float = 0.,
                           x_label: str = 'Epochs',
                           y_label: str = '',
                           clear: bool = False):
    """ Plots two metrics
    """
    plot_2_metrics_dict(history_metrics=history.history,
                        metric1=metric1,
                        metric2=metric2,
                        title=title,
                        metric1_label=metric1_label,
                        metric2_label=metric2_label,
                        metric1_style=metric1_style,
                        metric2_style=metric2_style,
                        metric1_smooth_factor=metric1_smooth_factor,
                        metric2_smooth_factor=metric2_smooth_factor,
                        x_label=x_label,
                        y_label=y_label,
                        clear=clear)


def plot_metric_history(history: History,
                        training_metric: str,
                        validation_metric: str,
                        title: str,
                        x_label: str,
                        y_label: str,
                        training_smooth_factor: float,
                        validation_smooth_factor: float):
    """
    Plots the training and validation metrics in the history object
    :param history:
    :param training_metric:
    :param validation_metric:
    :param title:
    :param x_label:
    :param y_label:
    :param training_smooth_factor:
    :param validation_smooth_factor:
    """
    if history is None:
        raise RuntimeError('No training History object trying to plot training loss evolution')

    if training_metric not in history.history:
        raise RuntimeError(
            'Error trying to plot metric: History object does not have {} metric'.format(training_metric))

    if validation_metric not in history.history:
        plot_1_metric_history(history=history,
                              metric=training_metric,
                              title=title,
                              x_label=x_label,
                              y_label=y_label,
                              smooth_factor=training_smooth_factor)
    else:
        plot_2_metrics_history(history=history,
                               title=title,
                               metric1=training_metric,
                               metric2=validation_metric,
                               metric1_label='Training',
                               metric2_label='Validation',
                               metric1_smooth_factor=training_smooth_factor,
                               metric2_smooth_factor=validation_smooth_factor,
                               x_label=x_label,
                               y_label=y_label)


def plot_loss(history: History,
              title: str = 'Loss',
              training_smooth_factor: float = 0.,
              validation_smooth_factor: float = 0.):
    """
    Plot the evolution of the loss function during training
    :param history: training history object, containing a dictionary of metrics
    :param title: plot title
    :param training_smooth_factor: exponential smooth factor applied to the training series
    :param validation_smooth_factor: exponential smooth factor applied to the validation series
    """
    plot_metric_history(history=history,
                        training_metric='loss',
                        validation_metric='val_loss',
                        title=title,
                        x_label='Epoch',
                        y_label='Loss',
                        training_smooth_factor=training_smooth_factor,
                        validation_smooth_factor=validation_smooth_factor)


def plot_accuracy(history: History,
                  title: str = 'Accuracy',
                  training_smooth_factor: float = 0.,
                  validation_smooth_factor: float = 0.):
    """
    Plot the evolution of the accuracy function during training
    :param history: training history object, containing a dictionary of metrics
    :param title: plot title
    :param training_smooth_factor: exponential smooth factor applied to the training series
    :param validation_smooth_factor: exponential smooth factor applied to the validation series
    """
    if history is None:
        raise RuntimeError('No training History object trying to plot accuracy evolution')

    if 'accuracy' in history.history:
        training_metric = 'accuracy'

    elif 'acc' in history.history:
        training_metric = 'acc'

    else:
        raise RuntimeError('History object doesn\'t have accuracy values')

    plot_metric_history(history=history,
                        training_metric=training_metric,
                        validation_metric='val_' + training_metric,
                        title=title,
                        x_label='Epoch',
                        y_label='Accuracy',
                        training_smooth_factor=training_smooth_factor,
                        validation_smooth_factor=validation_smooth_factor)


def plot_mae(history: History,
             title: str = 'Mean Absolute Error (MAE)',
             training_smooth_factor: float = 0.,
             validation_smooth_factor: float = 0.):
    """
    Plot the evolution of the Mean Absolute Error (MAE) during training
    :param history: training history object, containing a dictionary of metrics
    :param title: plot title
    :param training_smooth_factor: exponential smooth factor applied to the training series
    :param validation_smooth_factor: exponential smooth factor applied to the validation series
    """
    plot_metric_history(history=history,
                        training_metric='mae',
                        validation_metric='val_mae',
                        title=title,
                        x_label='Epoch',
                        y_label='Accuracy',
                        training_smooth_factor=training_smooth_factor,
                        validation_smooth_factor=validation_smooth_factor)


def plot_loss_list(history_metrics_list: list,
                   labels_list: list,
                   title: str = 'Loss Evolution',
                   plot_training: bool = True,
                   plot_validation: bool = True,
                   smooth_factor: float = 0.):
    """
    Plot the compared evolution of the loss functions
    :param history_metrics_list: list of History objects containing the metrics series
    :param labels_list: list of labels of the individual series
    :param title: plot title
    :param plot_training: plot training series
    :param plot_validation: plot validation series
    :param smooth_factor: smooth factor to be applied to all curves
    """
    clear = True
    metric_values_list = []

    for history in history_metrics_list:

        if plot_training and 'loss' in history.history:
            metric_values_list.append(history.history['loss'])

        if plot_validation and 'val_loss' in history.history:
            metric_values_list.append(history.history['val_loss'])

    plot_metrics_list(metric_values_list=metric_values_list,
                      metric_labels_list=labels_list,
                      metric_style_list=[],
                      title=title,
                      x_label='Epochs',
                      y_label='Loss',
                      metrics_smooth_factor=smooth_factor)


def plot_accuracy_list(history_metrics_list: list,
                       labels_list: list,
                       title: str = 'Accuracy Evolution',
                       plot_training: bool = True,
                       plot_validation: bool = True,
                       smooth_factor: float = 0.):
    """ Plot the compared evolution of the accuracies
    """
    clear = True
    metric_values_list = []

    for history in history_metrics_list:

        if plot_training and 'accuracy' in history.history:
            metric_values_list.append(history.history['accuracy'])

        if plot_validation and 'val_accuracy' in history.history:
            metric_values_list.append(history.history['val_accuracy'])

    plot_metrics_list(metric_values_list=metric_values_list,
                      metric_labels_list=labels_list,
                      metric_style_list=[],
                      title=title,
                      x_label='Epochs',
                      y_label='Accuracy',
                      metrics_smooth_factor=smooth_factor)


def merge_history_metrics(history_list: list):
    """
    Merges and averages all metrics from a list of training histories, typically from K-Fold cross validation
    :param history_list: list of History objects that resulted from many training processes
    :return: merged list of metrics
    """
    if list is None:
        raise ValueError('History list is None')

    if len(history_list) == 0:
        return None

    merged_metrics = {}
    merged_history = History()
    first_history: History = history_list[0]
    merged_history.model = first_history.model
    merged_history.epoch = first_history.epoch
    merged_history.params = first_history.params

    for history in history_list:
        if history.model != merged_history.model:
            raise ValueError('Cannot merge history from different models')

        if history.epoch != merged_history.epoch:
            raise ValueError('Cannot merge history with different epoch vectors')

        if history.params != merged_history.params:
            raise ValueError('Cannot merge history with different parameters')

        history_metrics = history.history

        # metrics loop
        for metric in history_metrics.keys():
            if metric in merged_metrics:
                merged_metrics[metric].append(history_metrics[metric])
            else:
                merged_metrics[metric] = [history_metrics[metric]]

    # metrics averaging
    average_history_metrics = {}
    for metric in merged_metrics.keys():
        average_history_metrics[metric] = np.mean(merged_metrics[metric], axis=0)

    merged_history.history = average_history_metrics
    return merged_history


def smooth_metric_values(metric_values: np.array, factor: float = 0.9):
    smoothed_points = np.empty(shape=np.shape(metric_values))
    for point in metric_values:
        if smoothed_points.any():
            previous = smoothed_points[-1]
            np.append(smoothed_points, factor * previous + (1 - factor) * point)
        else:
            np.append(smoothed_points, point)
    return smoothed_points


def smooth_metrics_dict(metrics_dict: dict, factor: float = 0.9):
    for metric in metrics_dict.keys():
        metric_values = metrics_dict[metric]
        smooth_metric_values(metric_values=metric_values, factor=factor)
