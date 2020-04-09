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
    """ Plots a single metric

    Args:
        metric_values (np.array): array of metric values
        title (str): plot title
        metric_style (str): color and style of the curve plot
        x_label (str): horizontal axis label
        y_label (str): vertical axis label
        smooth_factor (float): if > 0, applies smoothing to the curve
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
    plt.legend()
    plt.show(block=False)


def plot_1_metric_history(history: History,
                          metric: str,
                          title: str,
                          metric_style: str = 'b',
                          x_label: str = 'Epochs',
                          y_label: str = None,
                          smooth_factor: float = 0.):
    """ Plots a single metric based on a History object

    Args:
        history (History): training history object, containing the training metrics series
        metric (str): metric key of the series to be plotted
        title (str): plot title
        metric_style (str): color and style of the curve plot
        x_label (str): horizontal axis label
        y_label (str): vertical axis label
        smooth_factor (float): if > 0, applies smoothing to the curve
    """
    metric_values = history.history[metric]
    plot_1_metric(metric_values=metric_values,
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
    """ Plots two metrics (y1 and y2) sharing the same horizontal axis

    Args:
        metric1_values (np.array): first array of metric values
        metric2_values (np.array): second array of metric values
        title (str): plot title
        metric1_label (str): first metric legend label
        metric2_label (str): second metric legend label
        metric1_style (str): plot style of the first metric
        metric2_style (str): plot style of the second metric
        metric1_smooth_factor (float): if > 0, apply exponential smoothing to the first metric
        metric2_smooth_factor (float): if > 0, apply exponential smoothing to the second metric
        x_label (str): horizontal axis label
        y_label (str): vertical axis label
        clear (bool): clear last plot before current plot
    """
    if metric1_values is None:
        raise ValueError('Metric 1 array is empty')
    if metric2_values is None:
        raise ValueError('Metric 2 array is empty')
    len1 = len(metric1_values)
    len2 = len(metric2_values)
    if len1 != len2:
        raise ValueError('Metric 1 array is size {} and metric 2 array is size {}'.format(len1, len2))

    epochs = range(1, len1 + 1)
    if clear:
        plt.clf()

    if metric1_smooth_factor > 0.:
        plt.plot(epochs, smooth_curve(metric1_values, metric1_smooth_factor), metric1_style, label=metric1_label)
    else:
        plt.plot(epochs, metric1_values, metric1_style, label=metric1_label)

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
    plt.show(block=False)


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


def plot_loss(history: History,
              title: str = 'Loss',
              training_smooth_factor: float = 0.,
              validation_smooth_factor: float = 0.):
    """ Plot the evolution of the loss function

    Args:
        history (History): training history object, containing a dictionary of metrics
        title (str): plot title
        training_smooth_factor (float): exponential smooth factor applied to the training series
        validation_smooth_factor (float): exponential smooth factor applied to the validation series
    """
    plot_2_metrics_history(history=history,
                           title=title,
                           metric1='loss',
                           metric2='val_loss',
                           metric1_label='Training loss',
                           metric2_label='Validation loss',
                           metric1_smooth_factor=training_smooth_factor,
                           metric2_smooth_factor=validation_smooth_factor,
                           x_label='Epochs',
                           y_label='Loss')


def plot_loss_dict(history_metrics: dict,
                   title: str = 'Training and Validation Losses',
                   clear: bool = False):
    """ Plot the evolution of the loss function
    """
    loss_training = None
    loss_validation = None

    if 'loss' in history_metrics:
        loss_training = history_metrics['loss']

    if 'val_loss' in history_metrics:
        loss_validation = history_metrics['val_loss']

    plot_2_metrics(metric1_values=loss_training,
                   metric2_values=loss_validation,
                   title=title,
                   metric1_label='Training loss',
                   metric2_label='Validation loss',
                   y_label='Loss',
                   clear=clear)


def plot_accuracy(history: History,
                  title: str = 'Accuracy',
                  training_smooth_factor: float = 0.,
                  validation_smooth_factor: float = 0.):
    """ Plot the evolution of the loss function
    """
    plot_2_metrics_history(history=history,
                           title=title,
                           metric1='accuracy',
                           metric2='val_accuracy',
                           metric1_label='Training accuracy',
                           metric2_label='Validation accuracy',
                           metric1_smooth_factor=training_smooth_factor,
                           metric2_smooth_factor=validation_smooth_factor,
                           y_label='Accuracy')


def plot_accuracy_dict(history_metrics: dict,
                       title: str = 'Training and Validation Accuracies',
                       clear: bool = False):
    """ Plot the evolution of the accuracy
    """
    plot_2_metrics_dict(history_metrics=history_metrics,
                        metric1='accuracy',
                        metric2='val_accuracy',
                        title=title,
                        metric1_label='Training accuracy',
                        metric2_label='Validation accuracy',
                        y_label='Accuracy',
                        clear=clear)


def plot_loss_list(history_metrics_list: list,
                   labels_list: list,
                   title: str = 'Loss Evolution',
                   plot_training: bool = True,
                   plot_validation: bool = True,
                   smooth_factor: float = 0.):
    """Plot the compared evolution of the loss functions

    Args:
        history_metrics_list (list): list of History objects containing the metrics series
        labels_list (list): list of labels of the individual series
        title (list): plot title
        plot_training (bool): plot training series
        plot_validation (bool): plot validation series
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


def plot_mae(mae_training: np.array,
             mae_validation: np.array,
             title: str = 'Mean Absolute Error (MAE)'):
    """ Plot the evolution of the Mean Absolute Error (MAE)
    """
    plot_2_metrics(metric1_values=mae_training,
                   metric2_values=mae_validation,
                   title=title,
                   metric1_label='Training MAE',
                   metric2_label='Validation MAE',
                   metric1_style='b',
                   metric2_style='r',
                   y_label='Mean Absolute Error (MAE)')


def plot_mae_dict(history_metrics: dict,
                  title: str = 'Mean Absolute Error (MAE)'):
    """ Plot the evolution of the Mean Absolute Error (MAE)
    """
    mae_training = None
    mae_validation = None

    if 'mae' in history_metrics:
        mae_training = history_metrics['mae']

    if 'val_mae' in history_metrics:
        mae_validation = history_metrics['val_mae']

    plot_2_metrics(metric1_values=mae_training,
                   metric2_values=mae_validation,
                   title=title,
                   metric1_label='Training MAE',
                   metric2_label='Validation MAE',
                   x_label='Epochs',
                   y_label='Mean Absolute Error (MAE)')


def merge_history_metrics(history_list: list):
    """ Merges and averages all metrics from a list of train histories
    """
    merged_metrics = {}

    # history list loop
    for history in history_list:
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

    return average_history_metrics


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
