from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection


def get_metrics(task: str) -> MetricCollection:
    """
    Creates and returns a collection of metrics for evaluating model performance.

    Args:
        task (str): The type of classification task. Supported values are:
            - 'binary': Binary classification task.

    Returns:
        MetricCollection: A collection of metrics, including accuracy, precision, recall, and F1 score.

    Raises:
        ValueError: If the specified task is unsupported.
    """
    if task == 'binary':
        metrics = {
            'accuracy': Accuracy(task=task),
            'precision': Precision(task=task),
            'recall': Recall(task=task),
            'f1': F1Score(task=task),
        }
    else:
        raise ValueError(f'Unsupported task: {task}')
    return MetricCollection(metrics)
