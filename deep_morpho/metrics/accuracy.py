def accuracy(y_true, y_pred):
    """ Computes accuracy for multi-class prediction.

    Args:
        y_true (torch.Tensor): size (batch_size, n_classes) true values. One-hot encoding
        y_pred (torch.Tensor): size (batch_size, n_classes) output of preds, float values [0, 1]. One-hot encoding.

    Returns:
        float: accuracy
    """

    if y_true.ndim > 1:
        y_true = y_true.argmax(1)
    y_pred = y_pred.argmax(1)

    return (y_true == y_pred).float().mean().item()
