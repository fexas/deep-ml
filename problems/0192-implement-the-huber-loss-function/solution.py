def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	if not isinstance(y_true, list):
		y_true = [y_true]
	if not isinstance(y_pred, list):
		y_pred = [y_pred]

	n = len(y_true)

	huber_loss = []
	for i in range(n):
		yi_diff = abs(y_pred[i]-y_true[i])
		if yi_diff <= delta:
			huber_loss.append(yi_diff**2/2)
		else:
			huber_loss.append(delta*(yi_diff-delta/2))

	huber = 0
	for i in range(n):
		huber += huber_loss[i]
	huber = huber / n
	
	return huber