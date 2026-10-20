import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X = np.asarray(X, dtype = float)
	y = np.asarray(y, dtype = float)

	XtX = X.T @ X
	Xty = X.T @ y

	theta = np.linalg.solve(XtX, Xty)

	return theta