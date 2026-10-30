
import numpy as np

def rmse(y_true, y_pred):
	
	rmse_res = np.sqrt(np.mean((y_pred - y_true)**2))
	
	return round(rmse_res,3)
