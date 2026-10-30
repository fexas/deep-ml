
import numpy as np

def r_squared(y_true, y_pred):
	
	res = y_pred - y_true
	ssr = np.var(res)
	sst = np.var(y_true)
	r2 = 1 - ssr / sst
	
	return r2
