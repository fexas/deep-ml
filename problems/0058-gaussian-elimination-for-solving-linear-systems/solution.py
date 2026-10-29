import numpy as np

def gaussian_elimination(A, b):
	"""
	Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
	:param A: Coefficient matrix
	:param b: Right-hand side vector
	:return: Solution vector x
	"""
	rows_A, cols_A = A.shape
	b = b.reshape(-1,1)
	Augument = np.concatenate([A,b], axis=1)
	Augument.astype(float)

	iter_num = min(rows_A, cols_A)

	for i in range(iter_num):
		
		index = i 
		
		if Augument[i,i] == 0:
			
			while Augument[index,i] == 0 and index < rows_A:
				index += 1
		
		if index == rows_A:
			break
		
		if index != i and index < rows_A:
			Augument[[i,index]] = Augument[[index, i]]

		Augument[i,:] = Augument[i,:] / Augument[i,i]

		for j in range(rows_A):
			if j != i:
				Augument[j,:] = Augument[j,:] - Augument[i,:] * Augument[j,i]

	 
	sol = Augument[:,-1]

	return sol
