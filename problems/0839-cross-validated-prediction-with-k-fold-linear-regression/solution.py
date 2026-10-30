import numpy as np

def cross_val_predict(X, y, k):
    """
    Return out-of-fold predictions for every sample using k-fold
    cross-validation with a linear regression model (with intercept).
    """
    
    n = len(y)
    X = np.array(X)
    y = np.array(y)
    X =  X.astype(float)
    Y = y.astype(float)


    ones =np.ones((n,1))
    ones = ones.astype(float)
    X_aug = np.concatenate((X,ones),axis=1)
    Y = y.reshape(-1,1)
    iter_res = n % k
    batch_size = n // k 
    Y_pred_list = []

    for i in range(k):
        if iter_res != 0 and i < iter_res:
            Xi_test = X_aug[i*(batch_size+1): (i+1)*(batch_size+1), :]  
            Yi_test =  Y[i*(batch_size+1): (i+1)*(batch_size+1), :]
            
            Xi_train = np.delete(X_aug, slice(i*(batch_size+1), (i+1)*(batch_size+1)), axis = 0)
            Yi_train = np.delete(Y, slice(i*(batch_size+1), (i+1)*(batch_size+1)), axis = 0)
        elif iter_res !=0 and i >= iter_res:
            j =  i - iter_res
            Xi_test = X_aug[iter_res*(batch_size+1) + j * batch_size:iter_res*(batch_size+1) + (j+1) * batch_size,:]
            Yi_test = Y[iter_res*(batch_size+1) + j * batch_size:iter_res*(batch_size+1) + (j+1) * batch_size,:]

            Xi_train = np.delete(X_aug, slice(iter_res*(batch_size+1) + j * batch_size , iter_res*(batch_size+1) + (j+1) * batch_size), axis=0)
            Yi_train = np.delete(Y,slice(iter_res*(batch_size+1) + j * batch_size , iter_res*(batch_size+1) + (j+1) * batch_size), axis=0)
        else:
            Xi_test  = X_aug[i*(batch_size): (i+1)*batch_size,:]
            Yi_test = Y[i*(batch_size): (i+1)*batch_size,:]

            Xi_train = np.delete(X_aug, slice(i*(batch_size) , (i+1)*batch_size),axis=0)
            Yi_train = np.delete(Y, slice(i*(batch_size) , (i+1)*batch_size),axis=0)


        weights = np.linalg.solve(Xi_train.T @ Xi_train, Xi_train.T @ Yi_train)

        Yi_test_pred = Xi_test @ weights.reshape(-1,1)
        Yi_test_pred = Yi_test_pred.flatten()
        Y_pred_list.extend(Yi_test_pred.tolist())
            
    return Y_pred_list
