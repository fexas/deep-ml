import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """

    X_train = np.array(X_train, dtype = float)
    y_train = np.array(y_train, dtype = float)
    X_val = np.array(X_val, dtype = float)
    y_val = np.array(y_val, dtype = float)
    N = X_train.shape[0]

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1,1)
    if X_val.ndim == 1:
        X_val = X_val.reshape(-1,1)

    def poly_feature(Z, degree):
        n = Z.shape[0]
        Z_poly = np.ones((n,1), dtype=float)
    
        for i in range(degree):
            Z_ith_order = Z**(i+1)
            Z_poly = np.concatenate((Z_poly, Z_ith_order), axis=1)

        return Z_poly

    train_MSE_list = []
    val_MSE_list = []

    for train_size in train_sizes:
        X_train_poly = poly_feature(X_train[:train_size], degree)
        Y_train = y_train[:train_size].reshape(-1,1)

        weights = np.linalg.pinv(X_train_poly.T @ X_train_poly) @ X_train_poly.T @ Y_train

        Y_train_pred = X_train_poly @ weights
        train_MSE = np.mean( (Y_train_pred.reshape(-1,1)-Y_train)**2 )
        train_MSE_list.append(train_MSE)


        X_val_poly = poly_feature(X_val, degree)
        Y_val_pred = X_val_poly @ weights
        Y_val = y_val.reshape(-1,1)
        val_MSE = np.mean((Y_val_pred.reshape(-1,1) - Y_val)**2)
        val_MSE_list.append(val_MSE)

    train_diag = 0.0
    val_diag = 0.0

    if train_MSE_list[-1] > bias_threshold:
        train_diag = -1.0
    
    if val_MSE_list[-1] - train_MSE_list[-1] > variance_threshold:
        val_diag = -1.0
    
    if train_diag == -1.0:
        diag = 'high_bias'
    elif val_diag ==  -1.0:
        diag = 'high_variance'
    elif train_diag == 0 and val_diag == 0:
        diag = 'good_fit'
    else:
        raise ValueError('ValueError')

    
    return {
        'train_errors': [float(mse) for mse in train_MSE_list],
        'val_errors': [float(mse) for mse in val_MSE_list],
        'diagnosis': diag
    }

    