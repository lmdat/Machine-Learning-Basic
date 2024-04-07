import numpy as np
import matplotlib.pyplot as plt

class ViiLinearRegression:
    """
    Parameters:
        n_iters (int) : Number of iterations
        eta (float) : Learning rate
        w (numpy.ndarray(n,)) : Model parameter
        b (scalar) : Model parameter
    """
    
    def __init__(self, n_iters=10000, eta=0.01):
        self.n_iters = int(n_iters)
        self.eta = eta
        self.w = None
        self.b = None

    def fit(self, X, y):
        _w, _b = self._compute_gradient_descent(X, y)
        self.w = _w
        self.b = _b

    def predict(self, X):
        preds = X.dot(self.w) + self.b # X(mxn).w(nx1) => vector(mx1)
        return preds

    def mse(self, preds, y_test):
        """
        Compute the Mean Squared Error
        """
        return np.mean((preds - y_test) ** 2)
        
    #===============================================
    def _compute_cost(self, X, y_vector, w_vector, b):
        """
        Parameters:
            X (numpy.ndarray(m,n)): Feature data with m samples (observations) and n features (columns)
            y_vector (numpy.ndarray(m,)): Target values
            w_vector (numpy.ndarray(n,)):  Model parameter
            b (scalar): Model parameter
    
        Return:
            j_wb (float): The cost of using parameters w_vector, b for linear regression
                   to fit the data points in X and y_vector
        """
        j_wb = np.mean(((X.dot(w_vector) + b) - y_vector) ** 2) / 2
        return j_wb
        
    def _compute_partial_derivative(self, X, y_vector, w_vector, b):
        """
        Parameters:
            X (numpy.ndarray(m,n)): Feature data with m samples (observations) and n features (columns)
            y_vector (numpy.ndarray(m,)): Target values
            w_vector (numpy.ndarray(n,)):  Model parameter
            b (scalar): Model parameter
    
        Return:
            dj_dw (ndarray (n,)): The derivative value with respect to w
            dj_db (scalar): The derivative value with respect to b  
        """
        m_samples = X.shape[0]        

        dj_b = (X.dot(w_vector) + b) - y_vector # dj_b có shape(m,)
        dj_w = np.dot(X.T, dj_b) # dj_w has có shape(n,)

        dj_dw = dj_w / m_samples # dj_dw có shape(n,): mỗi phần tử của dj_dw là giá trị đạo hàm tương ứng của mỗi w trong w_vector
        dj_db = dj_b.mean() # dj_db là số thực
        return dj_dw, dj_db

    def _compute_gradient_descent(self, X, y_vector):
        """
        Parameters:
            X (numpy.ndarray(m,n)) : Feature data with m samples (observations)
            y_vector (numpy.ndarray(m,)) : Target values
    
        Return:
            w (numpy.ndarray(n,)) : Updated values of parameters w
            b (scalar) : Updated value of parameter b            
        """
        _, n_features = X.shape
        w = np.zeros(n_features)
        b = 0
                
        for _ in range(self.n_iters):
            dj_dw, dj_db = self._compute_partial_derivative(X, y_vector, w, b)
            w = w - self.eta * dj_dw # Cập nhật giá trị mới cho w
            b = b - self.eta * dj_db # Cập nhật giá trị mới cho b

        return w, b

#===============================================================
def plot_linear_regression_zero_one(X, y):
    # Plot the actual value points
    plt.scatter(X[y == 1], y[y == 1], s=50, c='red')
    plt.scatter(X[y == 0], y[y == 0], s=50)

    ligr = ViiLinearRegression(255, 0.01)
    ligr.fit(X.reshape(-1, 1), y)
    w = ligr.w
    b = ligr.b
    y_hat = ligr.predict(X.reshape(-1, 1))
    
    #Plot the prediction line
    plt.plot(X, y_hat, c='green')
    print(f"X train: {X}")
    print(f"y train: {y}")
    print(f"Prediction: {y_hat}")
    #Plot the threshold point
    y_05 = 0.5
    x_05 = (y_05 - b) / np.squeeze(w)
    plt.scatter(x_05, y_05)

    #Fill the seperated area
    plt.vlines(x=x_05, ymin=0, ymax=y_05, ls='dotted', colors='brown')
    # plt.axvline(x=x_05, ls='dotted', c='brown')
    plt.hlines(y=y_05, xmin=0, xmax=x_05, ls='dotted', colors='brown')
    _xlim = plt.xlim()
    _ylim = plt.ylim()
    plt.fill_between([_xlim[0], x_05], _ylim[1], color='darkblue', alpha=0.1, interpolate=True)
    plt.fill_between([x_05, _xlim[1]], _ylim[1], color='darkred', alpha=0.1, interpolate=True)
    plt.annotate(f"(x={x_05:.2f} y={y_05})", xy= [x_05, y_05], xytext=[5,-5], textcoords='offset points')
    plt.show()