import numpy as np

class ErrorMetrics:
    """ Class to calculate error metrics
        Parameters:
        x: actual value
        x_hat: predicted value 
    """
    def __init__(self, x, x_hat):
        self.x = x.tolist() if isinstance(x, np.ndarray) else x
        self.x_hat = x_hat.tolist() if isinstance(x_hat, np.ndarray) else x_hat

    def MSE(self):
        """ Calculate Mean Squared Error
            Outputs:
            mse - mean squared error 
        """
        if isinstance(self.x, float):
            return (self.x - self.x_hat)**2
        else:
            return sum((self.x[i] - self.x_hat[i])**2 for i in range(len(self.x))) / len(self.x)
    
    def MAE(self):
        """ Calculate Mean Absolute Error 
            Outputs:
            mae - mean absolute error """
        if isinstance(self.x, float):
            return abs(self.x - self.x_hat)
        else:
            return sum(abs(self.x[i] - self.x_hat[i]) for i in range(len(self.x))) / len(self.x)