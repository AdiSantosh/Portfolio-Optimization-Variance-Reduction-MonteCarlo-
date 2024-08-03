import numpy as np

class MeanVariancePortfolio:
    """ Class to calculate the minimum variance portfolio and simulate the efficient frontier

        Inputs:
        stock_data - stock data with the following columns: 'TICKER', 'DATE', 'RET'

        Methods:
        mean_variance - calculate the minimum variance portfolio using non-linear optimization
        simulate_efficient_frontier - simulate the efficient frontier for the portfolio
        simulated_portfolios - generate random inefficient portfolios for the efficient frontier simulation
        sample_selection - select historical data for the number of years and append the samples
        sample_selection_MM - modify samples to match the mean and std of the original returns
    """
    def __init__(self, stock_data):
        self.stock_data = stock_data
        self.tickers = stock_data['TICKER'].unique().tolist()
        self.stock_returns = {ticker: stock_data[stock_data['TICKER'] == ticker]['RET'].values for ticker in self.tickers}
        self.return_matrix = np.array([self.stock_returns[ticker] for ticker in self.stock_returns]).reshape(len(self.tickers),-1)
        self.average_returns = np.array([np.mean(self.stock_returns[ticker]) for ticker in self.tickers])
        self.covariance = np.array(np.cov(self.return_matrix, rowvar = True))
        self.I = np.ones(len(self.tickers))
        self.inv_cov = np.linalg.inv(self.covariance)
        self.n_stocks =  len(self.tickers)
        self.observations = self.return_matrix.shape[1]
        self.n_months = 12
        
    def mean_variance(self, mean = None, cov = None, new = False):
        """ Calculate the minimum variance portfolio using non-linear optimization 

            Inputs:
            mean - mean of the portfolio
            cov - covariance of the portfolio
            new - boolean to indicate if new mean and covariance are provided
            Outputs:
            Er - expected return of the portfolio
            W - weights of the portfolio
            var - variance of the portfolio
        """
        if new == False:
            mean = self.average_returns
            cov = self.covariance
        else:
            mean = mean
            cov = cov
            
        W = np.matmul(np.linalg.inv(cov), self.I) / np.matmul(np.matmul(self.I.T, np.linalg.inv(cov)), self.I)
        var = np.matmul(np.matmul(W.T, cov), W)
        Er = np.matmul(mean, W)
        return Er, W, var
    
    def simulate_efficient_frontier(self):
        """ Simulate the efficient frontier for the portfolio 

            Outputs:
            sigma_space - standard deviation of the efficient frontier
            return_space - return of the efficient frontier
        """
        return_space = np.linspace(-0.03, 0.05, 1000)
        A = np.matmul(np.matmul(self.I.T, self.inv_cov), self.average_returns)
        B = np.matmul(np.matmul(self.average_returns.T, self.inv_cov), self.average_returns)
        C = np.matmul(np.matmul(self.I.T, self.inv_cov), self.I)
        D = B * C - A**2

        sigma_space = [np.sqrt(1 / C + (C / D) * (Er - A / C)**2) for Er in return_space]
        return sigma_space, return_space

    def simulated_portfolios(self, iterations=1000):
        """ Generate random inefficient portfolios for the efficient frontier simulation

            Inputs: 
            iterations - number of random portfolios to generate
            Outputs:
            sigma_space - standard deviation of the random portfolios
            return_space - return of the random portfolios
            weights_space - weights of the random portfolios
        """
        return_space, sigma_space, weights_space = [], [], []

        for _ in range(iterations):
            weights = np.random.uniform(-0.2, 1, len(self.average_returns))
            weights /= weights.sum()
            weights = weights.reshape(-1, 1)

            ret = np.matmul(weights.T, self.average_returns)
            sig = np.sqrt(np.matmul(np.matmul(weights.T, self.covariance), weights))

            return_space.append(ret)
            sigma_space.append(sig)
            weights_space.append(weights)

        return sigma_space, return_space, weights_space

    def sample_selection(self, years, samples):
        """ Select historical data for the number of years and append the samples

            Inputs:
            years - number of years of historical data to select
            samples - random samples to append to the historical data
            Outputs:
            ret_array_samples - historical data with samples appended
            mean_samples - mean of the historical data with samples appended
            V_samples - covariance matrix of the historical data with samples appended
        """
        req_samples = (self.observations - self.n_months * years) * self.n_stocks  ## required samples from multivariate normal distribution

        temp_ret_array = np.array([self.stock_returns[key][:self.n_months * years] for key in self.stock_returns])
        temp_ret_array = temp_ret_array.reshape(self.n_stocks, self.n_months * years)

        samples = np.array(samples)  # Ensure samples is a numpy array
        samples = samples[:req_samples].reshape(self.n_stocks, -1)
        ret_array_ny = np.hstack((temp_ret_array, samples))

        mean_ny = np.mean(ret_array_ny, axis=1)
        V_ny = np.cov(ret_array_ny, rowvar=True)

        return ret_array_ny, mean_ny, V_ny


    def sample_selection_MM(self, years, samples):
        """ Modify samples to match the mean and std of the original returns

            Inputs:
            years - number of years of historical data to select
            samples - random samples to append to the historical
            Outputs:
            ret_array_samples - historical returns with samples appended
            mean_samples - mean of the historical returns with samples appended
            V_samples - covariance matrix of the historical returns with samples appended
        """
        req_samples = (self.observations - self.n_months * years) * self.n_stocks

        temp_ret_array = np.array([self.stock_returns[key][:self.n_months * years] for key in self.stock_returns])
        temp_ret_array = temp_ret_array.reshape(self.n_stocks, self.n_months * years)

        samples = np.array(samples)  # Ensure samples is a numpy array
        samples = samples[:req_samples].reshape(self.n_stocks, -1)
        ret_array_ny = np.hstack((temp_ret_array, samples))

        mean_ny, std_ny = np.mean(ret_array_ny, axis=1), np.std(ret_array_ny)

        samples = ((samples - mean_ny[:, None]) * (np.std(self.return_matrix) / std_ny)) + self.average_returns[:, None]

        ret_array_samples = np.hstack((temp_ret_array, samples))
        mean_samples = np.mean(ret_array_samples, axis=1)
        V_samples = np.cov(ret_array_samples, rowvar=True)

        return ret_array_samples, mean_samples, V_samples
    