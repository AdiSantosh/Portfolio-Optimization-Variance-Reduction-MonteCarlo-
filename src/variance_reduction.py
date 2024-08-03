import numpy as np
import math
import pandas as pd
from mean_variance_portfolio import MeanVariancePortfolio

class VarianceReduction(MeanVariancePortfolio):
    """ Class to implement variance reduction techniques to Monte Carlo simulation

        Parameters:
        iterations: Number of iterations
        samples: Number of samples to be generated 

        Methods:
        standard_normal: Generate samples from a standard normal distribution
        multivariate_normal: Generate samples from a multivariate normal distribution
        MC: Monte Carlo simulation to estimate the expected return, weights and variance of the portfolio
        MC_antithetic_variates: Monte Carlo simulation using antithetic variates to estimate the expected return, weights and variance of the portfolio
        MC_moment_matching: Monte Carlo simulation using moment matching to estimate the expected return, weights and variance of the portfolio
        MC_control_variates: Monte Carlo simulation using control variates to estimate the expected return, weights and variance of the portfolio
    """
    def __init__(self, MVP, iterations, samples):
        self.u1 = np.array([])
        self.u2 = np.array([])
        self.u1_dict = {i: [] for i in range(2, 10)}  ## range is 2 - 9 (yrs) as per the data used
        self.u2_dict = {i: [] for i in range(2, 10)}
        self.iterations = int(iterations)
        self.samples = int(samples / 2)
        self.stock_data = MVP.stock_data
        self.tickers = MVP.tickers
        self.stock_returns = MVP.stock_returns
        self.return_matrix = MVP.return_matrix
        self.average_returns = MVP.average_returns
        self.covariance = MVP.covariance
        self.observations = MVP.observations
        self.n_months = MVP.n_months
        
    def standard_normal(self):
        """ Generate samples from a standard normal distribution using Box-Muller method 

            Outputs:
            random_numbers - samples from a standard normal distribution
        """
        random_numbers = []
        x1 = np.random.rand(self.samples)
        x2 = np.random.rand(self.samples)
        
        self.u1 = x1
        self.u2 = x2
        
        for j in range(self.samples):
            z1 = math.sqrt(-2 * math.log(x1[j])) * math.cos(2 * math.pi * x2[j])
            z2 = math.sqrt(-2 * math.log(x1[j])) * math.sin(2 * math.pi * x2[j])
            random_numbers.extend([z1, z2])
        
        return random_numbers
    
    def multivariate_normal(self, mean = None, cov = None, new = False):
        """ Generate samples from a multivariate normal distribution

            Inputs:
            mean - mean of the distribution
            cov - covariance matrix of the distribution
            new - boolean to indicate if new mean and covariance matrix is provided
            Outputs:
            samples - samples from the multivariate normal distribution
        """
        samples = np.array([])
        if new == False:
            mean = self.average_returns
            cov = self.covariance             
        else:
            mean = mean
            cov = cov
            
        x = self.standard_normal()
        x = np.reshape(x, (-1, len(mean)))
        L = np.linalg.cholesky(cov)
        y = np.matmul(x, L.T) + mean
        
        samples = np.append(samples, y.flatten())
        
        return samples
    
    def MC(self):
        """ Monte Carlo simulation to estimate the expected return, weights and 
            variance of the portfolio overa range of years
 
            Outputs:
            Er - expected return of the portfolio
            W - weights of the portfolio
            var - variance of the portfolio 
        """
        Er = {i: [] for i in range(2, 10)}
        W = {i: [] for i in range(2, 10)}
        var = {i: [] for i in range(2, 10)}
        mvp = MeanVariancePortfolio(self.stock_data)
        
        for _ in range(self.iterations):
            k = self.multivariate_normal()
            for i in range(2, 10):
                t = int((len(self.average_returns) * (self.observations - self.n_months * i)) / 2)
                self.u1_dict[i].append(self.u1[:t])
                self.u2_dict[i].append(self.u2[:t])
                mean_ny, V_ny = mvp.sample_selection(i, k)[1:]
                Er_opt, W_opt, var_opt = mvp.mean_variance(mean_ny, V_ny, new=True)
                W[i].append(W_opt)
                var[i].append(var_opt)
                Er[i].append(Er_opt)
        
        Er = {i: np.mean(Er[i]) for i in Er}
        var = {i: np.mean(var[i]) for i in var}
        W = {i: np.mean(W[i], axis=0) for i in W}
        
        return Er, W, var
    
    def MC_antithetic_variates(self):
        """ Monte Carlo simulation using antithetic variates to estimate the expected return, weights and variance of the portfolio
            over a range of years

            Outputs:
            Er_anti - expected return of the portfolio
            W_anti - weights of the portfolio
            var_anti - variance of the portfolio
        """
        random_numbers = {i: [] for i in range(2, 10)}  
        Er_anti = {i: [] for i in range(2, 10)}
        W_anti = {i: [] for i in range(2, 10)}
        var_anti = {i: [] for i in range(2, 10)}
        mvp = MeanVariancePortfolio(self.stock_data)
        
        for j in range(self.iterations):
            for i in range(2, 10):
                t = int((len(self.average_returns) * (self.observations - self.n_months * i)) / 4)
                a_u1 = np.concatenate((self.u1_dict[i][j][:t], 1 - self.u1_dict[i][j][:t]))
                a_u2 = np.concatenate((self.u2_dict[i][j][:t], 1 - self.u2_dict[i][j][:t]))
                
                z = t * 2
                for k in range(z):
                    z1 = math.sqrt(-2 * math.log(a_u1[k])) * math.cos(2 * math.pi * a_u2[k])
                    z2 = math.sqrt(-2 * math.log(a_u1[k])) * math.cos(2 * math.pi * a_u2[k])
                    random_numbers[i].extend([z1, z2])
                
                s = random_numbers[i]
                s = np.reshape(s, (-1, len(self.average_returns)))
                L = np.linalg.cholesky(self.covariance)
                y = np.matmul(s, L.T) + self.average_returns

                y = np.append(y, y.flatten())

                mean_ny, V_ny = mvp.sample_selection(i, y)[1:]
                Er_opt, W_opt, var_opt = mvp.mean_variance(mean_ny, V_ny, new=True)
                W_anti[i].append(W_opt)
                var_anti[i].append(var_opt)
                Er_anti[i].append(Er_opt)
        
        Er_anti = {i: np.mean(Er_anti[i]) for i in Er_anti}
        var_anti = {i: np.mean(var_anti[i]) for i in var_anti}
        W_anti = {i: np.mean(W_anti[i], axis=0) for i in W_anti}
        
        return Er_anti, W_anti, var_anti
    
    def MC_moment_matching(self):
        """ Monte Carlo simulation using moment matching to estimate the expected return, weights and variance of the portfolio
            over a range of years

            Outputs:
            Er - expected return of the portfolio
            W - weights of the portfolio
            var - variance of the portfolio
        """
        Er = {i: [] for i in range(2, 10)}
        W = {i: [] for i in range(2, 10)}
        var = {i: [] for i in range(2, 10)}
        mvp = MeanVariancePortfolio(self.stock_data)
        
        for _ in range(self.iterations):
            k = self.multivariate_normal()
            for i in range(2, 10):
                mean_ny, V_ny = mvp.sample_selection_MM(i, k)[1:]
                Er_opt, W_opt, var_opt = mvp.mean_variance(mean_ny, V_ny, new=True)
                W[i].append(W_opt)
                var[i].append(var_opt)
                Er[i].append(Er_opt)
        
        Er = {i: np.mean(Er[i]) for i in Er}
        var = {i: np.mean(var[i]) for i in var}
        W = {i: np.mean(W[i], axis=0) for i in W}
        
        return Er, W, var
    
    def MC_control_variates(self, n=5):
        """ Monte Carlo simulation using control variates to estimate the expected return, weights and variance of the portfolio
            over a range of years

            Outputs:
            Er - expected return of the portfolio
            W - weights of the portfolio
            var - variance of the portfolio
        """
        mvp = MeanVariancePortfolio(self.stock_data)
        portfolio_weights = mvp.mean_variance()[1]
        portfolio_returns = np.matmul(self.return_matrix.T, portfolio_weights).reshape(-1, 1)

        ## corelation bw portfolio returns and individual stock returns
        corr = []
        for i in range(len(self.tickers)):
            temp = abs(np.corrcoef(portfolio_returns.flatten(), self.return_matrix[i].flatten()))
            corr.append(temp[0][1])

        ## top n stocks with highest correlation and get values of correlation
        top_n = np.argsort(corr)[-n:]
        top_n_stocks = [self.tickers[i] for i in top_n]

        ## Get the average returns and covariance matrix of the top 5 stocks
        top_n_returns = np.array([self.stock_returns[ticker] for ticker in top_n_stocks])
        top_n_returns = top_n_returns.reshape(len(top_n_stocks), -1)
        top_n_mean = np.array([np.mean(self.stock_returns[ticker]) for ticker in top_n_stocks])
        top_n_cov = np.array(np.cov(top_n_returns, rowvar = True))

        samples_top_n = self.multivariate_normal(mean = top_n_mean, cov = top_n_cov, new = True)
        samples_portfolio = self.multivariate_normal()
        c = -1 * np.cov(samples_top_n, samples_portfolio)[0][1] / np.var(samples_top_n)
        samples_portfolio = samples_portfolio.reshape(len(top_n_mean),-1) + c * (samples_top_n.reshape(len(top_n_mean),-1) - top_n_mean.reshape(-1,1))
        samples_portfolio = np.array(samples_portfolio).reshape(-1,1)

        Er = {i: [] for i in range(2, 10)}
        W = {i: [] for i in range(2, 10)}
        var = {i: [] for i in range(2, 10)}

        for _ in range(self.iterations):
            for i in range(2, 10):
                mean_ny, V_ny = mvp.sample_selection(i, samples_portfolio)[1:]
                Er_opt, W_opt, var_opt = mvp.mean_variance(mean_ny, V_ny, new=True)
                W[i].append(W_opt)
                var[i].append(var_opt)
                Er[i].append(Er_opt)

        Er = {i: np.mean(Er[i]) for i in Er}
        var = {i: np.mean(var[i]) for i in var}
        W = {i: np.mean(W[i], axis=0) for i in W}

        return Er, W, var
        