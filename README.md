# Mean-Variance Portfolio Optimization with Variance Reduction Techniques

This project explores Mean-Variance Portfolio Optimization and the application of variance reduction techniques to improve the efficiency of Monte Carlo simulations. It includes scripts for generating random portfolios, calculating the efficient frontier, and evaluating error metrics. The project is organized into Python scripts and Jupyter notebooks for ease of use and understanding.

## Project Structure

- `variance_reduction.py`: Contains functions and classes implementing various variance reduction techniques, such as antithetic variates, moment matching, and control variates. Including the functions to generate multivariate and univariate normal distributions for given mean and covariance.
- `mean_variance_portfolio.py`: Includes functions for generating random portfolios, calculating portfolio returns, risks, and the efficient frontier.
- `error_metrics.py`: Provides methods to calculate and evaluate error metrics to assess the effectiveness of the variance reduction techniques.
- `main_notebook.ipynb`: A Jupyter notebook that ties everything together. It provides a step-by-step walkthrough of the portfolio optimization process, demonstrating the use of variance reduction techniques and the calculation of error metrics.

## Installation and Requirements

To run this project, you need the following packages:

- Python 3.x
- numpy
- pandas
- matplotlib
- scipy
- jupyter

You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib scipy jupyter
```

## Usage

1. **Jupyter Notebook**: Start by running the `main_notebook.ipynb` file, which will guide you through the process of Mean-Variance Portfolio Optimization and the application of variance reduction techniques.

2. **Python Scripts**: You can also run the Python scripts directly:
   
   - `variance_reduction.py`: Provides tools to reduce variance in Monte Carlo simulations.
   - `mean_variance_portfolio.py`: Use this script to generate random portfolios and compute the efficient frontier.
   - `error_metrics.py`: Run this script to calculate error metrics and evaluate the performance of different variance reduction techniques.

## Project Highlights

- **Variance Reduction Techniques**: Implementations of antithetic variates, moment matching, and control variates to improve simulation efficiency.
- **Mean-Variance Portfolio Optimization**: Calculation of optimal portfolios on the efficient frontier.
- **Error Metrics**: Evaluation of the effectiveness of variance reduction techniques.

## Contributing

Not the most optimal methods used, contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential improvements. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact [Aditya Santosh](www.linkedin.com/in/aditya-santosh14) from Rutgers Business School, Master of Quantitative Finance program.
