# CarPredictor
### - Batch Data Training and Analysis for Predictive Modeling

## Overview

This project focuses on the development of a custom Stochastic Gradient Descent (SGD) algorithm for training and evaluating a machine learning model using an automotive dataset. The primary objectives include data preprocessing, model training, hyperparameter optimization, and data visualization.

## Key Achievements

### Data Preprocessing

- Obtained and cleaned the automotive dataset from the UCI Machine Learning Repository, addressing missing values.
- Employed Pandas and SimpleImputer for data cleaning and imputation.

### Custom SGD Implementation

- Developed a custom Stochastic Gradient Descent (SGD) algorithm from scratch, enabling batch data processing.
- Created a "SGD" class with loss and gradient computation functions for model training.

### Model Training and Evaluation

- Trained the SGD model to predict car prices using a single feature (Feature 13).
- Utilized batch processing to update model weights.
- Evaluated the model's performance using Root Mean Square Error (RMSE).

### Hyperparameter Tuning

- Conducted a systematic hyperparameter search to optimize model performance.
- Explored various combinations of `max_epochs`, `batch_size`, and learning rates (`lr`).

### Data Visualization

- Implemented visualizations to monitor the model's training progress, including scatter plots of actual vs. predicted prices.

### Exploratory Data Analysis (EDA)

- Conducted an initial exploratory data analysis to gain insights into the dataset.
- Created histograms for numerical features and bar plots for categorical features.

## Skills and Tools

- Python (NumPy, Pandas, Matplotlib)
- Data Cleaning and Preprocessing
- Stochastic Gradient Descent (SGD) Optimization
- Machine Learning Model Training
- Hyperparameter Tuning
- Batch Data Processing
- Data Visualization
- Exploratory Data Analysis (EDA)

## Project Structure

- `main.py`: Contains the main implementation of the custom SGD algorithm.
- `data_preprocessing.py`: Handles data cleaning and imputation tasks.
- `visualization.py`: Includes code for data visualization.
- `README.md`: The project README file.

## Usage

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the main script using `python main.py`.
4. Experiment with different hyperparameters to optimize the model's performance.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset source: [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data).

## Author

- [Gideon Wikina](https://github.com/gwikina)

Feel free to reach out if you have any questions or suggestions!

