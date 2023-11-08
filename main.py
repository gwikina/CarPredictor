from SGD import SGD
from data_preprocessing import data_preprocessing
from visualization import plot_histograms, plot_bar_plots

# Load and preprocess data
Data = data_preprocessing("http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data")

# Model training and evaluation
def train_and_evaluate(max_epochs, batch_size, lr):
    np.random.seed(42)
    sgd = SGD(max_epoch=max_epochs, lr=lr, batch_size=batch_size)
    sgd.fit(Data[[13]], Data[25])
    np.random.seed()

    # Calculate RMSE
    rmse = np.sqrt(sgd.batch_loss(sgd.X, sgd.y))
    
    return rmse

# Define the parameter combinations to test
max_epochs_values = [100, 1000]
batch_size_values = [0, 1, 10, 50]
lr_values = [1e-6, 1e-7, 1e-8]

# Iterate over parameter combinations and report the results
for max_epochs in max_epochs_values:
    for batch_size in batch_size_values:
        for lr in lr_values:
            rmse = train_and_evaluate(max_epochs, batch_size, lr)
            print(f"max_epochs={max_epochs}, batch_size={batch_size}, lr={lr}, RMSE={rmse}")

# Data Visualization
description = pd.read_csv("description.txt", delimiter=':', header=None)
Data.columns = description[0].str.strip()  # Delete extra spaces in the column names

# List of numerical columns
numerical_columns = Data.select_dtypes(exclude=['object']).columns.tolist()

# List of nominal (categorical) columns
nominal_columns = Data.select_dtypes(include=['object']).columns.tolist()

# Plot histograms for numerical columns
for column in numerical_columns:
    plot_histograms(Data, column)

# Plot bar plots for categorical columns
for column in nominal_columns:
    plot_bar_plots(Data, column)
