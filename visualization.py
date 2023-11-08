import matplotlib.pyplot as plt

def plot_histograms(data, column):
    plt.figure(figsize=(8, 6))
    data[column].hist()
    plt.xlabel(column)  # Use the column name for the x-axis label
    plt.title(f'Histogram of {column}')
    plt.show()

def plot_bar_plots(data, column):
    plt.figure(figsize=(8, 6))
    data[column].value_counts().plot(kind='bar')
    plt.xlabel(column)  # Use the column name for the x-axis label
    plt.title(f'Bar-Plot of {column}')
    plt.show()
