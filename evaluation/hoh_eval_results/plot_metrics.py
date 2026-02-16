import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def plot_error_metrics(mae_values, mse_values, metric_dir):
    """
    Plots Mean Absolute Error (MAE) and Mean Squared Error (MSE) over time or iterations.

    Parameters:
    - mae_values (list or array): List of Mean Absolute Error values.
    - mse_values (list or array): List of Mean Squared Error values.
    - metric_dir (str): Directory to save the plot.
    """

    output_file = os.path.join(metric_dir, 'error_metrics_plot.png')

    plt.figure(figsize=(10, 6))

    # Plotting MAE and MSE as lines
    plt.plot(mae_values, label='MAE', color='blue', marker='o')
    plt.plot(mse_values, label='MSE', color='orange', marker='o')

    plt.xlabel('Epochs')
    plt.ylabel('Error Values')
    plt.title('Error Metrics (MAE and MSE)')

    # Adding grid and legend
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Saving the plot
    plt.savefig(output_file)
    # plt.show()

    print(f"Plot saved to {output_file}")

def plot_accuracy(accuracy_values, metric_dir):
    """
    Plots Accuracy values as a line plot.

    Parameters:
    - accuracy_values (list or array): List or array of Accuracy values over time or iterations.
    - output_file (str): Path to save the plot.
    """
    output_file = os.path.join(metric_dir, 'accuracy_plot.png')
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy_values, marker='o', color='green', linestyle='-', linewidth=2, markersize=6)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Over Time or Iterations')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_file)
    #plt.show()
    print(f"Plot saved to {output_file}")

def save_metrics_to_csv(mae, mse, accuracy, metric_dir ):
    """
    Saves MAE, MSE, and Accuracy metrics to a CSV file.

    Parameters:
    - mae (float): Mean Absolute Error.
    - mse (float): Mean Squared Error.
    - accuracy (float): Accuracy value.
    - csv_file (str): Path to save the CSV file.
    """
    csv_file = os.path.join(metric_dir, 'metrics_data.csv')
    metrics_data = {
        'Metric': ['MAE', 'MSE', 'Accuracy'],
        'Value': [mae, mse, accuracy]
    }
    df = pd.DataFrame(metrics_data)
    df.to_csv(csv_file, index=False)

    print(f"Metrics saved to {csv_file}")