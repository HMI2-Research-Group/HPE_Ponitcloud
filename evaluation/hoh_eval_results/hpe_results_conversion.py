import os
import torch
import torch.nn as nn
import numpy as np
import struct
from torch.utils.data import DataLoader
from pointnet_model import PointNet
from plot_vertices import plot_vertices_3d, plot_kde
from plot_metrics import plot_error_metrics, plot_accuracy, save_metrics_to_csv
from sklearn.metrics import mean_absolute_error, mean_squared_error
import constant



mae_values = []
mse_values = []
accuracy_values = []

# Function to Read PLY File and Extract Vertices
def read_ply(ply_path):
    """Reads a PLY file and returns the vertices, ensuring exactly 21 vertices."""
    vertices = []
    try:
        with open(ply_path, 'rb') as file:
            is_header = True
            while is_header:
                line = file.readline()
                if b'end_header' in line:
                    is_header = False
            while True:
                data = file.read(12)  
                if not data:
                    break
                x, y, z = struct.unpack('fff', data)
                if np.isnan(x) or np.isnan(y) or np.isnan(z) or np.isinf(x) or np.isinf(y) or np.isinf(z):
                    continue
                vertices.append((x, y, z))
    except Exception:
        pass

    # Ensure exactly 21 vertices
    vertice = np.array(vertices)
    if len(vertice) > 21:
        # Trim to the first 21 vertices if more than 21
        vertices = vertice[:21].T
    elif len(vertice) < 21:
        # Pad with zeros if fewer than 21
        padding = np.zeros((21 - len(vertice), 3))
        vertices = np.vstack([vertice, padding]).T
    
    # Visualizing the vertices
    # plot_vertices_3d(vertice) # Ensure you put a break point before running the plot
    # plot_kde(vertice) # Ensure you put a break point before running the plot
    return vertices

# Function to Validate Vertices
def validate_vertices(vertices):
    """Check if vertices contain valid numeric values."""
    vertices_array = np.array(vertices)
    if np.isnan(vertices_array).any() or np.isinf(vertices_array).any():
        print("Error: Vertices contain NaN or Infinity values.")
        return False
    return True


# Save the Predictions 
def save_predictions_to_txt(predictions, txt_path):
    with open(txt_path, 'w') as f:
        ind = 0
        for idx, prediction in enumerate(predictions):
            for i, pred in enumerate(prediction):
                flattened_prediction = pred.flatten()                      
                output_line = [ind + 1] + flattened_prediction.tolist()                 
                output_str = ','.join(map(str, output_line))
                f.write(output_str + '\n')
                ind += 1 
    print(f"Predictions saved to {txt_path}")
    
def calculate_metrics(predictions, ground_truths):
    valid_pairs = []
    for pred, gt in zip(predictions, ground_truths):
        # print(f"Prediction shape: {pred.shape}, Ground truth shape: {gt.shape}")
        if pred.shape == gt.shape:
            valid_pairs.append((pred, gt))
        
    if not valid_pairs:
        raise ValueError("No valid pairs found with matching shapes.")
    
    all_predictions = np.concatenate([pred for pred, _ in valid_pairs], axis=0)
    all_ground_truths = np.concatenate([gt for _, gt in valid_pairs], axis=0)
    
    all_ground_truths = np.clip(all_ground_truths, -1e5, 1e5)
    all_predictions = np.nan_to_num(all_predictions, nan=0.0, posinf=1e5, neginf=-1e5)
    all_ground_truths = np.nan_to_num(all_ground_truths, nan=0.0, posinf=1e5, neginf=-1e5)

    
    mae = np.mean(np.abs(all_predictions - all_ground_truths))
    mse = np.mean((all_predictions - all_ground_truths) ** 2)
    correct_predictions = np.sum(np.isclose(all_predictions, all_ground_truths, atol=0.42))
    total_predictions = all_predictions.size
    accuracy = correct_predictions / total_predictions
    
    mae_values.append(mae)
    mse_values.append(mse)
    accuracy_values.append(accuracy)
    
    return mae, mse, accuracy

# Process all .ply files in a directory and accumulate predictions
def process_ply_files_in_directory(base_dir, output_prefix, max_points=5120):
    #predictions = []
    vertices = []
    print("Processing point cloud data....")
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            cleaned_path = os.path.join(folder_path, 'Cleaned')
            if os.path.isdir(cleaned_path):
                for file_name in os.listdir(cleaned_path):
                    if file_name.startswith(output_prefix):
                        ply_path = os.path.join(cleaned_path, file_name)
                        # print(f"Processing {ply_path}")                       
                        
                        # Read PLY to get vertices
                        vertice = read_ply(ply_path)
                        
                        # Validate vertices
                        if not validate_vertices(vertice):
                            continue
                        #else:
                        #    vertice = pad_point_cloud(vertice, max_points)
                        
                        vertices.append(vertice)
                        
                        # Predict the hand joints
                        #predicted_joints = predict_hand_joints_with_pointnet(vertices, model)
                        
                        # Accumulate predictions
                        #predictions.append(predicted_joints)
    
    return vertices


def train_pointnet(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, labels in enumerate(train_loader):
            processed_data = []
            lab = i + 1
            
            if torch.is_tensor(labels) and labels.numel() > 0:
                for label_tensor in labels:
                    if torch.is_tensor(label_tensor) and label_tensor.numel() > 0:
                        # Process each tensor within the label_tensor list
                        inner_processed_data = []
                        for t in label_tensor:
                            if not isinstance(t, torch.Tensor):
                                t = torch.tensor(t, dtype=torch.float32)
                            else:
                                t = t.float()  # Ensure all tensors are float32

                            t = torch.where(t.abs() > 1e30, torch.tensor(0.0, dtype=torch.float32), t)
                            min_val = t.min()
                            max_val = t.max()
                            if max_val != min_val:  
                                t = (t - min_val) / (max_val - min_val)
                                
                            inner_processed_data.append(t)
                        
                        # Stack the tensors within the label_tensor list
                        stacked_inner_tensor = torch.stack(inner_processed_data)
                        processed_data.append(stacked_inner_tensor)
                    else:
                        if not isinstance(label_tensor, torch.Tensor):
                            label_tensor = torch.tensor(label_tensor, dtype=torch.float32)
                        else:
                            label_tensor = label_tensor.float()  
                        processed_data.append(label_tensor)
                
                inputs = torch.stack(processed_data).float()  
            else:
                inputs = torch.tensor(labels, dtype=torch.float32)

            labels = torch.tensor(lab, dtype=torch.float32)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        mae, mse, accuracy, Validation_loss = validate_pointnet(model, val_loader, criterion)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {Validation_loss},  VAL MAE: {mae}, VAL MSE: {mse}')




def validate_pointnet(model, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for i, labels in enumerate(val_loader):
            inputs = labels
            lab = i + 1
            
            processed_data = []
            
            if torch.is_tensor(labels) and labels.numel() > 0:
                for label_tensor in labels:
                    if torch.is_tensor(label_tensor) and label_tensor.numel() > 0:
                        # Process each tensor within the label_tensor list
                        inner_processed_data = []
                        for t in label_tensor:
                            if not isinstance(t, torch.Tensor):
                                t = torch.tensor(t, dtype=torch.float32)
                            else:
                                t = t.float()  # Ensure all tensors are float32

                            t = torch.where(t.abs() > 1e30, torch.tensor(0.0, dtype=torch.float32), t)
                            min_val = t.min()
                            max_val = t.max()
                            if max_val != min_val:  
                                t = (t - min_val) / (max_val - min_val)
                                
                            inner_processed_data.append(t)
                        
                        # Stack the tensors within the label_tensor list
                        stacked_inner_tensor = torch.stack(inner_processed_data)
                        processed_data.append(stacked_inner_tensor)
                    else:
                        if not isinstance(label_tensor, torch.Tensor):
                            label_tensor = torch.tensor(label_tensor, dtype=torch.float32)
                        else:
                            label_tensor = label_tensor.float()  
                        processed_data.append(label_tensor)
                
                inputs = torch.stack(processed_data).float()  
            else:
                # If labels is not a list, assume it's already a tensor
                inputs = torch.tensor(labels, dtype=torch.float32)

            labels = torch.tensor(lab, dtype=torch.float32)  # Ensure labels are a tensor
            
                      
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            
            # Collect predictions and ground truths for metrics calculation
            all_predictions.append(outputs.numpy())
            all_ground_truths.append(inputs.numpy())
    
    mae, mse, accuracy = calculate_metrics(all_predictions, all_ground_truths)
    Validation_loss = val_loss/len(val_loader)
    #print(f'Validation Loss: {Validation_loss}')
    #print(f'Validation MAE: {mae}')
    #print(f'Validation MSE: {mse}')
    return mae, mse, accuracy, Validation_loss

def evaluate_pointnet(model, test_loader, metric_dir):
    model.eval()
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for labels in enumerate(test_loader):
            inputs = labels          
            processed_data = []
            
            if torch.is_tensor(labels[1]) and labels[1].numel() > 0:
                for label_tensor in labels[1]:
                    if torch.is_tensor(label_tensor) and label_tensor.numel() > 0:
                        # Process each tensor within the label_tensor list
                        inner_processed_data = []
                        for t in label_tensor:
                            if not isinstance(t, torch.Tensor):
                                t = torch.tensor(t, dtype=torch.float32)
                            else:
                                t = t.float()  # Ensure all tensors are float32

                            t = torch.where(t.abs() > 1e30, torch.tensor(0.0, dtype=torch.float32), t)
                            min_val = t.min()
                            max_val = t.max()
                            if max_val != min_val:  
                                t = (t - min_val) / (max_val - min_val)
                                
                            inner_processed_data.append(t)
                        
                        # Stack the tensors within the label_tensor list
                        stacked_inner_tensor = torch.stack(inner_processed_data)
                        processed_data.append(stacked_inner_tensor)
                    else:
                        if not isinstance(label_tensor, torch.Tensor):
                            label_tensor = torch.tensor(label_tensor, dtype=torch.float32)
                        else:
                            label_tensor = label_tensor.float()  
                        processed_data.append(label_tensor)
                
                inputs = torch.stack(processed_data).float()  
            else:
                inputs = torch.tensor(labels[1], dtype=torch.float32)
                       
            outputs = model(inputs)
            
            # Collect predictions and ground truths for metrics calculation
            all_predictions.append(outputs.numpy())
            all_ground_truths.append(inputs.numpy())
    
    mae, mse, accuracy = calculate_metrics(all_predictions, all_ground_truths)
    
    print(f'Test MAE: {mae}')
    print(f'Test MSE: {mse}')
    print(f'Test Accuracy: {accuracy}')
    print('Plotting the metrics and acuracy...')
    plot_error_metrics(mae_values, mse_values, metric_dir)
    plot_accuracy(accuracy_values, metric_dir)
    save_metrics_to_csv(mae_values, mse_values, accuracy_values, metric_dir)
    return all_predictions

def train_and_evaluate_pointnet(train_process, val_process, test_process, output_file, model, criterion, optimizer, metric_dir, num_epochs=constant.NUM_EPOCHS):
    # Create DataLoaders for train, validation, and test datasets
    train_loader = DataLoader(train_process, batch_size=constant.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_process, batch_size=constant.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_process, batch_size=constant.BATCH_SIZE, shuffle=False)

    print("Training the model....")
    train_pointnet(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    
    print("Evaluating the model on test data....")
    predictions =  evaluate_pointnet(model, test_loader, metric_dir)

    save_predictions_to_txt(predictions, output_file)

def main():
    base_train_dir = constant.BASE_TRAIN_DIR
    base_val_dir = constant.BASE_VAL_DIR
    base_test_dir = constant.BASE_TEST_DIR
    output_dir = constant.OUTPUT_DIR
    metric_dir_giver = constant.METRIC_DIR_GIVER
    metric_dir_receiver = constant.METRIC_DIR_RECEIVER

    
    #Initialize model
    pointnet = PointNet()
    
    # Set up criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(pointnet.parameters(), lr=constant.LR)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Process and train on giver files
    giver_train_process = process_ply_files_in_directory(base_train_dir, 'giver_')
    giver_val_process = process_ply_files_in_directory(base_val_dir, 'giver_')
    giver_test_process = process_ply_files_in_directory(base_test_dir, 'giver_')
    
    train_and_evaluate_pointnet(
        giver_train_process, 
        giver_val_process, 
        giver_test_process, 
        os.path.join(output_dir, 'giver_hpe_predicted_results_test.txt'), 
        pointnet, criterion, optimizer,
        metric_dir_giver
    )
    
    # Process and train on receiver files
    receiver_train_process = process_ply_files_in_directory(base_train_dir, 'receiver_')
    receiver_val_process = process_ply_files_in_directory(base_val_dir, 'receiver_')
    receiver_test_process = process_ply_files_in_directory(base_test_dir, 'receiver_')
    
    train_and_evaluate_pointnet(
        receiver_train_process, 
        receiver_val_process, 
        receiver_test_process, 
        os.path.join(output_dir, 'receiver_hpe_predicted_results_test.txt'), 
        pointnet, criterion, optimizer,
        metric_dir_receiver
    )

if __name__ == "__main__":
    main()

