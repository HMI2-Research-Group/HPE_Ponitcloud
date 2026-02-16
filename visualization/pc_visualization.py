import open3d as o3d 
import numpy as np

def load_point_cloud_o3d(path, color):
    # Load the point cloud using open3d
    point_cloud = o3d.io.read_point_cloud(path)
    if not point_cloud.has_points():
        print(f"Failed to load point cloud from {path}")
        return None
    colors = np.tile(color, (len(point_cloud.points), 1))
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    return point_cloud

def visualize_combined_point_clouds_o3d(paths, colors):
    # Load all point clouds with their respective colors
    point_clouds = [load_point_cloud_o3d(path, color) for path, color in zip(paths, colors)]
    point_clouds = [pc for pc in point_clouds if pc is not None]

    combined_point_cloud = o3d.geometry.PointCloud()
    for pc in point_clouds:
        print(f"Number of points in the current point cloud: {len(pc.points)}")
        combined_point_cloud += pc

    print(f"Total number of points in the combined point cloud: {len(combined_point_cloud.points)}")

    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Combined Point Clouds", width=1000, height=1000, left=50, top=50)
    vis.get_render_option().background_color = np.array([0, 0, 0])
    
    # Add the combined point cloud to the visualizer
    vis.add_geometry(combined_point_cloud)
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # Example paths to point cloud files
    giver_ply_path = "data/HOH/filtered/hoh_train/13/Cleaned/giver_frame17730.ply"
    object_ply_path = "data/HOH/filtered/hoh_train/13/Cleaned/object_frame17730.ply"
    receiver_ply_path = "data/HOH/filtered/hoh_train/13/Cleaned/receiver_frame17730.ply"

    # Define colors for each point cloud
    giver_color = [0, 0, 1]  # Blue
    object_color = [1, 0, 0]  # Red
    receiver_color = [1, 1, 0]  # Yellow

    # Visualize the combined point clouds
    paths = [giver_ply_path, object_ply_path, receiver_ply_path]
    colors = [giver_color, object_color, receiver_color]
    print("Visualizing combined point clouds with Open3D")
    visualize_combined_point_clouds_o3d(paths, colors)
