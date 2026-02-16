import numpy as np
import matplotlib.pyplot as plt
import constant

# Hand pose
default_joint_3d_receiving = constant.RECEIVER_HAND_POSE
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract X, Y, Z coordinates 
x_data = default_joint_3d_receiving[:, 0]
y_data = default_joint_3d_receiving[:, 1]
z_data = default_joint_3d_receiving[:, 2]

ax.scatter(x_data, y_data, z_data, c='r', marker='o')

# connect the joints with lines 
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Little
]

for connection in connections:
    ax.plot([x_data[connection[0]], x_data[connection[1]]],
            [y_data[connection[0]], y_data[connection[1]]],
            [z_data[connection[0]], z_data[connection[1]]], 'b')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Receiving Hand Pose')

plt.show()
