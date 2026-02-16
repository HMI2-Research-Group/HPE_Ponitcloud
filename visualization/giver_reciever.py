import numpy as np
import matplotlib.pyplot as plt
import constant

# Hand pose
default_joint_3d_receiving = constant.RECEIVER_HAND_POSE
default_joint_3d_handover = constant.GIVER_HAND_POSE
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Receiving pose
x_data_receiving = default_joint_3d_receiving[:, 0]
y_data_receiving = default_joint_3d_receiving[:, 1]
z_data_receiving = default_joint_3d_receiving[:, 2]
ax.scatter(x_data_receiving, y_data_receiving, z_data_receiving, c='r', marker='o', label='Receiver Pose')

# Handover pose
x_data_handover = default_joint_3d_handover[:, 0]
y_data_handover = default_joint_3d_handover[:, 1]
z_data_handover = default_joint_3d_handover[:, 2]
ax.scatter(x_data_handover, y_data_handover, z_data_handover, c='g', marker='o', label='Giver Pose')

# Connect the joints for both poses 
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Little
]

for connection in connections:
    ax.plot([x_data_receiving[connection[0]], x_data_receiving[connection[1]]],
            [y_data_receiving[connection[0]], y_data_receiving[connection[1]]],
            [z_data_receiving[connection[0]], z_data_receiving[connection[1]]], 'r')

    ax.plot([x_data_handover[connection[0]], x_data_handover[connection[1]]],
            [y_data_handover[connection[0]], y_data_handover[connection[1]]],
            [z_data_handover[connection[0]], z_data_handover[connection[1]]], 'g')
    
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Receiver and Giver Hand Poses')
ax.legend()

plt.show()
