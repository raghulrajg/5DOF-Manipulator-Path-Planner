import numpy as np

def dh_transformation_matrix(alpha, a, d, theta):
    """
    Computes the standard Denavit-Hartenberg (DH) homogeneous transformation matrix.
    
    Parameters:
    alpha : float - Link twist (radians)
    a     : float - Link length
    d     : float - Link offset
    theta : float - Joint angle (radians)
    """
    return np.array([
        [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
        [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),              d],
        [0,              0,                           0,                          1]
    ])

def get_effective_transformation(dh_table):
    """
    Computes the overall effective transformation matrix by multiplying joint matrices.
    """
    T_eff = np.eye(4) # Initialize as a 4x4 identity matrix
    
    for row in dh_table:
        alpha, a, d, theta = row
        T_next = dh_transformation_matrix(alpha, a, d, theta)
        T_eff = np.dot(T_eff, T_next) # Sequential matrix multiplication
        
    return T_eff

# --- Define your 5-DOF Manipulator DH Parameters ---
# Format: [alpha, a, d, theta]

"""
    Joint Limits (in degrees):
theta1:  -90 to 90
theta2:  -180 to 0
theta3:  0 to 180
theta4:  -90 to 90
theta5:  0 to 180
"""

theta1 = np.radians(30)  # Joint 1 angle in radians
theta2 = np.radians(-45)  # Joint 2 angle in radians
theta3 = np.radians(90)  # Joint 3 angle in radians
theta4 = np.radians(-90)  # Joint 4 angle in radians
theta5 = np.radians(0)  # Joint 5 angle in radians

dh_parameters = [
    #    alpha,           a,     d,     theta
    [np.radians(90),      0,    0.04,  theta1],                        # Joint 1
    [       0,          0.199,    0,   theta2 + np.radians(168.4)],    # Joint 2
    [       0,          0.14,     0,   theta3 - np.radians(168.4)],    # Joint 3
    [np.radians(90),      0,      0,   theta4 + np.radians(90)],       # Joint 4
    [       0,            0,    0.126, theta5]                         # Joint 5
]

# Calculate the effective transformation
effective_T = get_effective_transformation(dh_parameters)

# Output results
print("Effective 4x4 Transformation Matrix (Base to End-Effector):")
print(np.round(effective_T, 4))

print("\n--- Extracted Components ---")
print("End-Effector Position (X, Y, Z):")
print(np.round(effective_T[0:3, 3], 4))

print("\nEnd-Effector Orientation (3x3 Rotation Matrix):")
print(np.round(effective_T[0:3, 0:3], 4))
