import numpy as np
from kinematics import get_forward_kinematics, inverse_kinematics_dls

def main():
    # ---------------------------------------------------------
    # PART 1: Forward Kinematics Test (Using Default Input)
    # ---------------------------------------------------------
    q_default = [0, np.pi/2, np.pi, 0, 0]
    qh_fixed = 0.0
    
    T_effective = get_forward_kinematics(q_default, qh_fixed)
    
    print("Effective Transformation Matrix (T_0^5) for given q1 to q5:")
    with np.printoptions(precision=4, suppress=True):
        print(T_effective)
    print("-" * 50)

    # ---------------------------------------------------------
    # PART 2: Inverse Kinematics Test
    # ---------------------------------------------------------
    # 1. Define specific target matrix
    T_target = np.array([
    [ 0,     1.,     0.,    -0.   ],
    [-0.,    0.,    -1.,     -0.461],
    [-1.,     0.,     0.,     0.08],
    [ 0.,     0.,     0.,     1.   ]])
    
    # 2. Set an initial guess that biases the solver
    q_guess = [np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2]
    
    # 3. Run the Jacobian DLS Inverse Kinematics solver
    print("Running Jacobian Damped Least Squares IK...")
    q_solved = inverse_kinematics_dls(T_target, q_guess, qh_fixed)
    
    # 4. Display the results
    print("\nSolved Angles (q1 to q5 in radians):")
    print(np.round(np.rad2deg(q_solved)))
    
    # Verify by printing the Forward Kinematics of the solved angles
    print("\nVerification (Should closely match T_target):")
    T_verify = get_forward_kinematics(q_solved, qh_fixed)
    with np.printoptions(precision=4, suppress=True):
        print(T_verify)

if __name__ == "__main__":
    main()