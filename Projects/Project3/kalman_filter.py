import numpy as np

class PitchKalmanFilter:
    def __init__(self, dt=1/60):
        """
        State Vector x: [x_pos, y_pos, x_vel, y_vel, x_acc, y_acc]
        Measurement Vector z: [x_pos, y_pos]
        """
        self.dt = dt
        
        # State Transition Matrix (F)
        # x = x + v*t + 0.5*a*t^2
        # v = v + a*t
        # a = a
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],  # x_pos
            [0, 1, 0, dt, 0, 0.5*dt**2],  # y_pos
            [0, 0, 1, 0, dt, 0],          # x_vel
            [0, 0, 0, 1, 0, dt],          # y_vel
            [0, 0, 0, 0, 1, 0],           # x_acc
            [0, 0, 0, 0, 0, 1]            # y_acc
        ])

        # Measurement Matrix (H)
        # We only observe positions [x, y]
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        # Process Noise Covariance (Q)
        # Allow some flexibility in acceleration (jerk)
        q_var = 0.1
        self.Q = np.eye(6) * q_var

        # Measurement Noise Covariance (R)
        # YOLO noise (in pixels). Assume ~5-10 pixel error margin.
        r_var = 10.0 
        self.R = np.eye(2) * r_var

        # Initial State Covariance (P)
        self.P = np.eye(6) * 100.0
        
        # Initial State (x)
        self.x = np.zeros(6)

    def initialize(self, start_x, start_y):
        """Initialize state with the first detected ball position."""
        self.x = np.array([start_x, start_y, 0, 0, 0, 0])

    def predict(self):
        """Predict the next state based on physics model."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """
        Update state with new measurement.
        z: [x, y] from YOLO
        """
        # If measurement is NaN (missed detection), skip update
        if np.isnan(z).any():
            return

        y = z - (self.H @ self.x)       # Innovation (residual)
        S = self.H @ self.P @ self.H.T + self.R  # Innovation Covariance
        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman Gain

        self.x = self.x + (K @ y)
        self.P = (np.eye(6) - (K @ self.H)) @ self.P

    def process_sequence(self, trajectory):
        """
        Process an entire sequence of YOLO detections.
        trajectory: List or Array of shape (N, 2)
        """
        # 1. Initialize with first valid frame
        first_valid_idx = -1
        for i, point in enumerate(trajectory):
            if not np.isnan(point).any():
                self.initialize(point[0], point[1])
                first_valid_idx = i
                break
        
        if first_valid_idx == -1:
            return np.zeros(6) # Empty sequence

        # 2. Loop through the rest
        for i in range(first_valid_idx + 1, len(trajectory)):
            self.predict()
            self.update(trajectory[i])

        # Return final state [x, y, vx, vy, ax, ay]
        return self.x