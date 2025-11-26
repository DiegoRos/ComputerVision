import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
import json  # Added for saving/loading stats
from ultralytics import YOLO

class PitchDataset(Dataset):
    def __init__(self, data_frame, video_root_dir, yolo_model_path, transform=None, mode='train', fps=60, stats=None):
        """
        Args:
            data_frame (pd.DataFrame): DataFrame with annotations.
            video_root_dir (string): Directory with all the videos.
            yolo_model_path (string): Path to the trained YOLO .pt file.
            fps (int): Framerate of the video (usually 60 for MLB pitch clips).
            stats (dict or str, optional): Dictionary or Path to JSON file containing pre-computed means/stds. 
                                           CRITICAL: Must pass this for Validation/Test sets.
        """
        self.data_frame = data_frame
        self.video_root_dir = video_root_dir
        self.transform = transform
        self.mode = mode
        self.fps = fps

        # --- LAZY LOADING SETUP ---
        self.yolo_model_path = yolo_model_path
        self.yolo = None

        # Constants
        self.HOME_PLATE_CLASS_ID = 1
        self.BALL_CLASS_ID = 2
        self.NUM_FRAMES = 16

        # --- Normalization Stats ---
        self.norm_cols = [
            'release_speed', 'release_spin_rate', 'release_extension',
            'pfx_x', 'pfx_z', 'sz_top', 'sz_bot',
            'release_pos_x', 'release_pos_y', 'release_pos_z'
        ]

        self.target_cols = ['plate_x', 'plate_z']

        # --- LOGIC: LOAD STATS OR CALCULATE ---
        
        # 1. If stats is a file path, load it
        if isinstance(stats, str):
            if os.path.exists(stats):
                print(f"Loading normalization stats from {stats}...")
                with open(stats, 'r') as f:
                    stats = json.load(f)
            else:
                raise FileNotFoundError(f"Stats file not found at {stats}")

        # 2. Use provided stats (Validation/Test) or calculate new ones (Train)
        if stats:
            self.means = stats['means']
            self.stds = stats['stds']
            self.means_target = stats['means_target']
            self.stds_target = stats['stds_target']
        else:
            if self.mode == 'test':
                print("WARNING: You are initializing a TEST dataset without providing 'stats'.")
                print("         This will calculate stats from the Test set, which causes DATA LEAKAGE.")
                print("         Please load the stats saved from your Training set.")

            print("Calculating dataset statistics for normalization from current dataframe...")
            self.means = self.data_frame[self.norm_cols].mean().to_dict()
            self.stds = self.data_frame[self.norm_cols].std().to_dict()
            
            # For targets, we only calculate if we are in training mode or if targets exist
            if all(col in self.data_frame.columns for col in self.target_cols):
                self.means_target = self.data_frame[self.target_cols].mean().to_dict()
                self.stds_target = self.data_frame[self.target_cols].std().to_dict()
            else:
                # Fallback for inference if no targets exist in DF
                self.means_target = {k: 0.0 for k in self.target_cols}
                self.stds_target = {k: 1.0 for k in self.target_cols}

            # Avoid division by zero
            for k in self.stds:
                if self.stds[k] == 0: self.stds[k] = 1.0
            for k in self.stds_target:
                if self.stds_target[k] == 0: self.stds_target[k] = 1.0

    def save_stats(self, save_path):
        """
        Saves the current normalization statistics to a JSON file.
        Call this on your TRAINING dataset object.
        """
        # Helper to convert numpy types to native python types for JSON
        def convert(o):
            if isinstance(o, np.generic): return o.item()
            return o

        stats_dict = {
            'means': self.means,
            'stds': self.stds,
            'means_target': self.means_target,
            'stds_target': self.stds_target
        }
        
        with open(save_path, 'w') as f:
            json.dump(stats_dict, f, default=convert, indent=4)
        print(f"Normalization stats saved to {save_path}")

    def denormalize_targets(self, preds_tensor):
        """
        Convert model predictions (normalized) back to feet (real world).
        Args:
            preds_tensor: Torch tensor of shape (Batch, 2) -> [plate_x, plate_z]
        Returns:
            Numpy array of shape (Batch, 2) in feet.
        """
        preds = preds_tensor.cpu().detach().numpy()
        
        # plate_x is index 0, plate_z is index 1
        mean_x = self.means_target['plate_x']
        std_x = self.stds_target['plate_x']
        
        mean_z = self.means_target['plate_z']
        std_z = self.stds_target['plate_z']
        
        # Formula: Real = (Norm * Std) + Mean
        real_x = (preds[:, 0] * std_x) + mean_x
        real_z = (preds[:, 1] * std_z) + mean_z
        
        return np.stack([real_x, real_z], axis=1)

    def _get_yolo_model(self):
        """Lazy loader for the YOLO model."""
        if self.yolo is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.yolo = YOLO(self.yolo_model_path)
            self.yolo.to(device)
        return self.yolo

    def _get_box_center(self, box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def _apply_kalman_smoothing(self, trajectory):
        """
        Applies a Kalman Filter to smooth the 2D trajectory.
        Assumes trajectory is (N, 2) numpy array and has NO NaNs (interpolated beforehand).
        """
        # 4 dynamic params (x, y, dx, dy), 2 measurement params (x, y)
        kf = cv2.KalmanFilter(4, 2)
        
        # Measurement Matrix (We measure x and y)
        kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
        
        # Transition Matrix (Constant Velocity Model)
        # x_new = x_old + dx_old
        # y_new = y_old + dy_old
        kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
        
        # Process Noise Covariance (Q) - how much we trust the physics model
        # Smaller = smoother, Larger = more responsive to sudden changes
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        
        # Measurement Noise Covariance (R) - how much we trust the YOLO detections
        # Larger = ignore noise (smoother), Smaller = trust pixels exactly
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0 

        # Initialize state with the first measured point
        kf.statePost = np.array([[trajectory[0, 0]], 
                                 [trajectory[0, 1]], 
                                 [0], 
                                 [0]], dtype=np.float32)
        
        smoothed_traj = []
        for point in trajectory:
            measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
            
            # Predict step (Project state ahead)
            kf.predict()
            
            # Correct step (Update with measurement)
            estimated = kf.correct(measurement)
            
            # Store the smoothed position (x, y)
            smoothed_traj.append(estimated[:2].flatten())
            
        return np.array(smoothed_traj)

    def _process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((self.NUM_FRAMES, 2)), np.zeros(2), 1920, 1080, 16 

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames >= self.NUM_FRAMES:
            frame_indices = np.linspace(0, total_frames - 1, self.NUM_FRAMES).astype(int)
        else:
            frame_indices = np.arange(total_frames)

        ball_centers = []
        plate_centers = []

        current_frame_idx = 0
        target_idx_pointer = 0

        yolo_model = self._get_yolo_model()

        while cap.isOpened() and target_idx_pointer < len(frame_indices):
            ret, frame = cap.read()
            if not ret: break

            if current_frame_idx == frame_indices[target_idx_pointer]:
                results = yolo_model.predict(frame, verbose=False, conf=0.15, iou=0.15, classes=[1, 2])
                boxes = results[0].boxes

                ball_found = False
                if boxes is not None:
                    cls = boxes.cls.cpu().numpy().astype(int)
                    conf = boxes.conf.cpu().numpy()
                    xyxy = boxes.xyxy.cpu().numpy()

                    # Get Ball
                    ball_mask = (cls == self.BALL_CLASS_ID)
                    if np.any(ball_mask):
                        best_idx = np.argmax(conf[ball_mask])
                        real_indices = np.where(ball_mask)[0]
                        ball_box = xyxy[real_indices[best_idx]]
                        ball_centers.append(self._get_box_center(ball_box))
                        ball_found = True

                    # Get Plate
                    plate_mask = (cls == self.HOME_PLATE_CLASS_ID)
                    if np.any(plate_mask):
                         best_idx = np.argmax(conf[plate_mask])
                         real_indices = np.where(plate_mask)[0]
                         plate_box = xyxy[real_indices[best_idx]]
                         plate_centers.append(self._get_box_center(plate_box))

                if not ball_found:
                    ball_centers.append(np.array([np.nan, np.nan]))

                target_idx_pointer += 1
            current_frame_idx += 1
        cap.release()

        # Post Processing
        if len(plate_centers) > 0:
            plate_location = np.median(np.array(plate_centers), axis=0)
        else:
            plate_location = np.array([width / 2, height / 2])

        ball_traj = np.array(ball_centers)

        if np.all(np.isnan(ball_traj)):
             return np.zeros((self.NUM_FRAMES, 2)), plate_location, width, height, total_frames

        # 1. Linear Interpolation for NaNs (Gaps)
        df_temp = pd.DataFrame(ball_traj, columns=['x', 'y'])
        df_temp = df_temp.interpolate(method='linear', limit_direction='both')
        ball_traj = df_temp.to_numpy()
        ball_traj = np.nan_to_num(ball_traj) # Fallback if interpolation failed at very ends

        # 2. Kalman Smoothing (Jitter reduction)
        ball_traj = self._apply_kalman_smoothing(ball_traj)

        if len(ball_traj) < self.NUM_FRAMES:
            padding = np.zeros((self.NUM_FRAMES - len(ball_traj), 2))
            ball_traj = np.vstack([ball_traj, padding])

        return ball_traj, plate_location, width, height, total_frames

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        video_name = row['file_name']
        video_path = os.path.join(self.video_root_dir, video_name)

        # 1. Get Visual Data
        ball_traj_pixels, plate_loc_pixels, w, h, total_frames_in_clip = self._process_video(video_path)

        # --- PIXEL NORMALIZATION ---
        relative_traj = (ball_traj_pixels - plate_loc_pixels) 
        relative_traj[:, 0] = relative_traj[:, 0] / w
        relative_traj[:, 1] = relative_traj[:, 1] / h

        # 2. Get Physics Data
        stand = 1.0 if row['stand'] == 'R' else 0.0
        p_throws = 1.0 if row['p_throws'] == 'R' else 0.0

        def get_norm(col_name):
            val = float(row[col_name])
            return (val - self.means[col_name]) / (self.stds[col_name] + 1e-6)

        # --- DYNAMIC ToF CALCULATION ---
        mound_dist = 60.5
        total_tof = (mound_dist - row['release_extension']) / (row['release_speed'] * 1.467)
        time_elapsed = total_frames_in_clip / self.fps
        time_remaining = total_tof - time_elapsed
        
        # If time_remaining < 0 set to 0
        if time_remaining < 0:
            # print("Time remaining calculation < 0.") # Suppress spam
            time_remaining = 0

        physics_norm = np.array([
            get_norm('release_speed'),
            get_norm('release_spin_rate'),
            get_norm('release_extension'),
            get_norm('pfx_x'),
            get_norm('pfx_z'),
            stand,    
            p_throws, 
            get_norm('sz_top'),
            get_norm('sz_bot'),
            get_norm('release_pos_x'),
            get_norm('release_pos_y'),
            get_norm('release_pos_z'),
            time_remaining
        ], dtype=np.float32)

        relative_traj = np.nan_to_num(relative_traj, nan=0.0)
        physics_norm = np.nan_to_num(physics_norm, nan=0.0)

        data = {
            'trajectory': torch.tensor(relative_traj, dtype=torch.float32),
            'physics': torch.tensor(physics_norm, dtype=torch.float32),
            'plate_loc_raw': torch.tensor(plate_loc_pixels, dtype=torch.float32),
            'img_dims': torch.tensor([w, h], dtype=torch.float32)
        }

        if self.mode != 'test':
            def get_norm_target(col_name):
                val = float(row[col_name])
                return (val - self.means_target[col_name]) / (self.stds_target[col_name] + 1e-6)

            labels = np.array([
                get_norm_target('plate_x'),
                get_norm_target('plate_z')
            ], dtype=np.float32)

            data['labels'] = torch.tensor(labels, dtype=torch.float32)
            
            if 'pitch_class' in row:
                pitch_class = 1.0 if row['pitch_class'] == 'strike' else 0.0
                data['class_label'] = torch.tensor(pitch_class, dtype=torch.float32)
            if 'zone' in row:
                zone = float(row['zone']) if not pd.isna(row['zone']) else 14.0
                data['zone_label'] = torch.tensor(zone, dtype=torch.long)

        return data