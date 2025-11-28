import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
from ultralytics import YOLO

class PitchDataset(Dataset):
    def __init__(self, data_frame, video_root_dir, yolo_model_path, transform=None, mode='train'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            video_root_dir (string): Directory with all the videos.
            yolo_model_path (string): Path to the trained YOLO .pt file.
        """
        self.data_frame = data_frame
        self.video_root_dir = video_root_dir
        self.transform = transform
        self.mode = mode

        # --- LAZY LOADING SETUP ---
        # We DO NOT load the model here. We store the path.
        # This prevents the "Cannot re-initialize CUDA" error in multiprocess loaders.
        self.yolo_model_path = yolo_model_path
        self.yolo = None

        # Constants
        self.HOME_PLATE_CLASS_ID = 1
        self.BALL_CLASS_ID = 2
        self.NUM_FRAMES = 16

        # --- Normalization Stats ---
        # We only normalize continuous variables. Binary ones (stand, p_throws) stay 0/1.
        self.norm_cols = ['release_speed', 'release_spin_rate', 'release_extension',
                          'pfx_x', 'pfx_z', 'sz_top', 'sz_bot']

        self.target_cols = ['plate_x', 'plate_z']

        # sz_bot and sz_top are important for final guessing so keeping raw physics tensor is also important

        # Calculate stats for the entire dataset upfront
        print("Calculating dataset statistics for normalization...")
        self.means = self.data_frame[self.norm_cols].mean().to_dict()
        self.stds = self.data_frame[self.norm_cols].std().to_dict()

        self.means_target = self.data_frame[self.target_cols].mean().to_dict()
        self.stds_target = self.data_frame[self.target_cols].std().to_dict()

        # Avoid division by zero (just in case std is 0)
        for k in self.stds:
            if self.stds[k] == 0:
                self.stds[k] = 1.0

    def __len__(self):
        return len(self.data_frame)

    def _get_yolo_model(self):
        """
        Lazy loader for the YOLO model.
        Initializes the model only when needed by the specific worker process.
        """
        if self.yolo is None:
            # Determine device: Use CUDA if available in this process
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # print(f"Initializing YOLO on {device} (PID: {os.getpid()})...")
            self.yolo = YOLO(self.yolo_model_path)
            self.yolo.to(device)
        return self.yolo

    def _get_box_center(self, box):
        """Calculates (cx, cy) from [x1, y1, x2, y2]"""
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def _process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # Return zeros if video fails
            return np.zeros((self.NUM_FRAMES, 2)), np.zeros(2)

        # Get video dimensions for fallback logic
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Select 16 equidistant frames
        if total_frames >= self.NUM_FRAMES:
            frame_indices = np.linspace(0, total_frames - 1, self.NUM_FRAMES).astype(int)
        else:
            # Pad if video is too short (rare case)
            frame_indices = np.arange(total_frames)

        ball_centers = []
        plate_centers = []
        plate_boxes = []

        current_frame_idx = 0
        target_idx_pointer = 0

        # --- LAZY LOAD MODEL ---
        yolo_model = self._get_yolo_model()

        while cap.isOpened() and target_idx_pointer < len(frame_indices):
            ret, frame = cap.read()
            if not ret:
                break

            # Only process if this is one of our target frames
            if current_frame_idx == frame_indices[target_idx_pointer]:

                # Run Inference (using the worker-local model)
                results = yolo_model.predict(frame, verbose=False, conf=0.15, iou=0.15, classes=[1, 2])
                boxes = results[0].boxes

                # Extract Ball
                ball_found = False
                if boxes is not None:
                    # Filter for ball class
                    cls = boxes.cls.cpu().numpy().astype(int)
                    conf = boxes.conf.cpu().numpy()
                    xyxy = boxes.xyxy.cpu().numpy()

                    # Get Ball (Class 2) - Take highest confidence
                    ball_mask = (cls == self.BALL_CLASS_ID)
                    if np.any(ball_mask):
                        best_idx = np.argmax(conf[ball_mask])
                        # We have to map the subset index back to the full array index
                        real_indices = np.where(ball_mask)[0]
                        ball_box = xyxy[real_indices[best_idx]]
                        ball_centers.append(self._get_box_center(ball_box))
                        ball_found = True

                    # Get Plate (Class 1) - Collect all to average later
                    plate_mask = (cls == self.HOME_PLATE_CLASS_ID)
                    if np.any(plate_mask):
                         best_idx = np.argmax(conf[plate_mask])
                         real_indices = np.where(plate_mask)[0]
                         plate_box = xyxy[real_indices[best_idx]]
                         plate_boxes.append(plate_box)
                         plate_centers.append(self._get_box_center(plate_box))

                if not ball_found:
                    ball_centers.append(np.array([np.nan, np.nan])) # Mark as missing for interpolation

                target_idx_pointer += 1

            current_frame_idx += 1

        cap.release()

        # --- Post Processing ---

        # Handle Plate (Median)
        if len(plate_centers) > 0:
            plate_location = np.median(np.array(plate_centers), axis=0)
            # Find idx of the median plate in the original list to get box
            idx = np.argsort(np.linalg.norm(np.array(plate_centers) - plate_location, axis=1))[0]
            best_plate_box = plate_boxes[idx]

        else:
            # Fallback to center of image if no plate detected
            plate_location = np.array([width / 2, height / 2])
            best_plate_box = np.array([width / 2 - 50, height / 2 - 10, width / 2 + 50, height / 2 + 10]) # Dummy box

        # Handle Ball Trajectory (Interpolation)
        ball_traj = np.array(ball_centers)

        # If completely empty
        if np.all(np.isnan(ball_traj)):
            return np.zeros((self.NUM_FRAMES, 2)), plate_location

        # Linear Interpolation for NaNs
        df_temp = pd.DataFrame(ball_traj, columns=['x', 'y'])
        df_temp = df_temp.interpolate(method='linear', limit_direction='both')
        ball_traj = df_temp.to_numpy()

        # Fill remaining NaNs (edges) with 0 if interpolation failed at ends
        ball_traj = np.nan_to_num(ball_traj)

        # Pad if we didn't get 16 frames (short video edge case)
        if len(ball_traj) < self.NUM_FRAMES:
            padding = np.zeros((self.NUM_FRAMES - len(ball_traj), 2))
            ball_traj = np.vstack([ball_traj, padding])

        return ball_traj, plate_location, best_plate_box

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data_frame.iloc[idx]
        video_name = row['file_name']
        video_path = os.path.join(self.video_root_dir, video_name)

        # --- Extract Visual Data ---
        # Returns (16, 2) and (2,)
        ball_traj, plate_loc, best_plate_box = self._process_video(video_path)

        # Normalize Trajectory: Centered around plate
        # This makes the model robust to camera panning
        normalized_traj = ball_traj - plate_loc

        # --- Extract Physics Data ---
        # Binary features (do not normalize)
        stand = 1.0 if row['stand'] == 'R' else 0.0
        p_throws = 1.0 if row['p_throws'] == 'R' else 0.0

        # Helper to normalize specific columns
        def get_norm(col_name):
            val = float(row[col_name])
            return (val - self.means[col_name]) / (self.stds[col_name] + 1e-6)

        # Normalized Physics Tensor (for Model Input)
        physics_norm = np.array([
            get_norm('release_speed'),
            get_norm('release_spin_rate'),
            get_norm('release_extension'),
            get_norm('pfx_x'),
            get_norm('pfx_z'),
            stand,    # binary
            p_throws, # binary
            get_norm('sz_top'),
            get_norm('sz_bot')
        ], dtype=np.float32)

        # Raw Physics Tensor (for Denormalization or Analysis)
        physics_raw = np.array([
            row['release_speed'],
            row['release_spin_rate'],
            row['release_extension'],
            row['pfx_x'],
            row['pfx_z'],
            stand,
            p_throws,
            row['sz_top'],
            row['sz_bot']
        ], dtype=np.float32)

        physics_norm = np.nan_to_num(physics_norm, nan=0.0, posinf=0.0, neginf=0.0)
        physics_raw = np.nan_to_num(physics_raw, nan=0.0, posinf=0.0, neginf=0.0)
        normalized_traj = np.nan_to_num(normalized_traj, nan=0.0, posinf=0.0, neginf=0.0)

        if self.mode != 'test':
            # --- Extract Labels ---
            # Predict targets
            def get_norm_target(col_name):
                val = float(row[col_name])
                return (val - self.means_target[col_name]) / (self.stds_target[col_name] + 1e-6)

            labels = np.array([
                get_norm_target('plate_x'),
                get_norm_target('plate_z')
            ], dtype=np.float32)

            labels_raw = np.array([
                row['plate_x'],
                row['plate_z']
            ], dtype=np.float32)

            # Additional classification targets (optional, depending on loss function)
            pitch_class = 1.0 if row['pitch_class'] == 'strike' else 0.0
            zone = float(row['zone']) if not pd.isna(row['zone']) else 14.0 # Default to ball zone

            # Convert to Tensors
            return {
                'trajectory': torch.tensor(normalized_traj, dtype=torch.float32), # Shape [16, 2]
                'physics': torch.tensor(physics_norm, dtype=torch.float32),       # Shape [9] (Normalized)
                'physics_raw': torch.tensor(physics_raw, dtype=torch.float32),    # Shape [9] (Raw)
                'plate_loc_raw': torch.tensor(plate_loc, dtype=torch.float32),    # Shape [2] (Useful for debug)
                'labels': torch.tensor(labels, dtype=torch.float32),              # Shape [2] (plate_x, plate_z)
                'labels_raw' : torch.tensor(labels_raw, dtype=torch.float32),     # Shape [2] (plate_x, plate_z)
                'class_label': torch.tensor(pitch_class, dtype=torch.float32),    # Shape [1]
                'zone_label': torch.tensor(zone, dtype=torch.long)                # Shape [1]
            }

        else:
            # Convert to Tensors
            return {
                'trajectory': torch.tensor(normalized_traj, dtype=torch.float32), # Shape [16, 2]
                'physics': torch.tensor(physics_norm, dtype=torch.float32),       # Shape [9] (Normalized)
                'physics_raw': torch.tensor(physics_raw, dtype=torch.float32),    # Shape [9] (Raw)
                'plate_loc_raw': torch.tensor(plate_loc, dtype=torch.float32),    # Shape [2] (Useful for debug)
            }