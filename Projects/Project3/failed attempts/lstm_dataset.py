import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
import json
from ultralytics import YOLO
from tqdm import tqdm

VALID_ZONES = [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]

# Reverse map for training (Real Zone -> Model Index)
ZONE_TO_IDX = {z: i for i, z in enumerate(VALID_ZONES)}

class PitchDataset(Dataset):
    def __init__(self, data_frame, video_root_dir, yolo_model_path, transform=None, mode='train', fps=60, stats=None, cache_dir=None):
        """
        Args:
            data_frame (pd.DataFrame): DataFrame with annotations.
            video_root_dir (string): Directory with all the videos.
            yolo_model_path (string): Path to the trained YOLO .pt file.
            fps (int): Framerate.
            stats (dict/str): Stats for normalization.
            cache_dir (string, optional): Directory to save/load pre-processed trajectories.
        """
        self.data_frame = data_frame
        self.video_root_dir = video_root_dir
        self.transform = transform
        self.mode = mode
        self.fps = fps
        self.cache_dir = cache_dir

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

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


        # 1. Load Stats
        if isinstance(stats, str):
            if os.path.exists(stats):
                print(f"Loading normalization stats from {stats}...")
                with open(stats, 'r') as f:
                    stats = json.load(f)
            else:
                raise FileNotFoundError(f"Stats file not found at {stats}")

        if stats:
            self.means = stats['means']
            self.stds = stats['stds']
            self.means_target = stats['means_target']
            self.stds_target = stats['stds_target']
        else:
            if self.mode == 'test':
                print("WARNING: TEST dataset without stats provided. Possible DATA LEAKAGE.")
            
            print("Calculating dataset statistics for normalization...")
            self.means = self.data_frame[self.norm_cols].mean().to_dict()
            self.stds = self.data_frame[self.norm_cols].std().to_dict()
            
            if all(col in self.data_frame.columns for col in self.target_cols):
                self.means_target = self.data_frame[self.target_cols].mean().to_dict()
                self.stds_target = self.data_frame[self.target_cols].std().to_dict()
            else:
                self.means_target = {k: 0.0 for k in self.target_cols}
                self.stds_target = {k: 1.0 for k in self.target_cols}

            for k in self.stds: 
                if self.stds[k] == 0: self.stds[k] = 1.0
            for k in self.stds_target: 
                if self.stds_target[k] == 0: self.stds_target[k] = 1.0

    def __len__(self):
        return len(self.data_frame)

    def get_stats(self):
        return {
            'means': self.means, 'stds': self.stds,
            'means_target': self.means_target, 'stds_target': self.stds_target
        }

    def save_stats(self, save_path):
        def convert(o):
            if isinstance(o, np.generic): return o.item()
            return o
        stats_dict = self.get_stats()
        with open(save_path, 'w') as f:
            json.dump(stats_dict, f, default=convert, indent=4)
        print(f"Stats saved to {save_path}")

    def denormalize_targets(self, preds_tensor):
        preds = preds_tensor.cpu().detach().numpy()
        mean_x = self.means_target['plate_x']
        std_x = self.stds_target['plate_x']
        mean_z = self.means_target['plate_z']
        std_z = self.stds_target['plate_z']
        
        real_x = (preds[:, 0] * std_x) + mean_x
        real_z = (preds[:, 1] * std_z) + mean_z
        return np.stack([real_x, real_z], axis=1)

    def _get_yolo_model(self):
        if self.yolo is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.yolo = YOLO(self.yolo_model_path)
            self.yolo.to(device)
        return self.yolo

    def _get_box_center(self, box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    def _apply_kalman_smoothing(self, trajectory):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.05
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 10.0 
        kf.statePost = np.array([[trajectory[0, 0]], [trajectory[0, 1]], [0], [0]], dtype=np.float32)
        
        smoothed_traj = []
        for point in trajectory:
            measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
            kf.predict()
            estimated = kf.correct(measurement)
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

                    ball_mask = (cls == self.BALL_CLASS_ID)
                    if np.any(ball_mask):
                        best_idx = np.argmax(conf[ball_mask])
                        real_indices = np.where(ball_mask)[0]
                        ball_box = xyxy[real_indices[best_idx]]
                        ball_centers.append(self._get_box_center(ball_box))
                        ball_found = True

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

        if len(plate_centers) > 0:
            plate_location = np.median(np.array(plate_centers), axis=0)
        else:
            plate_location = np.array([width / 2, height / 2])

        ball_traj = np.array(ball_centers)
        if np.all(np.isnan(ball_traj)):
             return np.zeros((self.NUM_FRAMES, 2)), plate_location, width, height, total_frames

        df_temp = pd.DataFrame(ball_traj, columns=['x', 'y'])
        df_temp = df_temp.interpolate(method='linear', limit_direction='both')
        ball_traj = df_temp.to_numpy()
        ball_traj = np.nan_to_num(ball_traj)
        ball_traj = self._apply_kalman_smoothing(ball_traj)

        if len(ball_traj) < self.NUM_FRAMES:
            padding = np.zeros((self.NUM_FRAMES - len(ball_traj), 2))
            ball_traj = np.vstack([ball_traj, padding])

        return ball_traj, plate_location, width, height, total_frames

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        row = self.data_frame.iloc[idx]
        video_name = row['file_name']
        
       # --- Get Trajectory ---
        video_path = os.path.join(self.video_root_dir, video_name)
        ball_traj_pixels, plate_loc_pixels, w, h, total_frames_in_clip = self._process_video(video_path)

        # --- PIXEL NORMALIZATION (Fast) ---
        relative_traj = (ball_traj_pixels - plate_loc_pixels) 
        relative_traj[:, 0] = relative_traj[:, 0] / w
        relative_traj[:, 1] = relative_traj[:, 1] / h

        # --- PHYSICS DATA ---
        stand = 1.0 if row['stand'] == 'R' else 0.0
        p_throws = 1.0 if row['p_throws'] == 'R' else 0.0

        def get_norm(col_name):
            val = float(row[col_name])
            return (val - self.means[col_name]) / (self.stds[col_name] + 1e-6)

        mound_dist = 60.5
        total_tof = (mound_dist - row['release_extension']) / (row['release_speed'] * 1.467)
        time_elapsed = total_frames_in_clip / self.fps
        time_remaining = total_tof - time_elapsed
        if time_remaining < 0: time_remaining = 0

        physics_norm = np.array([
            get_norm('release_speed'), get_norm('release_spin_rate'), get_norm('release_extension'),
            get_norm('pfx_x'), get_norm('pfx_z'), stand, p_throws, 
            get_norm('sz_top'), get_norm('sz_bot'),
            get_norm('release_pos_x'), get_norm('release_pos_y'), get_norm('release_pos_z'),
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

            labels = np.array([get_norm_target('plate_x'), get_norm_target('plate_z')], dtype=np.float32)
            data['labels'] = torch.tensor(labels, dtype=torch.float32)
            
            if 'pitch_class' in row:
                pitch_class = 1.0 if row['pitch_class'] == 'strike' else 0.0
                data['class_label'] = torch.tensor(pitch_class, dtype=torch.float32)
            if 'zone' in row:
                raw_zone = float(row['zone']) if not pd.isna(row['zone']) else 14.0
                raw_zone = int(raw_zone)
                
                # Check if the zone is valid, otherwise default to 14 (Waste)
                if raw_zone not in ZONE_TO_IDX:
                    raw_zone = 14
                    
                # Map Real ID (1-14) -> Model Index (0-12)
                zone_idx = ZONE_TO_IDX[raw_zone]
                
                # Use LongTensor for Classification Targets
                data['zone_label'] = torch.tensor(zone_idx, dtype=torch.long)

        return data