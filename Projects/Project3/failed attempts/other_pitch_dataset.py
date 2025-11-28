import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
from ultralytics import YOLO

class PitchDataset(Dataset):
    def __init__(self, data_source, video_root_dir, yolo_model_path, mode='train', transform=None):
        # ... existing init code ...
        if isinstance(data_source, pd.DataFrame):
            self.data_frame = data_source.reset_index(drop=True)
        else:
            self.data_frame = pd.read_csv(data_source)
            
        self.video_root_dir = video_root_dir
        self.transform = transform
        self.mode = mode
        self.yolo_model_path = yolo_model_path
        self.yolo = None 
        self.HOME_PLATE_CLASS_ID = 1
        self.BALL_CLASS_ID = 2
        self.NUM_FRAMES = 16
        self.norm_cols = ['release_speed', 'release_spin_rate', 'release_extension', 
                          'pfx_x', 'pfx_z', 'sz_top', 'sz_bot']
                          
        # --- NEW: ZONE MAPPING ---
        # Maps raw zone IDs to 0-12 indices for CrossEntropyLoss
        self.zone_map = {
            1:0, 2:1, 3:2, 
            4:3, 5:4, 6:5, 
            7:6, 8:7, 9:8, 
            11:9, 12:10, 13:11, 14:12
        }

    # ... keep _get_yolo_model, _get_box_center, _process_video exactly as before ...
    def __len__(self):
        return len(self.data_frame)
    
    def _get_yolo_model(self):
        if self.yolo is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.yolo = YOLO(self.yolo_model_path)
            self.yolo.to(device)
        return self.yolo

    def _get_box_center(self, box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    
    def _process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros((self.NUM_FRAMES, 2)), np.zeros(2), (1280.0, 720.0)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width == 0: width = 1280.0
        if height == 0: height = 720.0
            
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
            if not ret:
                break
                
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
                        box = xyxy[real_indices[best_idx]]
                        ball_centers.append(self._get_box_center(box))
                        ball_found = True
                        
                    plate_mask = (cls == self.HOME_PLATE_CLASS_ID)
                    if np.any(plate_mask):
                         best_idx = np.argmax(conf[plate_mask])
                         real_indices = np.where(plate_mask)[0]
                         plate_centers.append(self._get_box_center(xyxy[real_indices[best_idx]]))

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
            return np.zeros((self.NUM_FRAMES, 2)), plate_location, (width, height)
            
        df_temp = pd.DataFrame(ball_traj, columns=['x', 'y'])
        ball_traj = df_temp.interpolate(method='linear', limit_direction='both').to_numpy()
        ball_traj = np.nan_to_num(ball_traj)
        
        if len(ball_traj) < self.NUM_FRAMES:
            padding = np.zeros((self.NUM_FRAMES - len(ball_traj), 2))
            ball_traj = np.vstack([ball_traj, padding])
            
        return ball_traj, plate_location, (width, height)

    def __getitem__(self, idx):
        # ... existing extraction logic ...
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.data_frame.iloc[idx]
        video_name = row['file_name']
        video_path = os.path.join(self.video_root_dir, video_name)
        
        ball_traj, plate_loc, (w, h) = self._process_video(video_path)
        centered_traj = ball_traj - plate_loc
        normalized_traj = centered_traj / np.array([w, h])
        normalized_traj = np.clip(normalized_traj, -1.0, 1.0)
        normalized_traj = np.nan_to_num(normalized_traj, nan=0.0)

        stand = 1.0 if row['stand'] == 'R' else 0.0
        p_throws = 1.0 if row['p_throws'] == 'R' else 0.0
        
        def get_val(col):
            std_col = f"{col}_std"
            val = float(row[std_col]) if std_col in row else float(row[col])
            if np.isnan(val) or np.isinf(val): return 0.0
            return val

        physics_norm = np.array([
            get_val('release_speed'), get_val('release_spin_rate'), get_val('release_extension'),
            get_val('pfx_x'), get_val('pfx_z'), stand, p_throws,
            get_val('sz_top'), get_val('sz_bot')
        ], dtype=np.float32)
        
        def get_raw(col):
             val = float(row[col])
             return 0.0 if np.isnan(val) else val
             
        physics_raw = np.array([
            get_raw('release_speed'), get_raw('release_spin_rate'), get_raw('release_extension'),
            get_raw('pfx_x'), get_raw('pfx_z'), stand, p_throws, get_raw('sz_top'), get_raw('sz_bot')
        ], dtype=np.float32)
        
        physics_norm = np.nan_to_num(physics_norm, nan=0.0)
        physics_raw = np.nan_to_num(physics_raw, nan=0.0)

        # Targets
        px = float(row['plate_x'])
        pz = float(row['plate_z'])
        if np.isnan(px): px = 0.0 
        if np.isnan(pz): pz = 2.5 
        labels = np.array([px, pz], dtype=np.float32)
        pitch_class = 1.0 if row['pitch_class'] == 'strike' else 0.0
        
        # --- NEW: MAP ZONE TO INDEX ---
        raw_zone = float(row['zone']) if not pd.isna(row['zone']) else 14.0
        # Default to index 12 (Zone 14) if mapping fails
        zone_idx = self.zone_map.get(int(raw_zone), 12)

        return {
            'trajectory': torch.tensor(normalized_traj, dtype=torch.float32),
            'physics': torch.tensor(physics_norm, dtype=torch.float32),
            'physics_raw': torch.tensor(physics_raw, dtype=torch.float32),
            'plate_loc_raw': torch.tensor(plate_loc, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.float32),
            'class_label': torch.tensor(pitch_class, dtype=torch.float32),
            'zone_label': torch.tensor(zone_idx, dtype=torch.long) # Now 0-12 index
        }
