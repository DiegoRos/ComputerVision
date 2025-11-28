import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os
from ultralytics import YOLO

class PitchDataset(Dataset):
    def __init__(self, data_frame, video_root_dir, yolo_model_path):
        self.data_frame = data_frame
        self.video_root_dir = video_root_dir
        self.yolo_model_path = yolo_model_path
        self.yolo = None
        self.HOME_PLATE_CLASS_ID = 1
        self.BALL_CLASS_ID = 2
        
        # We do NOT normalize here anymore. 
        # We will feed raw physics into XGBoost.
        
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

    def _get_box_width(self, box):
        return box[2] - box[0]

    def _process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.array([]), np.array([0,0]), 1.0

        ball_centers = []
        plate_centers = []
        plate_widths = []

        # Lazy load model
        yolo_model = self._get_yolo_model()

        # Read ALL frames until video ends or max limit
        # The prompt says clips are 16-21 frames, so we read them all.
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

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
                    plate_widths.append(self._get_box_width(plate_box))

            if not ball_found:
                ball_centers.append(np.array([np.nan, np.nan]))

        cap.release()

        # Post-Processing
        # 1. Plate Location & Scale
        if len(plate_centers) > 0:
            plate_loc = np.median(np.array(plate_centers), axis=0)
            avg_plate_width_px = np.median(np.array(plate_widths))
            
            # Plate is 17 inches = 1.4167 feet
            px_per_ft = avg_plate_width_px / 1.4167
            if px_per_ft < 1: px_per_ft = 1.0 # Safety
        else:
            # Fallback
            plate_loc = np.array([640, 240])
            px_per_ft = 50.0 # Guess

        # 2. Ball Trajectory (Keep NaNs for Kalman Filter to handle)
        ball_traj = np.array(ball_centers)

        return ball_traj, plate_loc, px_per_ft

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        video_name = row['file_name']
        video_path = os.path.join(self.video_root_dir, video_name)

        ball_traj, plate_loc, px_per_ft = self._process_video(video_path)

        # We return a dictionary of raw numpy data.
        # We don't need PyTorch tensors because we are feeding XGBoost.
        return {
            'file_name': video_name,
            'trajectory': ball_traj, # Shape [N_frames, 2]
            'plate_loc': plate_loc,  # [x, y]
            'px_per_ft': px_per_ft,  # Scalar
            'physics': row.to_dict(), # Raw physics row
            'labels': {
                'plate_x': row.get('plate_x', 0),
                'plate_z': row.get('plate_z', 0),
                'pitch_class': row.get('pitch_class', 'unknown'),
                'zone': row.get('zone', 0)
            }
        }