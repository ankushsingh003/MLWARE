import os
import json
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class SherlockVideoDataset(Dataset):
    def __init__(self, data_dir, labels_file=None, transform=None, is_train=True):
        """
        data_dir: directory containing video files (e.g., 'dataset/train' or 'dataset/test')
        labels_file: path to JSON containing labels (only used if is_train=True)
        transform: torchvision transforms applied to each frame
        is_train: if True, returns (frames, targets), otherwise returns (frames, video_id)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        self.video_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mp4')])
        
        self.labels = {}
        if self.is_train and labels_file is not None:
            with open(labels_file, 'r') as f:
                self.labels = json.load(f)

    def __len__(self):
        return len(self.video_files)
    
    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Force RGB conversion using cv2 but we will double check later
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def __getitem__(self, idx):
        video_filename = self.video_files[idx]
        video_id = os.path.splitext(video_filename)[0]
        video_path = os.path.join(self.data_dir, video_filename)
        
        frames = self.extract_frames(video_path)
        
        from PIL import Image
        frame_tensors = []
        for i, frame in enumerate(frames):
            # Ensure each frame is specifically RGB
            img = Image.fromarray(frame).convert('RGB')
            if self.transform:
                img = self.transform(img)
            else:
                # Default if no transform
                img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0
            frame_tensors.append(img)
            
        if len(frame_tensors) > 0:
            try:
                frame_tensors = torch.stack(frame_tensors)
            except RuntimeError as e:
                print(f"Error stacking frames for video {video_id}: {e}")
                # Print individual frame shapes for debugging
                for j, ft in enumerate(frame_tensors):
                    print(f"  Frame {j} shape: {ft.shape}")
                raise e
        else:
            frame_tensors = torch.empty(0)

        if self.is_train:
            target_order = self.labels.get(video_id, [])
            # Shift 1-indexed labels to 0-indexed
            target_order = [x - 1 for x in target_order]
            target = torch.tensor(target_order, dtype=torch.long)
            return frame_tensors, target
        else:
            return frame_tensors, video_id
