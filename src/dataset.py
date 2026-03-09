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
        """
        Extract frames from a video using OpenCV.
        Returns a numpy array of shape (num_frames, H, W, C) in RGB.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return np.array(frames)

    def __getitem__(self, idx):
        video_filename = self.video_files[idx]
        video_id = os.path.splitext(video_filename)[0]
        video_path = os.path.join(self.data_dir, video_filename)
        
        # Read frames
        frames = self.extract_frames(video_path)
        
        # Apply transforms if any (transforms usually expect PIL or Tensor, 
        # so we will assume transform handles numpy (H, W, C) or we loop)
        frame_tensors = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            frame_tensors.append(frame)
            
        # Stack frames to (num_frames, C, H, W)
        if len(frame_tensors) > 0 and isinstance(frame_tensors[0], torch.Tensor):
            frame_tensors = torch.stack(frame_tensors)
        else:
            # Fallback if no transform
            frame_tensors = torch.tensor(np.array(frames)).permute(0, 3, 1, 2).float() / 255.0

        if self.is_train:
            target_order = self.labels.get(video_id, [])
            target = torch.tensor(target_order, dtype=torch.long)
            return frame_tensors, target
        else:
            return frame_tensors, video_id
