import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

from dataset import SherlockVideoDataset
from model import FrameReorderingModel

def generate_submission(model_path="best_model.pth", data_dir="dataset/test", output_csv="submission.csv", num_videos=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for inference")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dataset = SherlockVideoDataset(data_dir=data_dir, labels_file=None, transform=transform, is_train=False)
    if num_videos:
        test_dataset.video_files = test_dataset.video_files[:num_videos]
    
    if len(test_dataset) == 0:
        print(f"No videos found in {data_dir}. Place test dataset files first.")
        return

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    model = FrameReorderingModel().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from {model_path}")
    else:
        print(f"Warning: {model_path} not found. Running with untrained weights!")

    model.eval()
    
    submission_data = []

    print("Generating predictions...")
    with torch.no_grad():
        for frames, video_id in tqdm(test_loader, desc="Testing"):
            frames = frames.to(device)
            vid = video_id[0]
            
            scores = model(frames) # (1, S)
            scores = scores.squeeze(0).cpu().numpy() # (S,)
            
            pred_order = scores.argsort().tolist()
            
            order_str = " ".join(map(str, pred_order))
            
            submission_data.append({
                "video_id": vid,
                "order": order_str
            })

    df = pd.DataFrame(submission_data)
    df.to_csv(output_csv, index=False)
    print(f"Saved submission to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference for Sherlock Frame Reordering Model')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='Path to model weights')
    parser.add_argument('--data_dir', type=str, default='dataset/test', help='Test data directory')
    parser.add_argument('--output_csv', type=str, default='submission.csv', help='Output CSV file')
    parser.add_argument('--num_videos', type=int, default=None, help='Limit number of videos for testing')
    
    args = parser.parse_args()
    
    if args.num_videos:
        generate_submission(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_csv=args.output_csv,
            num_videos=args.num_videos
        )
    else:
        generate_submission(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_csv=args.output_csv
        )
