import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm

from dataset import SherlockVideoDataset
from model import FrameReorderingModel
from loss import MarginRankingLossPairs, calculate_kendall_tau

def train(epochs=10, batch_size=4, lr=1e-4, data_dir='../dataset/train', labels_file='../dataset/train_labels.json'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(), # converts HWC [0,255] to CHW [0.0, 1.0]
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = SherlockVideoDataset(data_dir=data_dir, labels_file=labels_file, transform=transform, is_train=True)
    
    if len(full_dataset) == 0:
        print(f"No videos found in {data_dir}. Please place dataset files first.")
        return

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = FrameReorderingModel().to(device)
    criterion = MarginRankingLossPairs(margin=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_tau = -1.0

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (frames, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            frames = frames.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            scores = model(frames) # (B, S)
            
            loss = criterion(scores, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_tau = 0.0
        
        with torch.no_grad():
            for frames, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                frames = frames.to(device)
                targets = targets.to(device)
                
                scores = model(frames)
                loss = criterion(scores, targets)
                val_loss += loss.item()
                
                
                for b in range(scores.size(0)):
                    b_scores = scores[b].cpu().numpy()
                    b_targets = targets[b].cpu().numpy().tolist()
                    
                    pred_order = b_scores.argsort().tolist()
                    
                    tau = calculate_kendall_tau(pred_order, b_targets)
                    val_tau += tau

        val_loss /= len(val_loader)
        val_tau /= len(val_dataset)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Tau: {val_tau:.4f}")

        if val_tau > best_val_tau:
            best_val_tau = val_tau
            print(f"--> New best validation Tau: {best_val_tau:.4f}. Saving model...")
            torch.save(model.state_dict(), "best_model.pth")

if __name__ == '__main__':
    pass
