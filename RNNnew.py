import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np

# Define paths
video_paths = [
    r"C:\Users\krist\Data science\island\T809DATA_2024-master\project\How2Sign\video_level\train\rgb_front\raw_videos",
    r"C:\Users\krist\Data science\island\T809DATA_2024-master\project\How2Sign\video_level\val\rgb_front\raw_videos",
    r"C:\Users\krist\Data science\island\T809DATA_2024-master\project\How2Sign\video_level\test\rgb_front\raw_videos"
]

label_paths = [
    r"C:\Users\krist\Data science\island\T809DATA_2024-master\project\How2Sign\sentence_level\train\text\en\raw_text\re_aligned\how2sign_realigned_train.csv",
    r"C:\Users\krist\Data science\island\T809DATA_2024-master\project\How2Sign\sentence_level\test\text\en\raw_text\re_aligned\how2sign_realigned_test.csv",
    r"C:\Users\krist\Data science\island\T809DATA_2024-master\project\How2Sign\sentence_level\val\text\en\raw_text\re_aligned\how2sign_realigned_val.csv"
]

# Define a fixed number of frames
FIXED_FRAME_COUNT = 200  # Adjust this based on your dataset's average length

# Custom dataset class without using pandas
class VideoDataset(Dataset):
    def load_labels(self, label_file):
        labels = {}
        with open(label_file, 'r', encoding='utf-8') as f:
            header = f.readline()  # Read and skip the header
            for line in f:
                parts = line.strip().split('\t')  # Split by tab
                if len(parts) == 7:  # Ensure this matches the number of columns
                    video_name = parts[1].strip()  # Get the VIDEO_NAME
                    labels[video_name] = parts[6].strip()  # Store the SENTENCE
        print(f"Loaded labels: {list(labels.keys())}")  # Debug print
        return labels

    def __init__(self, video_dir, label_file):
        self.video_dir = video_dir
        self.labels = self.load_labels(label_file)
        self.video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        
        print(f"Video files found: {self.video_files}")  # Debug print

        # Normalize the comparison by stripping leading underscores
        stripped_video_files = [f.lstrip('_') for f in self.video_files]
        
        # Print the stripped video filenames for comparison
        print(f"Stripped video files: {stripped_video_files}")  # Debug print
        
        # Check if any of the stripped video filenames match the labels
        self.video_files = [f for f in self.video_files if f.lstrip('_') in self.labels]
        
        print(f"Filtered video files: {self.video_files}")  # Debug print

        if not self.video_files:
            raise ValueError("The dataset is empty. Check your video files and labels.")


    def __len__(self):
        return len(self.video_files)

    def pad_or_truncate(self, frames, target_length):
        frame_count = frames.shape[1]
        
        if frame_count > target_length:
            # Truncate to the target length
            frames = frames[:, :target_length, :, :]
        elif frame_count < target_length:
            # Pad with zeros to reach the target length
            padding = target_length - frame_count
            padding_shape = (frames.shape[0], padding, frames.shape[2], frames.shape[3])
            frames = torch.cat((frames, torch.zeros(padding_shape, dtype=frames.dtype)), dim=1)
        
        return frames

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        video_path = os.path.join(self.video_dir, video_file)
        
        # Read video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))  # Resize frame if necessary
            frames.append(frame)
        cap.release()
        
        # Check if we have frames to process
        if len(frames) == 0:
            return None, None
        
        # Convert frames to tensor and permute dimensions
        frames = torch.tensor(np.array(frames), dtype=torch.float32).permute(3, 0, 1, 2)  # (C, T, H, W)
        
        # Pad or truncate frames to a fixed length
        frames = self.pad_or_truncate(frames, FIXED_FRAME_COUNT)
        
        # Get label; convert sentence to a numeric label if necessary
        label = self.labels[video_file]  # This might need to be converted if you're using numeric labels
        
        return frames, label

# RNN model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = 224 * 224 * 3  # Assuming frame size is 224x224 and 3 color channels
hidden_size = 128
output_size = len(set(VideoDataset(video_paths[0], label_paths[0]).labels.values()))  # Unique labels count
num_epochs = 10
learning_rate = 0.001

# Load data
train_dataset = VideoDataset(video_paths[0], label_paths[0])
print(f"Number of samples in the dataset: {len(train_dataset)}")
if len(train_dataset) == 0:
    raise ValueError("The dataset is empty. Check your video files and labels.")
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
# Initialize model, loss function and optimizer
model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (videos, labels) in enumerate(train_loader):
        if videos is None or labels is None:
            continue  # Skip invalid samples
        
        videos = videos.view(videos.size(0), -1, input_size)  # Reshape videos for RNN
        
        # Forward pass
        outputs = model(videos)
        loss = criterion(outputs, labels)  # Ensure labels are numeric
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the trained model
model_save_path = r"C:\Users\krist\Data science\island\T809DATA_2024-master\project\rnn_model.pth"  # You can specify a different path if needed
torch.save(model.state_dict(), model_save_path)
print(f'Model saved to {model_save_path}')

print("Training complete.")
