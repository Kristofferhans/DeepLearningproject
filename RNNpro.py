import os
import glob
import torch
import torch.nn as nn
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

# Paths to video and label directories
train_video_path = r"C:\Users\krist\Data science\island\T809DATA_2024-master\project\How2Sign\video_level\train\rgb_front\raw_videos"
val_video_path = r"C:\Users\krist\Data science\island\T809DATA_2024-master\project\How2Sign\video_level\val\rgb_front\raw_videos"
test_video_path = r"C:\Users\krist\Data science\island\T809DATA_2024-master\project\How2Sign\video_level\test\rgb_front\raw_videos"
label_dir = r"C:\Users\krist\Data science\island\T809DATA_2024-master\project\How2Sign\sentence_level\train\text\en\raw_text\re_aligned"

# Path to the CSV file for loading labels
csv_file = os.path.join(label_dir, 'how2sign_realigned_train.csv')  # Ensure this file exists

class VideoDataset(Dataset):
    def __init__(self, video_dir, label_dir, csv_file, transform=None):
        self.video_dir = video_dir
        self.label_dir = label_dir
        self.transform = transform
        self.labels = self.load_labels_from_csv(csv_file)

        # Load all video files in the directory
        self.video_files = sorted(glob.glob(os.path.join(video_dir, "*.mp4")))

        # Print loaded labels for comparison
        print(f"Loaded labels: {list(self.labels.keys())[:10]}...")  # Display first 10 labels for comparison

        # Filter video files based on matching labels
        self.video_files = [vf for vf in self.video_files if os.path.splitext(os.path.basename(vf))[0] in self.labels]
        
        # Print the number of video files loaded after filtering
        print(f"Filtered video files: {self.video_files}")
        print(f"Loaded {len(self.video_files)} video files from {video_dir}")

    def load_labels_from_csv(self, csv_file):
        labels = {}
        with open(csv_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')  # Assuming tab delimiter
                if len(parts) < 7:
                    continue
                video_name = parts[0].strip()  # The name should be stripped of whitespace
                text = ' '.join(parts[6:])  # Join remaining fields for text
                labels[video_name] = text

        # Debugging output for loaded labels
        print(f"Loaded labels: {list(labels.keys())[:10]}...")  # Display first 10 labels
        return labels


    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_file = self.video_files[idx]
        cap = cv2.VideoCapture(video_file)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))  # Resize frame
            frame = frame / 255.0  # Normalize
            frames.append(frame)
        cap.release()
        frames = np.stack(frames)

        # Load the corresponding label
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        label = self.labels.get(video_name, "")

        if self.transform:
            frames = self.transform(frames)

        return torch.tensor(frames, dtype=torch.float32), label

# Encode labels using LabelEncoder
def encode_labels(dataset):
    labels = [dataset[i][1] for i in range(len(dataset))]
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    return encoded_labels, label_encoder

# Define the RNN model for video classification
class VideoRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(VideoRNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        output = self.softmax(output)
        return output

# Prepare data loaders for training, validation, and testing
def create_dataloaders(batch_size=4):
    train_dataset = VideoDataset(train_video_path, label_dir, csv_file)
    val_dataset = VideoDataset(val_video_path, label_dir, csv_file)
    test_dataset = VideoDataset(test_video_path, label_dir, csv_file)

    train_encoded_labels, train_label_encoder = encode_labels(train_dataset)
    val_encoded_labels, _ = encode_labels(val_dataset)
    test_encoded_labels, _ = encode_labels(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_label_encoder

# Initialize model, loss function, and optimizer
input_size = 224 * 224 * 3  # Adjust this if you use feature extraction from frames
hidden_size = 128
n_classes = len(set(os.listdir(label_dir)))  # Number of unique classes

model = VideoRNN(input_size, hidden_size, n_classes)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, criterion, optimizer, n_epochs=10):
    for epoch in range(n_epochs):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.view(inputs.size(0), -1, input_size)  # Reshape for LSTM
            optimizer.zero_grad()
            outputs = model(inputs)
            label_indices = torch.tensor([train_label_encoder.transform([label])[0] for label in labels])
            loss = criterion(outputs, label_indices)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {total_loss / len(train_loader):.4f}')

# Running training
train_loader, val_loader, test_loader, train_label_encoder = create_dataloaders()
train_model(model, train_loader, criterion, optimizer)
