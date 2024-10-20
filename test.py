import os
import csv
from collections import defaultdict

# Path to the directory containing CSV files
csv_folder = r"C:\Users\krist\Data science\island\T809DATA_2024-master\project\How2Sign\sentence_level\train\text\en\raw_text\re_aligned"
csv_file = os.path.join(csv_folder, 'how2sign_realigned_train.csv')  # Ensure this file exists

# Read the CSV and aggregate text for each video
video_base_path = r"C:\Users\krist\Data science\island\T809DATA_2024-master\project\How2Sign\video_level\test\rgb_front\raw_videos"

# Dictionary to hold video paths and their corresponding texts
video_texts = defaultdict(list)

try:
    with open(csv_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            fields = line.strip().split('\t')  # Split by tab
            video_id = fields[0]
            video_file = f"{video_id}.mp4"  # Assuming the video files are in .mp4 format
            video_path = os.path.join(video_base_path, video_file)  # Construct the full path

            # Join the remaining fields to get the text
            text = ' '.join(fields[6:])  # Assuming the text starts from the 7th field

            # Store the text in the dictionary with the video path as the key
            video_texts[video_path].append(text)

    # Print the video paths and their aggregated text
    for video_path, texts in video_texts.items():
        print(f"Video Path: {video_path}")
        for text in texts:
            print(f"Text: {text}\n")

except UnicodeDecodeError:
    print("There was an encoding error. Please check the file encoding.")
except FileNotFoundError:
    print(f"The file {csv_file} was not found. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")

