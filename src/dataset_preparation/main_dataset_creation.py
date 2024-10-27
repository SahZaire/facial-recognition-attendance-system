import os
import random
import shutil
import csv

# Paths
original_dataset_dir = "Minor Project/Facial Recognition Attendance Sysytem/Dataset/lfw_deepfunneled"
new_dataset_dir = "Minor Project/Facial Recognition Attendance Sysytem/final_dataset/TY DS/listed_100"
test_dataset_dir = "Minor Project/Facial Recognition Attendance Sysytem/final_dataset/TY DS/present"
csv_file_path = "Minor Project/Facial Recognition Attendance Sysytem/final_dataset/TY DS/listed_100_names.csv"

# Create directories if they don't exist
os.makedirs(new_dataset_dir, exist_ok=True)
os.makedirs(test_dataset_dir, exist_ok=True)

# List all people (directories) in the original dataset
people = [person for person in os.listdir(original_dataset_dir) if os.path.isdir(os.path.join(original_dataset_dir, person))]

# Randomly select 100 people
selected_people = random.sample(people, 100)

# Prepare CSV data
csv_data = []

# Copy selected people into the new dataset directory and collect CSV data
for person in selected_people:
    # Original person directory
    person_dir = os.path.join(original_dataset_dir, person)
    
    # New person directory
    new_person_dir = os.path.join(new_dataset_dir, person)
    os.makedirs(new_person_dir, exist_ok=True)
    
    # List all images of this person
    images = os.listdir(person_dir)
    
    # Copy images to the new dataset folder
    for img_file in images:
        src_file = os.path.join(person_dir, img_file)
        dst_file = os.path.join(new_person_dir, img_file)
        shutil.copy(src_file, dst_file)
    
    # Add data to CSV (name, number of images)
    csv_data.append([person, len(images)])

# Write CSV file
with open(csv_file_path, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["name", "images"])  # Header
    writer.writerows(csv_data)

print(f"Created listed_100_names.csv with data of 100 selected people.")

# Part 2: Create 'test/' folder with one image per person
test_image_counter = 1

for person in selected_people:
    # Original person directory
    person_dir = os.path.join(original_dataset_dir, person)
    
    # List all images of this person
    images = os.listdir(person_dir)
    
    # Select only one image randomly
    selected_image = random.choice(images)
    
    # Copy the selected image to the 'test/' folder with a new name
    src_file = os.path.join(person_dir, selected_image)
    dst_file = os.path.join(test_dataset_dir, f"image_{test_image_counter}.jpg")
    shutil.copy(src_file, dst_file)
    
    test_image_counter += 1

print(f"Created test folder with 100 images named image_1, image_2, ..., image_100.")
