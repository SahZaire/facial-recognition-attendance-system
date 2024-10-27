# Presence: Facial Recognition Attendance System

## Project Overview

**Presence** is a facial recognition-based attendance system designed to streamline and automate attendance marking in classrooms. Built with the **ResNet50 model** and a user-friendly web interface, Presence provides teachers with an efficient tool to record attendance quickly and accurately. This system reduces manual workload and increases classroom efficiency by instantly recognizing student faces and updating attendance records.

The system has achieved **88.57% accuracy** after fine-tuning the model for **500 epochs** and can reach **97% accuracy** on smaller datasets.

## Key Features

- **Real-Time Recognition**: The system captures and processes student images instantly, marking attendance in real-time.
- **Automated Record Updates**: Attendance is automatically logged in a backend CSV file, reducing errors and saving teachers valuable time.
- **User-Friendly Web Interface**: Designed to be intuitive, the web interface allows easy session setup and management.
- **Dynamic Database**: Records are updated instantly, and attendance data is saved and managed dynamically for easy access.
- **Data Analysis**: Provides insights on attendance trends, enabling teachers to better understand student engagement and attendance patterns.

## Dataset

The model is trained on the **Labeled Faces in the Wild (LFW) Dataset**, which includes numerous labeled facial images. This dataset allows for high-accuracy facial recognition, essential for the reliable functionality of Presence. Data augmentation techniques like random cropping and flipping have been applied to increase the robustness of the model.

## Process Flow

1. **User Interface Interaction**: Teachers log into the web interface, select the class and subject, and initiate the attendance session.
2. **Image Capture**: The webcam captures student images as they enter the classroom.
3. **Pre-processing**: Images are resized and augmented to ensure model accuracy.
4. **Facial Recognition**: The ResNet50 model extracts features from pre-processed images to identify students by matching them with the database.
5. **Attendance Marking**: Recognized students' attendance is marked, with only one entry per session even if multiple images are captured.
6. **Dynamic Record Updating**: Attendance is recorded in real-time, ensuring the data remains accurate and up-to-date.
7. **Overview Display**: After each session, an overview is generated to show total, present, and absent students.
8. **Data Analysis**: Attendance data can be analyzed to monitor trends and generate reports on student engagement.

## Expected Results

1. **High Accuracy**: Achieves **88.57% accuracy**, with up to **97% accuracy** on smaller datasets.
2. **Automated Record Updates**: Records are updated instantly, eliminating manual errors.
3. **Enhanced User Experience**: The intuitive interface ensures ease of use, even for non-technical users.
4. **Insightful Data Analysis**: Provides valuable insights into student attendance patterns and trends.
5. **Real-Time Attendance**: Attendance is marked instantly, saving time compared to traditional roll-call methods.

## Models Used

1. **ResNet50** - A convolutional neural network optimized for facial recognition, achieving high accuracy and reliable performance.
2. **OpenCV's face_recognition** - Used for detecting faces in real-time, enabling dynamic image capture and recognition.

## Future Work

Future developments include further model optimization and exploring additional features, such as **multi-class attendance tracking** and improved data visualization. The goal is to enhance model accuracy and provide more analytical insights for educators.

