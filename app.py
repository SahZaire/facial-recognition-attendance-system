from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, session
import os
import subprocess
from src.ex_attendance import model_testing
from src.ex_model_development import model_development
import time
import csv
import cv2
import numpy as np
import base64
import logging
import shutil, json

SUB = "ex_atn"

TEST = ""
BIODATA = ""
CLASS = ""

DATASET_FOLDER = 'final_dataset'
BIODATA_FILE = 'listed_100_biodata.csv'
NAMES_FILE = 'listed_100_names.csv'
PROFILE_IMAGE = 'resources/profile.jpg'

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = 'aP3h3@#P0d_j!aFgHDks%a9D'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_present_students():
    return session.get('present_students', 0)

def set_present_students(value):
    session['present_students'] = value

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/conduct', methods=['GET', 'POST'])
def conduct():
    dataset_path = 'final_dataset'
    class_names = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    print("Available classes:", class_names)
    
    if request.method == 'POST':
        selected_class = request.form['class']
        TEST = os.path.join('./final_dataset/', selected_class, '/present')
        BIODATA = os.path.join('./final_dataset/', selected_class, '/listed_100_biodata.csv')
        
        return redirect(url_for('processing', selected_class=selected_class))  
    
    return render_template('conduct.html', classes=class_names) 

@app.route('/detect_face', methods=['POST'])
def detect_face():
    try:
        data = request.get_json()
        img_data = data['imageData']
        subject = data['subject']

        # Strip off the data URL prefix if present (e.g., "data:image/jpeg;base64,")
        if img_data.startswith("data:image"):
            img_data = img_data.split(",")[1]

        # Decode base64 image data
        img = base64.b64decode(img_data)
        img_np = np.frombuffer(img, dtype=np.uint8)
        img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img_np is None:
            raise ValueError("Image data could not be decoded")

        # Perform face detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) > 0:
            # Draw bounding box around the first face
            (x, y, w, h) = faces[0]
            return jsonify({"faceDetected": True, "bbox": [int(x), int(y), int(w), int(h)], "captureImage": True})

        return jsonify({"faceDetected": False})
    except Exception as e:
        logging.exception("Error in /detect_face: %s", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/save_image', methods=['POST'])
def save_image():
    data = request.get_json()
    img_data = data['imageData']
    subject = data['subject']
    classss = data['selectedClass']

    # Create the subject's present directory if it doesn't exist
    save_dir = f"C:/A Drive/Machine Learning/Minor Project/Facial Recognition Attendance Sysytem/final_dataset/{classss}/present"
    os.makedirs(save_dir, exist_ok=True)

    # Decode the base64 image and save it
    img = base64.b64decode(img_data.split(',')[1])
    img_np = np.frombuffer(img, dtype=np.uint8)
    img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Save the image
    img_path = os.path.join(save_dir, f"{subject}_captured_{len(os.listdir(save_dir)) + 1}.jpg")
    cv2.imwrite(img_path, img_np)

    return jsonify({"status": "success", "imagePath": img_path})

@app.route('/get-csv-data', methods=['GET'])
def get_csv_data():
    selected_class = request.args.get('class')
    csv_file_path = f"C:/A Drive/Machine Learning/Minor Project/Facial Recognition Attendance Sysytem/final_dataset/{selected_class}/listed_100_biodata.csv"

    try:
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile)
            headers = next(csvreader) 
            rows = [row for row in csvreader]  

        return jsonify({
            'headers': headers,
            'rows': rows
        })
    except FileNotFoundError:
        return jsonify({'error': 'CSV file not found'}), 404

@app.route('/processing')
def processing():
    subject = request.args.get('subject')
    selected_class = request.args.get('class')
    return render_template('processing.html', subject=subject, selected_class=selected_class)

@app.route('/records')
def records():
    dataset_path = 'final_dataset'
    class_names = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    print("Available classes:", class_names)
    return render_template('records.html', classes=class_names)

@app.route('/database')
def database():
    dataset_path = 'final_dataset'
    class_names = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]
    
    return render_template('database.html', classes=class_names)

@app.route('/fonts/<path:filename>')
def serve_font(filename):
    return send_from_directory('Fonts', filename)

@app.route('/resources/<path:filename>')
def serve_resource(filename):
    return send_from_directory('Resources', filename)

@app.route('/start-attendance', methods=['POST'])
def start_attendance():
    try:
        data = request.get_json()
        sub = data.get('SUB')
        class_name = data.get('CLASS_NAME')

        print(sub, class_name)
        
        present_students = model_testing(sub, class_name)
        print("PRESENT_STUDENTS : ", present_students)
        set_present_students(present_students)
        print(get_present_students())
        print("hehehe")

        return jsonify({'success': True })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/start-deployment', methods=['POST'])
def start_deployment():
    try:
        data = request.get_json()
        class_name = data.get('CLASS_NAME')

        print(class_name)
        
        model_development(class_name)
        print("hehehe")

        return jsonify({'success': True })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get-overview-data', methods=['POST'])
def get_overview_data():
    print("done work")
    data = request.get_json()
    CLASS = data.get('CLASS_SSS')
    listed_100_folder = f"C:\\A Drive\\Machine Learning\\Minor Project\\Facial Recognition Attendance Sysytem\\final_dataset\\{CLASS}\\listed_100" 
    present_folder = f"C:\\A Drive\\Machine Learning\\Minor Project\\Facial Recognition Attendance Sysytem\\final_dataset\\{CLASS}\\present"
    print("bwork")
    print(f"Listed 100 Folder: {listed_100_folder}")
    print(f"Present Folder: {present_folder}")

    total_folders = len([name for name in os.listdir(listed_100_folder) if os.path.isdir(os.path.join(listed_100_folder, name))])

    present_students = get_present_students()
    present_images = len([file for file in os.listdir(present_folder) if os.path.isfile(os.path.join(present_folder, file))])
    print("wait a minute get it how u live it")
    print("\n Total folders : ", total_folders, "\n Present images : ", present_images)
    
    return jsonify({
        'total': total_folders,
        'present': present_students
    })

def run_attendance_script(sub, class_name):
    command = ["& C:/Users/sahil/.conda/envs/aco_pytorch/python.exe ", '"c:/A Drive/Machine Learning/Minor Project/Facial Recognition Attendance Sysytem/src/ex_attendance.py"', sub, class_name]  # Pass the variables here
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

@app.route('/get_student_data')
def get_student_data():
    class_name = request.args.get('class_name')
    student_name = request.args.get('name')

    try:
        biodata_file = f"{DATASET_FOLDER}/{class_name}/{BIODATA_FILE}"

        if not os.path.exists(biodata_file):
            return jsonify({"success": False, "message": "Class not found."}), 404

        with open(biodata_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            student = next((row for row in reader if row['name'] == student_name), None)

            if not student:
                return jsonify({"success": False, "message": "Student not found."}), 404

            student_folder = f"{DATASET_FOLDER}/{class_name}/listed_100/{student['name']}/"
            
            if not os.path.exists(student_folder):
                return jsonify({"success": False, "message": "Student folder not found."}), 404

            image_filenames = os.listdir(student_folder)
            if not image_filenames:
                return jsonify({"success": False, "message": "No images found for the student."}), 404

            image_filename = image_filenames[0]
            image_url = url_for('serve_image', filename=f"{class_name}/listed_100/{student['name']}/{image_filename}")

            return jsonify({
                "success": True,
                "name": student['name'],
                "roll_number": student['Roll_num'],
                "reg_number": student['Reg_num'],
                "image": image_url
            })

    except Exception as e:
        print(f"Error fetching student data: {e}")
        return jsonify({"success": False, "message": "Internal server error."}), 500

@app.route('/images/<path:filename>')
def serve_image(filename):
    try:
        return send_from_directory(DATASET_FOLDER, filename)
    except Exception as e:
        print(f"Error serving image: {e}")
        return "Error loading image", 404

@app.route('/get_all_students', methods=['GET'])
def get_all_students():
    class_name = request.args.get('class')
    biodata_file = os.path.join(DATASET_FOLDER, class_name, BIODATA_FILE)
    
    try:
        with open(biodata_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            students = [row['name'] for row in reader]
        return jsonify({'students': students})
    except Exception as e:
        print(f"Error fetching all students: {e}")
        return jsonify({"success": False, "message": "Error fetching students."}), 500

@app.route('/create_student', methods=['POST'])
def create_student():
    class_name = request.args.get('class')
    new_name = "Enter_Name"
    
    base_path = f'final_dataset/{class_name}/listed_100'
    biodata_path = f'{base_path}_biodata.csv'
    names_path = f'{base_path}_names.csv'
    
    # Prepare data for new student
    biodata = {
        'name': new_name,
        'images': '1',
        'Roll_num': 'Z00',
        'Reg_num': '00ZZZZ0000000'
    }
    for i in range(1, 12):
        biodata[f'subject_{i}'] = '0'
    
    # Add to biodata CSV
    with open(biodata_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=biodata.keys())
        if file.tell() == 0:  # If file is empty, write header
            writer.writeheader()
        writer.writerow(biodata)
    
    # Add to names CSV
    with open(names_path, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['name', 'images'])
        if file.tell() == 0:  # If file is empty, write header
            writer.writeheader()
        writer.writerow({'name': new_name, 'images': '1'})
    
    # Create folder
    new_folder = os.path.join(base_path, new_name)
    os.makedirs(new_folder, exist_ok=True)
    
    # Copy default profile image
    default_image = 'resources/profile.jpg'
    new_image_path = os.path.join(new_folder, f"{new_name}_1.jpg")
    shutil.copy(default_image, new_image_path)
    
    return jsonify({"success": True, "message": "New student created", "name": new_name})

def update_csv_file(file_path, old_name, new_data, is_biodata=True):
    updated_rows = []
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['name'] == old_name:
                # Update only the fields that are present in new_data
                for key, value in new_data.items():
                    if key in row:
                        row[key] = value
                updated_rows.append(row)
            else:
                updated_rows.append(row)
   
    with open(file_path, 'w', newline='') as file:
        fieldnames = reader.fieldnames  # Use the original fieldnames
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

@app.route('/update_student', methods=['POST'])
def update_student():
    class_name = request.args.get('class')
    data = json.loads(request.form.get('data'))
    
    old_name = data['oldName']
    new_name = data['name']
    
    base_path = f'final_dataset/{class_name}/listed_100'
    biodata_path = f'{base_path}_biodata.csv'
    names_path = f'{base_path}_names.csv'
    
    # Update biodata CSV
    update_csv_file(biodata_path, old_name, data, is_biodata=True)
    
    # Update names CSV if name or images changed
    if old_name != new_name or 'images' in data:
        names_data = {'name': new_name, 'images': data.get('images', '1')}
        update_csv_file(names_path, old_name, names_data, is_biodata=False)
    
    # Handle folder renaming
    old_folder = os.path.join(base_path, old_name)
    new_folder = os.path.join(base_path, new_name)
    
    if os.path.exists(old_folder) and old_name != new_name:
        os.rename(old_folder, new_folder)
    
    # Handle image update
    if 'image' in request.files:
        image = request.files['image']

        profile_image_path = os.path.join(new_folder, "Enter_Name_1.jpg")
        if os.path.exists(profile_image_path):
            os.remove(profile_image_path)

        image_path = os.path.join(new_folder, f"{new_name}_1.jpg")
        image.save(image_path)
    
    return jsonify({"success": True, "message": "Student data updated", "oldName": old_name, "newName": new_name})

if __name__ == '__main__':
    app.run(debug=True)