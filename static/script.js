document.addEventListener("DOMContentLoaded", function () {
    document.addEventListener("scroll", function () {
        const popElements = document.querySelectorAll(".aboutus-members");
        popElements.forEach((el) => {
            const rect = el.getBoundingClientRect();
            if (rect.top < window.innerHeight && rect.bottom >= 0) {
                el.classList.add("visible");
            }
        });
    });
});

let CLAAAAAS;

function getSelectedClass() {
    const classElement = document.getElementById("class");
    return classElement ? classElement.value : null;
}

let count = 1;
const subjects = [
    "DAA",
    "SEPM",
    "DL",
    "NLP",
    "CN",
    "APT",
    "EEIM",
    "MP",
    "DAA_PR",
    "DL_PR",
    "NLP_PR",
];

function updateSubjectCounters() {
    const selectedClass = getSelectedClass();
    
    if (!selectedClass) {
        console.log("No class selected or class element not found");
        return;
    }
    
    subjects.forEach((subject) => {
        const key = `${selectedClass}_${subject}`;
        let currentValue = localStorage.getItem(key);
        if (currentValue === null) {
            currentValue = 0;
        }

        try {
            const subjectElement = document.querySelector(`.sub${subject}`);
            if (subjectElement) {
                subjectElement.textContent = currentValue;
            }
        } catch (error) {
            console.log(`Skipping ${subject} for ${selectedClass}`);
        }
    });
}

// Call this function whenever the class selection changes
document.addEventListener("DOMContentLoaded", function() {
    const classElement = document.getElementById("class");
    if (classElement) {
        classElement.addEventListener("change", updateSubjectCounters);
        // Call this function initially to set up the counters
        updateSubjectCounters();
    } else {
        console.log("Class selection element not found on this page");
    }
});

let selectedClass;
function onDropChange() {
    const selectedClass = document.getElementById("class").value;

    fetch(`/get-csv-data?class=${selectedClass}`)
        .then((response) => response.json())
        .then((data) => {
            const table = document.getElementById("biodataTable");
            table.innerHTML = "";

            const headerRow = document.createElement("tr");
            data.headers.forEach((header) => {
                const th = document.createElement("th");
                th.textContent = header;
                headerRow.appendChild(th);
            });
            table.appendChild(headerRow);

            data.rows.forEach((row) => {
                const rowElement = document.createElement("tr");
                row.forEach((cell) => {
                    const td = document.createElement("td");
                    td.textContent = cell;
                    rowElement.appendChild(td);
                });
                table.appendChild(rowElement);
            });
        })
        .catch((error) => {
            console.error("Error fetching CSV data:", error);
        });
}
// updateSubjectCounters();
function proceedToProcessing(subject, URL) {
    const selectedClass = getSelectedClass();

    if (!selectedClass) {
        const errorMessage = document.querySelector(".error-message");
        if (errorMessage) {
            errorMessage.innerText = "Please select a class before proceeding.";
            setTimeout(() => {
                errorMessage.innerText = "";
            }, 3000);
        }
        return;
    }

    const key = `${selectedClass}_${subject}`;
    let currentValue = localStorage.getItem(key);

    if (currentValue === null) {
        localStorage.setItem(key, 1);
    } else {
        currentValue = parseInt(currentValue, 10) + 1;
        localStorage.setItem(key, currentValue);
    }

    console.log(subject, typeof subject);

    const url = `${URL}?subject=${subject}&class=${selectedClass}`;
    console.log(selectedClass);
    console.log(count);
    count += 1;

    // Update the counter display
    updateSubjectCounters();

    window.location.href = url;
}

function updateSubjectLinks() {
    const subjectLinks = document.querySelectorAll(".proceed-link");
    subjectLinks.forEach((link) => {
        link.href = `{{ url_for('processing') }}?class=${selectedClass}&subject=${link.dataset.subject}`;
    });
}

document.addEventListener("DOMContentLoaded", function() {
    document.querySelector("#class").addEventListener("change", onClassChangeDatabase);
    document.querySelector(".img-edit").addEventListener("click", handleImageEdit);
    document.querySelector(".create-btn").addEventListener("click", createNewStudent);
    document.querySelector(".next-btn").addEventListener("click", nextStudent);
    document.querySelector(".prev-btn").addEventListener("click", prevStudent);
    document.querySelector(".update-btn").addEventListener("click", updateStudentData);
    document.querySelector("#search-button").addEventListener("click", searchStudent);

    makeFieldsEditable();
});

function onClassChangeDatabase() {
    const selectedClass = document.querySelector("#class").value;
    fetchStudents(selectedClass);
}

function fetchStudents(selectedClass) {
    fetch(`/get_all_students?class=${selectedClass}`)
        .then(response => response.json())
        .then(data => {
            students = data.students;
            currentIndex = 0;
            displayStudentData(students[currentIndex]);
        })
        .catch(error => console.error("Error fetching students:", error));
}

async function displayStudentData(studentName) {
    const selectedClass = document.querySelector("#class").value;
    fetch(`/get_student_data?class_name=${selectedClass}&name=${studentName}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.querySelector(".student-img").innerHTML = `<img src="${data.image}" alt="Student Image" style="height: 40vh; width: 60vh; object-fit: contain;">`;
                document.querySelector(".name-para h1").textContent = data.name;
                document.querySelector(".roll-para h1").textContent = data.roll_number;
                document.querySelector(".reg-para h1").textContent = data.reg_number;
            } else {
                console.error("Failed to load student data:", data.message);
            }
        })
        .catch(error => console.error("Error fetching student data:", error));
}

function handleImageEdit() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (event) => {
        const file = event.target.files[0];
        const reader = new FileReader();
        reader.onload = (e) => {
            document.querySelector(".student-img img").src = e.target.result;
        };
        reader.readAsDataURL(file);
    };
    input.click();
}

async function createNewStudent() {
    const selectedClass = document.querySelector("#class").value;
    fetch(`/create_student?class=${selectedClass}`, { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                console.log("New student created successfully");
                students.push(data.name);
                currentIndex = students.length - 1;
                displayStudentData(data.name);
            } else {
                console.error("Failed to create new student:", data.message);
            }
        })
        .catch(error => console.error("Error creating new student:", error));
}

function nextStudent() {
    if (currentIndex < students.length - 1) {
        currentIndex++;
        displayStudentData(students[currentIndex]);
    }
}

function prevStudent() {
    if (currentIndex > 0) {
        currentIndex--;
        displayStudentData(students[currentIndex]);
    }
}

function updateStudentData() {
    const selectedClass = document.querySelector("#class").value;
    const currentName = students[currentIndex];
    const updatedData = {
        oldName: currentName,
        name: document.querySelector(".name-para h1").textContent,
        Roll_num: document.querySelector(".roll-para h1").textContent,
        Reg_num: document.querySelector(".reg-para h1").textContent
    };

    // Only include changed fields
    const originalData = {
        name: currentName,
        Roll_num: document.querySelector(".roll-para h1").getAttribute("data-original"),
        Reg_num: document.querySelector(".reg-para h1").getAttribute("data-original")
    };

    Object.keys(updatedData).forEach(key => {
        if (updatedData[key] === originalData[key]) {
            delete updatedData[key];
        }
    });

    console.log("Updating student data:", updatedData); // Debugging log

    const formData = new FormData();
    formData.append("data", JSON.stringify(updatedData));

    const imgElement = document.querySelector(".student-img img");
    if (imgElement.src.startsWith('data:image')) {
        fetch(imgElement.src)
            .then(res => res.blob())
            .then(blob => {
                formData.append("image", blob, "new_image.jpg");
                sendUpdateRequest(selectedClass, formData);
            });
    } else {
        sendUpdateRequest(selectedClass, formData);
    }
}

function sendUpdateRequest(selectedClass, formData) {
    console.log("Sending update request for class:", selectedClass); // Debugging log

    fetch(`/update_student?class=${selectedClass}`, {
        method: "POST",
        body: formData
    })
    .then(response => {
        console.log("Raw response:", response); // Debugging log
        return response.json();
    })
    .then(data => {
        console.log("Parsed response data:", data); // Debugging log
        if (data.success) {
            console.log("Student data updated successfully");
            // Update the students array with the new name if it was changed
            if (data.oldName !== data.newName) {
                const index = students.indexOf(data.oldName);
                if (index !== -1) {
                    students[index] = data.newName;
                }
            }
            // Refresh the current student's data
            displayStudentData(students[currentIndex]);
        } else {
            console.error("Failed to update student data:", data.message);
        }
    })
    .catch(error => console.error("Error updating student:", error));
}

function makeFieldsEditable() {
    const editableFields = ['.name-para h1', '.roll-para h1', '.reg-para h1'];
    editableFields.forEach(selector => {
        const element = document.querySelector(selector);
        element.addEventListener('click', function() {
            const input = document.createElement('input');
            input.type = 'text';
            input.value = this.textContent;
            input.style.fontSize = getComputedStyle(this).fontSize;
            input.style.fontWeight = getComputedStyle(this).fontWeight;
            input.style.textAlign = 'center';
            input.style.width = '100%';
            this.textContent = '';
            this.appendChild(input);
            input.focus();

            input.addEventListener('blur', function() {
                this.parentElement.textContent = this.value;
            });

            input.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    this.blur();
                }
            });
        });
    });
}

function searchStudent() {
    const searchInput = document.querySelector("#search-input").value.trim().toLowerCase();
    const foundIndex = students.findIndex(student => student.toLowerCase().includes(searchInput));
    
    if (foundIndex !== -1) {
        currentIndex = foundIndex;
        displayStudentData(students[currentIndex]);
    } else {
        alert("Student not found.");
    }
}




document.addEventListener("DOMContentLoaded", function () {
    function toggleDropdown() {
        var content = document.getElementById("dropdownContent");
        content.style.display =
        content.style.display === "block" ? "none" : "block";
    }
    console.log("bkl");  
        const videoElement = document.getElementById("camera-stream");
        
        videoElement.addEventListener('loadedmetadata', () => {
            
        detectFaceInStream(); 
    });

    let cameraStream = null;
    let isSessionActive = false;
    
    document
    .getElementById("startSessionBtn")
    .addEventListener("click", function () {
        const videoElement = document.getElementById("camera-stream");
        
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices
                .getUserMedia({ video: true })
                .then(function (stream) {
                    cameraStream = stream;
                    videoElement.srcObject = stream;
                    videoElement.play();
                    
                    console.log("Camera stream started");
                    isSessionActive = true;
                    detectFaceInStream(); // Start face detection
                })
                .catch(function (error) {
                    console.error("Error accessing the camera: ", error);
                });
            } else {
                alert("Your browser does not support camera access.");
        }
    });

    // End Session: Stop the camera feed
    document
    .getElementById("endSessionBtn")
    .addEventListener("click", function () {
            const videoElement = document.getElementById("camera-stream");
            
            isSessionActive = false;
            if (cameraStream) {
                let tracks = cameraStream.getTracks();
                tracks.forEach((track) => track.stop());
                videoElement.srcObject = null;
            }
        });

        const currentUrl = window.location.href;
    const url = new URL(currentUrl);
    const subject = url.searchParams.get("subject");
    const selectedClass = url.searchParams.get("class");
    
    // Capture video frames and send them for face detection
    function detectFaceInStream() {
        const videoElement = document.getElementById("camera-stream");
        const canvas = document.getElementById('overlayCanvas');
        const context = canvas.getContext('2d');
    
        console.log(canvas); // Check if the canvas is correctly obtained
        console.log(context);

        if (!canvas || !context) {
            console.error("Canvas or context is null");
            return; // Exit if canvas or context is null
        }
        console.log("canvas inputed")
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;    
        
        console.log("canvas issue fixed")
        function captureFrame() {
            // Ensure the video element has valid dimensions
            if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
                console.error("Video dimensions are zero. Waiting for video to load.");
                // requestAnimationFrame(captureFrame); // Try again
                return;
            }
    
            // Draw the video frame to the canvas
            context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    
            // Get the base64 image data
            const imageData = canvas.toDataURL("image/jpeg");
    
            // Check if the image data is valid
            if (!imageData || imageData.length < 100) {
                console.error("Captured image data is empty or invalid");
                return; // Exit if image data is not valid
            }
    
            // console.log("Captured image data: ", imageData); // Log the image data for debugging
    
            // Prepare to send the image data to the server
            const currentUrl = window.location.href;
            const url = new URL(currentUrl);
            const subject = url.searchParams.get("subject");
            const selectedClass = url.searchParams.get("class");
    
            // Send the image to the server for face detection
            setTimeout(() => {
                fetch("/detect_face", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({
                        imageData: imageData,
                        subject: subject, // Subject retrieved from the URL
                    }),
                })
                .then(response => response.json())
                .then(data => {
                    if (data.faceDetected) {
                        console.log("Face detected, bounding box:", data.bbox);
                        drawBoundingBox(data.bbox);
                        saveImage(imageData, data.bbox);
                    } else {
                        console.log("No face detected.");
                        drawBoundingBox(null);
                    }
                })
                .catch((error) => {
                    console.error("Server error:", error);
                });
        
                // Capture the next frame
                requestAnimationFrame(captureFrame);
            }, 2000);
            
        }

        if (isSessionActive==true) {
            console.log(`Video dimensions: ${videoElement.videoWidth} x ${videoElement.videoHeight}`);
            captureFrame();
        } else {
            return;
        }
    }

    // Draw the bounding box
	function drawBoundingBox(bbox) {
        const canvas = document.getElementById('overlayCanvas');
        if (!canvas) {
            console.error('Canvas overlay not found');
            return;
        }
        
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height); // Clear previous drawings
        context.beginPath();
        context.rect(bbox[0], bbox[1], bbox[2], bbox[3]); // bbox = [x, y, w, h]
        context.lineWidth = 3;
        context.strokeStyle = 'green'; // Change color as per requirement
        context.stroke();
    }

    function saveImage(imageData, bbox) {
        const img = new Image();
        img.onload = function() {
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
    
            // Set canvas dimensions to the bounding box size
            canvas.width = bbox[2]; // width from bbox
            canvas.height = bbox[3]; // height from bbox
    
            // Draw the cropped portion onto the new canvas
            context.drawImage(img, bbox[0], bbox[1], bbox[2], bbox[3], 0, 0, bbox[2], bbox[3]);
    
            // Convert cropped canvas to data URL
            const croppedImageData = canvas.toDataURL("image/jpeg");
    
            fetch("/save_image", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ imageData: croppedImageData, subject: subject, selectedClass: selectedClass }),
            })
            .then((response) => response.json())
            .then((data) => {
                console.log("Cropped image saved at:", data.imagePath);
            })
            .catch((error) => {
                console.error("Error saving image:", error);
            });
        };
        
        img.src = imageData; // Set the source to trigger onload
    }

    document
        .getElementById("modelProceedBtn")
        .addEventListener("click", function () {
            console.log("workinggggggg");

            const currentUrl = window.location.href;

            const url = new URL(currentUrl);

            const subject = url.searchParams.get("subject");
            const selectedClass = url.searchParams.get("class");
            CLAAAAAS = selectedClass;
            console.log(selectedClass);

            if (!selectedClass) {
                alert("Please select a class before proceeding.");
                return;
            }
            document.querySelector(".model-processingbar").style.display =
                "block";
            document.querySelector(".model-processingbar").textContent =
                "Processing...";

            fetch("/start-attendance", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    SUB: subject,
                    CLASS_NAME: selectedClass,
                }),
            })
                .then((response) => response.json())
                .then((result) => {
                    if (result.success) {
                        console.log(
                            "Attendance completed successfully",
                            result.output
                        );
                        startProgressBar();
                    } else {
                        console.error(
                            "Error starting attendance:",
                            result.error
                        );
                    }
                })
                .catch((error) => {
                    console.error("Error:", error);
                });
        });

    function startProgressBar() {
        const progressBar = document.querySelector(".model-processingbar");
        progressBar.style.display = "block";

        let progress = 0;
        const interval = setInterval(() => {
            if (progress < 100) {
                progress += 10;
                progressBar.innerHTML = `${progress}%`;
            } else {
                clearInterval(interval);
                progressBar.style.display = "none";
                document.querySelector(".overview").style.display = "block";
            }
        }, 200);
    }

    document
        .getElementById("overviewBtn")
        .addEventListener("click", function () {
            fetch("/get-overview-data", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    CLASS_SSS: CLAAAAAS,
                }),
            })
                .then((response) => {
                    if (!response.ok) {
                        throw new Error("Network response was not ok");
                    }
                    return response.json();
                })
                .then((data) => {
                    const totalFolders = data.total;
                    const presentImages = data.present;
                    const absentCount = totalFolders - presentImages;

                    const overviewTable =
                        document.querySelector(".overview-table");
                    overviewTable.style.display = "block";

                    const totalCell = overviewTable.querySelector(
                        "tr:nth-child(1) td:nth-child(2)"
                    );
                    const presentCell = overviewTable.querySelector(
                        "tr:nth-child(2) td:nth-child(2)"
                    );
                    const absentCell = overviewTable.querySelector(
                        "tr:nth-child(3) td:nth-child(2)"
                    );

                    totalCell.textContent = totalFolders;
                    presentCell.textContent = presentImages;
                    absentCell.textContent = absentCount;

                    document.querySelector(
                        ".model-processingbar"
                    ).style.display = "none";
                })
                .catch((error) => {
                    console.error("Error fetching overview data:", error);
                });
        });

});
