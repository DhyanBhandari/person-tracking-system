# create virtual environment
python -m venv venv
# activate 
.\venv\Scripts\Activate
# Set excution policy
Set-ExecutionPolicy RemoteSigned -Scope Process
# install the required libraries
pip install -r requirements.txt
 
 # 👤 Person Tracking System with Child/Adult Classification 🚸

This project is a real-time person detection and tracking system using OpenCV and a pre-trained MobileNet SSD model. The system identifies and tracks individuals in a video feed, classifying them as either a "Child" or an "Adult" based on their size in relation to the frame.

## 📂 Features
- Real-time person detection using **MobileNet SSD**.
- Non-Maximum Suppression to reduce overlapping bounding boxes.
- **Centroid Tracking** to track people across frames and assign unique IDs.
- Classification of people as **Child** or **Adult** based on bounding box height.
- Displays **Frames Per Second (FPS)** for performance monitoring.

## 🚀 How It Works
1. The video is processed frame-by-frame to detect objects using the **MobileNet SSD** model.
2. Detected bounding boxes are refined using **Non-Maximum Suppression** to remove overlaps.
3. A **Centroid Tracker** is used to assign unique IDs to detected persons and keep track of them over multiple frames.
4. The height of each detected person is compared with the frame height to classify them as a "Child" or "Adult".
5. The system displays the **ID** and **classification** for each tracked individual on the video.

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/DhyanBhandari/person-tracking-system.git

Install the required dependencies:
bash
Copy code
pip install -r requirements.txt

📦 Requirements
Python 3.7+
OpenCV
NumPy
Imutils

🏃‍♂️ Running the Project

Make sure you have the necessary model files:

MobileNetSSD_deploy.prototxt
MobileNetSSD_deploy.caffemodel
You can download them from here.

Run the program:

bash
"""
python track.py
"""

📑 Classifying Child vs Adult
The system uses the height of the bounding box in relation to the frame height. If a person's bounding box is less than 40% of the total frame height, they are classified as a Child; otherwise, they are classified as an Adult.

🎯 Future Improvements
Integrating Re-Identification (ReID) models for more accurate person tracking.
Allowing saving of labeled video output.
Enhancing the classification model with additional features like posture analysis or distance-based calibration.

🖼️ Example Output
The following image shows how the system labels detected persons with unique IDs and classifies them as either a "Child" or an "Adult":