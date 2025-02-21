 ARGUS: Face Recognition Alert System

ARGUS is a system for real-time face recognition with alerts via WhatsApp, useful in surveillance or tracking specific individuals.
 Features
- Detects faces from a live video feed
- Sends alerts with images via WhatsApp
- Simple Streamlit UI for input

 How to Set Up

1. Clone the repository:
   git clone https://github.com/baba-fried/ARGUS.git
   cd ARGUS
   
Install dependencies:
pip install -r requirements.txt

Download Large Files (e.g., shape predictor):

Download shape_predictor_68_face_landmarks.dat and place it in the root folder.

Run the app:
streamlit run app.py

Input Details in the UI:

Enter phone number and name to detect.
Project Structure
app.py # Main app ImagesAttendance/ # Folder with images for recognition requirements.txt # Dependencies shape_predictor_68_face_landmarks.dat # Model for landmarks README.md # Project info

Requirements
Python 3.7+
OpenCV
face_recognition
pywhatkit
Streamlit







