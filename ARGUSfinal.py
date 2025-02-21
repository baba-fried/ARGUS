import cv2
import numpy as np
import face_recognition
import pywhatkit as kit
import time
import os
import threading
import streamlit as st

class ARGUS:
    def __init__(self, phone_number, name_to_detect, path):
        self.phone_number = phone_number
        self.name_to_detect = name_to_detect
        self.path = path
        self.images = []
        self.class_names = []
        self.encode_list_known = []
        self.alert_sent = set()
        self.last_seen = {}

    def load_images(self):
        if os.path.exists(self.path):
            my_list = os.listdir(self.path)
            for img_name in my_list:
                cur_img = cv2.imread(f'{self.path}/{img_name}')
                self.images.append(cur_img)
                self.class_names.append(os.path.splitext(img_name)[0])
        else:
            st.error("The provided image path doesn't exist!")

    def find_encodings(self):
        encode_list = []
        for img in self.images:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img_rgb)[0]
            encode_list.append(encode)
        self.encode_list_known = encode_list

    def send_alert_with_image(self, message, img_path):
        try:
            time.sleep(5)
            kit.sendwhats_image(self.phone_number, img_path, message)
            time.sleep(10)

        except Exception as e:
            st.error(f"Error while sending message: {e}")

    def process_camera(self, camera_id, camera_name="PC Webcam"):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            st.warning(f"Warning: {camera_name} could not be opened.")
            return

        while True:
            success, img = cap.read()
            if not success:
                st.warning(f"Failed to capture image from {camera_name}.")
                break

            self.recognize_faces(img, camera_name)
            
            #camera feed window
            cv2.imshow(camera_name, img)

            if cv2.waitKey(1) == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def recognize_faces(self, img, camera_name):
        img_small = cv2.resize(img, (0, 0), None, fx=0.25, fy=0.25)
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb)
        encodings_cur_frame = face_recognition.face_encodings(img_rgb, face_locations)

        for encode_face, face_loc in zip(encodings_cur_frame, face_locations):
            matches = face_recognition.compare_faces(self.encode_list_known, encode_face)
            face_dis = face_recognition.face_distance(self.encode_list_known, encode_face)
            match_index = np.argmin(face_dis)

            if matches[match_index]:
                name = self.class_names[match_index].upper()
                self.handle_match(name, img, camera_name, face_loc)

    def handle_match(self, name, img, camera_name, face_loc):
        if name.lower() == self.name_to_detect.lower() and name not in self.alert_sent:
            st.write(f"{name} detected on {camera_name}!")

            timestamp = time.strftime("%d_%m_%Y_%H_%M_%S")
            screenshot_path = os.path.join(os.getcwd(), f"{name}_detected_{camera_name}_{timestamp}.jpg")

            cv2.imwrite(screenshot_path, img)

            #alert message
            current_datetime = time.localtime()
            formatted_date = time.strftime("%d-%m-%Y", current_datetime)
            formatted_time = time.strftime("%H:%M:%S", current_datetime)
            message = f"Alert: {name} detected on {camera_name} on {formatted_date} at {formatted_time}"
            alert_thread = threading.Thread(target=self.send_alert_with_image, args=(message, screenshot_path))
            alert_thread.start()
            self.alert_sent.add(name)

        self.draw_face_box(img, name, face_loc)

    def draw_face_box(self, img, name, face_loc):
        y1, x2, y2, x1 = face_loc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


#Streamlit
st.title('ARGUS')
st.sidebar.header('User Input')

#input
phone_number = st.sidebar.text_input('Enter Phone Number (with country code)')
name_to_detect = st.sidebar.text_input('Enter Name to Detect')
path = r'C:\Users\Neeraj\Desktop\ImagesAttendance'

if st.sidebar.button('Search'):
    if phone_number and name_to_detect:
        st.write("Starting live feed...")

        #initialize
        face_recognition_system = ARGUS(phone_number, name_to_detect, path)
        face_recognition_system.load_images()
        face_recognition_system.find_encodings()
        st.write('Encoding Complete')

        #camera
        face_recognition_system.process_camera(0)

    else:
        st.error("Please provide both phone number and name to detect.")
