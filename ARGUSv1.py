import cv2
import numpy as np
import face_recognition
import pywhatkit as kit
import time
import os
import threading
import streamlit as st

#img folder
path = r'C:\Users\Neeraj\Desktop\ImagesAttendance'


st.title('Face Recognition Alert System')
st.sidebar.header('User Input')

#i/p
phone_number = st.sidebar.text_input('Enter Phone Number (with country code)')
name_to_detect = st.sidebar.text_input('Enter Name to Detect')

#o/p
st.write(f"Phone Number: {phone_number}")
st.write(f"Name to Detect: {name_to_detect}")


if st.sidebar.button('Search'):
    if phone_number and name_to_detect:
        st.write("Starting live feed...")
        
        images = []
        classNames = []
        if os.path.exists(path):
            myList = os.listdir(path)
            for cl in myList:
                curImg = cv2.imread(f'{path}/{cl}')
                images.append(curImg)
                classNames.append(os.path.splitext(cl)[0])
        else:
            st.error("The provided image path doesn't exist!")

#encoding fucntion
        def findEncodings(images):
            encodeList = []
            for img in images:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encodeList.append(encode)
            return encodeList

#WA function
        def send_alert_with_image(phone_number, message, img_path):
            try:
            
                time.sleep(5)
                kit.sendwhats_image(phone_number, img_path, message)
                time.sleep(10) 
            except Exception as e:
                print(f"Error while sending message: {e}")

#initialize encoded faces
        encodeListKnown = findEncodings(images)
        st.write('Encoding Complete')

        camera_areas = {
            0: "My PC Webcam",
        }

#live feed function
        def process_camera(camera_id, camera_name):
            cap = cv2.VideoCapture(camera_id)
            if not cap.isOpened():
                st.warning(f"Warning: {camera_name} could not be opened. Skipping this camera.")
                return

            alert_sent = set()
            last_seen = {}

            while True:
                success, img = cap.read()
                if not success:
                    st.warning(f"Failed to capture image from {camera_name}. Continuing...")
                    break

#resize and color space conversion
                imgS = cv2.resize(img, (0, 0), None, fx=0.25, fy=0.25)
                imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
                faceCurFrame = face_recognition.face_locations(imgS)
                encodesCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

                current_frame_faces = set()

                for encodeFace, faceLoc in zip(encodesCurFrame, faceCurFrame):
                    matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                    faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
                    matchIndex = np.argmin(faceDis)

                    if matches[matchIndex]:
                        name = classNames[matchIndex].upper()

#alert on webpage
                        if name.lower() == name_to_detect.lower() and name not in alert_sent:
                            st.write(f"{name} detected on {camera_name}!")

                            timestamp = time.strftime("%d_%m_%Y_%H_%M_%S")
                            screenshot_path = f"{name}_detected_{camera_name}_{timestamp}.jpg"
                            cv2.imwrite(screenshot_path, img)

#importing date and time
                            current_datetime = time.localtime()
                            formatted_date = time.strftime("%d-%m-%Y", current_datetime)
                            formatted_time = time.strftime("%H:%M:%S", current_datetime)

#message alert
                            message = f"Alert: {name} detected on {camera_name} on {formatted_date} at {formatted_time}"
                            alert_thread = threading.Thread(target=send_alert_with_image, args=(phone_number, message, screenshot_path))
                            alert_thread.start()
                            alert_sent.add(name)

#face box and name
                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

                        # Update last seen time for the detected face
                        last_seen[name] = time.time()
                        current_frame_faces.add(name)

                for name in list(last_seen.keys()):
                    if name not in current_frame_faces and (time.time() - last_seen[name] > 5): 
                        if name in alert_sent:
                            alert_sent.remove(name)

                cv2.imshow(camera_name, img)

                if cv2.waitKey(1) == 27: 
                    break

            cap.release()
            cv2.destroyAllWindows()


        threads = []
        for idx, camera_name in camera_areas.items():
            thread = threading.Thread(target=process_camera, args=(idx, camera_name))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
    else:
        st.error("Please provide both phone number and name to detect.")
