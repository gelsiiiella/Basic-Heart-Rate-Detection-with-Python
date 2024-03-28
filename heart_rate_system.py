import cv2
import math
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

heart_rate = 70
last_color_signals = 0
time_elapsed = 0

def extract_forehead_roi(frame,face):
    x,y,w,h = face

    forehead_x = x + w // 4
    forehead_y = y + h // 10
    forehead_width = w // 2
    forehead_height = h // 5

    forehead_roi = frame[forehead_y: forehead_y + forehead_height, forehead_x: forehead_x + forehead_width]

    return forehead_roi

def capture_color_signals(roi):
    gray_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    main_intensity = int(gray_roi.mean())

    return main_intensity

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        forehead_roi = extract_forehead_roi (frame,(x,y,w,h))
        color_signals = capture_color_signals(forehead_roi)
        time_elapsed += 1
        heart_rate = int(70 + 10 * (1 + math.sin(time_elapsed/10)))

        last_color_signals = color_signals

        print("Simulated Heart Rate (BPM): ", heart_rate)

        cv2.imshow('Face Detection', forehead_roi)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break



cap.release()
cv2.destroyAllWindows()