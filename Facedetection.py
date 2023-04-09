import cv2
from deepface import DeepFace
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    cap=cv2.VideoCapture(1)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
while True:
    ret,frame=cap.read()
    result=DeepFace.analyze(frame,actions=['emotion','age','gender','race'],enforce_detection=True)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.1,4)
    font=cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(frame,(result[0]['region']['x'],result[0]['region']['y']),(result[0]['region']['x']+result[0]['region']['w'],result[0]['region']['y']+result[0]['region']['h']),(0,255,0),2)
    cv2.putText(frame,
                result[0]['dominant_emotion'], # type: ignore
                (50,50),
                font, 3,
                (0,10,255),
                2,
                cv2.LINE_4)
    cv2.putText(frame,
                result[0]['dominant_gender'], # type: ignore
                (150,100),
                font, 1,
                (0,234,255),
                2,
                cv2.LINE_4)
    cv2.imshow('Demo video',frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()