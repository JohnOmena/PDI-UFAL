import cv2 as cv

face_cascade = cv.CascadeClassifier('/home/john/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('/home/john/anaconda3/share/OpenCV/haarcascades/haarcascade_eye.xml')


img = cv.imread('jenny.jpg')
img01 = cv.imread('time.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray01 = cv.cvtColor(img01, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 5)
faces01 = face_cascade.detectMultiScale(gray01, 1.1, 5)

for (x,y,w,h) in faces:
    
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

for (x,y,w,h) in faces01:

    cv.rectangle(img01,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray01[y:y+h, x:x+w]
    roi_color = img01[y:y+h, x:x+w]
    
    

cv.imshow('img01',img01)
cv.imshow('img',img)


