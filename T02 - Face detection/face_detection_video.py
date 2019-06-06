import cv2
 
scale_factor = 1.2
min_neighbors = 3
min_size = (50, 50)
webcam=False
 
def detect(path):
 
    cascade = cv2.CascadeClassifier(path)
    
    if webcam:
        video_cap = cv2.VideoCapture(1)
    else:
        video_cap = cv2.VideoCapture("/home/john/Downloads/john.mp4")
    while True:
        
        ret, img = video_cap.read()
 
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
        rects = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)
        
        if len(rects) >= 0:
            
            for (x, y, w, h) in rects:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
            
            cv2.imshow('Face Detection on Video', img)
            
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
    video_cap.release()
 
def main():
    
    cascadeFilePath="/home/john/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"
    detect(cascadeFilePath)
 
 
if __name__ == "__main__":
    main()