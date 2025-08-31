import cv2
video = cv2.VideoCapture(0)
while True:
    success, frame = video.read()
    if success:
        cv2.imshow("Camera Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
