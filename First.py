import torch
import cv2

# Load YOLOv12 model
model = torch.hub.load('ultralytics/yolov12', 'yolov12m', pretrained=True)
cap = cv2.VideoCapture(0)  # 0 is the default camera
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    results = model(frame)

    # Render results on the frame
    frame = results.render()[0]

    # Display the frame
    cv2.imshow('YOLOv12', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
