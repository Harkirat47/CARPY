import torch
import cv2
import torchvision.transforms as T
import numpy as np

# Load model
model = torch.jit.load("best.torchscript").eval()

# Constants
NUM_CLASSES = 14  # change if more
IMG_SIZE = 640
CONF_THRESH = 0.3
IOU_THRESH = 0.4

# Preprocessing
transform = T.Compose([
    T.ToTensor()
])

# Decode predictions
def decode_output(output):
    boxes = []
    for i in range(output.shape[0]):
        pred = output[i]  # [8400]
        pred = pred.reshape(-1, 5 + NUM_CLASSES)  # [8400, 19]

        # Apply sigmoid to obj and class confs
        pred[:, 4:] = torch.sigmoid(pred[:, 4:])

        for row in pred:
            x, y, w, h = row[:4]
            obj_conf = row[4]
            class_confs = row[5:]
            class_id = torch.argmax(class_confs)
            class_score = class_confs[class_id]

            final_conf = obj_conf * class_score
            if final_conf > CONF_THRESH:
                # Convert (x, y, w, h) to (x1, y1, x2, y2)
                cx, cy, w, h = x, y, w, h
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                boxes.append([x1, y1, x2, y2, final_conf.item(), int(class_id.item())])
    return boxes

# Run NMS
def nms(boxes, iou_threshold):
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    final_boxes = []
    while boxes:
        chosen = boxes.pop(0)
        boxes = [b for b in boxes if iou(chosen, b) < iou_threshold]
        final_boxes.append(chosen)
    return final_boxes

# IoU helper
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

# Video feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        boxes = decode_output(output)
        boxes = nms(boxes, IOU_THRESH)

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        # Scale to original frame size
        x1 = int(x1 / IMG_SIZE * frame.shape[1])
        x2 = int(x2 / IMG_SIZE * frame.shape[1])
        y1 = int(y1 / IMG_SIZE * frame.shape[0])
        y2 = int(y2 / IMG_SIZE * frame.shape[0])

        label = f"Class {cls}: {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv12 - Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
