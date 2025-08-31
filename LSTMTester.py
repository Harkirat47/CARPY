import cv2
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO

class BaseYOLOv12:
    def __init__(self):
        self.model = YOLO('yolo12s.pt')

    def detect(self, image):
        results = self.model(image)
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            detections.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": conf,
                "class": cls
            })
        return detections

class BridgeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.fc(x)

class FinalNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.fc(x)

class LSTMControllerModel(nn.Module):
    def __init__(self, input_size=28, hidden_size=64, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hx=None):
        out, hx = self.lstm(x, hx)
        weights = self.softmax(self.fc(out[:, -1]))
        return weights, hx

class LSTMController:
    def __init__(self):
        self.yolo = BaseYOLOv12()
        self.bridge_model = BridgeNN()
        self.final_model = FinalNN()
        self.controller = None
        self.current_input_size = None
        self.hidden = None
        self.class_names = self.yolo.model.names

    def process_frame(self, frame):
        out = frame.copy()
        detections = self.yolo.detect(frame)
        final_boxes, sequence = [], []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls = det["class"]
            input_vec = torch.tensor([x1, y1, x2, y2, conf, cls], dtype=torch.float32).unsqueeze(0)

            delta = self.bridge_model(input_vec).detach().numpy()[0]
            bridge_box = [x1 + delta[0], y1 + delta[1], x2 + delta[2], y2 + delta[3]]

            new_box = self.final_model(input_vec).detach().numpy()[0]
            final_box = [int(x) for x in new_box]

            det["bridge_box"] = bridge_box
            det["final_box"] = final_box
            final_boxes.append(det)

            sequence.extend([x1, y1, x2, y2, conf, cls, *bridge_box, *final_box])

        if sequence:
            input_size = len(sequence)
            lstm_input = torch.tensor(sequence, dtype=torch.float32).reshape(1, 1, -1)

            if self.controller is None or self.current_input_size != input_size:
                self.controller = LSTMControllerModel(input_size=input_size)
                self.current_input_size = input_size
                self.hidden = None  # reset hidden state if structure changes

            weights, self.hidden = self.controller(lstm_input, self.hidden)
            w_yolo, w_bridge, w_final = weights[0].tolist()
        else:
            w_yolo, w_bridge, w_final = 1, 0, 0

        for det in final_boxes:
            x1, y1, x2, y2 = det["bbox"]
            bx1, by1, bx2, by2 = [int(v) for v in det["bridge_box"]]
            fx1, fy1, fx2, fy2 = det["final_box"]
            cls = det["class"]
            conf = det["confidence"]
            name = self.class_names.get(cls, str(cls))

            # YOLO box
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(out, f"YOLO: {name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Bridge box
            cv2.rectangle(out, (bx1, by1), (bx2, by2), (255, 100, 0), 1)
            cv2.putText(out, f"Bridge", (bx1, by1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

            # Final box
            green = int(255 * w_final)
            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (0, green, 0), 2)
            cv2.putText(out, f"Final: {name}", (fx1, fy2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, green, 0), 1)

        cv2.putText(out, f"Weights -> YOLO: {w_yolo:.2f}  Bridge: {w_bridge:.2f}  Final: {w_final:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        return out

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    ful = LSTMController()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_frame = ful.process_frame(frame)
        cv2.imshow("YOLO + LSTM Fusion", out_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
