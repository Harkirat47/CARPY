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
        self.criterion = nn.MSELoss()
        self.optimizer = None 

        def render_only_final(self, frame, detections, weights):
            out = frame.copy()
            w_yolo, w_bridge, w_final = weights[0].tolist()

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                conf = det["confidence"]
                cls = det["class"]
                input_vec = torch.tensor([x1, y1, x2, y2, conf, cls], dtype=torch.float32).unsqueeze(0)

                final_box = [int(x) for x in self.final_model(input_vec).detach().numpy()[0]]
                name = self.class_names.get(cls, str(cls))
                fx1, fy1, fx2, fy2 = final_box

                green = int(255 * w_final)
                cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (0, green, 0), 2)
                cv2.putText(out, f"Final (Fast): {name}", (fx1, fy2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, green, 0), 1)

            cv2.putText(out, f"Weights -> YOLO: {w_yolo:.2f}  Bridge: {w_bridge:.2f}  Final: {w_final:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            return out



    def process_frame(self, frame):  # ← FIXED: this is now indented inside the class
        h, w, _ = frame.shape
        grid_size = 80
        grid_boxes = [[x, y, x + grid_size, y + grid_size]
                      for y in range(0, h - grid_size, grid_size)
                      for x in range(0, w - grid_size, grid_size)]

        detections = self.yolo.detect(frame)
        full_boxes = detections + [{"bbox": box, "confidence": 0.0, "class": -1} for box in grid_boxes]

        sequence, det_output, pipeline_display = [], [], frame.copy()

        for det in full_boxes:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cls = det["class"]
            input_vec = torch.tensor([x1, y1, x2, y2, conf, cls], dtype=torch.float32).unsqueeze(0)

            delta = self.bridge_model(input_vec).detach().numpy()[0]
            bridge_box = [x1 + delta[0], y1 + delta[1], x2 + delta[2], y2 + delta[3]]
            final_box = self.final_model(input_vec).detach().numpy()[0]
            final_box = [int(x) for x in final_box]

            sequence.extend([x1, y1, x2, y2, conf, cls, *bridge_box, *final_box])
            det_output.append({
                "bbox": [x1, y1, x2, y2],
                "bridge_box": bridge_box,
                "final_box": final_box,
                "class": cls,
                "confidence": conf
            })

            cv2.rectangle(pipeline_display, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # LSTM weight calculation
        if sequence:
            lstm_input = torch.tensor(sequence, dtype=torch.float32).reshape(1, 1, -1)
            if self.controller is None or self.current_input_size != len(sequence):
                self.controller = LSTMControllerModel(input_size=len(sequence))
                self.current_input_size = len(sequence)
                self.hidden = None
                self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=1e-3)

            weights, self.hidden = self.controller(lstm_input, self.hidden)

            # Detach hidden state
            if self.hidden is not None:
                self.hidden = tuple(h.detach() for h in self.hidden)

            loss = self.criterion(weights, torch.tensor([[0, 0, 1]], dtype=torch.float32))
            self.controller.train()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            w_yolo, w_bridge, w_final = weights[0].tolist()
        else:
            w_yolo, w_bridge, w_final = 1, 0, 0

        # Draw final fused output
        out = frame.copy()
        for det in det_output:
            x1, y1, x2, y2 = det["bbox"]
            bx1, by1, bx2, by2 = [int(v) for v in det["bridge_box"]]
            fx1, fy1, fx2, fy2 = det["final_box"]
            cls = det["class"]
            name = self.class_names.get(cls, str(cls)) if cls >= 0 else "Unknown"

            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(out, f"YOLO: {name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

            cv2.rectangle(out, (bx1, by1), (bx2, by2), (255, 100, 0), 1)
            cv2.putText(out, "Bridge", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)

            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (0, int(255 * w_final), 0), 2)
            cv2.putText(out, "Final", (fx1, fy2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.putText(out, f"Weights → YOLO: {w_yolo:.2f} | Bridge: {w_bridge:.2f} | Final: {w_final:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("Fusion Output", out)
        cv2.imshow("Grid View", pipeline_display)
        cv2.imshow("Raw Input", frame)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    ful = LSTMController()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ful.process_frame(frame)


        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
