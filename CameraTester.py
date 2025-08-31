import cv2
import torch
from ultralytics import YOLO
from CameraCalibrator import CameraCalibrator

def main():
    # 1) Device auto-selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Tester] CUDA available? {torch.cuda.is_available()} → using {device}")

    # 2) Initialize YOLO
    yolo = YOLO("yolo12s.pt")   # update path if needed
    yolo.fuse()
    yolo.model.to(device)
    if device.startswith("cuda"):
        yolo.model.half()

    # 3) Initialize Calibrator
    calibrator = CameraCalibrator(device=device)

    # 4) Open webcam at lower resolution
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    LABELS = [
        "RGB Mean R", "RGB Mean G", "RGB Mean B",
        "RGB Std R",  "RGB Std G",  "RGB Std B",
        "Brightness",  "Contrast",   "Focus (Laplacian)",
        "Tenengrad",   "FFT Blur",   "Entropy",
        "Edge Magnitude", "Gamma",    "Skew"
    ]

    frame_idx  = 0
    SKIP       = 3   # run calibrator every 3 frames
    last_calib = None

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        frame_idx += 1

        # A) downsize for speed
        frame = cv2.resize(raw, (640, 480))

        # B) YOLO detection
        results = yolo(frame)
        dets = []
        for box in results[0].boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            dets.append({
                "bbox":       [x1,y1,x2,y2],
                "confidence": float(box.conf[0]),
                "class":      int(box.cls[0])
            })
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)

        # C) Calibrate every SKIP frames
        if frame_idx % SKIP == 0:
            last_calib = calibrator.predict(frame, dets)

        # D) Draw results
        if last_calib:
            pred  = last_calib["prediction"]
            tint  = last_calib["tint_hex"]
            blind = last_calib["blind_grid_coords"]

            # Draw first 5 metrics
            for i in range(min(5, len(pred))):
                cv2.putText(frame,
                            f"{LABELS[i]}: {pred[i]:.2f}",
                            (10, 25 + i*20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            # Draw tint
            cv2.putText(frame,
                        f"Tint: {tint}",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            # Highlight blind cells
            h,w = frame.shape[:2]
            gs  = 6
            gh,gw = h//gs, w//gs
            for i,j in blind:
                x1,y1 = j*gw, i*gh
                x2,y2 = x1+gw, y1+gh
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)

        cv2.imshow("Camera Tester", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
