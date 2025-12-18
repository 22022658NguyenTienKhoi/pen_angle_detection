import cv2
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# 1. Load your specific model
model = YOLO("best.pt")

def process_frame(img):
    """Applies YOLOv8-seg and PCA logic to a single frame."""
    results = model(img, stream=True)
    
    for result in results:
        if result.masks is None:
            continue

        # Process each detected pen
        for mask_data in result.masks.data:
            # Resize mask to match original image dimensions
            mask = mask_data.cpu().numpy()
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask = (mask > 0.5).astype(np.uint8) * 255

            # PCA for Orientation (Optimized approach)
            ys, xs = np.where(mask > 0)
            if len(xs) < 50: continue # Ignore noise
            
            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            mean = np.mean(pts, axis=0)
            pts_centered = pts - mean
            
            # Covariance and Eigen Decomposition
            cov = np.cov(pts_centered.T)
            eigvals, eigvecs = np.linalg.eig(cov)
            principal_axis = eigvecs[:, np.argmax(eigvals)]
            
            # Angle Calculation
            angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
            angle_deg = np.degrees(angle_rad) % 180

            # Visualization
            p1 = (int(mean[0]), int(mean[1]))
            p2 = (int(mean[0] + 100 * principal_axis[0]), 
                  int(mean[1] + 100 * principal_axis[1]))
            
            cv2.line(img, p1, p2, (0, 0, 255), 3)
            cv2.putText(img, f"{angle_deg:.1f} deg", (p1[0]+10, p1[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
    return img

def gen_frames():
    """Webcam video streaming generator."""
    cap = cv2.VideoCapture(1) # 0 is the default webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Apply our pen angle logic
            processed_frame = process_frame(frame)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            # Yield frame in a format Flask can stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)