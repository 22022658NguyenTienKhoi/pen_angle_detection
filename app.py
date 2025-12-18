import cv2
import numpy as np
from flask import Flask, render_template, Response
from ultralytics import YOLO

app = Flask(__name__)

# 1. Load your model
model = YOLO("best.pt")

def process_frame(img):
    """Applies YOLOv8-seg, PCA logic, and styled visualization."""
    # Use stream=True for faster processing in real-time
    results = model(img, stream=True)
    
    # We create a separate overlay for the semi-transparent segments
    overlay = img.copy()

    for result in results:
        if result.masks is None:
            continue

        for mask_data in result.masks.data:
            # 2. Prepare Mask
            mask = mask_data.cpu().numpy()
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            binary_mask = (mask > 0.5).astype(np.uint8) * 255

            # 3. Add Green Segment Overlay
            # This fills the pen area with a solid green color on the overlay
            overlay[binary_mask > 0] = [0, 255, 0] 

            # 4. PCA for Orientation
            ys, xs = np.where(binary_mask > 0)
            if len(xs) < 50: continue 
            
            pts = np.stack([xs, ys], axis=1).astype(np.float32)
            mean = np.mean(pts, axis=0)
            pts_centered = pts - mean
            
            cov = np.cov(pts_centered.T)
            eigvals, eigvecs = np.linalg.eig(cov)
            principal_axis = eigvecs[:, np.argmax(eigvals)]
            
            # Angle Calculation
            angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
            angle_deg = np.degrees(angle_rad) % 180

            # 5. Draw Angle Text (Smaller and Green)
            # fontScale=0.5 (smaller), color=(0, 255, 0) (green)
            p1 = (int(mean[0]), int(mean[1]))
            cv2.putText(
                img, 
                f"{angle_deg:.1f} deg", 
                (p1[0] + 15, p1[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5,           # Smaller font scale
                (0, 255, 0),   # Green color (BGR)
                1,             # Thin thickness for smaller text
                cv2.LINE_AA
            )

    # 6. Apply Transparency (0.3 = 30% green segment visibility)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            
    return img

def gen_frames():
    cap = cv2.VideoCapture(1)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            processed_frame = process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)