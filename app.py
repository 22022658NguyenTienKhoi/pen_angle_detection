import cv2
import numpy as np
import logging
import time
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# Configurable detection interval (in seconds)
detection_interval = 0.0  # 0 means process every frame

# Setup logging
logging.basicConfig(
    filename='detections.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 1. Load your model
model = YOLO("best.pt")

def draw_coordinate_system(img, origin=(50, 50), axis_length=80, thickness=2):
    """Draws xOy reference coordinate system on the frame."""
    ox, oy = origin

    # Draw X axis (horizontal, pointing right) - Red
    cv2.arrowedLine(img, (ox, oy), (ox + axis_length, oy), (0, 0, 255), thickness, tipLength=0.15)
    cv2.putText(img, "X", (ox + axis_length + 5, oy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # Draw Y axis (vertical, pointing down) - Green
    cv2.arrowedLine(img, (ox, oy), (ox, oy + axis_length), (0, 255, 0), thickness, tipLength=0.15)
    cv2.putText(img, "Y", (ox - 5, oy + axis_length + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw origin point and label
    cv2.circle(img, (ox, oy), 4, (255, 255, 255), -1)
    cv2.putText(img, "O", (ox - 15, oy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img

def process_frame(img):
    """Applies YOLOv8-seg, PCA logic, and styled visualization."""
    # Use stream=True for faster processing in real-time
    results = model(img, stream=True)

    # Draw coordinate reference system at top left
    draw_coordinate_system(img, origin=(50, 50))

    # We create a separate overlay for the semi-transparent segments
    overlay = img.copy()

    for result in results:
        if result.masks is None:
            continue

        # Get bounding boxes if available
        boxes = result.boxes if result.boxes is not None else []

        for idx, mask_data in enumerate(result.masks.data):
            # 2. Prepare Mask
            mask = mask_data.cpu().numpy()
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            binary_mask = (mask > 0.5).astype(np.uint8) * 255

            # 3. Draw Bounding Box and Coordinates
            if idx < len(boxes):
                box = boxes[idx]
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                # Draw bounding box rectangle (cyan)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

                # Draw corner coordinate markers (white)
                corner_size = 5
                # Top-left corner
                cv2.circle(img, (x1, y1), corner_size, (255, 255, 255), -1)
                cv2.putText(img, f"({x1},{y1})", (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                # Top-right corner
                cv2.circle(img, (x2, y1), corner_size, (255, 255, 255), -1)
                cv2.putText(img, f"({x2},{y1})", (x2 - 60, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                # Bottom-left corner
                cv2.circle(img, (x1, y2), corner_size, (255, 255, 255), -1)
                cv2.putText(img, f"({x1},{y2})", (x1 + 5, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                # Bottom-right corner
                cv2.circle(img, (x2, y2), corner_size, (255, 255, 255), -1)
                cv2.putText(img, f"({x2},{y2})", (x2 - 60, y2 + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                # Draw center point with coordinates (magenta)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), -1)
                cv2.putText(img, f"Center({cx},{cy})", (cx + 10, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)

            # 4. Add Segment Overlay (light blue)
            overlay[binary_mask > 0] = [255, 200, 100] 

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

            # 5. Detect pen tip direction (narrower end = tip)
            # Get secondary axis (perpendicular to principal)
            secondary_axis = eigvecs[:, np.argmin(eigvals)]

            # Project points onto principal axis to find endpoints
            projections = pts_centered @ principal_axis

            # Get points near each end (within 15% of total length)
            proj_range = projections.max() - projections.min()
            threshold = proj_range * 0.15

            # Points near the "min" end
            min_end_mask = projections < (projections.min() + threshold)
            min_end_pts = pts_centered[min_end_mask]

            # Points near the "max" end
            max_end_mask = projections > (projections.max() - threshold)
            max_end_pts = pts_centered[max_end_mask]

            # Measure width at each end (spread along secondary axis)
            if len(min_end_pts) > 0 and len(max_end_pts) > 0:
                min_end_width = np.std(min_end_pts @ secondary_axis)
                max_end_width = np.std(max_end_pts @ secondary_axis)

                # Tip is the narrower end - direction points toward tip
                if min_end_width < max_end_width:
                    tip_direction = -principal_axis  # Point toward min end
                else:
                    tip_direction = principal_axis   # Point toward max end

                # Draw direction arrow from center toward tip
                arrow_length = 60
                arrow_start = (int(mean[0]), int(mean[1]))
                arrow_end = (int(mean[0] + tip_direction[0] * arrow_length),
                            int(mean[1] + tip_direction[1] * arrow_length))

                cv2.arrowedLine(img, arrow_start, arrow_end, (0, 165, 255), 2, tipLength=0.3)
                cv2.putText(img, "Tip", (arrow_end[0] + 5, arrow_end[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)

            # 6. Draw Angle Text (white for visibility)
            p1 = (int(mean[0]), int(mean[1]))
            cv2.putText(
                img,
                f"{angle_deg:.1f} deg",
                (p1[0] + 15, p1[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),  # White color
                1,
                cv2.LINE_AA
            )

            # 7. Log detection with timestamp
            logging.info(f"Pen detected - Center: ({int(mean[0])}, {int(mean[1])}), Angle: {angle_deg:.1f} deg")

    # 6. Apply Transparency (0.3 = 30% green segment visibility)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
            
    return img

def gen_frames():
    global detection_interval
    cap = cv2.VideoCapture(1)
    last_detection_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()

        # Check if enough time has passed since last detection
        if detection_interval == 0 or (current_time - last_detection_time) >= detection_interval:
            processed_frame = process_frame(frame)
            last_detection_time = current_time
        else:
            # Show raw frame with coordinate system only (no detection)
            processed_frame = frame.copy()
            draw_coordinate_system(processed_frame, origin=(50, 50))

        ret, buffer = cv2.imencode('.jpg', processed_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/interval', methods=['GET', 'POST'])
def set_interval():
    global detection_interval
    if request.method == 'POST':
        data = request.get_json()
        detection_interval = float(data.get('interval', 0))
        return jsonify({'success': True, 'interval': detection_interval})
    return jsonify({'interval': detection_interval})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)