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


def draw_text_with_background(img, text, pos, font_scale=0.7, color=(255, 255, 255), thickness=2):
    """Draw text with dark background for better visibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # Draw dark background rectangle
    cv2.rectangle(img, (x - 5, y - text_h - 5), (x + text_w + 5, y + baseline + 5), (0, 0, 0), -1)
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


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
    draw_coordinate_system(img, origin=(0, 0))

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

            # 3. Apply morphological erosion to tighten the mask border
            kernel = np.ones((5, 5), np.uint8)
            tight_mask = cv2.erode(binary_mask, kernel, iterations=2)

            # Draw Segmentation Contour (edge border around pen)
            contours, _ = cv2.findContours(tight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)

                # Smooth the contour using approxPolyDP for cleaner edges
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                smooth_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

                # Draw the contour edge border (cyan color, thickness 2)
                cv2.drawContours(img, [smooth_contour], -1, (255, 255, 0), 2)

                # Calculate centroid using moments
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Draw center point with coordinates (magenta)
                    cv2.circle(img, (cx, cy), 6, (255, 0, 255), -1)
                    draw_text_with_background(img, f"Center({cx},{cy})", (cx + 15, cy + 5),
                                              font_scale=0.6, color=(255, 0, 255), thickness=2)

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
                    tip_direction = principal_axis  # Point toward max end

                # Draw direction arrow from center toward tip
                arrow_length = 60
                arrow_start = (int(mean[0]), int(mean[1]))
                arrow_end = (int(mean[0] + tip_direction[0] * arrow_length),
                             int(mean[1] + tip_direction[1] * arrow_length))

                cv2.arrowedLine(img, arrow_start, arrow_end, (0, 165, 255), 3, tipLength=0.3)
                draw_text_with_background(img, "Tip", (arrow_end[0] + 10, arrow_end[1]),
                                          font_scale=0.6, color=(0, 165, 255), thickness=2)

            # 6. Draw Angle Text with background for visibility
            p1 = (int(mean[0]), int(mean[1]))
            draw_text_with_background(img, f"{angle_deg:.1f} deg", (p1[0] + 20, p1[1] - 20),
                                      font_scale=0.8, color=(0, 255, 255), thickness=2)

            # 7. Log detection with timestamp
            logging.info(f"Pen detected - Center: ({int(mean[0])}, {int(mean[1])}), Angle: {angle_deg:.1f} deg")

    # 6. Apply Transparency (0.3 = 30% green segment visibility)
    alpha = 0.3
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    return img


def gen_frames():
    global detection_interval
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Ví dụ: 1280, 1920
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Ví dụ: 720, 1080
    last_detection_time = 0
    last_processed_frame = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()

        # Check if enough time has passed since last detection
        if detection_interval == 0 or (current_time - last_detection_time) >= detection_interval:
            processed_frame = process_frame(frame)
            last_processed_frame = processed_frame.copy()
            last_detection_time = current_time
        else:
            # Keep displaying the previous detection result
            if last_processed_frame is not None:
                processed_frame = last_processed_frame
            else:
                processed_frame = frame.copy()
                draw_coordinate_system(processed_frame, origin=(0, 0))

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