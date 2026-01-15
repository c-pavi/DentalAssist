import cv2
import mediapipe
import time
from collections import deque
from math import hypot

# Initialize MediaPipe Hands + Face Detection
capture_hands = mediapipe.solutions.hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)
capture_face = mediapipe.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6,
)
drawing_option = mediapipe.solutions.drawing_utils

# Open camera (try 0 or 1)
camera = cv2.VideoCapture(1)

PRE_COUNTDOWN_SECONDS = 3
BRUSH_DURATION_SECONDS = 120
NOTIFY_INTERVAL_SECONDS = 30
WARNING_COOLDOWN_SECONDS = 2

state = "idle"  # idle -> pre_countdown -> brushing -> done
pre_countdown_start = None
brush_start_time = None
last_notify_time = None
notified_markers = set()
last_warning_time = 0.0
warning_message = ""
warning_message_until = 0.0

motion_points = deque(maxlen=12)

while True:
    ret, image = camera.read()
    if not ret:
        continue

    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    output_hands = capture_hands.process(rgb_image)
    output_faces = capture_face.process(rgb_image)
    all_hands = output_hands.multi_hand_landmarks
    face_detections = output_faces.detections if output_faces else None

    now = time.time()
    hand_tip = None
    face_center = None
    face_size = None

    if face_detections:
        detection = face_detections[0]
        bbox = detection.location_data.relative_bounding_box
        keypoints = detection.location_data.relative_keypoints
        if len(keypoints) >= 3:
            face_center = (keypoints[2].x, keypoints[2].y)  # nose tip
        else:
            face_center = (bbox.xmin + bbox.width / 2, bbox.ymin + bbox.height / 2)
        face_size = max(bbox.width, bbox.height)

    if all_hands:
        for hand in all_hands:
            index_finger_tip = hand.landmark[8]
            hand_tip = (index_finger_tip.x, index_finger_tip.y)
            break

    hand_near_face = False
    if hand_tip and face_center and face_size:
        distance = hypot(hand_tip[0] - face_center[0], hand_tip[1] - face_center[1])
        hand_near_face = distance < (face_size * 1.2)

    if hand_tip:
        x_px = int(hand_tip[0] * image.shape[1])
        y_px = int(hand_tip[1] * image.shape[0])
        motion_points.append((now, x_px, y_px))
    else:
        motion_points.clear()

    if state == "idle":
        if face_center and hand_near_face:
            state = "pre_countdown"
            pre_countdown_start = now
    elif state == "pre_countdown":
        if not (face_center and hand_near_face):
            state = "idle"
            pre_countdown_start = None
        elif now - pre_countdown_start >= PRE_COUNTDOWN_SECONDS:
            state = "brushing"
            brush_start_time = now
            last_notify_time = now
            notified_markers = set()
    elif state == "brushing":
        elapsed = now - brush_start_time
        if elapsed >= BRUSH_DURATION_SECONDS:
            state = "done"
        else:
            for marker, message in (
                (30, "Change position"),
                (60, "Change sides"),
                (90, "Change position"),
            ):
                if elapsed >= marker and marker not in notified_markers:
                    notified_markers.add(marker)
                    warning_message = message
                    warning_message_until = now + 2.0
                    break

            if len(motion_points) >= 6 and now - last_warning_time >= WARNING_COOLDOWN_SECONDS:
                xs = [p[1] for p in motion_points]
                ys = [p[2] for p in motion_points]
                x_range = max(xs) - min(xs)
                y_range = max(ys) - min(ys)
                dx_total = 0.0
                dy_total = 0.0
                t_start = motion_points[0][0]
                t_end = motion_points[-1][0]
                for i in range(1, len(motion_points)):
                    dx_total += abs(motion_points[i][1] - motion_points[i - 1][1])
                    dy_total += abs(motion_points[i][2] - motion_points[i - 1][2])
                time_span = max(t_end - t_start, 0.001)
                horizontal_speed = dx_total / time_span

                is_vigorous_lr = (
                    x_range > 40
                    and x_range > y_range * 1.4
                    and horizontal_speed > 80
                )
                if is_vigorous_lr:
                    warning_message = "Brush in small circles"
                    warning_message_until = now + 2.0
                    last_warning_time = now
    elif state == "done":
        pass

    status_text = ""
    if state == "idle":
        status_text = "Bring hand to face to start"
    elif state == "pre_countdown":
        remaining = max(0, int(PRE_COUNTDOWN_SECONDS - (now - pre_countdown_start)) + 1)
        status_text = f"Starting in {remaining}"
    elif state == "brushing":
        remaining = max(0, int(BRUSH_DURATION_SECONDS - (now - brush_start_time)))
        status_text = f"Brushing... {remaining}s left"
    elif state == "done":
        status_text = "All done!"

    cv2.putText(
        image,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    if now < warning_message_until:
        cv2.putText(
            image,
            warning_message,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Brushing Monitor", image)

    if cv2.waitKey(100) & 0xFF == 27:  # ESC to exit
        break

camera.release()
cv2.destroyAllWindows()
