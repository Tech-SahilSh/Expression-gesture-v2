"""
Emotion-Responsive Robot Interaction
Author: (you)
Requirements:
  pip install opencv-python numpy fer
Run:
  python emotion_robot.py
"""

import cv2
import numpy as np
import math
from fer import FER   # FER ka pre-trained model use karega

# ---------- PARAMETERS ----------
CAM_W, CAM_H = 640, 480   # camera panel size
ROBOT_W, ROBOT_H = CAM_W, CAM_H
FPS_SLEEP = 1  # hum CPU pe framerate adjust kar sakte hain (if needed)

# Initialize camera and detector
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

detector = FER()   # default detector (works with OpenCV face detection backend)

# Helper: draw cute robot face on a canvas (emotion influences shapes)
def draw_robot_face(canvas, emotion='neutral', score=0.0, t=0):
    """
    canvas - numpy array (H x W x 3), will be modified in-place
    emotion - one of: 'happy','sad','angry','neutral','surprise'
    score - confidence (0..1)
    t - time/frame counter for animation
    """
    h, w = canvas.shape[:2]
    canvas[:] = (230, 240, 250)  # background (very light blue)

    # head
    cx, cy = w // 2, h // 2 - 20
    head_radius = min(w, h) // 3
    cv2.circle(canvas, (cx, cy), head_radius, (200, 220, 255), -1)       # fill head
    cv2.circle(canvas, (cx, cy), head_radius, (120, 140, 180), 4)        # outline

    # antenna
    ant_x = cx
    ant_y = cy - head_radius - 20
    cv2.line(canvas, (ant_x, ant_y), (ant_x, ant_y + 30), (120, 140, 180), 6)
    cv2.circle(canvas, (ant_x, ant_y), 10, (255, 120, 120), -1)
    # little shine on antenna
    cv2.circle(canvas, (ant_x+3, ant_y-2), 4, (255, 170, 170), -1)

    # cheeks (cute)
    cheek_off_x = 80
    cv2.circle(canvas, (cx - cheek_off_x, cy + 10), 18, (255, 190, 200), -1)
    cv2.circle(canvas, (cx + cheek_off_x, cy + 10), 18, (255, 190, 200), -1)

    # Eyes positions
    eye_y = cy - 30
    left_eye_x = cx - 60
    right_eye_x = cx + 60

    # Animation helpers
    blink = (t // 12) % 20  # controls blinking cycles
    pulse = 1.0 + 0.15 * math.sin(t * 0.2)  # for surprise mouth pulse

    # Default eye/pupil sizes
    eye_h = 26
    eye_w = 36
    pupil_r = 8

    # Emotion-specific appearance
    emotion = emotion.lower() if emotion else 'neutral'
    if emotion == 'happy':
        # eyes: crescent (smiling eyes) or blink occasionally
        if blink < 2:
            # closed eyes -> thin arcs
            cv2.ellipse(canvas, (left_eye_x, eye_y), (eye_w, 8), 0, 0, 180, (60,60,80), 6)
            cv2.ellipse(canvas, (right_eye_x, eye_y), (eye_w, 8), 0, 0, 180, (60,60,80), 6)
        else:
            # open smiling eyes (small arcs upward)
            cv2.ellipse(canvas, (left_eye_x, eye_y), (eye_w, 18), 0, 180, 360, (60,60,80), 6)
            cv2.ellipse(canvas, (right_eye_x, eye_y), (eye_w, 18), 0, 180, 360, (60,60,80), 6)
            # pupils
            cv2.circle(canvas, (left_eye_x, eye_y), pupil_r, (30,30,40), -1)
            cv2.circle(canvas, (right_eye_x, eye_y), pupil_r, (30,30,40), -1)

        # mouth: big smiling arc
        mouth_center = (cx, cy + 60)
        axes = (90, 50)
        cv2.ellipse(canvas, mouth_center, axes, 0, 180, 360, (30,30,45), -1)  # filled smile
        # little tongue for cuteness
        cv2.ellipse(canvas, (cx, cy + 70), (30, 15), 0, 0, 180, (255,150,170), -1)

    elif emotion == 'sad':
        # eyes: half-closed droopy eyes
        cv2.ellipse(canvas, (left_eye_x, eye_y), (eye_w, 18), 0, 20, 160, (60,60,80), 6)
        cv2.ellipse(canvas, (right_eye_x, eye_y), (eye_w, 18), 0, 20, 160, (60,60,80), 6)
        cv2.circle(canvas, (left_eye_x, eye_y+6), 6, (30,30,40), -1)
        cv2.circle(canvas, (right_eye_x, eye_y+6), 6, (30,30,40), -1)

        # mouth: small sad arc (frown)
        mouth_center = (cx, cy + 80)
        axes = (70, 30)
        cv2.ellipse(canvas, mouth_center, axes, 0, 10, 170, (30,30,45), -1)

    elif emotion == 'angry':
        # eyes: narrow angry eyes
        cv2.ellipse(canvas, (left_eye_x, eye_y), (eye_w, 18), -15, 200, 340, (60,60,80), -1)
        cv2.ellipse(canvas, (right_eye_x, eye_y), (eye_w, 18), 15, 200, 340, (60,60,80), -1)
        # pupils
        cv2.circle(canvas, (left_eye_x, eye_y), 7, (10,10,10), -1)
        cv2.circle(canvas, (right_eye_x, eye_y), 7, (10,10,10), -1)

        # eyebrows (slanted down inside)
        brow_y = eye_y - 28
        cv2.line(canvas, (left_eye_x - 40, brow_y + 10), (left_eye_x + 10, brow_y - 6), (30,30,40), 8)
        cv2.line(canvas, (right_eye_x + 40, brow_y + 10), (right_eye_x - 10, brow_y - 6), (30,30,40), 8)

        # mouth: small straight line or slight frown
        cv2.line(canvas, (cx - 45, cy + 80), (cx + 45, cy + 80), (30,30,45), 6)

    elif emotion == 'surprise' or emotion == 'surprised':
        # eyes: very wide open
        eye_big_r = int(22 * pulse)
        cv2.circle(canvas, (left_eye_x, eye_y), eye_big_r+8, (255,255,255), -1)
        cv2.circle(canvas, (right_eye_x, eye_y), eye_big_r+8, (255,255,255), -1)
        cv2.circle(canvas, (left_eye_x, eye_y), int(8*pulse), (30,30,40), -1)
        cv2.circle(canvas, (right_eye_x, eye_y), int(8*pulse), (30,30,40), -1)

        # eyebrows high (surprised)
        cv2.line(canvas, (left_eye_x - 40, eye_y - 36), (left_eye_x + 10, eye_y - 44), (30,30,40), 6)
        cv2.line(canvas, (right_eye_x + 40, eye_y - 36), (right_eye_x - 10, eye_y - 44), (30,30,40), 6)

        # mouth: open "O" shape (pulse)
        mouth_r = int(18 * (1 + 0.6 * abs(math.sin(t * 0.2))))
        cv2.circle(canvas, (cx, cy + 80), mouth_r + 8, (30,30,45), -1)
        cv2.circle(canvas, (cx, cy + 80), mouth_r, (240,200,200), -1)

    else:  # neutral / default
        # normal round eyes
        cv2.circle(canvas, (left_eye_x, eye_y), 16, (255,255,255), -1)
        cv2.circle(canvas, (right_eye_x, eye_y), 16, (255,255,255), -1)
        cv2.circle(canvas, (left_eye_x, eye_y), 7, (30,30,40), -1)
        cv2.circle(canvas, (right_eye_x, eye_y), 7, (30,30,40), -1)
        # mouth: simple straight line
        cv2.line(canvas, (cx - 35, cy + 80), (cx + 35, cy + 80), (30,30,45), 6)

    # Robot name / emotion text
    label = f"{emotion.capitalize()}  {score:.2f}"
    cv2.putText(canvas, label, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20,30,50), 2, cv2.LINE_AA)

    return canvas

# Main loop
frame_counter = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from camera. Exiting.")
        break

    # resize camera frame to our panel size
    cam_frame = cv2.resize(frame, (CAM_W, CAM_H))

    # Convert to RGB for detector (fer expects RGB)
    rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)

    # Detect emotions (we will use the first detected face)
    results = detector.detect_emotions(rgb)  # list of detections
    if results and len(results) > 0:
        # take the largest/first face
        face = results[0]
        (x, y, w, h_face) = face["box"]
        emotions = face["emotions"]
        # top emotion
        top_emotion = max(emotions, key=emotions.get)
        score = emotions[top_emotion]
        emotion_label = top_emotion
        # draw rectangle & label on camera feed
        # correct negative coordinates if any
        x1 = max(0, x); y1 = max(0, y)
        x2 = x1 + w; y2 = y1 + h_face
        cv2.rectangle(cam_frame, (x1, y1), (x2, y2), (0, 200, 100), 2)
        cv2.putText(cam_frame, f"{emotion_label} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,100), 2, cv2.LINE_AA)
    else:
        emotion_label = 'neutral'
        score = 0.0

    # Robot panel (blank)
    robot_canvas = np.zeros((ROBOT_H, ROBOT_W, 3), dtype=np.uint8)
    # draw robot based on detected emotion
    draw_robot_face(robot_canvas, emotion=emotion_label, score=score, t=frame_counter)

    # Combine camera and robot side-by-side
    combined = np.hstack((cam_frame, robot_canvas))

    # Show
    cv2.imshow("You (left)  |  Robot avatar (right)  -- press 'q' to quit", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    frame_counter += 1

cap.release()
cv2.destroyAllWindows()
