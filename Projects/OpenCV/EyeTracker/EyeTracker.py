import cv2
import mediapipe as mp

# Initialize Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
)

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands_detected = False

cap = cv2.VideoCapture(0)

is_looking_away = False

cv2.namedWindow("Web Cam Feed", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Web Cam Feed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Initial smoothed values (based on your straight gaze baselines)
smooth_l_ratio = 0.46
smooth_r_ratio = 0.45
smooth_l_v_ratio = 0.40
smooth_pitch = 0.33

# Smoothing factor: 
# 0.15 = very smooth & stable
# 0.30 = faster response, slightly more jitter
ALPHA = 0.20

while True:
    ret, frame = cap.read()
    if not ret:
        print("No Uggo  Found")

    # --- Digital Zoom / Crop Background ---
    h, w, _ = frame.shape

    # Define crop margins (e.g., crop 15% from top/bottom, 25% from left/right)
    crop_y = int(h * 0.15)
    crop_x = int(w * 0.25)

    # Slice array: [y1:y2, x1:x2]
    frame = frame[crop_y : h - crop_y, crop_x : w - crop_x]

    # Resize back to normal resolution so your coordinate math stays clean
    frame = cv2.resize(frame, (w, h))

    frame = cv2.flip(frame, 1)
    frame = cv2.convertScaleAbs(frame, alpha= 2, beta = 10)

    #Mediapipe uses RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #Process and detect face
    results = face_mesh.process(rgb_frame)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)

    hand_detected = bool(hand_results.multi_hand_landmarks)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        

        #Calculating position of left iris
        left_iris = landmarks[468]
        l_iris_x = int(left_iris.x * w)
        l_iris_y = int(left_iris.y * h)

        #Drawing Left Iris
        cv2.circle(frame, (l_iris_x, l_iris_y), 4, (0, 255, 255), -1)


        #Calculating position of right iris
        right_iris = landmarks[473]
        r_iris_x = int(right_iris.x * w)
        r_iris_y = int(right_iris.y * h) 

        #Drawing Right Iris
        cv2.circle(frame, (r_iris_x, r_iris_y), 4, (0, 255, 255), -1)


        #Left points
        l_outer = landmarks[33]
        l_outer_x = int(l_outer.x * w )
        l_outer_y = int(l_outer.y * h )

        #Drawing left outer point
        cv2.circle(frame, (l_outer_x, l_outer_y), 4, (0, 0, 255), -1)

        l_inner = landmarks[133]
        l_inner_x = int(l_inner.x * w )
        l_inner_y = int(l_inner.y * h )

        #Drawing left inner point
        cv2.circle(frame, (l_inner_x, l_inner_y), 4, (0, 0, 255), -1)        

        #Calculating Distances and ratios
        l_eye_width = abs(l_inner_x - l_outer_x)
        l_iris_dist = abs(l_iris_x - l_outer_x)
        l_ratio = l_iris_dist/l_eye_width

        #Looking Away Check
        if(l_ratio < 0.35 or l_ratio > 0.60):
            is_looking_away = True
            #cv2.putText(frame, "Looking Away!!!", (150, 350),
            #            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        else:
            is_looking_away = False
            #cv2.putText(frame, "FOCUSED", (200, 350),
            #            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.3
        color = (255, 0, 0)
        thickness = 1

        #Showing Left Ratio
        cv2.putText(frame, f"Left Ratio: {l_ratio:.2f}", (20, 30),
                            font, scale, color, thickness)

        #Right outer points
        r_outer = landmarks[263]
        r_outer_x = int(r_outer.x * w )
        r_outer_y = int(r_outer.y * h )

        #Drawing right outer points
        cv2.circle(frame, (r_outer_x, r_outer_y), 4, (0, 0, 255), -1)

        #Right inner points
        r_inner = landmarks[362]
        r_inner_x = int(r_inner.x * w )
        r_inner_y = int(r_inner.y * h )

        #Drawing right inner points
        cv2.circle(frame, (r_inner_x, r_inner_y), 4, (0, 0, 255), -1)

        #Calculating Right Ratios
        r_eye_width = abs(r_inner_x - r_outer_x)
        r_iris_dist = abs(r_iris_x - r_outer_x)
        r_ratio = r_iris_dist/r_eye_width

        #Looking Away Check
        if(r_ratio < 0.32 or r_ratio > 0.65):
            is_looking_away = True
            #cv2.putText(frame, "Looking Away!!!", (150, 350),
            #            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        else:
            is_looking_away = False
            #cv2.putText(frame, "FOCUSED", (200, 350),
            #            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        #Showing Right ratios
        cv2.putText(frame, f"Right Ratio: {r_ratio:.2f}", (20, 55),
                            font, scale, color, thickness)


        #Left Top point
        l_upper = landmarks[159]
        l_upper_x = int(l_upper.x * w )
        l_upper_y = int(l_upper.y * h )

        #Drawing left top point
        cv2.circle(frame, (l_upper_x, l_upper_y), 4, (0, 0, 255), -1)

        #Left bottom point
        l_lower = landmarks[145]
        l_lower_x = int(l_lower.x * w )
        l_lower_y = int(l_lower.y * h )

        #Drawing left bottom point
        cv2.circle(frame, (l_lower_x, l_lower_y), 4, (0, 0, 255), -1)

        #Calculating left vertical ratios
        l_eye_hight = abs(l_lower_y - l_upper_y)
        l_iris_vert_dist = abs(l_iris_y - l_upper_y)

        if(l_eye_hight == 0):
            is_looking_away = True
            l_vert_ratio = 0
        else:
            l_vert_ratio = l_iris_vert_dist/l_eye_hight

        #Looking Away Check
        # if(l_vert_ratio < 0.20 or l_vert_ratio > 0.65):
        #     is_looking_away = True
        #     cv2.putText(frame, "Looking Away!!!", (250, 300),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        #Showing Left ratios
        cv2.putText(frame, f"Left Vetical Ratio: {l_vert_ratio:.2f}", (20, 80),
                            font, scale, color, thickness)

        #Right top point
        r_upper = landmarks[386]
        r_upper_x = int(r_upper.x * w )
        r_upper_y = int(r_upper.y * h )

        #Drawing right top point
        cv2.circle(frame, (r_upper_x, r_upper_y), 4, (0, 0, 255), -1)

        #Right bottom point
        r_lower = landmarks[374]
        r_lower_x = int(r_lower.x * w )
        r_lower_y = int(r_lower.y * h )

        #Drawing right bottom point
        cv2.circle(frame, (r_lower_x, r_lower_y), 4, (0, 0, 255), -1)

        #Calculating right vertical ratio
        r_eye_hight = abs(r_lower_y - r_upper_y)
        r_iris_vert_dist = abs(r_iris_y - r_upper_y)

        if(r_eye_hight == 0):
            is_looking_away = True
            r_vert_ratio = 0
        else:
            r_vert_ratio = r_iris_vert_dist/r_eye_hight

        #Looking Away Check
        # if(r_vert_ratio < 0.28 or r_vert_ratio > 0.30):
        #     is_looking_away = True
        #     cv2.putText(frame, "Looking Away!!!", (250, 300),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        #Showing Right Vertical ratios
        cv2.putText(frame, f"Right Vetical Ratio: {r_vert_ratio:.2f}",(20, 105), 
                    font, scale, color, thickness)

        #NoseTip
        noseTip = landmarks[1]
        noseTip_x = int(noseTip.x * w )
        noseTip_y = int(noseTip.y * h )

        #Nose Bridge
        noseBridge = landmarks[168]
        noseBridge_x = int(noseBridge.x * w )
        noseBridge_y = int(noseBridge.y * h )

        #Chin
        chin = landmarks[152]
        chin_x = int(chin.x * w )
        chin_y = int(chin.y * h )


        #Drawing nose tip, bridge and chin point
        cv2.circle(frame, (noseTip_x, noseTip_y), 4, (0, 0, 255), -1)
        cv2.circle(frame, (noseBridge_x, noseBridge_y), 4, (0, 0, 255), -1)
        cv2.circle(frame, (chin_x, chin_y), 4, (0, 0, 255), -1)

        #Calculating shit
        face_height = abs(chin_y - noseBridge_y)
        nose_distan = abs(noseTip_y - noseBridge_y)

        pitch_ratio = nose_distan/face_height
        cv2.putText(frame, f"Pitch Ratio: {pitch_ratio:.2f}",(20, 130), 
                    font, 0.3, (255, 0, 0), 1)

        # --- Attention Threshold Checks ---

        # 1. Horizontal Eye Gaze (Left / Right)
        #looking_left  = l_ratio < 0.35
        #looking_right = l_ratio > 0.60

        # 2. Vertical Head Pitch (Down / Up)
        #looking_up   = pitch_ratio < 0.30   # Drops down to ~0.25
        #looking_down = pitch_ratio > 0.35   # Rises to 0.36+


        # Full Condition
        #is_looking_away = looking_left or looking_right or looking_down or looking_up

        # Apply Exponential Moving Average (EMA) to all 4 metrics
        smooth_l_ratio   = (ALPHA * l_ratio) + ((1 - ALPHA) * smooth_l_ratio)
        smooth_r_ratio   = (ALPHA * r_ratio) + ((1 - ALPHA) * smooth_r_ratio)
        smooth_l_v_ratio = (ALPHA * l_vert_ratio) + ((1 - ALPHA) * smooth_l_v_ratio)
        smooth_pitch     = (ALPHA * pitch_ratio) + ((1 - ALPHA) * smooth_pitch)

        # Threshold checks using the smoothed values
        looking_left  = smooth_l_ratio < 0.35
        looking_right = smooth_l_ratio > 0.60
        looking_up    = smooth_pitch < 0.28
        looking_down  = smooth_pitch > 0.35


        is_looking_away = (
            looking_left or 
            looking_right or 
            looking_up or 
            looking_down or 
            hand_detected
    )
        # Full condition
        #is_looking_away = looking_left or looking_right or looking_up or looking_down

        #Looking Away Check
        #status_text = "LOOKING AWAY!" if is_looking_away else "FOCUSED"
        #status_color = (0, 0, 255) if is_looking_away else (0, 0, 0)
        
        if hand_detected:
            status_text = "HAND DETECTED - NOT FOCUSED"
            status_color = (0, 0, 255)
        elif is_looking_away:
            status_text = "LOOKING AWAY!"
            status_color = (0, 0, 255)
        else:
            status_text = "FOCUSED"
            status_color = (0, 255, 0)

        (text_w, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        center_x = int((w - text_w) / 2)
        cv2.putText(frame, status_text, (center_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)

        #Tracking 
        cv2.putText(frame, "Tracking Iris", (250, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    else:
        face = "NO UGGO FOUND"
        (text_w, _), _ = cv2.getTextSize(face, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        center_x = int((w - text_w) / 2)
        cv2.putText(frame, face, (center_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        

    cv2.imshow("Web Cam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()