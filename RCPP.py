import cv2
import mediapipe as mp
import math

# Initialize Mediapipe Hand solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils


def calculate_distance(point1, point2):
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

def recognize_hand_gesture(landmarks):
    fingers = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20]
    ]
    thumb_tip = landmarks[4]
    thumb_base = landmarks[1]
    index_tip = landmarks[8]
    index_base = landmarks[5]
    middle_tip = landmarks[12]
    middle_base = landmarks[9]
    ring_tip = landmarks[16]
    ring_base = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_base = landmarks[17]
    wrist = landmarks[0]

    pointer_length = calculate_distance(landmarks[5], landmarks[6]) + calculate_distance(landmarks[6], landmarks[7]) + calculate_distance(landmarks[7], landmarks[8])
    palm_size = calculate_distance(landmarks[0], landmarks[5])
    pointer_dist = calculate_distance(index_tip, index_base)
    pointer_curl = pointer_length / pointer_dist

    min_curl = 0.95
    max_curl = 20

    pointer_curl_percentage = (math.log(pointer_curl) - math.log(min_curl)) / (math.log(max_curl) - math.log(min_curl)) * 100
    pointer_curl_percentage = max(0, min(pointer_curl_percentage, 100))

    return [
        f'Palmsize: {palm_size}',
        f'Pointer dist: {pointer_dist}',
        f'Pointer_curl: {pointer_curl}',
        f'Pointer_curl_percentage: {pointer_curl_percentage}'
    ]



cap = cv2.VideoCapture(0)
cv2.namedWindow("Rock, Paper, Scissors Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Rock, Paper, Scissors Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    if result.multi_hand_landmarks:
            for landmarks in result.multi_hand_landmarks:
                info = recognize_hand_gesture(landmarks.landmark)

                for i, data in zip(range(len(info)), info):
                    cv2.putText(frame, data, (10, 25*(i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)


    cv2.imshow("Rock, Paper, Scissors Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
