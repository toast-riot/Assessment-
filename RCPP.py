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

    finger_threstholds = [
        [1.05, 20.0],
        [1.05, 20.0],
        [1.05, 20.0],
        [1.05, 20.0],
        [1.05, 20.0]
    ]


    thumb_tip = landmarks[4]
    thumb_base = landmarks[1]
    wrist = landmarks[0]


    # METHOD 1
    finger_curl_percentages = []

    for finger in fingers:
        base = landmarks[finger[0]]
        tip = landmarks[finger[-1]]

        finger_length = sum(calculate_distance(landmarks[finger[i]], landmarks[finger[i+1]]) for i in range(len(finger) - 1))
        finger_dist = calculate_distance(tip, base)
        finger_curl = finger_length / finger_dist
        # finger_curl_percentage = (finger_length - finger_dist) / finger_dist

        min_curl, max_curl = finger_threstholds[fingers.index(finger)]

        finger_curl_percentage = (math.log(finger_curl) - math.log(min_curl)) / (math.log(max_curl) - math.log(min_curl)) * 100
        finger_curl_percentage = max(0, min(finger_curl_percentage, 100))

        finger_curl_percentages.append(finger_curl_percentage)

    # return [
    #     f"Thumb: {finger_curl_percentages[0]:.2f}%",
    #     f"Index: {finger_curl_percentages[1]:.2f}%",
    #     f"Middle: {finger_curl_percentages[2]:.2f}%",
    #     f"Ring: {finger_curl_percentages[3]:.2f}%",
    #     f"Pinky: {finger_curl_percentages[4]:.2f}%"
    # ]

    threshold = 2.5

    return [
        f"Thumb: {'True' if finger_curl_percentages[0] > threshold else 'False'}",
        f"Index: {'True' if finger_curl_percentages[1] > threshold else 'False'}",
        f"Middle: {'True' if finger_curl_percentages[2] > threshold else 'False'}",
        f"Ring: {'True' if finger_curl_percentages[3] > threshold else 'False'}",
        f"Pinky: {'True' if finger_curl_percentages[4] > threshold else 'False'}"
    ]


    # METHOD 2 (BROKEN)
    # finger_curl_checks = []

    # for finger in fingers:
    #     is_curling = True
    #     for i in range(len(finger) - 1):
    #         dist_current = calculate_distance(landmarks[finger[i]], landmarks[finger[i + 1]])
    #         dist_next = calculate_distance(landmarks[finger[i + 1]], landmarks[finger[i + 2]]) if i + 2 < len(finger) else float('inf')

    #         if dist_next >= dist_current:
    #             is_curling = False
    #             break

    #     finger_curl_checks.append(is_curling)

    # return [
    #     f"Thumb: {'Curling' if finger_curl_checks[0] else 'Not Curling'}",
    #     f"Index: {'Curling' if finger_curl_checks[1] else 'Not Curling'}",
    #     f"Middle: {'Curling' if finger_curl_checks[2] else 'Not Curling'}",
    #     f"Ring: {'Curling' if finger_curl_checks[3] else 'Not Curling'}",
    #     f"Pinky: {'Curling' if finger_curl_checks[4] else 'Not Curling'}"
    # ]




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
