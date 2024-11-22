import cv2
import mediapipe as mp
import random
import time

# Initialize Mediapipe Hand solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils


def recognize_hand_gesture(landmarks):
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

    thumb_up = (thumb_tip.y < thumb_base.y and abs(thumb_tip.x - thumb_base.x) < 0.1)
    fingers_folding = (index_tip.y > index_base.y and middle_tip.y > middle_base.y and
                       ring_tip.y > ring_base.y and pinky_tip.y > pinky_base.y)

    if thumb_up and fingers_folding:
        return "Thumb-Up"

    if not thumb_up and fingers_folding:
        return "Rock"
    elif thumb_up and not fingers_folding:
        return "Paper"
    elif index_tip.y < index_base.y and middle_tip.y < middle_base.y and ring_tip.y > ring_base.y and pinky_tip.y > pinky_base.y and thumb_tip.y < thumb_base.y:
        return "Scissors"
    elif index_tip.y > index_base.y and middle_tip.y < middle_base.y and ring_tip.y > ring_base.y and pinky_tip.y > pinky_base.y:
        return "THE FINGER"

    return "None"


def determine_winner(player_choice, computer_choice):
    if player_choice == computer_choice:
        return "It's a tie!"

    # Special case for "THE FINGER"
    if player_choice == "THE FINGER" and computer_choice in ["Rock", "Paper", "Scissors"]:
        return "You Win!"

    # Normal win conditions
    if (player_choice == "Rock" and computer_choice == "Scissors") or \
            (player_choice == "Paper" and computer_choice == "Rock") or \
            (player_choice == "Scissors" and computer_choice == "Paper"):
        return "You win!"

    # Default case: Computer wins
    return "Computer wins!"


cap = cv2.VideoCapture(0)
cv2.namedWindow("Rock, Paper, Scissors Game", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Rock, Paper, Scissors Game", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

state = "waiting_for_start"  # Game state
countdown = 4
start_time = None
player_choice = None
computer_choice = None
result_message = None

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

    if state == "waiting_for_start":
        cv2.putText(frame, "Press Thumb-Up to Start", (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        if result.multi_hand_landmarks:
            for landmarks in result.multi_hand_landmarks:
                gesture = recognize_hand_gesture(landmarks.landmark)
                if gesture == "Thumb-Up":
                    state = "countdown"
                    countdown = 4
                    start_time = None
                    computer_choice = random.choice(["Rock", "Paper", "Scissors"])
                    result_message = None
                    player_choice = None

    elif state == "countdown":
        if start_time is None:
            start_time = time.time()

        elapsed_time = time.time() - start_time
        time_left = int(countdown - elapsed_time)

        if time_left > 0:
            cv2.putText(frame, f'{time_left}...', (frame.shape[1] // 2 - 50, frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)
        else:
            state = "gesture_recognition"

    elif state == "gesture_recognition":
        if result.multi_hand_landmarks:
            for landmarks in result.multi_hand_landmarks:
                player_choice = recognize_hand_gesture(landmarks.landmark)

            if player_choice != "None":
                result_message = determine_winner(player_choice, computer_choice)
                state = "show_result"

    elif state == "show_result":
        cv2.putText(frame, f'Player: {player_choice}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, f'Computer: {computer_choice}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(frame, result_message, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press SPACE to Restart", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

        if cv2.waitKey(1) & 0xFF == ord(' '):  # Space bar pressed
            state = "waiting_for_start"

    cv2.imshow("Rock, Paper, Scissors Game", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
