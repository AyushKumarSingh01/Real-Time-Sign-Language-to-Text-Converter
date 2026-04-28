import cv2
import mediapipe as mp
import time
import math


#  MediaPipe setup
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
)

#  Landmark indices
WRIST                                               = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP          = 1, 2, 3, 4
INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP      = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP     = 9, 10, 11, 12
RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP       = 13, 14, 15, 16
PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP     = 17, 18, 19, 20



GESTURE_MAP = {


    (0, 0, 0, 0, 0): ("A",  (0,   200, 255)),   # Fist
    (0, 1, 1, 1, 1): ("B",  (0,   200, 100)),   # 4 fingers, no thumb
    (1, 1, 0, 1, 0): ("C",  (255, 180,  50)),   # Thumb + Index + Ring
    (0, 1, 0, 0, 0): ("D",  (255, 100,   0)),   # Index only
    (0, 0, 1, 1, 1): ("E",  (180,  80, 255)),   # Middle + Ring + Pinky
    (1, 1, 0, 0, 1): ("F",  (255,  20, 147)),   # Thumb + Index + Pinky
    (1, 0, 0, 0, 0): ("G",  (255, 200,   0)),   # Thumb only
    (1, 1, 1, 0, 0): ("H",  (100, 220, 255)),   # Thumb + Index + Middle
    (0, 0, 0, 0, 1): ("I",  (100, 255, 200)),   # Pinky only
    (0, 1, 0, 1, 1): ("J",  (255,  80,  80)),   # Index + Ring + Pinky
    (0, 1, 0, 1, 0): ("K",  (200, 255,  80)),   # Index + Ring
    (1, 1, 0, 0, 0): ("L",  (255,  80, 180)),   # Thumb + Index
    (0, 0, 1, 0, 0): ("M",  (80,  180, 255)),   # Middle only
    (0, 0, 0, 1, 0): ("N",  (180, 255, 100)),   # Ring only
    (1, 0, 1, 1, 0): ("O",  (255, 140,   0)),   # Thumb + Middle + Ring
    (0, 1, 1, 0, 1): ("P",  (150,  50, 255)),   # Index + Middle + Pinky
    (1, 0, 0, 1, 0): ("Q",  (255, 200, 100)),   # Thumb + Ring
    (0, 1, 0, 0, 1): ("R",  (255,  50,  50)),   # Index + Pinky
    (1, 0, 1, 1, 1): ("S",  (0,   150, 255)),   # Thumb + Middle + Ring + Pinky
    (1, 0, 1, 0, 0): ("T",  (0,   255, 150)),   # Thumb + Middle
    (0, 0, 0, 1, 1): ("U",  (180, 100, 255)),   # Ring + Pinky
    (0, 1, 1, 0, 0): ("V",  (180,   0, 255)),   # Index + Middle (peace)
    (0, 1, 1, 1, 0): ("W",  (255, 160,   0)),   # Index + Middle + Ring
    (0, 0, 1, 0, 1): ("X",  (255,  50, 150)),   # Middle + Pinky
    (1, 0, 0, 0, 1): ("Y",  (200,   0, 200)),   # Thumb + Pinky (shaka)
    (1, 1, 0, 1, 1): ("Z",  (100, 100, 255)),   # Thumb + Index + Ring + Pinky

    (1, 1, 1, 1, 1): ("Hello", (0,   255,   0)),   # All 5 fingers open

    (1, 0, 0, 1, 1): ("Space", (200, 200, 200)),   # Thumb + Ring + Pinky
    (0, 0, 1, 1, 0): ("Del",   (0,    50, 255)),   # Middle + Ring
    (1, 0, 1, 0, 1): ("Clear", (0,     0, 255)),   # Thumb + Middle + Pinky

    
}

# Labels that trigger special text actions (not added as a character)
SPECIAL_LABELS = {"Space", "Del", "Clear"}

# Labels that add the full word (not just first character)
WORD_LABELS    = {"Hello"}


#  Helper functions
def euclidean(p1, p2):
    """Straight-line distance between two MediaPipe landmark points."""
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def get_fingers_up(lm, hand_label):
    """
    Returns [thumb, index, middle, ring, pinky] as 0/1.
    Thumb uses x-axis comparison (moves sideways).
    Other four fingers use y-axis comparison (move up/down).
    hand_label: 'Left' or 'Right' from MediaPipe (camera is mirrored).
    """
    fingers = []

    # Thumb — x-axis check (flips for left vs right hand)
    if hand_label == "Right":
        fingers.append(1 if lm[THUMB_TIP].x < lm[THUMB_IP].x else 0)
    else:
        fingers.append(1 if lm[THUMB_TIP].x > lm[THUMB_IP].x else 0)

    # Index, Middle, Ring, Pinky — y-axis check
    # tip.y < pip.y  means finger is pointing UP (screen y=0 is top)
    tip_pip_pairs = [
        (INDEX_TIP,  INDEX_PIP),
        (MIDDLE_TIP, MIDDLE_PIP),
        (RING_TIP,   RING_PIP),
        (PINKY_TIP,  PINKY_PIP),
    ]
    for tip, pip in tip_pip_pairs:
        fingers.append(1 if lm[tip].y < lm[pip].y else 0)

    return fingers


def classify_gesture(fingers):
    """Return (label, color) from GESTURE_MAP or None if no match."""
    return GESTURE_MAP.get(tuple(fingers))


def draw_rounded_rect(img, x1, y1, x2, y2, radius, color, thickness=-1):
    """Draw a filled/stroked rounded rectangle (OpenCV workaround)."""
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    for cx, cy in [(x1 + radius, y1 + radius),
                   (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius),
                   (x2 - radius, y2 - radius)]:
        cv2.circle(img, (cx, cy), radius, color, thickness)


def draw_ui(img, sentence, current_gesture, gesture_color,
            hold_progress, fps, hand_present):
    h, w = img.shape[:2]

    # ── Semi-transparent top bar 
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    cv2.putText(img, f"FPS: {fps:.0f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 180), 2)
    cv2.putText(img, "Sign Language  to  Text  |  All 26 Letters",
                (w // 2 - 230, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # ── Finger state indicator (T I M R P) 
    if hand_present:
        labels = ["T", "I", "M", "R", "P"]
        for idx, lbl in enumerate(labels):
            bx = w - 160 + idx * 28
            by = 15
            cv2.putText(img, lbl, (bx, by + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # ── Detected gesture label 
    if current_gesture and hand_present:
        label_text = f"Detected: {current_gesture}"
        cv2.putText(img, label_text, (10, h - 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, gesture_color, 2)

        # Hold progress bar
        bar_x, bar_y, bar_w, bar_h = 10, h - 115, 300, 18
        cv2.rectangle(img, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
        fill = int(bar_w * hold_progress)
        if fill > 0:
            cv2.rectangle(img, (bar_x, bar_y),
                          (bar_x + fill, bar_y + bar_h), gesture_color, -1)
        cv2.rectangle(img, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
        cv2.putText(img, "Hold to register",
                    (bar_x + bar_w + 10, bar_y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    elif not hand_present:
        cv2.putText(img, "No hand detected", (10, h - 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 100, 100), 2)

    # ── Sentence output box 
    box_y = h - 85
    overlay2 = img.copy()
    cv2.rectangle(overlay2, (0, box_y), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.7, img, 0.3, 0, img)

    display_sentence = sentence if sentence else "_ _ _"
    max_chars = 48
    if len(display_sentence) > max_chars:
        display_sentence = "..." + display_sentence[-(max_chars - 3):]

    cv2.putText(img, "Text:", (10, box_y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1)
    cv2.putText(img, display_sentence, (70, box_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 200), 2)

    # ── Legend 
    legend = [
        "Q      - Quit",
        "Space  - gesture: Thumb+Ring+Pinky",
        "Del    - gesture: Middle+Ring",
        "Clear  - gesture: Thumb+Mid+Pinky",
        "Hello  - gesture: All 5 open",
    ]
    for i, tip in enumerate(legend):
        cv2.putText(img, tip, (w - 310, h - 100 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)


#  Quick-reference cheat sheet overlay
#  Press 'h' to toggle it on/off during the demo
def draw_cheatsheet(img):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    title = "GESTURE CHEAT SHEET  (press H to hide)"
    cv2.putText(img, title, (w // 2 - 220, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 200), 2)

    # All gestures as a grid
    entries = sorted(
        [(k, v[0]) for k, v in GESTURE_MAP.items()
         if v[0] not in SPECIAL_LABELS],
        key=lambda x: x[1]
    )

    col_w   = 200
    row_h   = 28
    cols    = 3
    start_x = 40
    start_y = 70

    for i, (pattern, label) in enumerate(entries):
        col = i % cols
        row = i // cols
        x   = start_x + col * col_w
        y   = start_y + row * row_h

        fingers_str = "".join(
            ["T" if pattern[0] else "-",
             "I" if pattern[1] else "-",
             "M" if pattern[2] else "-",
             "R" if pattern[3] else "-",
             "P" if pattern[4] else "-"]
        )
        color = GESTURE_MAP[pattern][1]
        cv2.putText(img, f"{label:8s}  [{fingers_str}]",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Controls section
    control_y = start_y + (len(entries) // cols + 2) * row_h
    cv2.putText(img, "CONTROLS:", (start_x, control_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    controls = [
        ("Space",  "Thumb+Ring+Pinky  [T--RP]"),
        ("Del  ",  "Middle+Ring       [-_MR-]"),
        ("Clear",  "Thumb+Mid+Pinky   [T-M-P]"),
    ]
    for i, (name, desc) in enumerate(controls):
        cv2.putText(img, f"{name}  {desc}",
                    (start_x, control_y + (i + 1) * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)


#  Main loop
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    sentence         = ""
    last_gesture     = None
    gesture_start    = 0.0
    last_reg_time    = 0.0
    HOLD_SECONDS     = 1.5
    COOLDOWN_SECONDS = 0.8
    show_cheatsheet  = False   # toggle with H key

    pTime = time.time()

    while True:
        success, img = cap.read()
        if not success:
            continue

        img    = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        cTime = time.time()
        fps   = 1.0 / (cTime - pTime + 1e-6)
        pTime = cTime

        hand_present    = False
        current_gesture = None
        gesture_color   = (255, 255, 255)
        hold_progress   = 0.0
        fingers         = [0, 0, 0, 0, 0]

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_present = True
            for hand_lms, handedness in zip(results.multi_hand_landmarks,
                                            results.multi_handedness):
                # Draw skeleton
                mp_drawing.draw_landmarks(
                    img, hand_lms, mp_hands.HAND_CONNECTIONS)

                lm         = hand_lms.landmark
                hand_label = handedness.classification[0].label
                fingers    = get_fingers_up(lm, hand_label)
                match      = classify_gesture(fingers)

                if match:
                    current_gesture, gesture_color = match

                    # Hold-to-register logic
                    if current_gesture == last_gesture:
                        held          = cTime - gesture_start
                        hold_progress = min(held / HOLD_SECONDS, 1.0)

                        if (held >= HOLD_SECONDS and
                                cTime - last_reg_time >= COOLDOWN_SECONDS):

                            # ── What gets added to sentence 
                            if current_gesture == "Space":
                                sentence += " "

                            elif current_gesture == "Del":
                                sentence = sentence[:-1]

                            elif current_gesture == "Clear":
                                sentence = ""

                            elif current_gesture in WORD_LABELS:
                                # Full word gestures (e.g. Hello)
                                sentence += current_gesture + " "

                            else:
                                # Single alphabet letter
                                sentence += current_gesture

                            last_reg_time = cTime
                            gesture_start = cTime  # re-hold needed for next
                    else:
                        last_gesture  = current_gesture
                        gesture_start = cTime
                        hold_progress = 0.0

                # Draw finger-state dots on fingertips
                tip_ids = [THUMB_TIP, INDEX_TIP,
                           MIDDLE_TIP, RING_TIP, PINKY_TIP]
                h_px, w_px = img.shape[:2]
                for tip_id, is_up in zip(tip_ids, fingers):
                    cx    = int(lm[tip_id].x * w_px)
                    cy    = int(lm[tip_id].y * h_px)
                    color = (0, 255, 0) if is_up else (0, 0, 255)
                    cv2.circle(img, (cx, cy), 10, color, cv2.FILLED)
        else:
            last_gesture  = None
            gesture_start = 0.0

        # Draw UI or cheat sheet
        if show_cheatsheet:
            draw_cheatsheet(img)
        else:
            draw_ui(img, sentence, current_gesture, gesture_color,
                    hold_progress, fps, hand_present)

        cv2.imshow("Sign Language to Text  |  Q=Quit  H=Cheatsheet", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("h"):
            show_cheatsheet = not show_cheatsheet

    cap.release()
    cv2.destroyAllWindows()
    print("\n── Final Sentence ──")
    print(sentence if sentence else "(empty)")


if __name__ == "__main__":
    main()