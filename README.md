Sign Language to Text Converter
Real-time Hand Gesture Recognition using Python, OpenCV & Google MediaPipe

A training-free, real-time system that converts
hand gestures to readable text using a standard webcam. 
Detects 21 hand landmarks per frame, maps finger positions to all 26
alphabet letters, and builds a full editable sentence — with
no machine learning training, no GPU, and no internet required.


📷 Webcam Frame
       │
       ▼
   cv2.flip()          ← Mirror for natural feel
       │
       ▼
   BGR → RGB           ← MediaPipe needs RGB, OpenCV gives BGR
       │
       ▼
  MediaPipe Hands      ← Detects 21 landmark points on hand
       │
       ├── ❌ No hand → skip frame, loop back
       │
       ▼
  get_fingers_up()
  ├── Thumb  → x-axis check  (tip.x vs ip.x)
  └── Others → y-axis check  (tip.y vs pip.y)
  Returns → [Thumb, Index, Middle, Ring, Pinky]  e.g. [0,1,0,0,0]
       │
       ▼
  classify_gesture()   ← GESTURE_MAP.get(tuple(fingers))
  O(1) dictionary lookup
       │
       ├── ❌ No match → skip
       │
       ▼
  Hold Timer (1.5 s)   ← Progress bar fills on screen
       │
       ▼
  Register to sentence
  ├── Space  → sentence += " "
  ├── Del    → sentence[:-1]
  ├── Clear  → sentence = ""
  ├── Hello  → sentence += "Hello "
  └── Letter → sentence += "A" / "B" / ... / "Z"
       │
       ▼
  draw_ui() → cv2.imshow()
       │
       └── 🔁 Loop back


Finger detection — Coordinate Geometry
python# Screen y=0 is TOP. Finger pointing UP = tip higher = smaller y value
# tip.y < pip.y  →  finger is UP   ✅
# tip.y > pip.y  →  finger is DOWN ❌

# Thumb moves sideways — use x-axis instead
# Right hand: THUMB_TIP.x < THUMB_IP.x  →  thumb extended ✅
# Left hand:  THUMB_TIP.x > THUMB_IP.x  →  thumb extended ✅
