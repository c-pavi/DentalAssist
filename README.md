# Brushly

Real-time brushing monitor using MediaPipe Hands + Face Detection. It starts when
your hand is brought to your face, counts down, tracks a 2-minute session, shows
timed prompts, and warns when motion is left-to-right instead of small circles.

## Features
- Face detection + hand proximity trigger
- 3-second start countdown
- 2-minute brushing timer
- Prompts at 30s, 60s, 90s
- Left-to-right motion warning
- Optional view without landmarks

## Requirements
- Python 3.9+
- A webcam

## Setup
1. Create a virtual environment (optional).
2. Install dependencies:

```
python3 -m pip install -r requirements.txt
```

## Run
With landmarks:
```
python3 Brushly.py
```

Without landmarks:
```
python3 Brushly_no_pointers.py
```

Press `ESC` to exit.

## Notes
- If the camera doesn't open, change the index in `cv2.VideoCapture(1)` to `0`.
- Motion warning sensitivity can be adjusted in `Brushly.py`.
