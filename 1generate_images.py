"""
Pipeline to generate sign language alphabet images.

Captures a fixed number of images per class (A-Z excluding J & Z)
from the webcam and saves images to a directory organized by class.
"""

import os
import string
import time
import cv2

# ---------------------------------------------------------------------------
# Global Variables
# ---------------------------------------------------------------------------
OUTPUT_DIR = "images"          # root folder where per-class subfolders are created
IMAGES_PER_CLASS = 100          # number of images to capture per class
CAPTURE_INTERVAL = 1         # seconds between saved frames during capture
CAMERA_INDEX = 0                # webcam device index
CLASSES = None                  # e.g. ["A", "B", "C"] to capture a subset; None = all 24 letters + numbers 1-10
# ---------------------------------------------------------------------------

def get_classes():
    """ Return the 24 static letter classes """
    return [c for c in string.ascii_uppercase if c not in ("J", "Z")] + [n for n in string.digits if n not in ("0")]

def ensure_dirs(output_dir, classes):
    for c in classes:
        os.makedirs(os.path.join(output_dir, c), exist_ok=True)

def next_index(class_dir):
    """ 
    Find the next free image index in a class folder. 
    Avoids restarting to first index and overwriting existing images.
    """
    existing = [f for f in os.listdir(class_dir) if f.endswith(".jpg")]
    if not existing:
        return 0
    indices = []
    for f in existing:
        try:
            indices.append(int(os.path.splitext(f)[0]))
        except ValueError:
            continue
    
    return max(indices) + 1 if indices else 0

def draw_overlay(frame, text, color, scale, y):
    cv2.putText(
        frame, text, (20,y),
        cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA
    )

def capture_class(cap, class_name, class_dir, num_images, capture_interval):
    """ Shows a ready screen, then capture num_images frames in one class. """
    start_index = next_index(class_dir)
    switch_point = num_images // 2  # capture first half on one hand, then switch

    # --- Ready Screen: wait for user to position their hand ---
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        
        draw_overlay(frame, f"Class: {class_name}", (0,255,255), 1.5, 60)
        draw_overlay(frame, "Press SPACE to start capture", (255,255,255), 0.8, 100)
        draw_overlay(frame, "Press Q to quit", (255,255,255), 0.8, 130)
        cv2.imshow("FSL Dataset", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key == ord("q"):
            return False        # signal quit
        
    
    # --- Countdown for positioning
    for i in range(3, 0, -1):
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        
        draw_overlay(frame, f"Class: {class_name}", (0,255,255), 1.5, 60)
        draw_overlay(frame, f"Starting in {i}...", (0,0,255), 1.2, 110)
        cv2.imshow("FSL Dataset", frame)
        cv2.waitKey(1)
        time.sleep(1)

    # --- Capture loop
    captured = 0
    already_switched = False
    last_capture_time = 0.0
    while captured < num_images:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame,1)

        # ---- Hand switch layer: pause once at a halfway point
        if captured == switch_point and not already_switched:
            while True:
                ok, switch_frame = cap.read()
                if not ok:
                    continue
                switch_frame = cv2.flip(switch_frame,1)
                draw_overlay(switch_frame, f"Class: {class_name}", (0,255,255), 1.5, 60)
                draw_overlay(switch_frame, f"Switch hand, then press SPACE", (0,165,255), 1.0, 100)
                draw_overlay(switch_frame, "Press Q to abort class", (255,255,255), 0.7, 460)
                cv2.imshow("FSL Dataset", switch_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    already_switched = True
                    last_capture_time = time.time()   # avoid instant capture right after resuming
                    break
                if key == ord('q'):
                    return False        # signal quit

        now = time.time()
        display = frame.copy()
        draw_overlay(display, f"Class: {class_name}", (0,255,255), 1.5, 60)
        draw_overlay(display, f"Captured: {captured}/{num_images}", (0,255,0), 1.0, 100)
        draw_overlay(display, "Press Q to abort class", (255,255,255), 0.7, 460)
        cv2.imshow("FSL Dataset", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        if now - last_capture_time >= capture_interval:
            filename = f"{start_index + captured:03d}.jpg"
            filepath = os.path.join(class_dir, filename)
            cv2.imwrite(filepath, frame)  # save clean frame, no overlay
            captured += 1
            last_capture_time = now

    return True  # continue to next class



def main():
    classes = CLASSES if CLASSES else get_classes()
    classes = [c.strip().upper() for c in classes]

    ensure_dirs(OUTPUT_DIR, classes)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {CAMERA_INDEX}")
    
    print(f"Capturing {IMAGES_PER_CLASS} images for {len(classes)} classes: {classes}")

    try:
        for class_name in classes:
            class_dir = os.path.join(OUTPUT_DIR, class_name)

            if len([f for f in os.listdir(class_dir) if f.endswith('.jpg')]) >= IMAGES_PER_CLASS:
                print(f"Skipping class '{class_name}' -- already has {IMAGES_PER_CLASS} images.")
                continue
            
            keep_going = capture_class(
                cap, class_name, class_dir, 
                IMAGES_PER_CLASS, CAPTURE_INTERVAL
            )
            if not keep_going:
                print("Quit requested. Stopping early")
                break
            print(f"Finished class '{class_name}' -> {class_dir}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Done.")

def main2():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {CAMERA_INDEX}")
    
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        flipped_frame = cv2.flip(frame,1)

        cv2.imshow("Live Feed", flipped_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()