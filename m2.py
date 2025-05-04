import cv2
from ultralytics import YOLO
import time
import os
from datetime import datetime
import threading
import numpy as np

# --- Configuration ---
MODEL_PATH = 'white.pt'
STREAM_URL = 'http://192.168.0.1:8080/video'
CONFIDENCE_THRESHOLD = 0.25
OUTPUT_DIR = 'output_frames'
SAVE_INTERVAL = 1  # Process and save frame every N seconds
# --- End Configuration ---

class StreamReader:
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.latest_frame = None
        self.stopped = False
        self.lock = threading.Lock() # Lock for thread-safe access to latest_frame
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            print(f"Error: Could not open stream at {self.stream_url}")
            self.stopped = True
            return

        while not self.stopped:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame. Stream might have ended.")
                # Optionally try to reconnect or simply stop
                self.stopped = True # Stop if stream fails
                break

            with self.lock:
                self.latest_frame = frame

        cap.release()
        print("Stream capture released.")

    def read(self):
        with self.lock:
            frame = self.latest_frame
        return frame # Return the latest frame (or None if not started/stopped)

    def stop(self):
        self.stopped = True
        if self.thread is not None:
            self.thread.join() # Wait for the thread to finish

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stream_reader = None # Initialize to None

    try:
        # Load YOLO model
        model = YOLO(MODEL_PATH)
        print(f"Successfully loaded model: {MODEL_PATH}")

        # Start stream reader thread
        stream_reader = StreamReader(STREAM_URL).start()
        print(f"Attempting to connect to stream: {STREAM_URL}")

        # Wait a moment for the stream to potentially connect and get the first frame
        time.sleep(2)

        if stream_reader.stopped:
             print("Stream reader failed to start. Exiting.")
             return

        print(f"Connected to stream. Processing frames...")
        last_save_time = time.time()

        while not stream_reader.stopped:
            # Get latest frame
            frame = stream_reader.read()
            if frame is None:
                # Wait briefly if no frame is available yet
                time.sleep(0.1)
                continue

            # Check if it's time to process and save a frame
            current_time = time.time()
            if current_time - last_save_time >= SAVE_INTERVAL:
                # Perform inference on a copy to avoid issues if frame updates mid-inference
                frame_copy = frame.copy()
                results = model.predict(frame_copy, conf=CONFIDENCE_THRESHOLD, verbose=False)

                # Process results and save
                for result in results:
                    annotated_frame = result.plot()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") # Added microseconds for uniqueness
                    output_path = os.path.join(OUTPUT_DIR, f"frame_{timestamp}.jpg")
                    cv2.imwrite(output_path, annotated_frame)
                    # print(f"Saved frame to: {output_path}") # Optional: reduce console spam

                last_save_time = current_time

            # Add a small delay to prevent busy-waiting, adjust as needed
            time.sleep(0.05)


    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if stream_reader is not None:
            print("Stopping stream reader...")
            stream_reader.stop()
        print("Stream connection closed.")

if __name__ == "__main__":
    main()