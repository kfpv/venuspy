import cv2
from ultralytics import YOLO
import time
import os
from datetime import datetime
import threading
from queue import Queue
import numpy as np

# --- Configuration ---
MODEL_PATH = 'white.pt'
STREAM_URL = 'http://192.168.0.1:8080/video'
CONFIDENCE_THRESHOLD = 0.25
OUTPUT_DIR = 'output_frames'
SAVE_INTERVAL = 1
QUEUE_SIZE = 1  # Max frames to keep in queue
# --- End Configuration ---

class StreamReader:
    def __init__(self, stream_url, queue_size=10):
        self.stream_url = stream_url
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        
    def start(self):
        thread = threading.Thread(target=self.update, daemon=True)
        thread.start()
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
                print("Error: Failed to grab frame.")
                break
                
            # Clear queue if full
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except:
                    pass
                    
            self.queue.put(frame)
        
        cap.release()
    
    def read(self):
        return self.queue.get() if not self.queue.empty() else None
    
    def stop(self):
        self.stopped = True

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Load YOLO model
        model = YOLO(MODEL_PATH)
        print(f"Successfully loaded model: {MODEL_PATH}")

        # Start stream reader thread
        stream_reader = StreamReader(STREAM_URL, QUEUE_SIZE).start()
        print(f"Connected to stream. Processing frames...")

        last_save_time = time.time()

        while not stream_reader.stopped:
            # Get latest frame
            frame = stream_reader.read()
            if frame is None:
                continue

            # Check if it's time to process and save a frame
            # current_time = time.time()
            # if current_time - last_save_time >= SAVE_INTERVAL:
                # Perform inference
            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            for result in results:
                #annotated_frame = result.plot()
                print(result.obb.xywhr)
                # if result.obb.xywhr:
                #     xywhr = result.obb.xywhr
                #     print(xywhr)
                # else:
                #     print("no det")
                #timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                #output_path = os.path.join(OUTPUT_DIR, f"frame_{timestamp}.jpg")
                #cv2.imwrite(output_path, annotated_frame)
                #print(f"Saved frame to: {output_path}")
            
            # last_save_time = current_time

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if 'stream_reader' in locals():
            stream_reader.stop()
        print("Stream connection closed.")

if __name__ == "__main__":
    main()
