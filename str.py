import cv2
from ultralytics import YOLO
import time
import os
from datetime import datetime
import threading
from queue import Queue
import numpy as np
from flask import Flask, Response, render_template_string # Added Flask imports

# --- Configuration ---
MODEL_PATH = 'white.pt'
STREAM_URL = 'http://192.168.0.1:8080/video'
CONFIDENCE_THRESHOLD = 0.25
OUTPUT_DIR = 'output_frames'
SAVE_INTERVAL = 1 # Process every frame for real-time display
QUEUE_SIZE = 1  # Max frames to keep in queue
# --- End Configuration ---

# --- Global variable for the latest annotated frame ---
latest_annotated_frame = None
frame_lock = threading.Lock() # To safely update the frame
# --- End Global variable ---

# --- Flask App Setup ---
app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    # Simple HTML page with an img tag pointing to the video feed
    return render_template_string(
        '<html><head><title>YOLO Real-time Detection</title></head>'
        '<body><h1>YOLO Real-time Detection</h1>'
        '<img src="{{ url_for(\'video_feed\') }}" width="640" height="480">'
        '</body></html>'
    )

def gen_frames():
    """Video streaming generator function."""
    global latest_annotated_frame
    while True:
        with frame_lock:
            frame_to_show = latest_annotated_frame
        
        if frame_to_show is None:
            # Optional: Display a placeholder or wait message
            # For simplicity, we'll just skip if no frame is ready
            time.sleep(0.1) # Avoid busy-waiting
            continue

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame_to_show)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(1/30) # Limit frame rate slightly

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# --- End Flask App Setup ---


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
                # Optional: Attempt to reconnect or simply stop
                time.sleep(1) # Wait a bit before retrying or stopping
                # Re-try connection logic could go here
                self.stopped = True # For now, just stop
                break
                
            # Clear queue if full to prioritize latest frame
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except:
                    pass
                    
            self.queue.put(frame)
        
        print("Stream reader thread stopping.")
        cap.release()
    
    def read(self):
        # Get the latest frame, potentially discarding older ones if queue fills up
        return self.queue.get() if not self.queue.empty() else None
    
    def stop(self):
        self.stopped = True

def run_flask_app():
    # Run Flask app in a separate thread
    # Use host='0.0.0.0' to make it accessible on your network
    print("Starting Flask server on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)

def main():
    global latest_annotated_frame
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Still useful for potential saving later

    # Start Flask server in a background thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()

    stream_reader = None # Initialize to None
    try:
        # Load YOLO model
        model = YOLO(MODEL_PATH)
        print(f"Successfully loaded model: {MODEL_PATH}")

        # Start stream reader thread
        stream_reader = StreamReader(STREAM_URL, QUEUE_SIZE).start()
        print(f"Attempting to connect to stream: {STREAM_URL}")

        # Wait briefly for the stream reader to potentially connect
        time.sleep(2) 
        if stream_reader.stopped:
             print("Stream reader failed to start. Exiting.")
             return # Exit if stream couldn't open

        print(f"Connected to stream. Processing frames...")

        while not stream_reader.stopped:
            # Get latest frame
            frame = stream_reader.read()
            if frame is None:
                time.sleep(0.01) # Wait briefly if no frame is available
                continue

            # Perform inference
            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

            # Assume only one result for simplicity, get the annotated frame
            if results:
                annotated_frame = results[0].plot()
                # Update the global frame for the web server
                with frame_lock:
                    latest_annotated_frame = annotated_frame.copy() # Use copy to avoid race conditions
            else:
                 # If no results, still update with the original frame
                 with frame_lock:
                    latest_annotated_frame = frame.copy()


            # Optional: Add a small delay if processing is too fast
            # time.sleep(0.01)

    except Exception as e:
        print(f"An error occurred in the main loop: {e}")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        if stream_reader:
            stream_reader.stop()
        print("Stream processing stopped.")
        # Note: Flask thread is a daemon, it will exit when the main thread exits.

if __name__ == "__main__":
    main()