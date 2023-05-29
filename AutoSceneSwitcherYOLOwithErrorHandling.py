import cv2
import numpy as np
from obswebsocket import obsws, requests
import time

# Set up OBS WebSocket connection
try:
    ws = obsws(host, port, password)
    ws.connect()
except Exception as e:
    print(f"Failed to connect to OBS WebSocket: {e}")
    exit(1)

# Load YOLO model and class labels
def load_yolo_model():
    try:
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        return net, classes
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        exit(1)

net, classes = load_yolo_model()

# Define scene names and camera indexes
scene_mapping = {'USB Cam': 0, 'IP Cam 1': 'rtsp://username:password@1.1.1.1', 'IP Cam 2': 'rtsp://username:password@1.1.1.1', 'IP Cam 3': 'rtsp://username:password@1.1.1.1', 'IP Cam 4': 'rtsp://username:password@1.1.1.1', 'IP Cam 5': 'rtsp://username:password@1.1.1.1', 'IP Cam 6': 'rtsp://username:password@1.1.1.1'}

# Open cameras
cameras = []
for i in range(len(scene_mapping)):
    try:
        camera = cv2.VideoCapture(i)
        if not camera.isOpened():
            raise Exception(f"Failed to open camera {i}")
        cameras.append(camera)
    except Exception as e:
        print(f"Failed to open camera {i}: {e}")

if not cameras:
    print("No cameras available. Exiting...")
    exit(1)

# Main loop
counter = 0  # Frame counter
while True:
    cat_occupancy = {}  # Store cat occupancy for each camera

    # Capture frames from cameras
    for scene_name, camera_index in scene_mapping.items():
        try:
            camera = cameras[camera_index]
            ret, frame = camera.read()

            if not ret:
                print(f"Failed to capture frame from camera {camera_index}")
                continue

            # Perform object detection on frame
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            layer_outputs = net.forward(net.getUnconnectedOutLayers())

            # Initialize variables for cat detection
            cat_count = 0

            # Loop over each output layer
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # Check if the detected object is a cat
                    if classes[class_id] == 'cat' and confidence > 0.5:
                        cat_count += 1

            # Update cat occupancy dictionary
            cat_occupancy[scene_name] = cat_count

            # Display frame with bounding boxes
            cv2.imshow(scene_name, frame)

        except Exception as e:
            print(f"Error processing camera {camera_index}: {e}")

    # Find scene with highest cat occupancy every 20 frames
    if counter == 20:
        try:
            max_occupancy_scene = max(cat_occupancy, key=cat_occupancy.get)

            # Switch to new scene if different from current scene
            current_scene = ws.call(requests.GetCurrentScene()).getName()
            if max_occupancy_scene != current_scene:
                ws.call(requests.SetCurrentScene(max_occupancy_scene))
                time.sleep(2)  # Delay in seconds between scene switches

            counter = 0  # Reset frame counter

        except Exception as e:
            print(f"Error during scene switching: {e}")

    else:
        counter += 1

    # Release frames from cameras
    for camera in cameras:
        try:
            camera.grab()
        except Exception as e:
            print(f"Failed to release frame from camera: {e}")

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
for camera in cameras:
    camera.release()
ws.disconnect()
cv2.destroyAllWindows()
