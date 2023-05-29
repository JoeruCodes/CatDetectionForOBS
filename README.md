# Cat Detection Script Documentation (Python Script)

This script performs real-time cat detection using the YOLO (You Only Look Once) object detection algorithm and controls scene switching in OBS Studio based on cat occupancy.

## Prerequisites

Before running the script, make sure you have the following installed:

- Python 3.x
- OpenCV (`pip install opencv-python`)
- numpy (`pip install numpy`)
- obs-websocket-py (`pip install obs-websocket-py`)

## Usage

1. Make sure OBS Studio is running and the OBS WebSocket plugin is installed and enabled.
2. Modify the following variables in the script according to your OBS WebSocket connection details:
   - `host`: The hostname or IP address of the machine running OBS Studio.
   - `port`: The port number on which the OBS WebSocket server is listening.
   - `password`: The password set in OBS WebSocket settings.
4. Modify the `scene_mapping` dictionary to define your scene names and camera indexes/RTSP URLs. Add or remove entries as needed.
5. Run the script using the command `python3 AutoSceneSwitcherYOLOwithVisualizerAndErrorHandling.py`.
6. The script will open a window for each camera defined in `scene_mapping` and display real-time video with bounding boxes around detected cats.
7. The script will automatically switch to the scene with the highest cat occupancy every 20 frames.
8. To exit the script, press 'q' on the keyboard.

## Error Handling

The script includes error handling to handle common issues during execution. If an error occurs, the script will print an error message and exit gracefully. Make sure to check the console output for any error messages.

Note: Ensure that your system meets the hardware requirements for running the YOLO model and has the necessary dependencies installed.

## License

This script is provided under the MIT License. Feel free to modify and use it according to your needs.

## Disclaimer

The accuracy of cat detection depends on the performance and quality of the YOLO model. It may not detect all cats or may produce false positives. Use the script responsibly and verify the results before taking any actions based on the detected cat occupancy.

Please refer to the official documentation of OBS Studio, OpenCV, and obs-websocket-py for more details on their usage and features.

**Author**: Joel Medicala
