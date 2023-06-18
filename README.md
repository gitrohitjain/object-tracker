The goal of this project is to perform object tracking of objects in a video. 
We will use [this](https://drive.google.com/file/d/1YDa0_kFIOYrC9408sP_mL2ammKHvMQVu/view?usp=sharing) traffic video as input for the object tracking.

Tracking is a two step process-
1. Object detection where we classify and localize objects in an image.  For the task of object we use pretrained [YOLOv5 weights](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt)

2. Assigning ID to objects by temporally tracking them across frames. For tracking and ID assignment we use [DeepSORT](https://pypi.org/project/deep-sort-realtime/) algorithm.

**__DeepSORT Algorithm__** <br>
In DeepSORT (Deep Simple Online Realtime Tracking) has these four major steps-

`Object detection`: Initially, an object detection algorithm is used to detect and localize objects of interest in each frame of a video.

`Feature extraction`: A deep neural network is employed to extract appearance features or embeddings from the detected objects. These embeddings encode the appearance characteristics of the objects.

`Data association`: The Kalman filter is used to predict the state of each object and estimate its position in the current frame by assuming object propogates with constant velocity. The estimated positions, along with the appearance features, are then used to associate objects across frames. The association is performed based on a distance metric (cosine distance) between the appearance embeddings.

`Track management`: Each object is assigned a unique identity based on the associations made in the previous step. The identities are maintained over time, and the state estimates are continually updated using the Kalman filter.


**__How to run this code?__**
1. Git clone this repo: `git clone https://github.com/gitrohitjain/object-tracker.git`
2. Create a python environment: `python -m venv workenv`
3. Activate environment: `source workenv/bin/activate`
4. Install packages: `pip install -r requirements.txt`
5. Update the paths for input video and to save the output.
6. Run the script: `python detect_and_track.py`


**__How to interpret the saved output video__**
1. On the top left on the video you'll see:-
    * Count of total number of objects detected in current frame with confidence greater than threshold
    * Count of total number of objects detected from start till current frame
    * Count of total number of individual objects detected from start till current frame

2. For each detected object there is a localizing bounding box, a class label, unique id assigned by tracker, and confidence score of detection.


Output video can be viewed [here](https://drive.google.com/file/d/1xIQ8TjHUZMdl8hBhwhiJyp4EL0zEX9ub/view?usp=sharing)




