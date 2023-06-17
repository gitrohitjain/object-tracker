import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

def run_detector(model, frame, device='cpu'):
    model.to(device)
    preds = model(frame)
    labels = preds.xyxyn[0][:, -1]
    coords = preds.xyxyn[0][:, :-1]
    return labels, coords


def unwrap_detector_output(obj_to_track, yolo_classes, labels, coords, height, width, conf_threshold):
    detected_objs = []
    for i in range(len(labels)):
        x1, y1, x2, y2, conf = coords[i]
        x = int(x1*width)
        y = int(y1*height)
        w = int((x2-x1)*width)
        h = int((y2-y1)*height)
        conf = float(conf.item())
        label = yolo_classes[int(labels[i])]
        if conf >= conf_threshold:
            if label in obj_to_track:
                detected_objs.append(([x, y, w, h], conf, label))
    return detected_objs


if __name__ == "__main__":
    detector_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    yolo_classes = detector_model.names
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tracker = DeepSort()

    IN_VIDEO_PATH = 'traffic.mp4'
    cap = cv2.VideoCapture(IN_VIDEO_PATH)

    OUT_VIDEO_PATH = 'out.mp4'
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    success, _img = cap.read()
    height, width, _ = _img.shape
    fps = 5
    video_writer = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps, (width, height))

    obj_to_track = ['car','truck','bus','person','bicycle','motorcycle']
    counter = 0
    visited_ids = set()
    obj_count_tracker = {obj:0 for obj in obj_to_track}

    while True:
        if counter == 400:
            break
        print(f'Processing {counter + 1} frame')
        counter += 1
        success, frame = cap.read()
        if not success:
            break
        labels, coords = run_detector(detector_model, frame, device)
        detected_objs = unwrap_detector_output(obj_to_track, yolo_classes, labels, coords, height, width, conf_threshold=0.5)
        tracked_objs = tracker.update_tracks(detected_objs, frame=frame)

        detected_count = 0
        for track_obj in tracked_objs:
            if not track_obj.is_confirmed():
                continue
            detected_class = track_obj.det_class
            detection_conf = track_obj.det_conf 
            track_id = track_obj.track_id
            x1, y1, x2, y2 = track_obj.to_ltrb()
            x1 , y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if detected_class is not None and detection_conf is not None:
                if track_id not in visited_ids:
                    obj_count_tracker[detected_class] += 1
                    visited_ids.add(track_id)
                detected_count += 1
                ### update frames with its ID, bounding boxes, labels and confidence scores
                cv2.rectangle(frame,(x1, y1),(x2, y2),(0,0,0),2)
                cv2.putText(frame, f"id:{track_id}" , (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
                cv2.putText(frame, f"{detected_class.upper()}" , (x1, y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 180 , 0), 4)
                cv2.putText(frame, f"{round(detection_conf*100)}%" , (x2 - 60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128,0, 128), 5)

        ### summarize object count on frame

        #update number of objects in current frame
        cv2.putText(frame, f'Current Frame object Count: {detected_count}', (22,75), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,0,0), 5)

        #update total number of objects detected from start till current frame
        total_count = sum(obj_count_tracker.values())
        cv2.putText(frame, f'Total Object Count: {total_count}', (22, 75*2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,255,0), 5)
        
        #update total count of individual objects detected from start till current frame
        for obj in obj_to_track:
            if obj_count_tracker[obj] > 0:
                cv2.putText(frame, f'{obj.upper()}: {obj_count_tracker[obj]}', (22, 75 + 75*(obj_to_track.index(obj)+2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 5)
        video_writer.write(frame)

    cap.release()
    video_writer.release()
    

