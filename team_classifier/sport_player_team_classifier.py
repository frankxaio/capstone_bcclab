import os
import csv
import cv2
import random
import numpy as np
from ultralytics import YOLO
import supervision as sv
from sports.common.team import TeamClassifier

# ====== Settings ======
VIDEO_PATH = r"sample.mp4"
OUTPUT_VIDEO_PATH = r"box.mp4"
CSV_PATH = r"box.csv"
COURT_FILE = r"court_detection/court.txt"
FRAME_STRIDE = 32
DEVICE = "cuda"
FRAME_LIMIT = None  # None processes the whole video
BOX_COLORS = [(0, 0, 255), (255, 0, 0)]  # Team 0: red, Team 1: blue
CONF_THRESH = 0.6  # confidence threshold
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 293, 252, 996, 668
MIN_CROPS = 3500      # target number of player crops to fit the classifier
MIN_FIT_CROPS = 100   # below this, clustering is unreliable: stop with an error
CROP_ROUNDS = 5       # maximum rounds of random frame sampling
# ====================

def set_roi(x1, y1, x2, y2):
    global ROI_X1, ROI_Y1, ROI_X2, ROI_Y2
    ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = x1, y1, x2, y2

## Consideration: Use the real corner coordinates as ROI
def read_corner_set_roi(file_path):

    def parse_point(line_str):
        x_str, y_str = line_str.split(';')
        return float(x_str), float(y_str)
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        raise ValueError("Less than 3 lines")
    
    corner1, corner2 = lines[0].strip(), lines[2].strip()
    x1, y1 = parse_point(corner1)
    x2, y2 = parse_point(corner2)
    set_roi(x1, y1, x2, y2)

def box_overlaps_roi(x1, y1, x2, y2):
    # ============================== TODO [Ex 3.8a] ==============================
    # Return True if box (x1, y1, x2, y2) overlaps the court ROI
    # (ROI_X1, ROI_Y1, ROI_X2, ROI_Y2) at all, False if it is fully outside.
    # A box that only touches the ROI border counts as overlapping.
    # ============================================================================
    raise NotImplementedError("TODO [Ex 3.8a]")

# avoid too small boxes such as rackets
def test_box_area(x1, y1, x2, y2, threshold = 3000):
    area = (x2 - x1) * (y2 - y1)
    return area > threshold

def select_player_boxes(detections):
    '''
    Keep the YOLO detections that are probably players.

    detections: sv.Detections with `.xyxy` (N, 4) and `.confidence` (N,)
    Returns: list of the kept xyxy boxes, in detection order.
    '''
    # ============================== TODO [Ex 3.8b] ==============================
    # Keep a detection only if it is
    #   (a) confident:  confidence >= CONF_THRESH
    #   (b) on court:   box_overlaps_roi(x1, y1, x2, y2)
    #   (c) big enough: test_box_area(x1, y1, x2, y2)   (rejects rackets etc.)
    # Use the box corners converted to int for (b) and (c), but return the
    # original xyxy boxes unchanged.
    # ============================================================================
    raise NotImplementedError("TODO [Ex 3.8b]")

def crop_upper_body(frame, xyxy, ratio=0.5):
    """
    Crop the upper body from the original bbox (xyxy).
    ratio=0.5 keeps the top 50% of the box.
    """
    x1, y1, x2, y2 = map(int, xyxy)
    height = y2 - y1
    upper_y2 = int(y1 + height * ratio)

    # keep the indices inside the frame
    upper_y2 = min(upper_y2, frame.shape[0])
    x1 = max(x1, 0)
    x2 = min(x2, frame.shape[1])

    return frame[y1:upper_y2, x1:x2]

def get_random_frames(video_path, total=20):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        return []
    chosen_frames = sorted(random.sample(range(frame_count), min(total, frame_count)))
    frames = []
    for f_idx in chosen_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

def get_crops_from_rallies(model, rally_video_paths, frames_per_rally=30):
    crops = []
    for clip in os.listdir(rally_video_paths):
        video_path = f'{rally_video_paths}/{clip}/{clip}.mp4'
        frames = get_random_frames(video_path, total=frames_per_rally)
        for frame in frames:
            result = model(frame, imgsz=640, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            for xyxy in select_player_boxes(detections):
                crops.append(sv.crop_image(frame, xyxy))
    return crops

def get_player_crops(model, video_path, stride=32):
    crops = []
    frame_gen = sv.get_video_frames_generator(source_path=video_path, stride=stride)
    for idx, frame in enumerate(frame_gen):
        result = model(frame, imgsz=640, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        print(f"[Debug] Frame {idx}: Detected {len(detections.xyxy)} objects")
        for xyxy in select_player_boxes(detections):
            crops.append(crop_upper_body(frame, xyxy, ratio=0.6))
    return crops

def draw_boxes(frame, detections, team_ids):
    for xyxy, team_id in zip(detections.xyxy, team_ids):
        x1, y1, x2, y2 = map(int, xyxy)
        if box_overlaps_roi(x1, y1, x2, y2):
            color = BOX_COLORS[int(team_id)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"Team {team_id}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2, cv2.LINE_AA)
    return frame

## Consideration: use the full video with fallback
def train_yolo(video_path_1, court_file=COURT_FILE):
    model = YOLO("team_classifier/yolov8s.pt").to(DEVICE)
    classifier = TeamClassifier(device=DEVICE, batch_size=1024)

    # Use the court detected for this video, not the bundled sample court
    read_corner_set_roi(court_file)

    ## fallback: sample more random frames, for a bounded number of rounds
    crops = []
    for _ in range(CROP_ROUNDS):
        crops += get_crops_from_rallies(model, video_path_1, frames_per_rally=35)
        print(f"Collected {len(crops)} player crops")
        if len(crops) >= MIN_CROPS:
            break
        print("The data is not enough, sampling more frames")
    if len(crops) < MIN_FIT_CROPS:
        raise RuntimeError(f"Only {len(crops)} player crops after {CROP_ROUNDS} rounds; "
                           "check the court ROI and that the video has several rallies")
    if len(crops) < MIN_CROPS:
        print(f"[WARN] Fitting the team classifier on {len(crops)} crops (target {MIN_CROPS})")

    classifier.fit(crops)
    return classifier

def predict_teams(clip_dir, clip, classifier, start_frame_number, draw = False):
    '''
    print("[1] Loading the model...")
    model = YOLO("yolov8s.pt").to(DEVICE)
    classifier = TeamClassifier(device=DEVICE)

    print("[2] Collecting player crops that overlap the ROI...")
    crops = get_player_crops(model, video_path, stride=FRAME_STRIDE)
    print(f"Collected {len(crops)} player crops")

    print("[3] Training the TeamClassifier...")
    classifier.fit(crops)
    '''
    model = YOLO("team_classifier/yolov8s.pt").to(DEVICE)

    video_path = f'{clip_dir}/{clip}.mp4'
    output_video_path = f'{clip_dir}/{clip}_teams.mp4'
    csv_path = f'{clip_dir}/{clip}_teams.csv'

    print("[4] Analysing and drawing the video...")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if draw:
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    csv_rows = [["frame", "player_id", "x1", "y1", "x2", "y2", "team_id"]]
    frame_idx = start_frame_number

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (FRAME_LIMIT is not None and frame_idx > FRAME_LIMIT):
            break

        result = model(frame, imgsz=640, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        valid_xyxy = select_player_boxes(detections)

        if valid_xyxy:
            crops = [sv.crop_image(frame, xyxy) for xyxy in valid_xyxy]
            team_ids = classifier.predict(crops)

            for pid, (xyxy, tid) in enumerate(zip(valid_xyxy, team_ids)):
                x1, y1, x2, y2 = map(int, xyxy)
                csv_rows.append([frame_idx, pid, x1, y1, x2, y2, int(tid)])

            detections.xyxy = np.array(valid_xyxy)

            if draw:
                frame = draw_boxes(frame, detections, team_ids)

        if draw:
            writer.write(frame)
        frame_idx += 1

    cap.release()
    if draw:
        writer.release()

    print(f"[5] Output video: {output_video_path}")
    print(f"[6] Exported CSV: {csv_path}")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
