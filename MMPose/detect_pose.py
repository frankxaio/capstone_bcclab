import cv2
import csv
import os
import argparse
import numpy as np
import pandas as pd
from mmpose.apis import MMPoseInferencer
# from mayavi import mlab
from tqdm import tqdm
from pathlib import Path


def normalize_joints(
    arr: np.ndarray,
    bbox: np.ndarray,
    v_height=None,
    center_align=False,
):
    '''
    - `arr`: (m, J, 2), m=2.
    - `bbox`: (m, 4), m=2.
    
    Output: (m, J, 2), m=2.
    '''
    # ============================== TODO [Ex 3.2d] ==============================
    # Same function as TemPose/prepare_train.py normalize_joints (Ex 3.2d).
    # Only used by the commented-out demos below; implement it or import yours.
    # ============================================================================
    raise NotImplementedError("TODO [Ex 3.2d]")

'''

def demo_human_2d_and_3d(img_path):
    inferencer_2d = MMPoseInferencer('human')
    inferencer_3d = MMPoseInferencer(pose3d='human3d')
    result_generator_2d = inferencer_2d(img_path, show=False)
    result_generator_3d = inferencer_3d(img_path, show=False)
    for result_2d, result_3d in zip(result_generator_2d, result_generator_3d):
        for e_2d, e_3d in zip(result_2d['predictions'][0],
                              result_3d['predictions'][0]):  # batch_size=1 (default)
            
            ## 2d
            keypoints_2d = np.array(e_2d['keypoints'])[None, :]
            bbox = np.concatenate([
                keypoints_2d.min(1),
                keypoints_2d.max(1)
            ], axis=-1)
            
            keypoints_2d_normalized = normalize_joints(
                keypoints_2d, bbox,
                center_align=True
            )[0]

            coords = np.concatenate([
                keypoints_2d_normalized[:, 0:1],
                keypoints_2d_normalized[:, 1:],
                np.zeros((len(e_2d['keypoints']), 1)),
            ], axis=1)
            
            # create a 3D figure
            fig = mlab.figure(figure='My Figure')

            # scatter plot
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.03)

            # label each point
            for i, (x, y, z) in enumerate(coords):
                mlab.text3d(x, y, z, str(i), scale=0.02)

            # show the figure
            mlab.show(stop=True)

            ## 3d
            coords = np.array(e_3d['keypoints'])

            # x
            print('0 -> 1 :', coords[1] - coords[0])
            print('0 -> 4 :', coords[4] - coords[0])

            # y
            print('12 -> 13 :', coords[13] - coords[12])

            # z
            print('0 -> 7 :', coords[7] - coords[0])

            # create a 3D figure
            fig = mlab.figure(figure='My Figure')

            # scatter plot
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.1)

            # label each point
            for i, (x, y, z) in enumerate(coords):
                mlab.text3d(x, y, z, str(i), scale=0.1)

            # show the figure
            mlab.show(stop=True)


def demo_human_2d(img_path):
    inferencer = MMPoseInferencer('human')
    x = []
    y = []
    result_generator = inferencer(img_path, show=False)
    for result in result_generator:
        for e in result['predictions'][0]:  # batch_size=1 (default)
            keypoints_2d = np.array(e['keypoints'])[None, :]
            bbox = np.concatenate([
                keypoints_2d.min(1),
                keypoints_2d.max(1)
            ], axis=-1)
            
            keypoints_2d_normalized = normalize_joints(
                keypoints_2d, bbox,
                center_align=True
            )[0]

            coords = np.concatenate([
                keypoints_2d_normalized[:, 0:1],
                keypoints_2d_normalized[:, 1:],
                np.zeros((len(e['keypoints']), 1)),
            ], axis=1)
            
            # create a 3D figure
            fig = mlab.figure(figure='My Figure')

            # scatter plot
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.03)

            # label each point
            for i, (x, y, z) in enumerate(coords):
                mlab.text3d(x, y, z, str(i), scale=0.02)

            # show the figure
            mlab.show(stop=True)


def demo_human_3d(img_path):
    inferencer = MMPoseInferencer(pose3d='human3d')
    result_generator = inferencer(img_path, show=False)

    for result in result_generator:
        for e in result['predictions'][0]:  # batch_size=1 (default)
            coords = np.array(e['keypoints'])

            # x
            print('0 -> 1 :', coords[1] - coords[0])
            print('0 -> 4 :', coords[4] - coords[0])

            # y
            print('12 -> 13 :', coords[13] - coords[12])

            # z
            print('0 -> 7 :', coords[7] - coords[0])

            # create a 3D figure
            fig = mlab.figure(figure='My Figure')

            # scatter plot
            pts = mlab.points3d(coords[:, 0], coords[:, 1], coords[:, 2], scale_factor=0.05)

            # label each point
            for i, (x, y, z) in enumerate(coords):
                mlab.text3d(x, y, z, str(i), scale=0.05)

            # show the figure
            mlab.show(stop=True)

'''


def test_bug(inferencer, p):
    result_generator = inferencer(str(p), show=False)
    for result in result_generator:
        pass


def no_bug(p):
    inferencer = MMPoseInferencer(pose3d='human3d')
    result_generator = inferencer(str(p), show=False)
    for result in result_generator:
        pass

def save_player_keypoints(player_data, output_path):
    """Save player keypoints into a CSV file."""
    with open(output_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        headers = ['frame']
        num_keypoints = len(player_data[0]['keypoints']) if player_data else 0
        headers += [f'x{i}' for i in range(num_keypoints)] + [f'y{i}' for i in range(num_keypoints)]
        csv_writer.writerow(headers)
        
        for frame_data in player_data:
            frame_idx = frame_data['frame']
            keypoints = frame_data['keypoints']
            row = [frame_idx]
            for kp in keypoints:
                row.extend(kp)  # Append x, y, and confidence
            csv_writer.writerow(row)

    print(f"Keypoints saved to {output_path}")

def save_player_bbox(bbox, frame_id, output_path):
    df = pd.DataFrame(bbox, columns=["x1", "y1", "x2", "y2"])
    df["frame"] = frame_id
    df = df[["frame", "x1", "y1", "x2", "y2"]]
    df.to_csv(output_path, index=False)

def read_court_corners(file_path):
    """
    Read the first four court corner coordinates from a .txt file.
    The first four lines should contain two numbers separated by a semicolon:
    1st line: Upper-left corner
    2nd line: Bottom-left corner
    3rd line: Bottom-right corner
    4th line: Upper-right corner
    """
    corners = []
    
    with open(file_path, 'r') as file:
        for _ in range(4):  # Read only the first 4 lines
            line = file.readline().strip()
            if not line:
                break  # Stop if there are fewer than 4 lines
            x, y = map(float, line.split(';'))
            corners.append((x, y))
    
    if len(corners) == 4:
        return np.array(corners, dtype=np.int32)
    else:
        print("Error: Could not read all 4 court corners.")
        return np.array(corners, dtype=np.int32)

def is_inside_court(foot_positions, court_polygon, threshold=15):
    """Check if a player is inside the court based on their foot positions."""
    # ============================== TODO [Ex 1.7a] ==============================
    # A player is "inside" if EITHER heel is inside the court polygon or within
    # `threshold` pixels of it. Hint: cv2.pointPolygonTest(polygon, pt, True)
    # returns a signed distance (positive inside, negative outside).
    # ============================================================================
    raise NotImplementedError("TODO [Ex 1.7a]")

def visualize_video_estimated(inferencer, in_path, csv_output_dir='pose_data.csv', cap=None, court_corners=None, start_frame_number = 0, draw = False):
    in_path = str(in_path)
    # inferencer = MMPoseInferencer('human')

    video_name = os.path.splitext(os.path.basename(in_path))[0]
    os.makedirs(csv_output_dir, exist_ok=True)
    output_video_path = os.path.join(csv_output_dir, f"{video_name}_visualized_output.mp4")

    # Read all frames from the video
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    if len(frames) == 0:
        print("Error: No frames read from the video.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if draw:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    result_generator = inferencer(frames, show=False)

    # Data for CSVs
    bottom_player_data = []
    top_player_data = []
    bottom_bbox_data = [] 
    top_bbox_data = []
    frame_id = []

    # Process each frame
    for frame_idx, (result, frame) in enumerate(tqdm(zip(result_generator, frames), total=len(frames), desc=f'Processing {video_name}')):

        frame_idx += start_frame_number

        frame_id.append(frame_idx)
        people = result['predictions'][0]  # the list of dict of 1 person

        players = []
        for person in people:
            keypoints = person['keypoints']
            bbox = person['bbox'][0]
            if len(keypoints) < 17:
                continue  # 17 keypoints are required

            # Extract keypoints
            players.append({
                'foot_position': (keypoints[15], keypoints[16]),  # (left heel, right heel)
                'keypoints': keypoints,
                'bbox': bbox
            })

        # Check if there are at least two players inside the court
        # ============================== TODO [Ex 1.7b] ==============================
        # Filter out non-players (umpires, line judges, audience): keep only the
        # entries of `players` that are inside the court (use is_inside_court).
        # Store the result in `players_inside_court`.
        # ============================================================================
        raise NotImplementedError("TODO [Ex 1.7b]")

        # fill the frame with empty keypoints if less than 2 players
        if len(players_inside_court) < 2:
            if draw:
                out_video.write(frame)
            bottom_player_data.append({'frame': frame_idx, 'keypoints': [(0, 0)] * 17})
            top_player_data.append({'frame': frame_idx, 'keypoints': [(0, 0)] * 17})
            bottom_bbox_data.append((0, 0, 0, 0))
            top_bbox_data.append((0, 0, 0, 0))
            continue

        # Sort players by their foot positions (y-coordinate) to find the top and bottom players
        # ============================== TODO [Ex 1.7c] ==============================
        # Pick `bottom_player` and `top_player` from `players_inside_court`:
        # sort the players by heel y-coordinate (see "foot_position"), LARGEST first.
        # The first one is the bottom player (closest to the camera), the second one
        # is the top player.
        # ============================================================================
        raise NotImplementedError("TODO [Ex 1.7c]")

        # save keypoints
        bottom_player_data.append({'frame': frame_idx, 'keypoints': bottom_player['keypoints']})
        top_player_data.append({'frame': frame_idx, 'keypoints': top_player['keypoints']})
        bottom_bbox_data.append(bottom_player['bbox'])
        top_bbox_data.append(top_player['bbox'])

        # draw keypoints
        if draw:
            for player, color in [(top_player, (0, 255, 0)), (bottom_player, (0, 0, 255))]:
                for kp in player['keypoints']:
                    x, y = map(int, kp[:2])
                    cv2.circle(frame, (x, y), 5, color, -1)

            out_video.write(frame)

    # Release resources
    cap.release()

    if draw:
        out_video.release()
        print(f"Visualized video saved to {output_video_path}")

    bottom_file = os.path.join(csv_output_dir, f"{video_name}_bottom.csv")
    top_file = os.path.join(csv_output_dir, f"{video_name}_top.csv")
    bottom_bbox_file = os.path.join(csv_output_dir, f"{video_name}_bottom_bbox.csv")
    top_bbox_file = os.path.join(csv_output_dir, f"{video_name}_top_bbox.csv")

    save_player_keypoints(bottom_player_data, bottom_file)
    save_player_keypoints(top_player_data, top_file)  
    save_player_bbox(bottom_bbox_data, frame_id, bottom_bbox_file)
    save_player_bbox(top_bbox_data, frame_id, top_bbox_file)

def process_pose(inferencer, video_path, csv_output_dir, court_file, start_frame_number, draw = False):

    # Read court corners
    court_corners = read_court_corners(court_file)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        visualize_video_estimated(
            inferencer=inferencer,
            in_path=video_path,
            csv_output_dir = csv_output_dir,
            cap=cap,
            court_corners=court_corners,
            start_frame_number = start_frame_number,
            draw=draw
        )
    cap.release()

