# Some functions in this file are adapted from ai_badminton (Adobe Research)
# Original source: https://github.com/jhwang7628/monotrack/blob/main/modified-tracknet/train-hitnet.ipynb
# Licensed under the Adobe Research License - for non-commercial research use only.

import os
import sys
import cv2
import pandas as pd
import numpy as np
from skimage.transform import resize
from ai_badminton.pose import read_player_poses
from ai_badminton.trajectory import Trajectory

window_size = 7
half = window_size // 2
speed = 1.0

COURT_OUTPUT = 'court_detection/court.txt'

def read_court(court_path):
    # Open the file and read the first four lines
    with open(court_path, "r") as file:
        lines = file.readlines()
        # Extract the first four lines (court corner coordinates)
    corners = []
    for i in range(4):  # Read only the first four lines
        x, y = map(float, lines[i].strip().split(";"))  # Convert to float
        corners.append((x, y))  # Store as (x, y) tuple
    return corners

def fetch_data(basedir: str, rally: str, for_train = False):
    '''
    Fetch the corners, trajectory, hits, poses from a rally

    Argument:
        basedir: the directiory path of clip
        rally: the name of rally, it's something like 'clip_1'
    
    Return:
        corners: the coordinate of 4 court corners
        trajectory: the processed trajectory
        hit: the label (0 = no hit, 1 = hit)
        bottom_player, top_player: poses data
    '''
    court_path = COURT_OUTPUT
    traj_path = os.path.join(basedir, f'{rally}_ball.csv')  # E.g. videos/sample/clip_1/clip_1_ball.csv
    pose_path = os.path.join(basedir, rally)                # E.g. videos/sample/clip_1/clip_1


    # Note: Consider using the same court for all rallies
    if not os.path.exists(court_path):
        print(f'{court_path} not found')
        return None
    court_pts = read_court(court_path)
    corners = np.array([court_pts[1], court_pts[2], court_pts[0], court_pts[3]]).flatten()

    trajectory = Trajectory(traj_path, interp = False)
    poses = read_player_poses(pose_path)
    bottom_player, top_player = poses[0], poses[1]

    data_dict = {'corners': corners, 'trajectory': trajectory, 'bottom_player': bottom_player, 'top_player': top_player}

    if for_train:
        hit_path = os.path.join(basedir, 'hits', f'{rally}_hit.csv')
        hit = pd.read_csv(hit_path)
        hit = hit.values[:, 1]
        data_dict['hit'] = hit

    return data_dict

def resample(series, s):
    flatten = False
    if len(series.shape) == 1:
        # reshape (not ndarray.resize): works for views and does not modify the caller's array
        series = series.reshape(-1, 1).astype('float64')
        flatten = True
    series = resize(
        series, (int(s * series.shape[0]), series.shape[1]),
    )
    if flatten:
        series = series.flatten()
    return series  

def process(basedir: str, rally: str, for_train = False):
    '''
    Prepare frame-level input data with fixed-size sliding window (7 frames) and edge padding.

    Args:
        basedir (str): Path to match directory
        rally (str): Rally name like 'clip_1'

    Returns:
        x_t (ndarray): Feature vectors of shape (num_frames, 7 * feature_dim)
        y_t (ndarray): Binary labels per frame (num_frames,)
    '''
    import numpy as np
    import cv2

    video_path = f'{basedir}/rally_video/{rally}.mp4'
    x_list, y_list = [], []

    # fetch rally data
    data_dict = fetch_data(basedir, rally, for_train)
    if data_dict is None:
        return None

    trajectory = data_dict['trajectory']
    
    bottom_player = data_dict['bottom_player']
    top_player = data_dict['top_player']

    # resample all inputs
    trajectory.X = resample(np.array(trajectory.X), speed)
    trajectory.Y = resample(np.array(trajectory.Y), speed)
    
    bottom_player = resample(bottom_player.values, speed)
    top_player = resample(top_player.values, speed)

    num_frames = len(trajectory.X)

    if for_train:
        hit = data_dict['hit']
        hit = resample(hit, speed).round().astype(int)
        num_frames = len(hit)

        # label smoothing for temporal context
        # ============================== TODO [Ex 2.1a] ==============================
        # Label smoothing: hits are rare (about 1 frame per shot). Build `smooth_hit`,
        # a copy of `hit` where the frames right before and after every hit are
        # also labelled 1. Watch the array boundaries.
        # ============================================================================
        raise NotImplementedError("TODO [Ex 2.1a]")

    def get_padded_window(arr, t):
        # ============================== TODO [Ex 2.1b] ==============================
        # Return a window of exactly `window_size` rows centred on frame t
        # (rows t - half ... t + half). Near the edges, pad by REPEATING the first
        # row of `arr` (at the start of the clip) or the last row (at the end), so
        # the length is always window_size.
        # ============================================================================
        raise NotImplementedError("TODO [Ex 2.1b]")

    for t in range(num_frames):
        # ============================== TODO [Ex 2.1c] ==============================
        # Build the feature matrix `x` for frame t, shape (window_size, feature_dim):
        #   [shuttle X, shuttle Y | bottom player pose | top player pose]
        # Use get_padded_window on trajectory.X, trajectory.Y, bottom_player, top_player.
        # ============================================================================
        raise NotImplementedError("TODO [Ex 2.1c]")
        x_list.append(x)

        if for_train:                    
            y_list.append(smooth_hit[t])              # binary label

    x_t = np.array(x_list)
    y_t = np.array(y_list) if for_train else None
    return x_t, y_t

def make_data_for_train(matches_dir, matches_lst):
    X_lst, y_lst = [], []
    for match in matches_lst:
        basedir = f'{matches_dir}/{match}'
        for video in os.listdir(f'{basedir}/rally_video/'):
            rally = video.split('.')[0]
            data = process(basedir, rally, for_train = True)
            if data is None:
                continue
            x, y = data
            X_lst.append(x)
            y_lst.append(y)

    X = np.vstack(X_lst)
    y = np.hstack(y_lst)
    
    return X, y

def make_data_for_predict(clip_dir, rally):
    X_lst = []
    data = process(clip_dir, rally, for_train = False)
    if data:
        x, _ = data
        X_lst.append(x)

    X = np.vstack(X_lst)
    
    return X