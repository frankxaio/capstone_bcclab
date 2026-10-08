import numpy as np
import pandas as pd
from pathlib import Path
from TemPose.utils import normalize_joints, normalize_position, get_court_info, to_court_coordinate

# 19 skeleton bone connections
COCO_BONES = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (7, 9), (6, 8),
    (8,10), (5, 6), (5,11), (6,12), (11,12),
    (11,13), (13,15), (12,14), (14,16),
]

def load_keypoints(csv_path):
    df = pd.read_csv(csv_path)
    data = df.iloc[:, 1:].values    # skip the frame column
    return data.reshape(len(df), -1, 2)  # (T, J, 2)

def load_bbox(csv_path):
    df = pd.read_csv(csv_path)
    data = df.iloc[:, 1:].values    # skip the frame column
    return data                     # (T, 4)

def pad_to_length(data, T_max):
    T = data.shape[0]
    if T >= T_max:
        return data[:T_max]
    pad_shape = (T_max - T,) + data.shape[1:]
    return np.concatenate([data, np.zeros(pad_shape)], axis=0)

def compute_bones(kps, bones_idx):
    # returns (T, B, 2)
    # ============================== TODO [Ex 3.3a] ==============================
    # Bone vectors: for every (i, j) pair in `bones_idx` compute the vector
    # from joint i to joint j in every frame. kps: (T, J, 2) -> (T, B, 2).
    # ============================================================================
    raise NotImplementedError("TODO [Ex 3.3a]")

def process_clip(clip_name, tempose_root, court_txt, T_max=30):
    """
    tempose_root\
    clip_1\
        clip_1_top.csv
        clip_1_bottom.csv
        clip_1_ball.csv
    """
    clip_dir = Path(tempose_root)

    top_csv         = clip_dir / f"{clip_name}_top.csv"
    bot_csv         = clip_dir / f"{clip_name}_bottom.csv"
    top_bbox_csv    = clip_dir / f"{clip_name}_top_bbox.csv"
    bottom_bbox_csv = clip_dir / f"{clip_name}_bottom_bbox.csv"
    ball_csv        = clip_dir / f"{clip_name}_ball.csv"

    # read and normalise the keypoints
    top_key     = load_keypoints(top_csv)
    bottom_key  = load_keypoints(bot_csv)
    top_bbox    = load_bbox(top_bbox_csv)
    bottom_bbox = load_bbox(bottom_bbox_csv)

    # court info
    court_info = get_court_info(court_txt)

    # keep only X, Y (columns 2 and 3) and normalise
    ball_df = pd.read_csv(ball_csv)
    ball_xy = ball_df.iloc[:, 2:4].values / np.array([1280, 720])

    # trim to the shortest length and pad
    n = min(len(top_key), len(bottom_key), len(ball_xy), len(top_bbox), len(bottom_bbox))
    top, bottom, ball_xy, top_bbox, bottom_bbox = top_key[:n], bottom_key[:n], ball_xy[:n], top_bbox[:n], bottom_bbox[:n]
    '''
    top    = pad_to_length(top, T_max)      # (T_max, J, 2)
    bottom = pad_to_length(bottom, T_max)   # (T_max, J, 2)
    ball_xy= pad_to_length(ball_xy, T_max)
    top_bbox    = pad_to_length(top_bbox, T_max)
    bottom_bbox = pad_to_length(bottom_bbox, T_max)
    '''

    # position
    top_feet, bottom_feet = top[:, -2:, :], bottom[:, -2:, :]       # (T_max, J=2, 2)
    top_feet = to_court_coordinate(top_feet, court_info['H'])       # (T_max, J=2, 2)
    bottom_feet = to_court_coordinate(bottom_feet, court_info['H']) # (T_max, J=2, 2)
    
    # normalize
    top_pos = normalize_position(top_feet, court_info)
    bottom_pos = normalize_position(bottom_feet, court_info)
    pos = np.concatenate([top_pos, bottom_pos], axis=1)  # (T_max, 2, 2)

    # normalize keypoints
    top = normalize_joints(top, top_bbox)           # (T_max, 1, 2)
    bottom = normalize_joints(bottom, bottom_bbox)  # (T_max, 1, 2)

    # bone vectors
    # ============================== TODO [Ex 3.3b] ==============================
    # Build `human_pose` with shape (1, T, 2, J+B, 2):
    #   for each player concatenate joints and COCO_BONES bone vectors along the
    #   joint axis -> (T, J+B, 2); stack [top, bottom] on a new axis 1; add a
    #   leading batch dimension.
    # ============================================================================
    raise NotImplementedError("TODO [Ex 3.3b]")
    pos        = np.expand_dims(pos, axis=0)
    shuttle    = ball_xy[None, ...]
    # videos_len = np.array([n])
    # labels     = np.array([0], dtype=int)

    # save to clip_dir
    np.save(clip_dir / 'JnB_bone.npy', human_pose)
    np.save(clip_dir / 'pos.npy', pos)
    np.save(clip_dir / 'shuttle.npy', shuttle)
    # np.save(clip_dir / 'videos_len.npy', videos_len)
    # np.save(clip_dir / 'labels.npy', labels)

    print(f"[✓] Saved .npy files into {clip_dir}")