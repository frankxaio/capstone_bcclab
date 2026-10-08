import numpy as np
import pandas as pd

def normalize_joints(
    arr: np.ndarray,  # shape: (T, J, 2)
    bbox: np.ndarray,  # shape: (T, 4)
    v_height=None,
    center_align=True,
):
    """
    Normalize joints over T frames for a single player.
    
    - arr: (T, J, 2)
    - bbox: (T, 4)
    - v_height: scalar or None
    - center_align: bool
    Return:
    - normalized_arr: (T, J, 2)
    """
    # ============================== TODO [Ex 3.2a] ==============================
    # Make joints invariant to where the player stands and how big they look:
    #   - dist = v_height / 4 if v_height is given, else the bbox diagonal length
    #   - x -> (x - bbox_x1) / dist, y -> (y - bbox_y1) / dist,
    #     BUT a coordinate that is exactly 0 (missing joint) stays 0
    #   - if center_align: afterwards subtract the normalised bbox centre
    #     ((centre - top_left) / dist) from EVERY joint (missing ones included),
    #     so the bbox centre maps to (0, 0)
    # Return (T, J, 2).
    # ============================================================================
    raise NotImplementedError("TODO [Ex 3.2a]")

def normalize_position(arr: np.ndarray, court_info: dict):
    '''
    Normalized by court boundary.

    arr:    (T, 2, 2). There are N 'x' and N 'y'.
    Output: (T, 1, 2). Every 'x', 'y' in-court should be in [0, 1].
    '''
    # calculate the mean of arr
    # ============================== TODO [Ex 3.2b] ==============================
    # Average the P points per frame -> (T, 1, 2) (e.g. the two heels), then
    # normalise by the court borders so the court spans [0, 1] on both axes:
    #   x -> (x - x_mean) / (border_R - border_L) + 0.5   (same for y with U/D)
    # ============================================================================
    raise NotImplementedError("TODO [Ex 3.2b]")

def get_corner_camera(court_txt):
    with open(court_txt, 'r') as f:
        lines = f.readlines()
    
    corners = []
    for i in range(4):
        parts = lines[i].strip().split(';')
        x = float(parts[0])
        y = float(parts[1])
        corners.append([x, y])
    
    corners = np.array(corners).T  # shape: (2, 4)
    return corners


def get_H(court_txt):
    with open(court_txt, 'r') as f:
        lines = f.readlines()

    # find the line "Homography Matrix:"
    h_start = next(i for i, line in enumerate(lines) if "Homography Matrix:" in line)
    h_lines = lines[h_start + 1 : h_start + 4]

    H = []
    for line in h_lines:
        row = [float(x.strip()) for x in line.strip().replace('[', '').replace(']', '').replace(';', '').split(',')]
        H.append(row)

    return np.array(H)

def convert_homogeneous(arr: np.ndarray):
    '''
    The shape of 2D `arr` is (2, N). => The output will be (3, N).
    '''
    # print(arr.shape)
    return np.concatenate((arr, np.ones((1, arr.shape[1]))), axis=0)

def project(H: np.ndarray, P_prime: np.ndarray):
    '''
    Transform coordinates from the camera system to the court system.
    
    H: (3, 3)
    P_prime: (3, N)
    Output: (2, N)
    '''
    # ============================== TODO [Ex 3.1a] ==============================
    # Apply homography H to homogeneous points P_prime (3, N) and return the
    # Cartesian result (2, N): multiply, then divide x and y by w (last row).
    # ============================================================================
    raise NotImplementedError("TODO [Ex 3.1a]")

def get_court_info(court_txt):
    '''
    Get the homography matrix and the 4 corners of the court in the court coordinate corresponding to the video.
    '''
    H = get_H(court_txt)
    corner_camera = get_corner_camera(court_txt)
    corner_camera = convert_homogeneous(corner_camera)

    corner_court = project(H, corner_camera)

    xs = corner_court[0, :]
    ys = corner_court[1, :]
    
    return {
        'H': H,
        'border_L': xs.min(),  
        'border_R': xs.max(),  
        'border_U': ys.min(),  
        'border_D': ys.max(),  
    }

def to_court_coordinate(
    arr_camera: np.ndarray,
    H
):
    '''
    Convert the camera coordinate system to the court coordinate system.
    
    Input:
        arr_camera: np.ndarray of shape (30, 2, 2)
            30 frames, each with 2 points (e.g., top/bottom), each point (x, y)
        H: homography matrix

    Output:
        np.ndarray of shape (30, 2, 2)
            Projected court coordinates
    '''
    # ============================== TODO [Ex 3.1b] ==============================
    # Project every frame of `arr_camera` (T, P, 2) - P points of (x, y) -
    # into court coordinates with `H`, returning (T, P, 2).
    # Careful with the axes: convert_homogeneous/project work on (2, N) arrays
    # (one COLUMN per point), while arr_camera[i] is (P, 2) (one ROW per point).
    # ============================================================================
    raise NotImplementedError("TODO [Ex 3.1b]")
