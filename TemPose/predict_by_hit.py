# predict_by_hit.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from TemPose.TemPoseII import TemPoseII_TF
from TemPose.data.dataset import create_bones, interpolate_joints, get_stroke_types

def slice_and_pad(arr, center, T):
    # ============================== TODO [Ex 2.5a] ==============================
    # Cut rows  center - T//2 (inclusive) ... center + T//2 (exclusive)  out of
    # `arr` (axis 0), clipped to the array boundaries. If fewer than T rows are
    # left, zero-pad at the END so the result always has shape
    # (T, *arr.shape[1:]) and the same dtype as `arr`.
    # ============================================================================
    raise NotImplementedError("TODO [Ex 2.5a]")

def per_hit_predict(model, clip_dir: Path, hit_csv: Path, cfg: dict, device: torch.device):
    hp  = torch.from_numpy(np.load(clip_dir/"JnB_bone.npy")).float().to(device)   # (1, T, 2, J+B, 2)
    pos = torch.from_numpy(np.load(clip_dir/"pos.npy")).float().to(device)        # (1, T, 2, 2)
    sh  = torch.from_numpy(np.load(clip_dir/"shuttle.npy")).float().to(device)    # (1, T, 2)

    df_hits = pd.read_csv(hit_csv)
    hit_frames = df_hits.loc[df_hits['hit']==1, 'frame'].astype(int).tolist()
    first_frame_num = df_hits.loc[0]['frame']

    T = cfg["model"]["sequence_length"]
    stroke_names = get_stroke_types()  # names of classes 0..34

    print(f"Found {len(hit_frames)} hits, window size {T}")

    hfs = []
    strokes = []

    for i, hf in enumerate(hit_frames):
        relative_frame = hf - first_frame_num

        seg_hp  = slice_and_pad(hp[0].cpu().numpy(),  relative_frame, T)[None]   # (1, T, 2, J+B, 2)
        seg_pos = slice_and_pad(pos[0].cpu().numpy(), relative_frame, T)[None]   # (1, T, 2, 2)
        seg_sh  = slice_and_pad(sh[0].cpu().numpy(),  relative_frame, T)[None]   # (1, T, 2)

        window_start = max(0, relative_frame - T//2)
        window_end = min(hp.shape[1], relative_frame + T//2)
        valid_len = window_end - window_start
        t_pad = torch.tensor([valid_len], device=device)

        x = torch.tensor(seg_hp,  dtype=torch.float32, device=device)
        p = torch.tensor(seg_pos, dtype=torch.float32, device=device)
        s = torch.tensor(seg_sh,  dtype=torch.float32, device=device)

        # ============================== TODO [Ex 2.5b] ==============================
        # Reshape the tensors into what TemPoseII_TF expects:
        #   x : (1, T, 2, J+B, 2)          -> (1, T, 2, (J+B)*2)   joint+bone features
        #   sp: positions p (1, T, 2, 2) flattened to (1, T, 4), then concatenated
        #       with shuttle s (1, T, 2)  -> (1, T, 6)
        # ============================================================================
        raise NotImplementedError("TODO [Ex 2.5b]")
        # sp = sp.view(b, t, -1)
        
        with torch.no_grad():
            pred_idx = model.predict(x, sp, t_pad)
        name = stroke_names[pred_idx.item()]
        hfs.append(hf)
        strokes.append(name)

    return hfs, strokes
