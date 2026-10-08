import math
import torch
import numpy as np

def adjust_lr(optimizer,LR_max,LR_min,e,num_epochs,warm_e,period=1):
    "adjusted learning cosine learning rate with linear warm-up"
    # ============================== TODO [Ex 3.6a] ==============================
    # Learning-rate schedule for epoch e -> store it in `lr`:
    #   - linear warm-up from 0 to LR_max during the first warm_e epochs
    #   - then cosine annealing from LR_max down to LR_min at num_epochs:
    #       LR_min + 0.5 * (LR_max - LR_min) * (1 + cos(period * pi * progress))
    #     where progress = (e - warm_e) / (num_epochs - warm_e)
    # ============================================================================
    raise NotImplementedError("TODO [Ex 3.6a]")
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group: # account for learning rate scale in optimizer
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    
def euclid_dist(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

def create_motion_robust(keypoints, pairs):
    num_steps = 3
    limbs = []
    for pose in keypoints:
        pose_limbs = []
        for start, end in pairs:
            start_point = pose[start]
            end_point = pose[end]
            if start_point[0] <= 0.01 or start_point[1] <= 0.01 or end_point[0] <= 0.01 or end_point[1] <= 0.01:
                limb = torch.zeros((num_steps, 2))
            else:
                x_linspace = torch.linspace(start_point[0], end_point[0], steps=num_steps)
                y_linspace = torch.linspace(start_point[1], end_point[1], steps=num_steps)
                limb = torch.stack([x_linspace, y_linspace], dim=1)
            pose_limbs.append(limb)
        limbs.append(torch.stack(pose_limbs))
    return torch.stack(limbs)

def get_2d_sincos_pos_embed(embed_dim, seq_len, cls_token=False):
    "Used from MAE paper"
    pos = np.arange(seq_len, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    "Used from MAE paper"
    # ============================== TODO [Ex 3.6b] ==============================
    # Fixed sin-cos positional embedding (Transformer / MAE style).
    # pos: positions (M,)  ->  emb: (M, embed_dim)
    #   omega_i = 1 / 10000^(i / (embed_dim/2)),  i = 0 .. embed_dim/2 - 1
    #   emb = [ sin(pos * omega) | cos(pos * omega) ]   (concatenate on axis 1)
    # ============================================================================
    raise NotImplementedError("TODO [Ex 3.6b]")


def main():
    return 1
        
if __name__ == '__main__':
    main()