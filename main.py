#!/usr/bin/env python
import os
import yaml
import torch
import pandas as pd
from pathlib import Path
import cv2
from collections import Counter
from mmpose.apis import MMPoseInferencer

from rally_clipping.get_result import timepoints_clipping
from TrackNetV3.predict       import predict_traj, load_tracknet_model
from TrackNetV3.denoise       import smooth
from MMPose.detect_pose        import process_pose
from HitNet.predict           import predict as hitnet_detect

from TemPose.generate_npy     import process_clip
from TemPose.predict_by_hit   import per_hit_predict
from TemPose.TemPoseII        import TemPoseII_TF

from team_classifier.sport_player_team_classifier import predict_teams, train_yolo

from utils import *
from court_detection.runner import detect_court

def main(video_path=Path("full_36.mp4"), *, court_executable=None, court_file=None,
         court_attempts=5, court_timeout=120):
    # ——— 0. Paths & config —————————————————————————————————————
    video_path = Path(video_path)
    name       = video_path.stem
    RALLY_OUTPUT_DIR = Path('videos') / name
    RESULT_OUTPUT_DIR = Path('results') / name

    COURT_OUTPUT  = RALLY_OUTPUT_DIR / 'court.txt'
    COURT_IMAGE   = RALLY_OUTPUT_DIR / 'court_image.png'

    # take the correct fps
    fps = get_video_fps(str(video_path))

    draw = False # True if you want to visualize the predictions
    logs = {}    # for recording execution time

    # mkdir RESULT_OUTPUT_DIR
    os.makedirs(RESULT_OUTPUT_DIR, exist_ok = True)
    
    # ——— 1. Rally clipping ——————————————————————————————————————————————
    print("\n[Message] Start rally clipping\n")
    recording_execution_time(logs, "Start Rally Clipping")
    start_frame = timepoints_clipping(video_path, fps)
    print("[Message] Rally clipping finished\n")
    recording_execution_time(logs, "End Rally Clipping")
    
    # ——— 2. Court Detection ——————————————————————————————————————————
    print("\n[Message] Start court detection")  
    recording_execution_time(logs, "Start Court Detection")

    detect_court(
        [RALLY_OUTPUT_DIR / clip / f"{clip}.mp4" for clip in start_frame],
        COURT_OUTPUT, COURT_IMAGE, check_court_coordinates,
        executable=court_executable, court_file=court_file,
        attempts=court_attempts, timeout=court_timeout,
    )
    print("[Message] Court detection finished\n")
    recording_execution_time(logs, "End Court Detection")
    
    # ——— 3. Trajectory & Pose Prediction —————————————————————————————————————
    print("\n[Message] Start trajectory & pose prediction\n")
    recording_execution_time(logs, "Start Trajectory & Pose Prediction")
    track_model = load_tracknet_model()
    inferencer = MMPoseInferencer('human', device='cuda')
    for clip in os.listdir(RALLY_OUTPUT_DIR):
        if not clip.startswith("clip_"):
            continue
        clip_dir  = RALLY_OUTPUT_DIR / clip
        clip_path = clip_dir / f"{clip}.mp4"
        start_frame_number = start_frame[clip]

        traj_csv = predict_traj(clip_path, str(clip_dir), track_model, start_frame_number=start_frame_number, verbose=False, draw=draw)

        ## Consideration: unable smooth is acceptable
        # df       = pd.read_csv(traj_csv, encoding="utf-8")
        # smooth(traj_csv, df)

        process_pose(inferencer, clip_path, str(clip_dir), COURT_OUTPUT, start_frame_number, draw)
    print("[Message] Trajectory & pose prediction finished\n")
    recording_execution_time(logs, "End Trajectory & Pose Prediction")
    
    # ——— 4. HitNet ————————————————————————————————————————
    print("\n[Message] Start hit detection\n")
    recording_execution_time(logs, "Start Hit Detection")
    hitnet_detect(RALLY_OUTPUT_DIR, start_frame)
    print("[Message] Hit detection finished\n")
    recording_execution_time(logs, "End Hit Detection")
    
    # ——— 5. Team Classification ——————————————————————————————————————
    ## Consideration: choose the longest clip to train the team classifier, instead of clip 11 and 101
    print("\n[Message] Start team classification\n")
    recording_execution_time(logs, "Start Team Classification")
    # clip_path = RALLY_OUTPUT_DIR / "clip_11" / f"clip_11.mp4"
    classifier = train_yolo(RALLY_OUTPUT_DIR, court_file=COURT_OUTPUT)
    for clip in os.listdir(RALLY_OUTPUT_DIR):
        if not clip.startswith("clip_"):
            continue
        clip_dir = RALLY_OUTPUT_DIR / clip
        print(f"\n[Team] Processing {clip} …")
        start_frame_number = start_frame[clip]
        predict_teams(clip_dir, clip, classifier, start_frame_number, draw)
    print("[Message] Team classification finished\n")
    recording_execution_time(logs, "End Team Classification")
    
    # ——— 6. TemPose  ——————————————————————————————————————
    # load model config
    print("\n[Message] Start TemPose\n")
    recording_execution_time(logs, "Start TemPose")
    cfg = yaml.safe_load(Path("TemPose/config/config.yml").read_text())
    mcfg = cfg["model"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TemPoseII_TF(
        poses_numbers=mcfg["input_dim"],
        time_steps=   mcfg["sequence_length"],
        num_people=   mcfg["num_people"],
        num_classes=  mcfg["output_dim"],
        dim=          mcfg["model_dim"],
        depth=        mcfg["depth_t"],
        depth_int=    mcfg["depth_n"],
        dim_head=     mcfg["head_dim"],
        emb_dropout=  cfg["hyperparameters"]["dropout"],
    ).to(device)
    ck = torch.load("TemPose/model.pt", map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    # Prepare containers for summarization
    timeline       = []
    all_hit_frames = []
    all_strokes    = []
    all_players    = []

    # For each clip: gen npy + per‐hit predict 
    for clip in sorted(os.listdir(RALLY_OUTPUT_DIR)):
        if not clip.startswith("clip_"):
            continue
        clip_dir = RALLY_OUTPUT_DIR / clip
        print(f"\n[TemPose] Processing {clip} …")
        # ① generate all the .npy required for TemPose
        process_clip(clip, str(clip_dir), COURT_OUTPUT, T_max=mcfg["sequence_length"])

        # ② do per‐hit classification
        hit_csv = clip_dir / f"{clip}_hits.csv"
        if not hit_csv.exists():
            print(f"[WARN] {hit_csv} not found; skipping.")
            continue

        # This returns a list of (frame_idx, stroke_name) for all hits in this clip
        hit_frames, strokes = per_hit_predict(model, clip_dir, hit_csv, cfg, device)

        # accumulate
        team_file = Path(f"{clip}_teams.csv")
        team_path = clip_dir / team_file
        ball_file = clip_dir / f"{clip}_ball.csv"

        # Read team data and determine Top/Bottom mapping
        df_team = pd.read_csv(team_path)
        df_ball = pd.read_csv(ball_file)
        
        for hit_frame, stroke in zip(hit_frames, strokes):

            # Check the y1 of bbox to determine which team is at the top or bottom
            # ============================== TODO [Ex 3.9] ==============================
            # Decide WHO hit the shot -> store the team id in `true_player`
            # (keep 'X' if it cannot be decided).
            #   1. From the FIRST frame of df_team, the box with the smallest y1 is the
            #      top player, the largest y1 is the bottom player (take their team_id).
            #   2. If the stroke is known (not "unknown"), its prefix tells the side:
            #      "Top_..." -> top player, "Bottom_..." -> bottom player.
            #   3. Unknown stroke: look up the shuttle (X, Y) at hit_frame in df_ball
            #      (first matching row). If there is none, print a warning and keep 'X' -
            #      do NOT skip the hit, all_players must stay aligned with hit_frames.
            #      Also keep 'X' when Visibility != 1 or X/Y are non-finite.
            #      Keep 'X' if no valid player box exists at hit_frame. A valid box
            #      has finite x1/y1/x2/y2, x2 > x1 and y2 > y1.
            #      Otherwise pick the team whose valid box at hit_frame is closest
            #      to the shuttle (use calculate_ball_bbox_dist).
            #   4. Known stroke: keep 'X' when the first frame does not contain
            #      two valid player boxes; a single box cannot establish both sides.
            # ===========================================================================
            raise NotImplementedError("TODO [Ex 3.9]")

            all_players.append(true_player)

            # Not use now
            '''
            ts = frame_to_timestamp(video_file, frame_idx, original_video_path=video_path)
            timeline.append({
                "clip": clip,
                "frame": frame_idx,
                "timestamp": ts,
                "stroke": stroke,
                "player_id": true_player
            })
            '''

        all_hit_frames.extend(hit_frames)
        all_strokes.extend(strokes)

    print("[Message] TemPose finished\n")
    recording_execution_time(logs, "End TemPose")
    
    # ——— 7. Summarize & print —————————————————————————————————————
    recording_execution_time(logs, "Start Summarize")
    '''
    if not timeline:
        print("\nNo hits detected across all clips.")
        return
    '''
    # hit_frames & strokes & players
    df_events = pd.DataFrame({"frame": all_hit_frames, "stroke": all_strokes, "player": all_players}).sort_values("frame").reset_index(drop=True)
    df_merged_events, df_stroke_count = merge_consecutuve_frame(df_events)
    
    # a) Stroke counts
    print("\n=== Stroke counts ===")
    for row in df_stroke_count.itertuples(index=False):
        print(f"{row.stroke:<20} {row.count:>5}")

    # b) Full hit timeline
    output_video_dir = RESULT_OUTPUT_DIR
    
    # Optionally save CSVs:
    df_stroke_count.to_csv(RESULT_OUTPUT_DIR / "summary_counts.csv", index=False)
    df_merged_events.to_csv(RESULT_OUTPUT_DIR / "hit_events.csv",  index=False)

    # Annotate video with hit events
    '''
    df_tl = pd.DataFrame(timeline)
    df_tl = df_tl.sort_values(["clip", "frame"]).reset_index(drop=True)
    print("\n=== Hit timeline ===")
    print("  timestamp stroke")
    # for entry in timeline:
    #    print(f"{entry['timestamp']}   {entry['stroke']}")
    '''
    print("\n[Message] Visualizing hits in video...")
    
    output_video_path = output_video_dir / f"{name}_annotated.mp4"
    visualize_hits_in_video(video_path, df_merged_events, output_path=output_video_path)

    #print(f"\n[Done] Summaries written to {RALLY_OUTPUT / 'summary_counts.csv'} and hit_timeline.csv")
    recording_execution_time(logs, "End Summarize")
    print_execution_time_with_plot(logs, output_video_dir)
    print("\n[All done]")

if __name__ == "__main__":
    main()
