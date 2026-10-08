#!/usr/bin/env python3
"""Compare results/<name>/hit_events.csv with the ShuttleSet labels in data/shuttleset/<name>.csv."""
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from scipy.special import kl_div

# label
class_ls = [
    'net shot', 'return net', 'smash', 'wrist smash', 'lob', 'defensive return lob',
    'clear', 'drive', 'back-court drive', 'drop', 'passive drop', 'push',
    'rush', 'defensive return drive', 'cross-court net shot', 'short service', 'long service', 'unknown'
]

def label2num(label):
    return class_ls.index(label)

def num2label(num):
    return class_ls[num]

def evaluate_event_level(prediction_df, truth_df, player_mapping, class_ls, save_path=None):
    """
    Event-level evaluation (player + stroke).
    
    Args:
        prediction_df: DataFrame with columns [start_frame, end_frame, stroke, player]
        truth_df: DataFrame with columns [frame, type, player]
        player_mapping: dict mapping prediction player -> truth player, e.g. {"0": "A", "1": "B", "X": "X"}
        class_ls: list of stroke types
        save_path: path to save results (confusion matrix & report), optional
        
    Returns:
        report (dict), accuracy (float), macro_f1 (float),
        skip_stats (dict: {'skip_zero': ..., 'skip_multi': ...})
    """
    
    import seaborn as sns
    import matplotlib.pyplot as plt

    truth_labels, pred_labels = [], []
    matches = match_hit_events(prediction_df, truth_df)
    matched_truth = {truth for _, truth in matches}
    skip_zero = skip_multi = 0
    for truth, row in enumerate(truth_df.itertuples(index=False)):
        if truth in matched_truth:
            continue
        covered = ((prediction_df['start_frame'] <= row.frame_num) &
                   (row.frame_num <= prediction_df['end_frame'])).any()
        if covered:
            skip_multi += 1
        else:
            skip_zero += 1
    for prediction, truth in matches:
        row = truth_df.iloc[truth]
        pred = prediction_df.iloc[prediction]
        pred_type = pred['stroke'].split('_', 1)[-1]
        pred_player = player_mapping[str(pred['player'])]
        truth_labels.append(f"{row['player']}_{row['type']}")
        pred_labels.append(f"{pred_player}_{pred_type}")

    # Event-level class set
    event_classes = []
    for p in sorted(set(truth_df['player']) | {'X'}):
        for s in class_ls:
            event_classes.append(f"{p}_{s}")
    
    # classification report
    report = classification_report(
        truth_labels, pred_labels,
        labels=event_classes,
        target_names=event_classes,
        digits=4,
        output_dict=True,
        zero_division=0
    )
    
    accuracy = accuracy_score(truth_labels, pred_labels) if truth_labels else 0.0        
    macro_f1 = f1_score(truth_labels, pred_labels, average='macro', zero_division=0) if truth_labels else 0.0
    
    # Skip statistics
    total_truth = len(truth_df)
    skip_stats = {
        "skip_zero": {"count": skip_zero, "ratio": skip_zero / total_truth if total_truth else 0.0},
        "skip_multi": {"count": skip_multi, "ratio": skip_multi / total_truth if total_truth else 0.0}
    }

    # transform player + stroke --> stroke
    truth_stroke_labels = [t.split('_')[-1] for t in truth_labels]
    pred_stroke_labels = [p.split('_')[-1] for p in pred_labels]
    stroke_report = classification_report(
        truth_stroke_labels, pred_stroke_labels,
        labels=class_ls,
        target_names=class_ls,
        digits=4,
        output_dict=True,
        zero_division=0
    )

    # confusion matrix for stroke
    conf = confusion_matrix(truth_stroke_labels, pred_stroke_labels, labels=class_ls)
    plt.figure(figsize=(12, 10))
    sns.heatmap(conf, annot=False, xticklabels=class_ls, yticklabels=class_ls, cmap="Blues")
    plt.title("Stroke-type Confusion Matrix")
    if save_path:
        plt.savefig(f"{save_path}/stroke_confusion_matrix.png")
    else:
        plt.show()
    plt.close()
    
    # save classification report
    report_df = pd.DataFrame(report).transpose()
    stroke_report_df = pd.DataFrame(stroke_report).transpose()
    if save_path:
        report_df.to_csv(f"{save_path}/events_classification_report.csv")
        stroke_report_df.to_csv(f"{save_path}/stroke_classification_report.csv")
    
    return report, accuracy, macro_f1, skip_stats

def match_hit_events(prediction_df, truth_df):
    """Match each prediction and truth row at most once, using inclusive intervals.

    An interval containing zero or multiple truth hits is ineligible. Duplicate
    predictions compete for their sole truth hit; the shortest interval wins,
    then the earliest start, then input order. Returned indices are row positions.
    """
    frames = truth_df['frame_num'].to_numpy(dtype=float)
    intervals = prediction_df[['start_frame', 'end_frame']].to_numpy(dtype=float)
    if not np.isfinite(frames).all() or not np.isfinite(intervals).all():
        raise ValueError('Hit frames and interval bounds must be finite')
    if (frames < 0).any() or (intervals < 0).any():
        raise ValueError('Hit frames and interval bounds must be nonnegative')
    if (frames != np.floor(frames)).any() or (intervals != np.floor(intervals)).any():
        raise ValueError('Hit frames and interval bounds must be integers')
    if (intervals[:, 0] > intervals[:, 1]).any():
        raise ValueError('start_frame must not exceed end_frame')
    candidates = []
    for prediction, (start, end) in enumerate(intervals):
        hits = np.flatnonzero((frames >= start) & (frames <= end))
        if len(hits) == 1:
            candidates.append((end - start, start, prediction, int(hits[0])))
    matched, used_truth = [], set()
    for _, _, prediction, truth in sorted(candidates):
        if truth not in used_truth:
            matched.append((prediction, truth))
            used_truth.add(truth)
    return sorted(matched)


def evaluate_hit_detection(prediction_df, truth_df, save_path=None):
    """Report one-to-one event counts. Event detection has no defined TN/accuracy.

    Rejected multi-hit intervals count as FP and leave all their truth hits
    unmatched unless another eligible prediction matches them.
    """
    tp = len(match_hit_events(prediction_df, truth_df))
    fp, fn = len(prediction_df) - tp, len(truth_df) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    metrics = {'TP': tp, 'FP': fp, 'FN': fn, 'precision': precision,
               'recall': recall, 'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    if save_path:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.bar(['TP', 'FP', 'FN'], [tp, fp, fn])
        ax.set_title('One-to-one hit-event counts')
        ax.set_ylabel('Events')
        fig.savefig(Path(save_path) / 'hit_event_counts.png')
        plt.close(fig)
        pd.DataFrame([metrics]).to_csv(Path(save_path) / 'hit_detection_metrics.csv', index=False)
    return metrics

def evaluate_stroke_distribution(prediction_df, truth_df, class_ls, player_mapping, save_path=None, epsilon=1e-8):
    """
    Evaluate stroke distribution with KL divergence.
    
    Args:
        prediction_df: DataFrame with ['stroke', 'player']
        truth_df: DataFrame with ['type', 'player']
        class_ls: list of all stroke types
        player_mapping: dict, e.g. {'0': 'A', '1': 'B', 'X': 'X'} (predict player -> truth player)
        epsilon: small value to avoid log(0) in KL
    
    Returns:
        result_dict: {
            'KL_A': float,
            'KL_B': float,
            'KL_baseline': float,
            'pred_dist_A': pd.Series,
            'pred_dist_B': pd.Series,
            'truth_dist_A': pd.Series,
            'truth_dist_B': pd.Series
        }
    """
    import matplotlib.pyplot as plt

    # Map player ids to ShuttleSet players
    prediction_df = prediction_df.copy()
    prediction_df['player'] = prediction_df['player'].astype(str).map(player_mapping)

    # drop unknown stroke
    prediction_df = prediction_df[prediction_df['stroke']!= 'unknown']
    truth_df = truth_df[truth_df['type']!= 'unknown']
    
    class_ls = [label for label in class_ls if label != 'unknown']

    # Ground truth A, B
    truth_df = truth_df.copy()
    truth_A = truth_df[truth_df['player'] == 'A']['type'].value_counts().reindex(class_ls, fill_value=0)
    truth_B = truth_df[truth_df['player'] == 'B']['type'].value_counts().reindex(class_ls, fill_value=0)
    
    # Prediction A, B
    prediction_df = prediction_df.copy()
    prediction_df['stroke'] = prediction_df['stroke'].apply(lambda x: x.split('_')[-1])
    pred_A = prediction_df[prediction_df['player'] == 'A']['stroke'].value_counts().reindex(class_ls, fill_value=0)
    pred_B = prediction_df[prediction_df['player'] == 'B']['stroke'].value_counts().reindex(class_ls, fill_value=0)
    
    # Convert counts to probability distributions
    truth_A_prob = (truth_A + epsilon) / (truth_A.sum() + epsilon * len(class_ls))
    truth_B_prob = (truth_B + epsilon) / (truth_B.sum() + epsilon * len(class_ls))
    pred_A_prob = (pred_A + epsilon) / (pred_A.sum() + epsilon * len(class_ls))
    pred_B_prob = (pred_B + epsilon) / (pred_B.sum() + epsilon * len(class_ls))

    # Uniform baseline, remove unknown stroke
    uniform_prob = np.ones(len(class_ls)) / len(class_ls)
    
    # KL divergence
    KL_A = np.sum(kl_div(truth_A_prob, pred_A_prob))
    KL_B = np.sum(kl_div(truth_B_prob, pred_B_prob))
    KL_A_baseline = np.sum(kl_div(truth_A_prob, uniform_prob))
    KL_B_baseline = np.sum(kl_div(truth_B_prob, uniform_prob))

    # bar plot between KL_A, KL_b, KL_baseline
    labels = ['Player A', 'Player B']
    model_vals = [KL_A, KL_B]
    baseline_vals = [KL_A_baseline, KL_B_baseline]

    x = np.arange(len(labels))  # A, B
    width = 0.35  # bar width
    plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, model_vals, width, label='Model', color='#4C72B0')
    plt.bar(x + width/2, baseline_vals, width, label='Uniform Baseline', color='#C44E52')
    plt.xticks(x, labels)
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence per Player (Model vs Baseline)")
    plt.legend()

    if save_path:
        plt.savefig(f"{save_path}/stroke_distribution_KL.png")
    else:
        plt.show()
    plt.close()
    
    result_dict = {
        'KL_A': KL_A,
        'KL_B': KL_B,
        'KL_A_baseline': KL_A_baseline,
        'KL_B_baseline': KL_B_baseline,
        'pred_dist_A': pred_A_prob,
        'pred_dist_B': pred_B_prob,
        'truth_dist_A': truth_A_prob,
        'truth_dist_B': truth_B_prob
    }
    
    return result_dict

def evaluate(prediction_df, truth_df, player_mapping, class_ls, save_path=None):
    report, accuracy, macro_f1, skip_stats = evaluate_event_level(prediction_df, truth_df, player_mapping, class_ls, save_path=save_path)
    metrics_dict = evaluate_hit_detection(prediction_df, truth_df, save_path=save_path)
    result_dict = evaluate_stroke_distribution(prediction_df, truth_df, class_ls, player_mapping, save_path=save_path)

    # Event-level evaluation
    print(f"Event-type evaluation (player + stroke, matched events only):")
    print(f"\tAccuracy = {accuracy:.4f}, Macro F1 = {macro_f1:.4f}")

    # Skip stats
    print(f"Skip stats:")
    print(f"\t{skip_stats['skip_zero']['count']} has been skipped ({(skip_stats['skip_zero']['ratio'] * 100):.2f}%) since no prediction found.")
    print(f"\t{skip_stats['skip_multi']['count']} has been skipped ({(skip_stats['skip_multi']['ratio'] * 100):.2f}%) since no eligible one-to-one match was found.")

    # Hit-event evaluation
    print("Hit-event evaluation (one-to-one):")
    print(f"\tTP = {metrics_dict['TP']}, FP = {metrics_dict['FP']}, FN = {metrics_dict['FN']}, Precision = {metrics_dict['precision']:.4f}, Recall = {metrics_dict['recall']:.4f}, F1 = {metrics_dict['f1']:.4f}")

    # Stroke distribution evaluation
    print(f"Stroke distribution evaluation:")
    print(f"\tKL_A = {result_dict['KL_A']:.4f}, KL_A_baseline = {result_dict['KL_A_baseline']:.4f}")
    print(f"\tKL_B = {result_dict['KL_B']:.4f}, KL_B_baseline = {result_dict['KL_B_baseline']:.4f}")


def detect_player_mapping(prediction_df, truth_df):
    """Pick the team id -> ShuttleSet player mapping that agrees with more labelled hits.

    In ShuttleSet, player A is the winner of the match. The pipeline's team ids come
    from clustering, so 0 and 1 can swap between runs and videos.
    """
    candidates = [{'0': 'A', '1': 'B', 'X': 'X'}, {'0': 'B', '1': 'A', 'X': 'X'}]

    matches = match_hit_events(prediction_df, truth_df)

    def agreement(mapping):
        return sum(mapping[str(prediction_df.iloc[p]['player'])] == truth_df.iloc[t]['player']
                   for p, t in matches)

    return max(candidates, key=agreement)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', help='video stem, for example full_36')
    name = parser.parse_args().name
    root = Path(__file__).resolve().parents[1]
    save_path = root / 'results' / name
    prediction_df = pd.read_csv(save_path / 'hit_events.csv')
    # Players are 0, 1, or X (unknown); a single X makes pandas read the column as text.
    prediction_df['player'] = prediction_df['player'].astype(str)
    truth_df = pd.read_csv(root / 'data' / 'shuttleset' / f'{name}.csv')
    player_mapping = detect_player_mapping(prediction_df, truth_df)

    print("Video name:", name)
    print(f"Player mapping: team 0 -> {player_mapping['0']}, team 1 -> {player_mapping['1']}")
    print("==============================================================================")
    evaluate(prediction_df, truth_df, player_mapping, class_ls, save_path=save_path)
    print("==============================================================================")
