import pdb
import math
import numpy as np
from collections import OrderedDict, defaultdict
import json
import time
import torch.nn.functional as F
import copy
import pathlib 
import torch
from run_on_video import clip, vid2clip, txt2clip
import logging
from utils.basic_utils import l2_normalize_np_array
import argparse
import torch.backends.cudnn as cudnn
from pathlib import Path
import multiprocessing as mp
from main.config import TestOptions, setup_model
from sklearn.metrics import precision_recall_curve
from config import PROMPT_TYPE, STORY_LIKE, NUM_CLIPS, THRESHOLD_LIST, NOISE_THRES, FPS, MODEL_VERSION, OUTPUT_FEAT_SIZE
from helper import load_txt, save_txt, save_json
def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def compute_hl_hit1(qid2preds, qid2gt_scores_binary):

    qid2max_scored_clip_idx = {k: np.argmax(v["saliency"]) for k, v in qid2preds.items()} # the most salient video clips in prediction for each query

    hit_scores = np.zeros((len(qid2preds), 1)) # query dimension, score dimension --> element wise
    qids = list(qid2preds.keys())
    for idx, qid in enumerate(qids):
        pred_clip_idx = qid2max_scored_clip_idx[qid] # for each query, find the most salient video clip idx
        gt_scores_binary = qid2gt_scores_binary[qid]   # (#clips, 3) # binary ground truth whether it is salient or not
        if pred_clip_idx < len(gt_scores_binary):
            hit_scores[idx] = gt_scores_binary[pred_clip_idx] # hit)scores: whther it is salient in GT or not
    # aggregate scores from 3 separate annotations (3 workers) by taking the max. --> I only have one worker
    # then average scores from all queries.
    hit_at_one = float(f"{100 * np.mean(np.max(hit_scores, 1)):.2f}")
    return hit_at_one


def compute_hl_ap(qid2preds, qid2gt_scores_binary, num_workers=8, chunksize=50):
    qid2pred_scores = {k: v["saliency"] for k, v in qid2preds.items()}
    ap_scores = np.zeros((len(qid2preds)))   # (#preds, 3)
    qids = list(qid2preds.keys())
    input_tuples = []
    for idx, qid in enumerate(qids):
        # for w_idx in range(1):  # annotation score idx: I only have one annotator
        # print(f"qid = {qid}") # correct id
        y_true = qid2gt_scores_binary[qid]
        y_predict = qid2pred_scores[qid]
        # F.softmax(y_predict, dim=0)
        # I only have one annotator
        # print(f"y_true = {len(y_true)}, y_pred = {len(y_predict)}")
        input_tuples.append((idx, y_true, y_predict)) # qid, annotator, GT, prediction

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for idx, score in pool.imap_unordered(
                    compute_ap_from_tuple, input_tuples, chunksize=chunksize):
                ap_scores[idx] = score
    else:
        for input_tuple in input_tuples:
            idx, score = compute_ap_from_tuple(input_tuple)
            ap_scores[idx] = score
    # print(f"len of ap_score = {ap_scores.shape}") # 474
    # it's the same if we first average across different annotations, then average across queries
    # since all queries have the same #annotations.
    mean_ap = float(f"{100 * np.mean(ap_scores):.2f}")
    return mean_ap

def get_ap(y_true, y_predict, interpolate=True, point_11=False):
    """
    Average precision in different formats: (non-) interpolated and/or 11-point approximated
    point_11=True and interpolate=True corresponds to the 11-point interpolated AP used in
    the PASCAL VOC challenge up to the 2008 edition and has been verfied against the vlfeat implementation
    The exact average precision (interpolate=False, point_11=False) corresponds to the one of vl_feat

    :param y_true: list/ numpy vector of true labels in {0,1} for each element
    :param y_predict: predicted score for each element
    :param interpolate: Use interpolation?
    :param point_11: Use 11-point approximation to average precision?
    :return: average precision

    ref: https://github.com/gyglim/video2gif_dataset/blob/master/v2g_evaluation/__init__.py

    """
    # Check inputs
    assert len(y_true) == len(y_predict), "Prediction and ground truth need to be of the same length"
    if len(set(y_true)) == 1:
        if y_true[0] == 0:
            return 0  # True labels are all zeros
            # raise ValueError('True labels cannot all be zero')
        else:
            return 1
    else:
        assert sorted(set(y_true)) == [0, 1], "Ground truth can only contain elements {0,1}"

    # Compute precision and recall
    # should be scores or probability
    precision, recall, _ = precision_recall_curve(y_true, y_predict)
    recall = recall.astype(np.float32)

    if interpolate:  # Compute the interpolated precision
        for i in range(1, len(precision)):
            precision[i] = max(precision[i - 1], precision[i])

    if point_11:  # Compute the 11-point approximated AP
        precision_11 = [precision[np.where(recall >= t)[0][-1]] for t in np.arange(0, 1.01, 0.1)]
        return np.mean(precision_11)
    else:  # Compute the AP using precision at every additionally recalled sample
        indices = np.where(np.diff(recall))
        return np.mean(precision[indices])
    

def compute_ap_from_tuple(input_tuple):
    idx, y_true, y_predict = input_tuple
    if len(y_true) < len(y_predict):
        # print(f"len(y_true) < len(y_predict) {len(y_true), len(y_predict)}")
        y_predict = y_predict[:len(y_true)]
    elif len(y_true) > len(y_predict):
        # print(f"len(y_true) > len(y_predict) {len(y_true), len(y_predict)}")
        _y_predict = np.zeros(len(y_true))
        _y_predict[:len(y_predict)] = y_predict
        y_predict = _y_predict

    score = get_ap(y_true, y_predict)
    return idx, score


def mk_gt_scores(gt_data, clip_length=1):
    """
    deal with one query at one time
    should load GT saliency data: output shape: [#video len, 1]
    """
    # ground truth:{"1iuQjyMP8_c_0": {"query": " The sea. Many of us have a deep longing to be by the water. Perhaps more so than ever during the pandemic.", "saliency": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}
    saliency_list = np.array(gt_data["saliency"])
    saliency = np.expand_dims(saliency_list, 1)  # [#videoltn, 1]
    return  saliency # [#videoltn, 1]
    # """gt_data, dict, """
    # num_clips = int(gt_data["duration"] / clip_length)
    # saliency_scores_full_video = np.zeros((num_clips, 1))
    # relevant_clip_ids = np.array(gt_data["relevant_clip_ids"])  # (#relevant_clip_ids, )
    # saliency_scores_relevant_clips = np.array(gt_data["saliency_scores"])  # (#relevant_clip_ids, 3)
    # saliency_scores_full_video[relevant_clip_ids] = saliency_scores_relevant_clips
    # return saliency_scores_full_video  # (#clips_in_video, 3)  the scores are in range [0, 4] --> [#videoltn, 1]


def eval_highlight(submission, ground_truth, verbose=True):
    """
    Args:
        submission:
        ground_truth:
        verbose:
    """
    # qid2preds: {qid: {"pred_saliency_scores": saliency score}}
    # qid2preds = {list(d.keys())[0]: d for d in submission} # qid and predicted score
    # qid2gt_scores_full_range = {list(d.keys())[0]: mk_gt_scores(d) for d in ground_truth}  # qid and GT score
    qid2preds =  {key: value for d in submission for key, value in d.items()}
    qid2gt_scores_full_range = {key: value["saliency"] for d in ground_truth for key, value in d.items()}
    # gt_saliency_score_min: int, in [0, 1, 2, 3, 4]. The minimum score for a positive clip.
    gt_saliency_score_min_list = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    saliency_score_names = ["0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    highlight_det_metrics = {}
    for gt_saliency_score_min, score_name in zip(gt_saliency_score_min_list, saliency_score_names):
        # 2 -- Fair, 3 -- Good, 4 -- VeryGood
        start_time = time.time()
        print(f"gt_saliency_score_min = {gt_saliency_score_min}")
        qid2gt_scores_binary = {
            k: [float(element >= gt_saliency_score_min) for element in v]
            for k, v in qid2gt_scores_full_range.items()
        }
        hit_at_one = compute_hl_hit1(qid2preds, qid2gt_scores_binary)
        mean_ap = compute_hl_ap(qid2preds, qid2gt_scores_binary)
        highlight_det_metrics[f"HL-min-{score_name}"] = {"HL-mAP": mean_ap, "HL-Hit1": hit_at_one}
        if verbose:
            print(f"Calculating highlight scores with min score {gt_saliency_score_min} ({score_name})")
            print(f"Time cost {time.time() - start_time:.2f} seconds")
    return highlight_det_metrics

def eval_main():
    # qid : {"pred_saliency_scores": saliency score}
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/home/weihanx/videogpt/deepx_data6/hl_eval/finetune/feature')
    parser.add_argument('--resume', type=str, default='/home/weihanx/videogpt/workspace/UniVTG/results/hl-documentary/qvhl-clip-clip-2024_07_21_12/model_general_best.ckpt')
    parser.add_argument("--submission_path", type=str, help="path to generated prediction file")
    parser.add_argument("--gt_path", type=str, help="path to GT file") # /home/weihanx/videogpt/deepx_data6/dataset/selected_annotation_old_nopad.jsonl
    parser.add_argument("--save_path", type=str, help="path to save the results")
    parser.add_argument("--not_verbose", action="store_true")
    args = parser.parse_args()
    verbose = not args.not_verbose
    submission = load_jsonl(args.submission_path)
    gt = load_jsonl(args.gt_path)
    results = eval_highlight(submission, gt, verbose=verbose)
    if verbose:
        print(json.dumps(results, indent=4))

    with open(Path(args.save_path)/ "hl_eval.txt", "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == '__main__':
    pass