# Human-Object-Human (HOH) Evaluation
# Copyright (C) 2024 HMI^2 Lab Santa Clara University


"""Running (Hand Pose Estimation) HPE evaluations on HOH dataset."""

import argparse
import os
import sys
from hpe.hpe_hoh_eval import HPEEvaluator


def parse_args():
    if '--hand_type' not in sys.argv:
        sys.argv += ['--hand_type', 'giver'] # giver hand by default if no hand_type selected
    parser = argparse.ArgumentParser(description="Run evaluation.")
    parser.add_argument(
        "--eval_type",
        help="Evaluation type: hpe",
        default="hpe",
        type=str,
    )
    parser.add_argument(
        "--hand_type",
        help="Hand type: giver or receiver",
        default="giver",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--name", 
        help="Dataset name", 
        default=None, 
        type=str
    )
    parser.add_argument(
        "--res_giver_file",
        help="Path to pointnet giver result file", 
        default='evaluation/hoh_eval_results/giver_hpe_predicted_results_test.txt',
        type=str
    ) 
    parser.add_argument(
        "--res_receiver_file",
        help="Path to pointnet receiver result file", 
        default='evaluation/hoh_eval_results/receiver_hpe_predicted_results_test.txt',
        type=str
    )     
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.name is None:
        args.name = 'hoh_test'
    
    if args.hand_type == 'giver':
        if args.res_giver_file is None:
            args.res_giver_file = os.path.join(
                os.path.dirname(__file__),
                "evaluation",
                "hoh_eval_results",
                "giver_hpe_predicted_results_test.txt",
            )
        hpe_eval = HPEEvaluator(args.name, args.hand_type)
        hpe_eval.evaluates(args.res_giver_file)
    elif args.hand_type =='receiver':
        if args.res_receiver_file is None:
            args.res_receiver_file = os.path.join(
                os.path.dirname(__file__),
                "evaluation",
                "hoh_eval_results",
                "receiver_hpe_predicted_results_test.txt",
            )
        hpe_eval = HPEEvaluator(args.name, args.hand_type)
        hpe_eval.evaluates(args.res_receiver_file)
    else:
        print("Unknown hand type. Please type 'giver' or 'receiver'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
