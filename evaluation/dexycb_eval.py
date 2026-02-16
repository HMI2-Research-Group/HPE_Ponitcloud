# DexYCB  Evaluation
# Copyright (C) 2024 HMI^2 Lab Santa Clara University

"""Running HPE evaluations on DexYCB."""

import argparse
import os
import sys
from dex_ycb_toolkit.hpe_eval import HPEEvaluator



def parse_args():
    if '--eval_type' not in sys.argv:
        sys.argv += ['--eval_type', 'hpe']
    parser = argparse.ArgumentParser(description='Run evaluation.')
    parser.add_argument('--eval_type', help='Evaluation type: hpe', default='hpe', required=True, type=str)
    parser.add_argument('--name', help='Dataset name', default=None, type=str)
    parser.add_argument('--res_file', help='Path to result file', default='evaluation/dexycb_sample_results/example_results_hpe_s0_test.txt', type=str) 
    parser.add_argument('--out_dir', help='Directory to save eval output', default='evaluation/dexycb_sample_results/', type=str)
    parser.add_argument('--visualize', action='store_true', default=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.name is None:
        args.name = 's0_test'

    if args.eval_type == 'hpe':
        if args.res_file is None:
            args.res_file = os.path.join(os.path.dirname(__file__), "..", "results", "example_results_hpe_{}.txt".format(args.name))
        hpe_eval = HPEEvaluator(args.name)
        hpe_eval.evaluate(args.res_file, out_dir=args.out_dir)

    else:
        print("Unknown evaluation type")
        sys.exit(1)


if __name__ == '__main__':
    main()
