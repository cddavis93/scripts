#!/usr/bin/env bash

#conda activate nlm

export PATH=~/Desktop/neural-logic-machines-master/third_party/Jacinle-master/bin:$PATH

jac-run scripts/graph/learn_policy_profiler.py --task path --test-only --test-number-end 10 --test-epoch-size 100 --use-gpu >> results.txt
