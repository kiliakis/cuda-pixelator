#!/bin/bash
for v in CPU OPENMP CUDA; do
    for size in 256x256 1024x1024 2048x2048 4096x4096; do
        echo $v $size;
        ./bin/release/AdaptiveHistogramOptimization $V samples/$size.png -b &> results/$v-$size.txt;
    done
done

