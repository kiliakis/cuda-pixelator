#!/bin/bash
for v in CPU OPENMP CUDA; do
    for size in 256x256.png 1024x1024.png 2048x2048.png 4096x4096.jpg; do
        echo $v $size;
        ./bin/release/AdaptiveHistogramOptimisation $v samples/$size -b > results/$v-$size.txt;
    done
done

