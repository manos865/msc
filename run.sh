#!/bin/bash

for size in 5 10 25 50 100 250 500 750 1000 2500 5000
do
    for conn in 0.01 0.025 0.05 0.10 0.25 0.5 0.75 0.90
    do
        python3 detect.py $size $conn
    done
done
