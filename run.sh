#!/bin/bash

for size in 5 10 25 50 100 250 500 1000 2500
do
    for conn in 0.05 0.10 0.25 0.5 0.75
    do
        python3 detect.py $size $conn
    done
done
