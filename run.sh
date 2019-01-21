#!/bin/bash
for c in {0,0.2,0.4,0.6,0.8,1};
do
    for i in {1..10};
    do
	python MTL.py -m "models/$i" -c $c &
	pids[${i}]=$! # store pids in an array
    done

    # wait for all pids
    # for pid in ${pids[*]}; do
    # 	wait $pid
    # done
done
