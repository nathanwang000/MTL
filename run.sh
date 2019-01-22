#!/bin/bash
for c in {0,0.2,0.4,0.6,0.8,1};
do
    counter=1
    for a in {0.1,0.5,1,3}
    do
	for i in {1..10};
	do
	    python MTL.py -m "models/$i" -c $c -a $a &
	    pids[${counter}]=$! # store pids in an array
	    counter=$((counter+1))
	done
    done

    # wait for all pids
    for pid in ${pids[*]}; do
    	wait $pid
    done
done
