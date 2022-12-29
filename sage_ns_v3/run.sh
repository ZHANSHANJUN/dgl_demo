#!/bin/bash
starttime=$(date +%s)

day=$1
if [ -z "$day" ]; then
    echo "miss date"
    exit -1
fi

/data/anaconda3/bin/coscmd download -rf /live/rgcn/${day}/friend/ ./input/
cat ./input/part-* > ./input/friend.txt
rm -rf ./input/part-*

/data/anaconda3/bin/python sage_ns.py

endtime=$(date +%s)
cost=$((endtime - starttime))
echo $day total $cost s
