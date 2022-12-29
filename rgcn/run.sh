#!/bin/bash
starttime=$(date +%s)

day=$1
if [ -z "$day" ]; then
    echo "miss date"
    exit -1
fi

/data/anaconda3/bin/coscmd download -rf /live/rgcn/${day}/live/ ./input/live/
/data/anaconda3/bin/coscmd download -rf /live/rgcn/${day}/friend/ ./input/friend/
cat ./input/live/part-* > ./input/live.txt
cat ./input/friend/part-* > ./input/friend.txt
rm -rf ./input/*/part-*

/data/anaconda3/bin/python graph.py

endtime=$(date +%s)
cost=$((endtime - starttime))
echo $day total $cost s
