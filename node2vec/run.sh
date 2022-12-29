#!/bin/bash
starttime=$(date +%s)

day=$1
if [ -z "$day" ]; then
    echo "miss date"
    exit -1
fi

/data/zsj/miniconda3/bin/coscmd download -rf /live/rgcn/${day}/ ./input/
cat ./input/live/part-* > ./input/live.txt
cat ./input/sing/part-* > ./input/sing.txt
cat ./input/friend/part-* > ./input/friend.txt
rm -rf ./input/*/part-*
#/data/zsj/miniconda3/bin/coscmd delete -rf /live/rgcn/${day}/

cat conf.py
/data/zsj/python node2vec.py
/home/worker/anaconda3/bin/python knn.py

endtime=$(date +%s)
cost=$((endtime - starttime))
echo $day total $cost s
