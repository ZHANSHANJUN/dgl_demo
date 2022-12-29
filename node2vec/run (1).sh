#!/bin/bash
starttime=$(date +%s)

day=$1
if [ -z "$day" ]; then
    echo "miss date"
    exit -1
fi

/data/zsj/miniconda3/bin/coscmd download -rf /live/pinsage/${day}/ ./input/
cat ./input/part-* > ./input/graph.txt
rm -rf ./input/part-*
/data/zsj/miniconda3/bin/coscmd delete -rf /live/pinsage/${day}/

cat conf.py
/data/zsj/python train.py
/home/worker/anaconda3/bin/python knn.py

endtime=$(date +%s)
cost=$((endtime - starttime))
echo $day total $cost s
