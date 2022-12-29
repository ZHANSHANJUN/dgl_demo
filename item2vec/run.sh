#!/bin/bash
starttime=$(date +%s)

day=$1
if [ -z "$day" ]; then
    echo "miss date"
    exit -1
fi

/data/zsj/miniconda3/bin/coscmd download -rf /live/item_session/${day}/ ./input/
cat ./input/part-* > ./input/item_session.txt
rm -rf ./input/part-*
/data/zsj/miniconda3/bin/coscmd delete -rf /live/item_session/${day}/
/data/zsj/python item2vec.py

endtime=$(date +%s)
cost=$((endtime - starttime))
echo $day total $cost s
