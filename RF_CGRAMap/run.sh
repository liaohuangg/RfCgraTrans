ulimit -c unlimited

#!/bin/bash
op=$1
if [ "$op" == "start" ]; then
  
    ./rf_tcad --dfg_file=data/test.txt  --II=1  --childNum=4 
#   nohup ./rf_tcad --dfg_file=data/test.txt --II=1  --childNum=4 >> log/test.log 2>&1 &
 
  
  
  
 
 
elif [ "$op" == "clear" ]; then
  rm -rf log/*
else
  echo "./run (start | clear)"
fi
