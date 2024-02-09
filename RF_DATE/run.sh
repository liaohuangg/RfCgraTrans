ulimit -c unlimited
BASE=/home/huangl/workspace/script/polybench-c-4.2.1-beta/1_8_8_Unroll_New
EXAMPLE=jacobi-1d
#jacobi-1d
#!/bin/bash
op=$1
if [ "$op" == "start" ]; then
  export LD_LIBRARY_PATH=/usr/local/lib

  rm /home/huangl/workspace/RF_DATE_before/log/test.log
  for ((i=0;i<=200;i++)) 
  do
    dir="map0_solu"$i".txt"
    echo $BASE/$EXAMPLE/$dir
  ./rf_tcad --dfg_file=$BASE/$EXAMPLE/$dir --II=1 --childNum=4 --pea_column=8 --pea_row=8 >> log/test.log 2>&1 &
  
  #./rf_tcad --dfg_file=data/test.txt  --II=2  --childNum=4 --pea_column=8 --pea_row=8
#   nohup ./rf_tcad --dfg_file=data/test.txt --II=1  --childNum=4 >> log/test.log 2>&1 &
 sleep 0.2s
done

elif [ "$op" == "clear" ]; then
  rm -rf log/*
else
  echo "./run (start | clear)"
fi
