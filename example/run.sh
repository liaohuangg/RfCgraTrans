
#!/bin/bash
# set -o errexit
# set -o pipefail
# set -o nounset
export PATH=/home/huangl/workspace/mlir-clang/build/bin:/home/huangl/workspace/RfCgraTrans/build/bin:/home/huangl/workspace/RfCgraTrans/pluto:/home/huangl/workspace/polygeist/build/bin:$PATH
export C_INCLUDE_PATH=/home/huangl/workspace/mlir-clang/build/projects/openmp/runtime/src
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/huangl/workspace/RfCgraTrans/build/pluto/lib:/home/huangl/workspace/mlir-clang/build/lib:/home/huangl/workspace/RfCgraTrans/glpk/glpk-5.0/src
stdinclude=/home/huangl/workspace/mlir-clang/llvm/../clang/lib/Headers
CFLAGS="-march=native -I /home/huangl/workspace/script/polybench-c-4.2.1-beta/utilities -I $stdinclude -D POLYBENCH_TIME -D POLYBENCH_NO_FLUSH_CACHE -D EXTRALARGE_DATASET "

TOOLS="RfCgraTrans"

function run()
{ 
  TOOL="$1"
  TEST="$2"
  OUT=$TEST.$TOOL.ll

  case $TOOL in

    mlir-clang)
      mlir-clang $CFLAGS -emit-llvm $TEST.c -o $OUT
      ;;

    pluto)
      if [[ $2 == "adi" ]]
      then
        return
      fi
      # NOTE: in recent version pluto use --tile and --parallel as def.
      polycc --silent --tile --noparallel --noprevector --nounrolljam $TEST.c -o $TEST.$TOOL.c &> /dev/null
      clang $CFLAGS -O3 -S -emit-llvm $TEST.$TOOL.c -o - -fno-vectorize -fno-unroll-loops | sed 's/llvm.loop.unroll.disable//g' > $OUT
      ;;

    RfCgraTrans)
      mlir-clang $CFLAGS $TEST.c -o $TEST.$TOOL.in.mlir
      # RfCgraTrans-opt -reg2mem \
      # -insert-redundant-load \
      # -extract-scop-stmt \
      # -canonicalize \
      # -pluto-opt="dump-clast-after-pluto=$TEST.$TOOL.cloog" \
      # -canonicalize $TEST.$TOOL.in.mlir 2>/dev/null > $TEST.$TOOL.out.mlir
      RfCgraTrans-opt -reg2mem \
      -insert-redundant-load \
      -extract-scop-stmt \
      -canonicalize \
      -pluto-opt="dump-clast-after-pluto=$TEST.$TOOL.cloog" \
      -canonicalize $TEST.$TOOL.in.mlir 2>/dev/null > $TEST.$TOOL.out.mlir

      mlir-opt -lower-affine -convert-scf-to-std -canonicalize -convert-std-to-llvm $TEST.$TOOL.out.mlir 
      #mlir-translate -mlir-to-llvmir > $OUT
      ;;

    *)
      echo "Illegal tool $TOOL"
      exit 1
      ;;
  esac	
}

BASE=$(pwd)
dirList="2mm 3mm atax gemm gemver gesummv jacobi-1d jacobi-2d mvt bicg advect-3d fdtd-2d"
for dir in $dirList; 
do
    echo "rm start"
    echo $dir
    cd "$BASE/$dir"
    rm $dir.*.RfCgraTrans.out.mlir
    rm *.RfCgraTrans*.cloog
    rm DFGInformation.out
    rm ScheduleInformation.out
    rm unrollInformation.out
    rm MapInformation.out
    rm min_dependence_distance_schedule.out
    rm AfterScheduleDFGInformation.out
    rm Schedule*.out
    rm map*.txt
    rm simpleSchedule*.out
done

for dir in $dirList; 
do
    echo "run start"
    echo $dir
    cd "$BASE/$dir"
    for t in $TOOLS; 
    do
        run $t $dir
    done
done
