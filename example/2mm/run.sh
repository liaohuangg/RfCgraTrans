

# set -o errexit
# set -o pipefail
# set -o nounset

export PATH=/home/huangl/workspace/mlir-clang/build/bin:/home/huangl/workspace/RfCgraTrans/build/bin:/home/huangl/workspace/RfCgraTrans/pluto:/home/huangl/workspace/polygeist/build/bin:$PATH
export C_INCLUDE_PATH=/home/huangl/workspace/mlir-clang/build/projects/openmp/runtime/src
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/huangl/workspace/RfCgraTrans/build/pluto/lib:/home/huangl/workspace/mlir-clang/build/lib
stdinclude=/home/huangl/workspace/mlir-clang/llvm/../clang/lib/Headers
CFLAGS="-march=native -I /home/huangl/workspace/script/polybench-c-4.2.1-beta/utilities -I $stdinclude -D POLYBENCH_TIME -D POLYBENCH_NO_FLUSH_CACHE -D EXTRALARGE_DATASET "

TOOLS="RfCgraTrans"

function run()
{ 
  TOOL="$1"
  TEST="2mm"
  OUT=$TEST.$TOOL.ll

  case $TOOL in

    clangsing)
      clang $CFLAGS -O1 -S -emit-llvm -Xclang -disable-llvm-passes $TEST.c -o $OUT
      ;;

    clang)
      clang $CFLAGS -O3 -S -emit-llvm $TEST.c -o - -fno-vectorize -fno-unroll-loops | sed 's/llvm.loop.unroll.disable//g' > $OUT
      ;;

    mlir-clang)
      mlir-clang $CFLAGS -emit-llvm $TEST.c -o $OUT
      ;;

    polly)
      clang $CFLAGS -O3 -S -emit-llvm $TEST.c -mllvm -polly -mllvm -polly-pattern-matching-based-opts=false -mllvm -polly-vectorizer=none -o - -fno-vectorize -fno-unroll-loops | sed 's/llvm.loop.unroll.disable//g' > $OUT
      ;;

    pollypar)
      clang $CFLAGS -O3 -S -emit-llvm $TEST.c -mllvm -polly -mllvm -polly-pattern-matching-based-opts=false -mllvm -polly-vectorizer=none -mllvm -polly-parallel -mllvm -polly-parallel-force -mllvm -polly-omp-backend=LLVM -mllvm -polly-scheduling=static -o - -fno-vectorize -fno-unroll-loops | sed 's/llvm.loop.unroll.disable//g' > $OUT
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

    plutopar)
      if [[ $2 == "adi" ]]
      then
        return
      fi
      polycc --silent --parallel --tile --noprevector --nounrolljam $TEST.c -o $TEST.$TOOL.c &> /dev/null
      clang $CFLAGS -O3 -fopenmp -S -emit-llvm $TEST.$TOOL.c -o - -fno-vectorize -fno-unroll-loops | sed 's/llvm.loop.unroll.disable//g' > $OUT
      ;;

    RfCgraTrans)
      mlir-clang $CFLAGS $TEST.c -o $TEST.$TOOL.in.mlir
      
      RfCgraTrans-opt -reg2mem \
      -insert-redundant-load \
      -extract-scop-stmt \
      -canonicalize \
      -pluto-opt="dump-clast-after-pluto=$TEST.$TOOL.cloog" \
      -canonicalize $TEST.$TOOL.in.mlir 2>/dev/null > $TEST.$TOOL.out.mlir
      
      mlir-opt -lower-affine -convert-scf-to-std -canonicalize -convert-std-to-llvm $TEST.$TOOL.out.mlir |\
        mlir-translate -mlir-to-llvmir > $OUT
      ;;

      polygeist)
      # mlir-clang $CFLAGS $TEST.c -o $TEST.$TOOL.in.mlir
      
      # RfCgraTrans-opt -reg2mem \
      # -insert-redundant-load \
      # -extract-scop-stmt \
      # -canonicalize \
      # -pluto-opt="dump-clast-after-pluto=$TEST.$TOOL.cloog" \
      # -canonicalize $TEST.$TOOL.in.mlir 2>/dev/null > $TEST.$TOOL.out.mlir
      polygeist-opt --parallel-lower --cudart-lower --split-input-file $TEST.$TOOL.in.mlir 2>/dev/null > $TEST.$TOOL.out.mlir
      # mlir-opt -lower-affine -convert-scf-to-std -canonicalize -convert-std-to-llvm $TEST.$TOOL.out.mlir |\
      #   mlir-translate -mlir-to-llvmir > $OUT
      ;;

    RfCgraTranspar)
      
      
      mlir-clang $CFLAGS $TEST.c -o $TEST.$TOOL.in.mlir
      
      RfCgraTrans-opt --demote-loop-reduction \
           --extract-scop-stmt \
           --pluto-opt='parallelize=1' \
           --inline \
           --canonicalize $TEST.$TOOL.in.mlir 2>/dev/null > $TEST.$TOOL.out.mlir

      mlir-opt -mem2reg -detect-reduction -mem2reg -canonicalize -affine-parallelize -lower-affine -convert-scf-to-openmp -convert-scf-to-std -convert-openmp-to-llvm $TEST.$TOOL.out.mlir | mlir-translate -mlir-to-llvmir > $OUT
      ;;

    *)
      echo "Illegal tool $TOOL"
      exit 1
      ;;
  esac
  
  # if [[ $1 == "clang" || $1 == "polly" || $1 == "pluto" || $1 == "mlir-clang" || $1 == "RfCgraTrans" || $1 == "clangsing" ]] 
  # then 
  # 	clang $BASE/utilities/polybench.c -O3 -march=native $OUT -o $TEST.$TOOL.exe -lm -D POLYBENCH_TIME  -D POLYBENCH_NO_FLUSH_CACHE -D EXTRALARGE_DATASET
  # else 
	# clang $BASE/utilities/polybench.c -O3 -march=native $OUT -o $TEST.$TOOL.exe -lm -fopenmp -D POLYBENCH_TIME -D POLYBENCH_NO_FLUSH_CACHE -D EXTRALARGE_DATASET
  # fi	
}

# for dir in $dirList; do
#   cd "$BASE/$dir"
#   for subDir in `ls`; do
#     cd "$BASE/$dir/$subDir" 
#     echo $(pwd)
#     for t in $TOOLS; do
#       run $t $subDir
#     done

#     for i in 1 2 3 4 5; do
#       for t in $TOOLS; do
# 	      if [[ $subDir == "adi" && ( $t == "pluto" || $t == "plutopar" ) ]]
# 	      then
# 	        echo $t:$subDir:nan
# 	      else
#         	time=$(taskset -c 1-8 numactl -i all ./$subDir.$t.exe)
# 		echo $t:$subDir:$time
# 	      fi
#       done
#     done 

#   done
# done
echo "start!"
for t in $TOOLS; do
      run $t
done
echo "end!"