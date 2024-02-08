export PATH=/home/huangl/workspace/mlir-clang/build/bin:/home/huangl/workspace/RfCgraTrans/build/bin:/home/huangl/workspace/RfCgraTrans/pluto:/home/huangl/workspace/polygeist/build/bin:$PATH
export C_INCLUDE_PATH=/home/huangl/workspace/mlir-clang/build/projects/openmp/runtime/src
export LD_LIBRARY_PATH=/home/huangl/workspace/RfCgraTrans/build/pluto/lib:/home/huangl/workspace/mlir-clang/build/lib:$LD_LIBRARY_PATH
stdinclude=/home/huangl/workspace/mlir-clang/llvm/../clang/lib/Headers
CFLAGS="-march=native -I /home/huangl/workspace/script/polybench-c-4.2.1-beta/utilities -I $stdinclude -D POLYBENCH_TIME -D POLYBENCH_NO_FLUSH_CACHE -D EXTRALARGE_DATASET "
# mlir-clang $CFLAGS aff.c -o aff.RfCgraTrans.in.mlir

#RfCgraTrans-opt -reg2mem -insert-redundant-load -extract-scop-stmt -canonicalize -pluto-opt="dump-clast-after-pluto=aff.RfCgraTrans.cloog" aff.RfCgraTrans.in.mlir 2>/dev/null > aff.RfCgraTrans.out.mlir
#RfCgraTrans-opt -reg2mem -insert-redundant-load -pluto-opt="dump-clast-after-pluto=aff.RfCgraTrans.cloog" -extract-scop-stmt -canonicalize aff.RfCgraTrans.in.mlir 2>/dev/null > aff.RfCgraTrans.out.mlir
# RfCgraTrans-opt -pluto-opt='parallelize=1' -canonicalize aff.RfCgraTrans.in.mlir 2>/dev/null > aff.RfCgraTranspar.out.mlir
TOOLS="mlir-clang"
function run()
{
  TOOL="$1"
  OUT=aff.$TOOL.ll

  case $TOOL in
    
    mlir-clang)
      #mlir-clang $CFLAGS -emit-llvm aff.c -o $OUT
      mlir-clang $CFLAGS --function=kernel_deriche aff.c -o aff.$TOOL.in.mlir
      ;;

    pluto)
      # NOTE: in recent version pluto use --tile and --parallel as def.
      polycc --silent --tile --noparallel --noprevector --nounrolljam aff.c -o aff.$TOOL.c
      clang $CFLAGS -O3 -S -emit-llvm aff.$TOOL.c -o - -fno-vectorize -fno-unroll-loops | sed 's/llvm.loop.unroll.disable//g' > $OUT
      ;;
    RfCgraTrans)
      mlir-clang $CFLAGS aff.c -o aff.$TOOL.in.mlir
      
      RfCgraTrans-opt -reg2mem \
      -insert-redundant-load \
      -canonicalize \
      -pluto-opt="dump-clast-after-pluto=aff.$TOOL.cloog" \
      -canonicalize aff.$TOOL.in.mlir 2>/dev/null > aff.$TOOL.out.mlir
      
      mlir-opt -lower-affine -convert-scf-to-std -canonicalize -convert-std-to-llvm aff.$TOOL.out.mlir |\
        mlir-translate -mlir-to-llvmir > $OUT
      ;;

    RfCgraTrans-pall)
      mlir-clang $CFLAGS aff.c -o aff.$TOOL.in.mlir
      
      RfCgraTrans-opt -reg2mem \
      -insert-redundant-load \
      -canonicalize \
      -pluto-opt="parallelize=1" \
      -canonicalize aff.$TOOL.in.mlir 2>/dev/null > aff.$TOOL.out.mlir
      
      mlir-opt -lower-affine -convert-scf-to-std -canonicalize -convert-std-to-llvm aff.$TOOL.out.mlir |\
        mlir-translate -mlir-to-llvmir > $OUT
      ;;

    cgeist)
      cgeist aff.c -S > aff.$TOOL.out.mlir
      #polygeist-opt aff.$TOOL.in.mlir 2>/dev/null > aff.$TOOL.out.mlir
      ;;
      
       *)
      echo "Illegal tool $TOOL"
      exit 1
      ;;
       esac
    
}
echo "start!"
for t in $TOOLS; do
      run $t
done
echo "end!"