#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c2800_i32 = constant 2800 : i32
    %c1000_i32 = constant 1000 : i32
    %c0_i32 = constant 0 : i32
    %0 = memref.alloc() : memref<2800x2800xf64>
    %1 = memref.alloc() : memref<2800x2800xf64>
    call @polybench_timer_start() : () -> ()
    %2 = memref.cast %0 : memref<2800x2800xf64> to memref<?x2800xf64>
    %3 = memref.cast %1 : memref<2800x2800xf64> to memref<?x2800xf64>
    call @kernel_jacobi_2d(%c1000_i32, %c2800_i32, %2, %3) : (i32, i32, memref<?x2800xf64>, memref<?x2800xf64>) -> ()
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    memref.dealloc %0 : memref<2800x2800xf64>
    memref.dealloc %1 : memref<2800x2800xf64>
    return %c0_i32 : i32
  }
  func private @polybench_timer_start()
  func private @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<?x2800xf64>, %arg3: memref<?x2800xf64>) {
    %cst = constant 2.000000e-01 : f64
    %0 = index_cast %arg1 : i32 to index
    affine.for %arg4 = 1 to #map()[%0] {
      affine.for %arg5 = 1 to #map()[%0] {
        %1 = affine.load %arg2[%arg4, %arg5] : memref<?x2800xf64>
        %2 = affine.load %arg2[%arg4, %arg5 - 1] : memref<?x2800xf64>
        %3 = addf %1, %2 : f64
        %4 = affine.load %arg2[%arg4, %arg5 + 1] : memref<?x2800xf64>
        %5 = addf %3, %4 : f64
        %6 = affine.load %arg2[%arg4 + 1, %arg5] : memref<?x2800xf64>
        %7 = addf %5, %6 : f64
        %8 = affine.load %arg2[%arg4 - 1, %arg5] : memref<?x2800xf64>
        %9 = addf %7, %8 : f64
        %10 = mulf %cst, %9 : f64
        affine.store %10, %arg3[%arg4, %arg5] : memref<?x2800xf64>
      }
    }
    affine.for %arg4 = 1 to #map()[%0] {
      affine.for %arg5 = 1 to #map()[%0] {
        %1 = affine.load %arg3[%arg4, %arg5] : memref<?x2800xf64>
        %2 = affine.load %arg3[%arg4, %arg5 - 1] : memref<?x2800xf64>
        %3 = addf %1, %2 : f64
        %4 = affine.load %arg3[%arg4, %arg5 + 1] : memref<?x2800xf64>
        %5 = addf %3, %4 : f64
        %6 = affine.load %arg3[%arg4 + 1, %arg5] : memref<?x2800xf64>
        %7 = addf %5, %6 : f64
        %8 = affine.load %arg3[%arg4 - 1, %arg5] : memref<?x2800xf64>
        %9 = addf %7, %8 : f64
        %10 = mulf %cst, %9 : f64
        affine.store %10, %arg2[%arg4, %arg5] : memref<?x2800xf64>
      }
    }
    return
  }
  func private @polybench_timer_stop()
  func private @polybench_timer_print()
}
