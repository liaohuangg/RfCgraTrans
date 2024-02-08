#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c1000_i32 = constant 1000 : i32
    %c2000_i32 = constant 2000 : i32
    %c2600_i32 = constant 2600 : i32
    %c0_i32 = constant 0 : i32
    %0 = memref.alloc() : memref<2000x2600xf64>
    %1 = memref.alloc() : memref<2000x2600xf64>
    %2 = memref.alloc() : memref<2000x2600xf64>
    %3 = memref.alloc() : memref<1000xf64>
    call @polybench_timer_start() : () -> ()
    %4 = memref.cast %0 : memref<2000x2600xf64> to memref<?x2600xf64>
    %5 = memref.cast %1 : memref<2000x2600xf64> to memref<?x2600xf64>
    %6 = memref.cast %2 : memref<2000x2600xf64> to memref<?x2600xf64>
    %7 = memref.cast %3 : memref<1000xf64> to memref<?xf64>
    call @kernel_fdtd_2d(%c1000_i32, %c2000_i32, %c2600_i32, %4, %5, %6, %7) : (i32, i32, i32, memref<?x2600xf64>, memref<?x2600xf64>, memref<?x2600xf64>, memref<?xf64>) -> ()
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    memref.dealloc %0 : memref<2000x2600xf64>
    memref.dealloc %1 : memref<2000x2600xf64>
    memref.dealloc %2 : memref<2000x2600xf64>
    memref.dealloc %3 : memref<1000xf64>
    return %c0_i32 : i32
  }
  func private @polybench_timer_start()
  func private @kernel_fdtd_2d(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x2600xf64>, %arg4: memref<?x2600xf64>, %arg5: memref<?x2600xf64>, %arg6: memref<?xf64>) {
    %cst = constant 5.000000e-01 : f64
    %cst_0 = constant 0.69999999999999996 : f64
    %0 = index_cast %arg2 : i32 to index
    affine.for %arg7 = 0 to %0 {
      %2 = affine.load %arg6[0] : memref<?xf64>
      affine.store %2, %arg4[0, %arg7] : memref<?x2600xf64>
    }
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg7 = 1 to %1 {
      affine.for %arg8 = 0 to %0 {
        %2 = affine.load %arg4[%arg7, %arg8] : memref<?x2600xf64>
        %3 = affine.load %arg5[%arg7, %arg8] : memref<?x2600xf64>
        %4 = affine.load %arg5[%arg7 - 1, %arg8] : memref<?x2600xf64>
        %5 = subf %3, %4 : f64
        %6 = mulf %cst, %5 : f64
        %7 = subf %2, %6 : f64
        affine.store %7, %arg4[%arg7, %arg8] : memref<?x2600xf64>
      }
    }
    affine.for %arg7 = 0 to %1 {
      affine.for %arg8 = 1 to %0 {
        %2 = affine.load %arg3[%arg7, %arg8] : memref<?x2600xf64>
        %3 = affine.load %arg5[%arg7, %arg8] : memref<?x2600xf64>
        %4 = affine.load %arg5[%arg7, %arg8 - 1] : memref<?x2600xf64>
        %5 = subf %3, %4 : f64
        %6 = mulf %cst, %5 : f64
        %7 = subf %2, %6 : f64
        affine.store %7, %arg3[%arg7, %arg8] : memref<?x2600xf64>
      }
    }
    affine.for %arg7 = 0 to #map()[%1] {
      affine.for %arg8 = 0 to #map()[%0] {
        %2 = affine.load %arg5[%arg7, %arg8] : memref<?x2600xf64>
        %3 = affine.load %arg3[%arg7, %arg8 + 1] : memref<?x2600xf64>
        %4 = affine.load %arg3[%arg7, %arg8] : memref<?x2600xf64>
        %5 = subf %3, %4 : f64
        %6 = affine.load %arg4[%arg7 + 1, %arg8] : memref<?x2600xf64>
        %7 = addf %5, %6 : f64
        %8 = affine.load %arg4[%arg7, %arg8] : memref<?x2600xf64>
        %9 = subf %7, %8 : f64
        %10 = mulf %cst_0, %9 : f64
        %11 = subf %2, %10 : f64
        affine.store %11, %arg5[%arg7, %arg8] : memref<?x2600xf64>
      }
    }
    return
  }
  func private @polybench_timer_stop()
  func private @polybench_timer_print()
}
