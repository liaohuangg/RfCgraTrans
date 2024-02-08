#map = affine_map<()[s0] -> (s0 - 1)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c308_i32 = constant 308 : i32
    %c0_i32 = constant 0 : i32
    %0 = memref.alloc() : memref<308x308x308xf64>
    %1 = memref.alloc() : memref<308x308x308xf64>
    %2 = memref.alloc() : memref<308x308x308xf64>
    %3 = memref.alloc() : memref<308x308x308xf64>
    %4 = memref.alloc() : memref<308x308x308xf64>
    %5 = memref.alloc() : memref<308x308x308xf64>
    %6 = memref.alloc() : memref<308x308x308xf64>
    %7 = memref.alloc() : memref<308x308x308xf64>
    call @polybench_timer_start() : () -> ()
    %8 = memref.cast %0 : memref<308x308x308xf64> to memref<?x308x308xf64>
    %9 = memref.cast %1 : memref<308x308x308xf64> to memref<?x308x308xf64>
    %10 = memref.cast %2 : memref<308x308x308xf64> to memref<?x308x308xf64>
    %11 = memref.cast %3 : memref<308x308x308xf64> to memref<?x308x308xf64>
    %12 = memref.cast %4 : memref<308x308x308xf64> to memref<?x308x308xf64>
    %13 = memref.cast %5 : memref<308x308x308xf64> to memref<?x308x308xf64>
    %14 = memref.cast %6 : memref<308x308x308xf64> to memref<?x308x308xf64>
    %15 = memref.cast %7 : memref<308x308x308xf64> to memref<?x308x308xf64>
    call @kernel_advect_3d(%c308_i32, %8, %9, %10, %11, %12, %13, %14, %15) : (i32, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>) -> ()
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    memref.dealloc %0 : memref<308x308x308xf64>
    memref.dealloc %1 : memref<308x308x308xf64>
    memref.dealloc %2 : memref<308x308x308xf64>
    memref.dealloc %3 : memref<308x308x308xf64>
    memref.dealloc %4 : memref<308x308x308xf64>
    memref.dealloc %5 : memref<308x308x308xf64>
    memref.dealloc %6 : memref<308x308x308xf64>
    memref.dealloc %7 : memref<308x308x308xf64>
    return %c0_i32 : i32
  }
  func private @polybench_timer_start()
  func private @kernel_advect_3d(%arg0: i32, %arg1: memref<?x308x308xf64>, %arg2: memref<?x308x308xf64>, %arg3: memref<?x308x308xf64>, %arg4: memref<?x308x308xf64>, %arg5: memref<?x308x308xf64>, %arg6: memref<?x308x308xf64>, %arg7: memref<?x308x308xf64>, %arg8: memref<?x308x308xf64>) {
    %cst = constant 2.000000e-01 : f64
    %cst_0 = constant 5.000000e-01 : f64
    %cst_1 = constant 3.000000e-01 : f64
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg9 = 4 to %0 {
      affine.for %arg10 = 4 to #map()[%0] {
        affine.for %arg11 = 4 to #map()[%0] {
          %1 = affine.load %arg4[%arg9 - 1, %arg10, %arg11] : memref<?x308x308xf64>
          %2 = affine.load %arg4[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
          %3 = addf %1, %2 : f64
          %4 = mulf %cst, %3 : f64
          %5 = affine.load %arg4[%arg9 - 2, %arg10, %arg11] : memref<?x308x308xf64>
          %6 = affine.load %arg4[%arg9 + 1, %arg10, %arg11] : memref<?x308x308xf64>
          %7 = addf %5, %6 : f64
          %8 = mulf %cst_0, %7 : f64
          %9 = addf %4, %8 : f64
          %10 = affine.load %arg4[%arg9 - 3, %arg10, %arg11] : memref<?x308x308xf64>
          %11 = affine.load %arg4[%arg9 + 2, %arg10, %arg11] : memref<?x308x308xf64>
          %12 = addf %10, %11 : f64
          %13 = mulf %cst_1, %12 : f64
          %14 = addf %9, %13 : f64
          %15 = mulf %14, %cst_1 : f64
          %16 = affine.load %arg6[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
          %17 = mulf %15, %16 : f64
          affine.store %17, %arg1[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
        }
      }
    }
    affine.for %arg9 = 4 to #map()[%0] {
      affine.for %arg10 = 4 to %0 {
        affine.for %arg11 = 4 to #map()[%0] {
          %1 = affine.load %arg4[%arg9, %arg10 - 1, %arg11] : memref<?x308x308xf64>
          %2 = affine.load %arg4[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
          %3 = addf %1, %2 : f64
          %4 = mulf %cst, %3 : f64
          %5 = affine.load %arg4[%arg9, %arg10 - 2, %arg11] : memref<?x308x308xf64>
          %6 = affine.load %arg4[%arg9, %arg10 + 1, %arg11] : memref<?x308x308xf64>
          %7 = addf %5, %6 : f64
          %8 = mulf %cst_0, %7 : f64
          %9 = addf %4, %8 : f64
          %10 = affine.load %arg4[%arg9, %arg10 - 3, %arg11] : memref<?x308x308xf64>
          %11 = affine.load %arg4[%arg9, %arg10 + 2, %arg11] : memref<?x308x308xf64>
          %12 = addf %10, %11 : f64
          %13 = mulf %cst_1, %12 : f64
          %14 = addf %9, %13 : f64
          %15 = mulf %14, %cst_1 : f64
          %16 = affine.load %arg7[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
          %17 = mulf %15, %16 : f64
          affine.store %17, %arg2[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
        }
      }
    }
    affine.for %arg9 = 4 to #map()[%0] {
      affine.for %arg10 = 4 to #map()[%0] {
        affine.for %arg11 = 4 to %0 {
          %1 = affine.load %arg4[%arg9, %arg10, %arg11 - 1] : memref<?x308x308xf64>
          %2 = affine.load %arg4[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
          %3 = addf %1, %2 : f64
          %4 = mulf %cst, %3 : f64
          %5 = affine.load %arg4[%arg9, %arg10, %arg11 - 2] : memref<?x308x308xf64>
          %6 = affine.load %arg4[%arg9, %arg10, %arg11 + 1] : memref<?x308x308xf64>
          %7 = addf %5, %6 : f64
          %8 = mulf %cst_0, %7 : f64
          %9 = addf %4, %8 : f64
          %10 = affine.load %arg4[%arg9, %arg10, %arg11 - 3] : memref<?x308x308xf64>
          %11 = affine.load %arg4[%arg9, %arg10, %arg11 + 2] : memref<?x308x308xf64>
          %12 = addf %10, %11 : f64
          %13 = mulf %cst_1, %12 : f64
          %14 = addf %9, %13 : f64
          %15 = mulf %14, %cst_1 : f64
          %16 = affine.load %arg8[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
          %17 = mulf %15, %16 : f64
          affine.store %17, %arg3[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
        }
      }
    }
    affine.for %arg9 = 4 to #map()[%0] {
      affine.for %arg10 = 4 to #map()[%0] {
        affine.for %arg11 = 4 to #map()[%0] {
          %1 = affine.load %arg4[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
          %2 = affine.load %arg2[%arg9, %arg10 + 1, %arg11] : memref<?x308x308xf64>
          %3 = affine.load %arg2[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
          %4 = subf %2, %3 : f64
          %5 = addf %1, %4 : f64
          %6 = affine.load %arg1[%arg9 + 1, %arg10, %arg11] : memref<?x308x308xf64>
          %7 = affine.load %arg1[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
          %8 = subf %6, %7 : f64
          %9 = addf %5, %8 : f64
          %10 = affine.load %arg3[%arg9, %arg10, %arg11 + 1] : memref<?x308x308xf64>
          %11 = affine.load %arg3[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
          %12 = subf %10, %11 : f64
          %13 = addf %9, %12 : f64
          affine.store %13, %arg5[%arg9, %arg10, %arg11] : memref<?x308x308xf64>
        }
      }
    }
    return
  }
  func private @polybench_timer_stop()
  func private @polybench_timer_print()
}
