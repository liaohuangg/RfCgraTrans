filename = advect-3d.RfCgraTransFinal.cloog 
#map0 = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0 - 1)>
#set = affine_set<()[s0] : (s0 - 6 >= 0)>
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
    call @kernel_advect_3d_opt(%c308_i32, %8, %9, %10, %11, %12, %13, %14, %15) : (i32, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>) -> ()
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
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg9 = 4 to %0 {
      affine.for %arg10 = 4 to #map0()[%0] {
        affine.for %arg11 = 4 to #map0()[%0] {
          call @S0(%arg1, %arg9, %arg10, %arg11, %arg6, %arg4) : (memref<?x308x308xf64>, index, index, index, memref<?x308x308xf64>, memref<?x308x308xf64>) -> ()
        }
      }
    }
    affine.for %arg9 = 4 to #map0()[%0] {
      affine.for %arg10 = 4 to %0 {
        affine.for %arg11 = 4 to #map0()[%0] {
          call @S1(%arg2, %arg9, %arg10, %arg11, %arg7, %arg4) : (memref<?x308x308xf64>, index, index, index, memref<?x308x308xf64>, memref<?x308x308xf64>) -> ()
        }
      }
    }
    affine.for %arg9 = 4 to #map0()[%0] {
      affine.for %arg10 = 4 to #map0()[%0] {
        affine.for %arg11 = 4 to %0 {
          call @S2(%arg3, %arg9, %arg10, %arg11, %arg8, %arg4) : (memref<?x308x308xf64>, index, index, index, memref<?x308x308xf64>, memref<?x308x308xf64>) -> ()
        }
      }
    }
    affine.for %arg9 = 4 to #map0()[%0] {
      affine.for %arg10 = 4 to #map0()[%0] {
        affine.for %arg11 = 4 to #map0()[%0] {
          call @S3(%arg5, %arg9, %arg10, %arg11, %arg3, %arg1, %arg2, %arg4) : (memref<?x308x308xf64>, index, index, index, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>) -> ()
        }
      }
    }
    return
  }
  func private @polybench_timer_stop()
  func private @S0(%arg0: memref<?x308x308xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<?x308x308xf64>, %arg5: memref<?x308x308xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e-01 : f64
    %cst_0 = constant 5.000000e-01 : f64
    %cst_1 = constant 3.000000e-01 : f64
    %0 = affine.load %arg5[symbol(%arg1) - 1, symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %1 = affine.load %arg5[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %2 = addf %0, %1 : f64
    %3 = mulf %cst, %2 : f64
    %4 = affine.load %arg5[symbol(%arg1) - 2, symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %5 = affine.load %arg5[symbol(%arg1) + 1, symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %6 = addf %4, %5 : f64
    %7 = mulf %cst_0, %6 : f64
    %8 = addf %3, %7 : f64
    %9 = affine.load %arg5[symbol(%arg1) - 3, symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %10 = affine.load %arg5[symbol(%arg1) + 2, symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %11 = addf %9, %10 : f64
    %12 = mulf %cst_1, %11 : f64
    %13 = addf %8, %12 : f64
    %14 = mulf %13, %cst_1 : f64
    %15 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %16 = mulf %14, %15 : f64
    affine.store %16, %arg0[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    return
  }
  func private @S1(%arg0: memref<?x308x308xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<?x308x308xf64>, %arg5: memref<?x308x308xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e-01 : f64
    %cst_0 = constant 5.000000e-01 : f64
    %cst_1 = constant 3.000000e-01 : f64
    %0 = affine.load %arg5[symbol(%arg1), symbol(%arg2) - 1, symbol(%arg3)] : memref<?x308x308xf64>
    %1 = affine.load %arg5[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %2 = addf %0, %1 : f64
    %3 = mulf %cst, %2 : f64
    %4 = affine.load %arg5[symbol(%arg1), symbol(%arg2) - 2, symbol(%arg3)] : memref<?x308x308xf64>
    %5 = affine.load %arg5[symbol(%arg1), symbol(%arg2) + 1, symbol(%arg3)] : memref<?x308x308xf64>
    %6 = addf %4, %5 : f64
    %7 = mulf %cst_0, %6 : f64
    %8 = addf %3, %7 : f64
    %9 = affine.load %arg5[symbol(%arg1), symbol(%arg2) - 3, symbol(%arg3)] : memref<?x308x308xf64>
    %10 = affine.load %arg5[symbol(%arg1), symbol(%arg2) + 2, symbol(%arg3)] : memref<?x308x308xf64>
    %11 = addf %9, %10 : f64
    %12 = mulf %cst_1, %11 : f64
    %13 = addf %8, %12 : f64
    %14 = mulf %13, %cst_1 : f64
    %15 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %16 = mulf %14, %15 : f64
    affine.store %16, %arg0[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    return
  }
  func private @S2(%arg0: memref<?x308x308xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<?x308x308xf64>, %arg5: memref<?x308x308xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e-01 : f64
    %cst_0 = constant 5.000000e-01 : f64
    %cst_1 = constant 3.000000e-01 : f64
    %0 = affine.load %arg5[symbol(%arg1), symbol(%arg2), symbol(%arg3) - 1] : memref<?x308x308xf64>
    %1 = affine.load %arg5[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %2 = addf %0, %1 : f64
    %3 = mulf %cst, %2 : f64
    %4 = affine.load %arg5[symbol(%arg1), symbol(%arg2), symbol(%arg3) - 2] : memref<?x308x308xf64>
    %5 = affine.load %arg5[symbol(%arg1), symbol(%arg2), symbol(%arg3) + 1] : memref<?x308x308xf64>
    %6 = addf %4, %5 : f64
    %7 = mulf %cst_0, %6 : f64
    %8 = addf %3, %7 : f64
    %9 = affine.load %arg5[symbol(%arg1), symbol(%arg2), symbol(%arg3) - 3] : memref<?x308x308xf64>
    %10 = affine.load %arg5[symbol(%arg1), symbol(%arg2), symbol(%arg3) + 2] : memref<?x308x308xf64>
    %11 = addf %9, %10 : f64
    %12 = mulf %cst_1, %11 : f64
    %13 = addf %8, %12 : f64
    %14 = mulf %13, %cst_1 : f64
    %15 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %16 = mulf %14, %15 : f64
    affine.store %16, %arg0[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    return
  }
  func private @S3(%arg0: memref<?x308x308xf64>, %arg1: index, %arg2: index, %arg3: index, %arg4: memref<?x308x308xf64>, %arg5: memref<?x308x308xf64>, %arg6: memref<?x308x308xf64>, %arg7: memref<?x308x308xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg7[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %1 = affine.load %arg6[symbol(%arg1), symbol(%arg2) + 1, symbol(%arg3)] : memref<?x308x308xf64>
    %2 = affine.load %arg6[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %3 = subf %1, %2 : f64
    %4 = addf %0, %3 : f64
    %5 = affine.load %arg5[symbol(%arg1) + 1, symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %6 = affine.load %arg5[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %7 = subf %5, %6 : f64
    %8 = addf %4, %7 : f64
    %9 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3) + 1] : memref<?x308x308xf64>
    %10 = affine.load %arg4[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    %11 = subf %9, %10 : f64
    %12 = addf %8, %11 : f64
    affine.store %12, %arg0[symbol(%arg1), symbol(%arg2), symbol(%arg3)] : memref<?x308x308xf64>
    return
  }
  func private @kernel_advect_3d_opt(%arg0: i32, %arg1: memref<?x308x308xf64>, %arg2: memref<?x308x308xf64>, %arg3: memref<?x308x308xf64>, %arg4: memref<?x308x308xf64>, %arg5: memref<?x308x308xf64>, %arg6: memref<?x308x308xf64>, %arg7: memref<?x308x308xf64>, %arg8: memref<?x308x308xf64>) {
    %c4 = constant 4 : index
    %0 = index_cast %arg0 : i32 to index
    affine.if #set()[%0] {
      affine.for %arg9 = 4 to #map0()[%0] {
        affine.for %arg10 = 4 to #map0()[%0] {
          affine.for %arg11 = 4 to %0 {
            call @S2(%arg3, %arg9, %arg10, %arg11, %arg8, %arg4) : (memref<?x308x308xf64>, index, index, index, memref<?x308x308xf64>, memref<?x308x308xf64>) -> ()
          }
        }
      }
      affine.for %arg9 = 4 to #map0()[%0] {
        affine.for %arg10 = 4 to #map0()[%0] {
          affine.for %arg11 = 4 to %0 {
            call @S1(%arg2, %arg9, %arg11, %arg10, %arg7, %arg4) : (memref<?x308x308xf64>, index, index, index, memref<?x308x308xf64>, memref<?x308x308xf64>) -> ()
          }
        }
      }
      affine.for %arg9 = 4 to #map0()[%0] {
        affine.for %arg10 = 4 to #map0()[%0] {
          call @S0(%arg1, %c4, %arg9, %arg10, %arg6, %arg4) : (memref<?x308x308xf64>, index, index, index, memref<?x308x308xf64>, memref<?x308x308xf64>) -> ()
          affine.for %arg11 = 5 to %0 {
            call @S0(%arg1, %arg11, %arg9, %arg10, %arg6, %arg4) : (memref<?x308x308xf64>, index, index, index, memref<?x308x308xf64>, memref<?x308x308xf64>) -> ()
            %1 = affine.apply #map1(%arg11)
            call @S3(%arg5, %1, %arg9, %arg10, %arg3, %arg1, %arg2, %arg4) : (memref<?x308x308xf64>, index, index, index, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>, memref<?x308x308xf64>) -> ()
          }
        }
      }
    }
    return
  }
  func private @polybench_timer_print()
}

