filename = fdtd-2d.RfCgraTransFinal.cloog 
#map0 = affine_map<()[s0] -> (s0 - 1)>
#map1 = affine_map<(d0) -> (d0 - 1)>
#set = affine_set<()[s0, s1] : (s0 - 2 >= 0, s1 - 2 >= 0)>
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
    call @kernel_fdtd_2d_opt(%c1000_i32, %c2000_i32, %c2600_i32, %4, %5, %6, %7) : (i32, i32, i32, memref<?x2600xf64>, memref<?x2600xf64>, memref<?x2600xf64>, memref<?xf64>) -> ()
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
    %0 = index_cast %arg2 : i32 to index
    affine.for %arg7 = 0 to %0 {
      call @S0(%arg4, %arg7, %arg6) : (memref<?x2600xf64>, index, memref<?xf64>) -> ()
    }
    %1 = index_cast %arg1 : i32 to index
    affine.for %arg7 = 1 to %1 {
      affine.for %arg8 = 0 to %0 {
        call @S1(%arg4, %arg7, %arg8, %arg5) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>) -> ()
      }
    }
    affine.for %arg7 = 0 to %1 {
      affine.for %arg8 = 1 to %0 {
        call @S2(%arg3, %arg7, %arg8, %arg5) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>) -> ()
      }
    }
    affine.for %arg7 = 0 to #map0()[%1] {
      affine.for %arg8 = 0 to #map0()[%0] {
        call @S3(%arg5, %arg7, %arg8, %arg4, %arg3) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>, memref<?x2600xf64>) -> ()
      }
    }
    return
  }
  func private @polybench_timer_stop()
  func private @S0(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: memref<?xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg2[0] : memref<?xf64>
    affine.store %0, %arg0[0, symbol(%arg1)] : memref<?x2600xf64>
    return
  }
  func private @S1(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index, %arg3: memref<?x2600xf64>) attributes {scop.stmt} {
    %cst = constant 5.000000e-01 : f64
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %1 = affine.load %arg3[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %2 = affine.load %arg3[symbol(%arg1) - 1, symbol(%arg2)] : memref<?x2600xf64>
    %3 = subf %1, %2 : f64
    %4 = mulf %cst, %3 : f64
    %5 = subf %0, %4 : f64
    affine.store %5, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @S2(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index, %arg3: memref<?x2600xf64>) attributes {scop.stmt} {
    %cst = constant 5.000000e-01 : f64
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %1 = affine.load %arg3[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %2 = affine.load %arg3[symbol(%arg1), symbol(%arg2) - 1] : memref<?x2600xf64>
    %3 = subf %1, %2 : f64
    %4 = mulf %cst, %3 : f64
    %5 = subf %0, %4 : f64
    affine.store %5, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @S3(%arg0: memref<?x2600xf64>, %arg1: index, %arg2: index, %arg3: memref<?x2600xf64>, %arg4: memref<?x2600xf64>) attributes {scop.stmt} {
    %cst = constant 0.69999999999999996 : f64
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %1 = affine.load %arg4[symbol(%arg1), symbol(%arg2) + 1] : memref<?x2600xf64>
    %2 = affine.load %arg4[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %3 = subf %1, %2 : f64
    %4 = affine.load %arg3[symbol(%arg1) + 1, symbol(%arg2)] : memref<?x2600xf64>
    %5 = addf %3, %4 : f64
    %6 = affine.load %arg3[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    %7 = subf %5, %6 : f64
    %8 = mulf %cst, %7 : f64
    %9 = subf %0, %8 : f64
    affine.store %9, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2600xf64>
    return
  }
  func private @kernel_fdtd_2d_opt(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: memref<?x2600xf64>, %arg4: memref<?x2600xf64>, %arg5: memref<?x2600xf64>, %arg6: memref<?xf64>) {
    %0 = index_cast %arg1 : i32 to index
    %1 = index_cast %arg2 : i32 to index
    affine.if #set()[%1, %0] {
      affine.for %arg7 = 0 to %1 {
        call @S0(%arg4, %arg7, %arg6) : (memref<?x2600xf64>, index, memref<?xf64>) -> ()
      }
      affine.for %arg7 = 0 to %0 {
        affine.for %arg8 = 1 to %1 {
          call @S2(%arg3, %arg7, %arg8, %arg5) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>) -> ()
        }
      }
      affine.for %arg7 = 0 to #map0()[%1] {
        affine.for %arg8 = 1 to %0 {
          call @S1(%arg4, %arg8, %arg7, %arg5) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>) -> ()
          %2 = affine.apply #map1(%arg8)
          call @S3(%arg5, %2, %arg7, %arg4, %arg3) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>, memref<?x2600xf64>) -> ()
        }
      }
      affine.for %arg7 = 1 to %0 {
        %2 = affine.apply #map0()[%1]
        call @S1(%arg4, %arg7, %2, %arg5) : (memref<?x2600xf64>, index, index, memref<?x2600xf64>) -> ()
      }
    }
    return
  }
  func private @polybench_timer_print()
}

