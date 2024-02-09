filename = jacobi-2d.RfCgraTransFinal.cloog 
#map = affine_map<()[s0] -> (s0 - 1)>
#set = affine_set<()[s0] : (s0 - 3 >= 0)>
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
    call @kernel_jacobi_2d_opt(%c1000_i32, %c2800_i32, %2, %3) : (i32, i32, memref<?x2800xf64>, memref<?x2800xf64>) -> ()
    call @polybench_timer_stop() : () -> ()
    call @polybench_timer_print() : () -> ()
    memref.dealloc %0 : memref<2800x2800xf64>
    memref.dealloc %1 : memref<2800x2800xf64>
    return %c0_i32 : i32
  }
  func private @polybench_timer_start()
  func private @kernel_jacobi_2d(%arg0: i32, %arg1: i32, %arg2: memref<?x2800xf64>, %arg3: memref<?x2800xf64>) {
    %0 = index_cast %arg1 : i32 to index
    affine.for %arg4 = 1 to #map()[%0] {
      affine.for %arg5 = 1 to #map()[%0] {
        call @S0(%arg3, %arg4, %arg5, %arg2) : (memref<?x2800xf64>, index, index, memref<?x2800xf64>) -> ()
      }
    }
    affine.for %arg4 = 1 to #map()[%0] {
      affine.for %arg5 = 1 to #map()[%0] {
        call @S1(%arg2, %arg4, %arg5, %arg3) : (memref<?x2800xf64>, index, index, memref<?x2800xf64>) -> ()
      }
    }
    return
  }
  func private @polybench_timer_stop()
  func private @S0(%arg0: memref<?x2800xf64>, %arg1: index, %arg2: index, %arg3: memref<?x2800xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e-01 : f64
    %0 = affine.load %arg3[symbol(%arg1), symbol(%arg2)] : memref<?x2800xf64>
    %1 = affine.load %arg3[symbol(%arg1), symbol(%arg2) - 1] : memref<?x2800xf64>
    %2 = addf %0, %1 : f64
    %3 = affine.load %arg3[symbol(%arg1), symbol(%arg2) + 1] : memref<?x2800xf64>
    %4 = addf %2, %3 : f64
    %5 = affine.load %arg3[symbol(%arg1) + 1, symbol(%arg2)] : memref<?x2800xf64>
    %6 = addf %4, %5 : f64
    %7 = affine.load %arg3[symbol(%arg1) - 1, symbol(%arg2)] : memref<?x2800xf64>
    %8 = addf %6, %7 : f64
    %9 = mulf %cst, %8 : f64
    affine.store %9, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2800xf64>
    return
  }
  func private @S1(%arg0: memref<?x2800xf64>, %arg1: index, %arg2: index, %arg3: memref<?x2800xf64>) attributes {scop.stmt} {
    %cst = constant 2.000000e-01 : f64
    %0 = affine.load %arg3[symbol(%arg1), symbol(%arg2)] : memref<?x2800xf64>
    %1 = affine.load %arg3[symbol(%arg1), symbol(%arg2) - 1] : memref<?x2800xf64>
    %2 = addf %0, %1 : f64
    %3 = affine.load %arg3[symbol(%arg1), symbol(%arg2) + 1] : memref<?x2800xf64>
    %4 = addf %2, %3 : f64
    %5 = affine.load %arg3[symbol(%arg1) + 1, symbol(%arg2)] : memref<?x2800xf64>
    %6 = addf %4, %5 : f64
    %7 = affine.load %arg3[symbol(%arg1) - 1, symbol(%arg2)] : memref<?x2800xf64>
    %8 = addf %6, %7 : f64
    %9 = mulf %cst, %8 : f64
    affine.store %9, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x2800xf64>
    return
  }
  func private @kernel_jacobi_2d_opt(%arg0: i32, %arg1: i32, %arg2: memref<?x2800xf64>, %arg3: memref<?x2800xf64>) {
    %0 = index_cast %arg1 : i32 to index
    affine.if #set()[%0] {
      affine.for %arg4 = 1 to #map()[%0] {
        affine.for %arg5 = 1 to #map()[%0] {
          call @S0(%arg3, %arg5, %arg4, %arg2) : (memref<?x2800xf64>, index, index, memref<?x2800xf64>) -> ()
        }
      }
      affine.for %arg4 = 1 to #map()[%0] {
        affine.for %arg5 = 1 to #map()[%0] {
          call @S1(%arg2, %arg5, %arg4, %arg3) : (memref<?x2800xf64>, index, index, memref<?x2800xf64>) -> ()
        }
      }
    }
    return
  }
  func private @polybench_timer_print()
}

