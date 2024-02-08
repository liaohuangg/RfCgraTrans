#set = affine_set<()[s0, s1, s2] : (s0 - 1 >= 0, s1 - 1 >= 0, s2 - 1 >= 0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c1000_i32 = constant 1000 : i32
    %c1100_i32 = constant 1100 : i32
    %c1200_i32 = constant 1200 : i32
    %c0_i32 = constant 0 : i32
    %0 = memref.alloca() : memref<1xf64>
    %1 = memref.alloca() : memref<1xf64>
    %2 = memref.alloc() : memref<1000x1100xf64>
    %3 = memref.alloc() : memref<1000x1200xf64>
    %4 = memref.alloc() : memref<1200x1100xf64>
    %5 = affine.load %0[0] : memref<1xf64>
    %6 = affine.load %1[0] : memref<1xf64>
    %7 = memref.cast %2 : memref<1000x1100xf64> to memref<?x1100xf64>
    %8 = memref.cast %3 : memref<1000x1200xf64> to memref<?x1200xf64>
    %9 = memref.cast %4 : memref<1200x1100xf64> to memref<?x1100xf64>
    call @kernel_gemm_opt(%c1000_i32, %c1100_i32, %c1200_i32, %5, %6, %7, %8, %9) : (i32, i32, i32, f64, f64, memref<?x1100xf64>, memref<?x1200xf64>, memref<?x1100xf64>) -> ()
    memref.dealloc %2 : memref<1000x1100xf64>
    memref.dealloc %3 : memref<1000x1200xf64>
    memref.dealloc %4 : memref<1200x1100xf64>
    return %c0_i32 : i32
  }
  func private @S0(%arg0: memref<?x1100xf64>, %arg1: index, %arg2: index, %arg3: f64) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x1100xf64>
    %1 = mulf %0, %arg3 : f64
    affine.store %1, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x1100xf64>
    return
  }
  func private @S1(%arg0: memref<?x1100xf64>, %arg1: index, %arg2: index, %arg3: memref<?x1100xf64>, %arg4: index, %arg5: f64, %arg6: memref<?x1200xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x1100xf64>
    %1 = affine.load %arg6[symbol(%arg1), symbol(%arg4)] : memref<?x1200xf64>
    %2 = mulf %arg5, %1 : f64
    %3 = affine.load %arg3[symbol(%arg4), symbol(%arg2)] : memref<?x1100xf64>
    %4 = mulf %2, %3 : f64
    %5 = addf %0, %4 : f64
    affine.store %5, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x1100xf64>
    return
  }
  func private @kernel_gemm_opt(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<?x1100xf64>, %arg6: memref<?x1200xf64>, %arg7: memref<?x1100xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %arg2 : i32 to index
    affine.if #set()[%0, %1, %2] {
      affine.for %arg8 = 0 to %0 {
        affine.for %arg9 = 0 to %1 {
          call @S0(%arg5, %arg8, %arg9, %arg4) : (memref<?x1100xf64>, index, index, f64) -> ()
        }
      }
      affine.for %arg8 = 0 to %0 {
        affine.for %arg9 = 0 to %2 {
          affine.for %arg10 = 0 to %1 {
            call @S1(%arg5, %arg8, %arg10, %arg7, %arg9, %arg3, %arg6) : (memref<?x1100xf64>, index, index, memref<?x1100xf64>, index, f64, memref<?x1200xf64>) -> ()
          }
        }
      }
    }
    return
  }
  func private @kernel_gemm(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: f64, %arg4: f64, %arg5: memref<?x1100xf64>, %arg6: memref<?x1200xf64>, %arg7: memref<?x1100xf64>) {
    %0 = index_cast %arg0 : i32 to index
    %1 = index_cast %arg1 : i32 to index
    %2 = index_cast %arg2 : i32 to index
    affine.for %arg8 = 0 to %0 {
      affine.for %arg9 = 0 to %1 {
        call @S0(%arg5, %arg8, %arg9, %arg4) : (memref<?x1100xf64>, index, index, f64) -> ()
      }
      affine.for %arg9 = 0 to %2 {
        affine.for %arg10 = 0 to %1 {
          call @S1(%arg5, %arg8, %arg10, %arg7, %arg9, %arg3, %arg6) : (memref<?x1100xf64>, index, index, memref<?x1100xf64>, index, f64, memref<?x1200xf64>) -> ()
        }
      }
    }
    return
  }
}

