filename = gemver.RfCgraTransFinal.cloog 
#set = affine_set<()[s0] : (s0 - 1 >= 0)>
module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %c4000_i32 = constant 4000 : i32
    %c0_i32 = constant 0 : i32
    %0 = memref.alloca() : memref<1xf64>
    %1 = memref.alloca() : memref<1xf64>
    %2 = memref.alloc() : memref<4000x4000xf64>
    %3 = memref.alloc() : memref<4000xf64>
    %4 = memref.alloc() : memref<4000xf64>
    %5 = memref.alloc() : memref<4000xf64>
    %6 = memref.alloc() : memref<4000xf64>
    %7 = memref.alloc() : memref<4000xf64>
    %8 = memref.alloc() : memref<4000xf64>
    %9 = memref.alloc() : memref<4000xf64>
    %10 = memref.alloc() : memref<4000xf64>
    %11 = affine.load %0[0] : memref<1xf64>
    %12 = affine.load %1[0] : memref<1xf64>
    %13 = memref.cast %2 : memref<4000x4000xf64> to memref<?x4000xf64>
    %14 = memref.cast %3 : memref<4000xf64> to memref<?xf64>
    %15 = memref.cast %4 : memref<4000xf64> to memref<?xf64>
    %16 = memref.cast %5 : memref<4000xf64> to memref<?xf64>
    %17 = memref.cast %6 : memref<4000xf64> to memref<?xf64>
    %18 = memref.cast %7 : memref<4000xf64> to memref<?xf64>
    %19 = memref.cast %8 : memref<4000xf64> to memref<?xf64>
    %20 = memref.cast %9 : memref<4000xf64> to memref<?xf64>
    %21 = memref.cast %10 : memref<4000xf64> to memref<?xf64>
    call @kernel_gemver_opt(%c4000_i32, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21) : (i32, f64, f64, memref<?x4000xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>) -> ()
    memref.dealloc %2 : memref<4000x4000xf64>
    memref.dealloc %3 : memref<4000xf64>
    memref.dealloc %4 : memref<4000xf64>
    memref.dealloc %5 : memref<4000xf64>
    memref.dealloc %6 : memref<4000xf64>
    memref.dealloc %7 : memref<4000xf64>
    memref.dealloc %8 : memref<4000xf64>
    memref.dealloc %9 : memref<4000xf64>
    memref.dealloc %10 : memref<4000xf64>
    return %c0_i32 : i32
  }
  func private @S0(%arg0: memref<?x4000xf64>, %arg1: index, %arg2: index, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x4000xf64>
    %1 = affine.load %arg6[symbol(%arg1)] : memref<?xf64>
    %2 = affine.load %arg5[symbol(%arg2)] : memref<?xf64>
    %3 = mulf %1, %2 : f64
    %4 = addf %0, %3 : f64
    %5 = affine.load %arg4[symbol(%arg1)] : memref<?xf64>
    %6 = affine.load %arg3[symbol(%arg2)] : memref<?xf64>
    %7 = mulf %5, %6 : f64
    %8 = addf %4, %7 : f64
    affine.store %8, %arg0[symbol(%arg1), symbol(%arg2)] : memref<?x4000xf64>
    return
  }
  func private @S1(%arg0: memref<?xf64>, %arg1: index, %arg2: memref<?xf64>, %arg3: index, %arg4: f64, %arg5: memref<?x4000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1)] : memref<?xf64>
    %1 = affine.load %arg5[symbol(%arg3), symbol(%arg1)] : memref<?x4000xf64>
    %2 = mulf %arg4, %1 : f64
    %3 = affine.load %arg2[symbol(%arg3)] : memref<?xf64>
    %4 = mulf %2, %3 : f64
    %5 = addf %0, %4 : f64
    affine.store %5, %arg0[symbol(%arg1)] : memref<?xf64>
    return
  }
  func private @S2(%arg0: memref<?xf64>, %arg1: index, %arg2: memref<?xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1)] : memref<?xf64>
    %1 = affine.load %arg2[symbol(%arg1)] : memref<?xf64>
    %2 = addf %0, %1 : f64
    affine.store %2, %arg0[symbol(%arg1)] : memref<?xf64>
    return
  }
  func private @S3(%arg0: memref<?xf64>, %arg1: index, %arg2: memref<?xf64>, %arg3: index, %arg4: f64, %arg5: memref<?x4000xf64>) attributes {scop.stmt} {
    %0 = affine.load %arg0[symbol(%arg1)] : memref<?xf64>
    %1 = affine.load %arg5[symbol(%arg1), symbol(%arg3)] : memref<?x4000xf64>
    %2 = mulf %arg4, %1 : f64
    %3 = affine.load %arg2[symbol(%arg3)] : memref<?xf64>
    %4 = mulf %2, %3 : f64
    %5 = addf %0, %4 : f64
    affine.store %5, %arg0[symbol(%arg1)] : memref<?xf64>
    return
  }
  func private @kernel_gemver_opt(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x4000xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.if #set()[%0] {
      affine.for %arg12 = 0 to %0 {
        affine.for %arg13 = 0 to %0 {
          call @S0(%arg3, %arg13, %arg12, %arg7, %arg6, %arg5, %arg4) : (memref<?x4000xf64>, index, index, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>) -> ()
        }
      }
      affine.for %arg12 = 0 to %0 {
        affine.for %arg13 = 0 to %0 {
          call @S1(%arg9, %arg13, %arg10, %arg12, %arg2, %arg3) : (memref<?xf64>, index, memref<?xf64>, index, f64, memref<?x4000xf64>) -> ()
        }
      }
      affine.for %arg12 = 0 to %0 {
        call @S2(%arg9, %arg12, %arg11) : (memref<?xf64>, index, memref<?xf64>) -> ()
      }
      affine.for %arg12 = 0 to %0 {
        affine.for %arg13 = 0 to %0 {
          call @S3(%arg8, %arg13, %arg9, %arg12, %arg1, %arg3) : (memref<?xf64>, index, memref<?xf64>, index, f64, memref<?x4000xf64>) -> ()
        }
      }
    }
    return
  }
  func private @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x4000xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        call @S0(%arg3, %arg12, %arg13, %arg7, %arg6, %arg5, %arg4) : (memref<?x4000xf64>, index, index, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>) -> ()
      }
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        call @S1(%arg9, %arg12, %arg10, %arg13, %arg2, %arg3) : (memref<?xf64>, index, memref<?xf64>, index, f64, memref<?x4000xf64>) -> ()
      }
    }
    affine.for %arg12 = 0 to %0 {
      call @S2(%arg9, %arg12, %arg11) : (memref<?xf64>, index, memref<?xf64>) -> ()
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        call @S3(%arg8, %arg12, %arg9, %arg13, %arg1, %arg3) : (memref<?xf64>, index, memref<?xf64>, index, f64, memref<?x4000xf64>) -> ()
      }
    }
    return
  }
}

