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
    call @kernel_gemver(%c4000_i32, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21) : (i32, f64, f64, memref<?x4000xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>, memref<?xf64>) -> ()
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
  func private @kernel_gemver(%arg0: i32, %arg1: f64, %arg2: f64, %arg3: memref<?x4000xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>) {
    %0 = index_cast %arg0 : i32 to index
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        %1 = affine.load %arg3[%arg12, %arg13] : memref<?x4000xf64>
        %2 = affine.load %arg4[%arg12] : memref<?xf64>
        %3 = affine.load %arg5[%arg13] : memref<?xf64>
        %4 = mulf %2, %3 : f64
        %5 = addf %1, %4 : f64
        %6 = affine.load %arg6[%arg12] : memref<?xf64>
        %7 = affine.load %arg7[%arg13] : memref<?xf64>
        %8 = mulf %6, %7 : f64
        %9 = addf %5, %8 : f64
        affine.store %9, %arg3[%arg12, %arg13] : memref<?x4000xf64>
      }
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        %1 = affine.load %arg9[%arg12] : memref<?xf64>
        %2 = affine.load %arg3[%arg13, %arg12] : memref<?x4000xf64>
        %3 = mulf %arg2, %2 : f64
        %4 = affine.load %arg10[%arg13] : memref<?xf64>
        %5 = mulf %3, %4 : f64
        %6 = addf %1, %5 : f64
        affine.store %6, %arg9[%arg12] : memref<?xf64>
      }
    }
    affine.for %arg12 = 0 to %0 {
      %1 = affine.load %arg9[%arg12] : memref<?xf64>
      %2 = affine.load %arg11[%arg12] : memref<?xf64>
      %3 = addf %1, %2 : f64
      affine.store %3, %arg9[%arg12] : memref<?xf64>
    }
    affine.for %arg12 = 0 to %0 {
      affine.for %arg13 = 0 to %0 {
        %1 = affine.load %arg8[%arg12] : memref<?xf64>
        %2 = affine.load %arg3[%arg12, %arg13] : memref<?x4000xf64>
        %3 = mulf %arg1, %2 : f64
        %4 = affine.load %arg9[%arg13] : memref<?xf64>
        %5 = mulf %3, %4 : f64
        %6 = addf %1, %5 : f64
        affine.store %6, %arg8[%arg12] : memref<?xf64>
      }
    }
    return
  }
}
