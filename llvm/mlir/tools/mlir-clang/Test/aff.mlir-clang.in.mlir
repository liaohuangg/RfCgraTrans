module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"}  {
  func @kernel_deriche(%arg0: i32, %arg1: i32, %arg2: f64, %arg3: memref<?xmemref<?xf64>>) {
    %0 = index_cast %arg1 : i32 to index
    %1 = index_cast %arg0 : i32 to index
    affine.for %arg4 = 0 to %1 {
      affine.for %arg5 = 0 to %0 {
        %2 = affine.load %arg3[%arg4] : memref<?xmemref<?xf64>>
        affine.store %arg2, %2[-%arg5 + symbol(%0) - 1] : memref<?xf64>
      }
    }
    return
  }
}
