// RUN: mlir-clang %s --function=kernel_deriche | FileCheck %s

int kernel_deriche(int *a) {
    a[3]++;
    return a[1];
}

// CHECK:  func @kernel_deriche(%arg0: memref<?xi32>) -> i32 {
// CHECK-NEXT:    %c1_i32 = constant 1 : i32
// CHECK-NEXT:    %0 = affine.load %arg0[3] : memref<?xi32>
// CHECK-NEXT:    %1 = addi %0, %c1_i32 : i32
// CHECK-NEXT:    affine.store %1, %arg0[3] : memref<?xi32>
// CHECK-NEXT:    %2 = affine.load %arg0[1] : memref<?xi32>
// CHECK-NEXT:    return %2 : i32
// CHECK-NEXT:  }
