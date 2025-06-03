// RUN: mlir-clang %s --function=kernel_deriche | FileCheck %s

void sub(int *a) {
    *a = 3;
}

void kernel_deriche() {
    int a;
    sub(&a);
}

// CHECK:  func @kernel_deriche() {
// CHECK-NEXT:    %0 = memref.alloca() : memref<1xi32>
// CHECK-NEXT:    %1 = memref.cast %0 : memref<1xi32> to memref<?xi32>
// CHECK-NEXT:    call @sub(%1) : (memref<?xi32>) -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }

// CHECK:  func @sub(%arg0: memref<?xi32>) {
// CHECK-NEXT:    %c3_i32 = constant 3 : i32
// CHECK-NEXT:    affine.store %c3_i32, %arg0[0] : memref<?xi32>
// CHECK-NEXT:    return
// CHECK-NEXT:  }
