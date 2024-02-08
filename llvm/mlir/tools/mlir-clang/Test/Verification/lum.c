// RUN: mlir-clang %s --function=test | FileCheck %s

int test() {
    return -3;
}

// CHECK:  func @test() -> i32 {
// CHECK-NEXT:    %c-3_i32 = constant -3 : i32
// CHECK-NEXT:    return %c-3_i32 : i32
// CHECK-NEXT:  }