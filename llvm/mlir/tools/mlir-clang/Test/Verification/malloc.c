// RUN: mlir-clang %s --function=caller | FileCheck %s

#include <stdlib.h>

void sum(double *result);

void caller(int size) {
    double* array = (double*)malloc(sizeof(double) * size);
    sum(array);
    free(array);
}

// CHECK:  func @caller(%arg0: i32) {
// CHECK-NEXT:    %c8 = constant 8 : index
// CHECK-NEXT:    %0 = zexti %arg0 : i32 to i64
// CHECK-NEXT:    %1 = index_cast %0 : i64 to index
// CHECK-NEXT:    %2 = muli %1, %c8 : index
// CHECK-NEXT:    %3 = divi_unsigned %2, %c8 : index
// CHECK-NEXT:    %4 = memref.alloc(%3) : memref<?xf64>
// CHECK-NEXT:    call @sum(%4) : (memref<?xf64>) -> ()
// CHECK-NEXT:    memref.dealloc %4 : memref<?xf64>
// CHECK-NEXT:    return
// CHECK-NEXT:  }