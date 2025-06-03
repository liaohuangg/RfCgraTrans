// RUN: RfCgraTrans-opt %s -pluto-opt | FileCheck %s

func @load_store_dep_tiling() {
  %A = alloc() : memref<64x64xf32>

  affine.for %i0 = 1 to 64 {
    affine.for %j0 = 1 to 64 {
      %i1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%i0)
      %j1 = affine.apply affine_map<(d0) -> (d0 - 1)>(%j0)

      %0 = affine.load %A[%i0, %j1] : memref<64x64xf32>
      %1 = affine.load %A[%i1, %j0] : memref<64x64xf32>
      %2 = addf %0, %1 : f32
      affine.store %2, %A[%i0, %j0] : memref<64x64xf32>
    }
  }

  return
}

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0) -> (d0 - 1)>
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0) -> (1, d0 * 32)>
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0) -> (d0 * 32 + 32)>
//
//
// CHECK:      func @main(%[[ARG0:.*]]: memref<?x?xf32>) {
// CHECK-NEXT:   affine.for %[[ARG1:.*]] = 0 to 2 {
// CHECK-NEXT:     affine.for %[[ARG2:.*]] = 0 to 2 {
// CHECK-NEXT:       affine.for %[[ARG3:.*]] = max #[[MAP2]](%[[ARG1]]) to #[[MAP3]](%[[ARG1]]) {
// CHECK-NEXT:         affine.for %[[ARG4:.*]] = max #[[MAP2]](%[[ARG2]]) to #[[MAP3]](%[[ARG2]]) {
// CHECK-NEXT:           %[[VAL0:.*]] = affine.apply #[[MAP0]](%[[ARG3]])
// CHECK-NEXT:           %[[VAL1:.*]] = affine.load %[[ARG0]][%[[VAL0]], %[[ARG4]]] : memref<?x?xf32>
// CHECK-NEXT:           %[[VAL2:.*]] = affine.apply #[[MAP0]](%[[ARG4]])
// CHECK-NEXT:           %[[VAL3:.*]] = affine.load %[[ARG0]][%[[ARG3]], %[[VAL2]]] : memref<?x?xf32>
// CHECK-NEXT:           %[[VAL4:.*]] = addf %[[VAL3]], %[[VAL1]] : f32
// CHECK-NEXT:           affine.store %[[VAL4]], %[[ARG0]][%[[ARG3]], %[[ARG4]]] : memref<?x?xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
