// RUN: mlir-opt -allow-unregistered-dialect %s -split-input-file -pass-pipeline='func(canonicalize)' | FileCheck %s
// XFAIL: *
// -----

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<(d0) -> (d0 - 1)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<(d0) -> (d0 + 1)>

// CHECK-LABEL: func @compose_affine_maps_1dto2d_no_symbols() {
func @compose_affine_maps_1dto2d_no_symbols() {
  %0 = memref.alloc() : memref<4x4xf32>

  affine.for %i0 = 0 to 15 {
    // Test load[%x, %x]

    %x0 = affine.apply affine_map<(d0) -> (d0 - 1)> (%i0)
    %x1_0 = affine.apply affine_map<(d0, d1) -> (d0)> (%x0, %x0)
    %x1_1 = affine.apply affine_map<(d0, d1) -> (d1)> (%x0, %x0)

    // CHECK: %[[I0A:.*]] = affine.apply #[[$MAP0]](%{{.*}})
    // CHECK-NEXT: %[[V0:.*]] = memref.load %0[%[[I0A]], %[[I0A]]]
    %v0 = memref.load %0[%x1_0, %x1_1] : memref<4x4xf32>

    // Test store[%y, %y]
    %y0 = affine.apply affine_map<(d0) -> (d0 + 1)> (%i0)
    %y1_0 = affine.apply affine_map<(d0, d1) -> (d0)> (%y0, %y0)
    %y1_1 = affine.apply affine_map<(d0, d1) -> (d1)> (%y0, %y0)

    // CHECK-NEXT: %[[I1A:.*]] = affine.apply #[[$MAP1]](%{{.*}})
    // CHECK-NEXT: memref.store %[[V0]], %0[%[[I1A]], %[[I1A]]]
    memref.store %v0, %0[%y1_0, %y1_1] : memref<4x4xf32>

    // Test store[%x, %y]
    %xy_0 = affine.apply affine_map<(d0, d1) -> (d0)> (%x0, %y0)
    %xy_1 = affine.apply affine_map<(d0, d1) -> (d1)> (%x0, %y0)

    // CHECK-NEXT: memref.store %[[V0]], %0[%[[I0A]], %[[I1A]]]
    memref.store %v0, %0[%xy_0, %xy_1] : memref<4x4xf32>

    // Test store[%y, %x]
    %yx_0 = affine.apply affine_map<(d0, d1) -> (d0)> (%y0, %x0)
    %yx_1 = affine.apply affine_map<(d0, d1) -> (d1)> (%y0, %x0)
    // CHECK-NEXT: memref.store %[[V0]], %0[%[[I1A]], %[[I0A]]]
    memref.store %v0, %0[%yx_0, %yx_1] : memref<4x4xf32>
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP4:.*]] = affine_map<(d0) -> (d0 - 4)>
// CHECK-DAG: #[[$MAP7:.*]] = affine_map<(d0) -> (d0 * 2 - 3)>
// CHECK-DAG: #[[$MAP7a:.*]] = affine_map<(d0) -> (d0 * 2 + 1)>

// CHECK-LABEL: func @compose_affine_maps_1dto2d_with_symbols() {
func @compose_affine_maps_1dto2d_with_symbols() {
  %0 = memref.alloc() : memref<4x4xf32>

  affine.for %i0 = 0 to 15 {
    // Test load[%x0, %x0] with symbol %c4
    %c4 = constant 4 : index
    %x0 = affine.apply affine_map<(d0)[s0] -> (d0 - s0)> (%i0)[%c4]

    // CHECK: %[[I0:.*]] = affine.apply #[[$MAP4]](%{{.*}})
    // CHECK-NEXT: %[[V0:.*]] = memref.load %{{.*}}[%[[I0]], %[[I0]]]
    %v0 = memref.load %0[%x0, %x0] : memref<4x4xf32>

    // Test load[%x0, %x1] with symbol %c4 captured by '%x0' map.
    %x1 = affine.apply affine_map<(d0) -> (d0 + 1)> (%i0)
    %y1 = affine.apply affine_map<(d0, d1) -> (d0+d1)> (%x0, %x1)
    // CHECK-NEXT: %[[I1:.*]] = affine.apply #[[$MAP7]](%{{.*}})
    // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I1]], %[[I1]]]
    memref.store %v0, %0[%y1, %y1] : memref<4x4xf32>

    // Test store[%x1, %x0] with symbol %c4 captured by '%x0' map.
    %y2 = affine.apply affine_map<(d0, d1) -> (d0 + d1)> (%x1, %x0)
    // CHECK-NEXT: %[[I2:.*]] = affine.apply #[[$MAP7]](%{{.*}})
    // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I2]], %[[I2]]]
    memref.store %v0, %0[%y2, %y2] : memref<4x4xf32>

    // Test store[%x2, %x0] with symbol %c4 from '%x0' and %c5 from '%x2'
    %c5 = constant 5 : index
    %x2 = affine.apply affine_map<(d0)[s0] -> (d0 + s0)> (%i0)[%c5]
    %y3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)> (%x2, %x0)
    // CHECK: %[[I3:.*]] = affine.apply #[[$MAP7a]](%{{.*}})
    // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I3]], %[[I3]]]
    memref.store %v0, %0[%y3, %y3] : memref<4x4xf32>
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP8:.*]] = affine_map<(d0, d1) -> (d1 + (d0 ceildiv 4) * 4 - (d1 floordiv 4) * 4)>
// CHECK-DAG: #[[$MAP8a:.*]] = affine_map<(d0, d1) -> (d1 + (d0 ceildiv 8) * 8 - (d1 floordiv 8) * 8)>

// CHECK-LABEL: func @compose_affine_maps_2d_tile() {
func @compose_affine_maps_2d_tile() {
  %0 = memref.alloc() : memref<16x32xf32>
  %1 = memref.alloc() : memref<16x32xf32>

  %c4 = constant 4 : index
  %c8 = constant 8 : index

  affine.for %i0 = 0 to 3 {
    %x0 = affine.apply affine_map<(d0)[s0] -> (d0 ceildiv s0)> (%i0)[%c4]
    affine.for %i1 = 0 to 3 {
      %x1 = affine.apply affine_map<(d0)[s0] -> (d0 ceildiv s0)> (%i1)[%c8]
      affine.for %i2 = 0 to 3 {
        %x2 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)> (%i2)[%c4]
        affine.for %i3 = 0 to 3 {
          %x3 = affine.apply affine_map<(d0)[s0] -> (d0 mod s0)> (%i3)[%c8]

          %x40 = affine.apply affine_map<(d0, d1, d2, d3)[s0, s1] ->
            ((d0 * s0) + d2)> (%x0, %x1, %x2, %x3)[%c4, %c8]
          %x41 = affine.apply affine_map<(d0, d1, d2, d3)[s0, s1] ->
            ((d1 * s1) + d3)> (%x0, %x1, %x2, %x3)[%c4, %c8]
          // CHECK: %[[I0:.*]] = affine.apply #[[$MAP8]](%{{.*}}, %{{.*}})
          // CHECK: %[[I1:.*]] = affine.apply #[[$MAP8a]](%{{.*}}, %{{.*}})
          // CHECK-NEXT: %[[L0:.*]] = memref.load %{{.*}}[%[[I0]], %[[I1]]]
          %v0 = memref.load %0[%x40, %x41] : memref<16x32xf32>

          // CHECK-NEXT: memref.store %[[L0]], %{{.*}}[%[[I0]], %[[I1]]]
          memref.store %v0, %1[%x40, %x41] : memref<16x32xf32>
        }
      }
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP4b:.*]] = affine_map<(d0) -> (d0 - 7)>
// CHECK-DAG: #[[$MAP9:.*]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-DAG: #[[$MAP10:.*]] = affine_map<(d0) -> (d0 * 3)>
// CHECK-DAG: #[[$MAP11:.*]] = affine_map<(d0) -> ((d0 + 3) ceildiv 3)>
// CHECK-DAG: #[[$MAP12:.*]] = affine_map<(d0) -> (d0 * 7 - 49)>

// CHECK-LABEL: func @compose_affine_maps_dependent_loads() {
func @compose_affine_maps_dependent_loads() {
  %0 = memref.alloc() : memref<16x32xf32>
  %1 = memref.alloc() : memref<16x32xf32>

  affine.for %i0 = 0 to 3 {
    affine.for %i1 = 0 to 3 {
      affine.for %i2 = 0 to 3 {
        %c3 = constant 3 : index
        %c7 = constant 7 : index

        %x00 = affine.apply affine_map<(d0, d1, d2)[s0, s1] -> (d0 + s0)>
            (%i0, %i1, %i2)[%c3, %c7]
        %x01 = affine.apply affine_map<(d0, d1, d2)[s0, s1] -> (d1 - s1)>
            (%i0, %i1, %i2)[%c3, %c7]
        %x02 = affine.apply affine_map<(d0, d1, d2)[s0, s1] -> (d2 * s0)>
            (%i0, %i1, %i2)[%c3, %c7]

        // CHECK: %[[I0:.*]] = affine.apply #[[$MAP9]](%{{.*}})
        // CHECK: %[[I1:.*]] = affine.apply #[[$MAP4b]](%{{.*}})
        // CHECK: %[[I2:.*]] = affine.apply #[[$MAP10]](%{{.*}})
        // CHECK-NEXT: %[[V0:.*]] = memref.load %{{.*}}[%[[I0]], %[[I1]]]
        %v0 = memref.load %0[%x00, %x01] : memref<16x32xf32>

        // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I0]], %[[I2]]]
        memref.store %v0, %0[%x00, %x02] : memref<16x32xf32>

        // Swizzle %i0, %i1
        // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I1]], %[[I0]]]
        memref.store %v0, %0[%x01, %x00] : memref<16x32xf32>

        // Swizzle %x00, %x01 and %c3, %c7
        %x10 = affine.apply affine_map<(d0, d1)[s0, s1] -> (d0 * s1)>
           (%x01, %x00)[%c3, %c7]
        %x11 = affine.apply affine_map<(d0, d1)[s0, s1] -> (d1 ceildiv s0)>
           (%x01, %x00)[%c3, %c7]

        // CHECK-NEXT: %[[I2A:.*]] = affine.apply #[[$MAP12]](%{{.*}})
        // CHECK-NEXT: %[[I2B:.*]] = affine.apply #[[$MAP11]](%{{.*}})
        // CHECK-NEXT: memref.store %[[V0]], %{{.*}}[%[[I2A]], %[[I2B]]]
        memref.store %v0, %0[%x10, %x11] : memref<16x32xf32>
      }
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP13A:.*]] = affine_map<(d0) -> ((d0 + 6) ceildiv 8)>
// CHECK-DAG: #[[$MAP13B:.*]] = affine_map<(d0) -> ((d0 * 4 - 4) floordiv 3)>

// CHECK-LABEL: func @compose_affine_maps_diamond_dependency
func @compose_affine_maps_diamond_dependency(%arg0: f32, %arg1: memref<4x4xf32>) {
  affine.for %i0 = 0 to 15 {
    %a = affine.apply affine_map<(d0) -> (d0 - 1)> (%i0)
    %b = affine.apply affine_map<(d0) -> (d0 + 7)> (%a)
    %c = affine.apply affine_map<(d0) -> (d0 * 4)> (%a)
    %d0 = affine.apply affine_map<(d0, d1) -> (d0 ceildiv 8)> (%b, %c)
    %d1 = affine.apply affine_map<(d0, d1) -> (d1 floordiv 3)> (%b, %c)
    // CHECK: %[[I0:.*]] = affine.apply #[[$MAP13A]](%{{.*}})
    // CHECK: %[[I1:.*]] = affine.apply #[[$MAP13B]](%{{.*}})
    // CHECK-NEXT: memref.store %arg0, %arg1[%[[I0]], %[[I1]]]
    memref.store %arg0, %arg1[%d0, %d1] : memref<4x4xf32>
  }

  return
}

// -----

// CHECK-DAG: #[[$MAP14:.*]] = affine_map<()[s0, s1] -> (((s1 + s0) * 4) floordiv s0)>

// CHECK-LABEL: func @compose_affine_maps_multiple_symbols
func @compose_affine_maps_multiple_symbols(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] -> (s0 + d0)> (%arg0)[%arg1]
  %c = affine.apply affine_map<(d0) -> (d0 * 4)> (%a)
  %e = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)> (%c)[%arg1]
  // CHECK: [[I0:.*]] = affine.apply #[[$MAP14]]()[%{{.*}}, %{{.*}}]
  return %e : index
}

// -----

// CHECK-LABEL: func @arg_used_as_dim_and_symbol
func @arg_used_as_dim_and_symbol(%arg0: memref<100x100xf32>, %arg1: index, %arg2: f32) {
  %c9 = constant 9 : index
  %1 = memref.alloc() : memref<100x100xf32, 1>
  %2 = memref.alloc() : memref<1xi32>
  affine.for %i0 = 0 to 100 {
    affine.for %i1 = 0 to 100 {
      %3 = affine.apply affine_map<(d0, d1)[s0, s1] -> (d1 + s0 + s1)>
        (%i0, %i1)[%arg1, %c9]
      %4 = affine.apply affine_map<(d0, d1, d3) -> (d3 - (d0 + d1))>
        (%arg1, %c9, %3)
      // CHECK: memref.store %arg2, %{{.*}}[%{{.*}}, %{{.*}}]
      memref.store %arg2, %1[%4, %arg1] : memref<100x100xf32, 1>
    }
  }
  return
}

// -----

// CHECK-LABEL: func @trivial_maps
func @trivial_maps() {
  // CHECK-NOT: affine.apply

  %0 = memref.alloc() : memref<10xf32>
  %c0 = constant 0 : index
  %cst = constant 0.000000e+00 : f32
  affine.for %i1 = 0 to 10 {
    %1 = affine.apply affine_map<()[s0] -> (s0)>()[%c0]
    memref.store %cst, %0[%1] : memref<10xf32>
    %2 = memref.load %0[%c0] : memref<10xf32>

    %3 = affine.apply affine_map<()[] -> (0)>()[]
    memref.store %cst, %0[%3] : memref<10xf32>
    memref.store %2, %0[%c0] : memref<10xf32>
  }
  return
}

// -----

// CHECK-DAG: #[[$MAP15:.*]] = affine_map<()[s0] -> (s0 - 42)>

// CHECK-LABEL: func @partial_fold_map
func @partial_fold_map(%arg1: index, %arg2: index) -> index {
  // TODO: Constant fold one index into affine.apply
  %c42 = constant 42 : index
  %2 = affine.apply affine_map<(d0, d1) -> (d0 - d1)> (%arg1, %c42)
  // CHECK: [[X:.*]] = affine.apply #[[$MAP15]]()[%{{.*}}]
  return %2 : index
}

// -----

// CHECK-DAG: #[[$MAP_symbolic_composition_a:.*]] = affine_map<()[s0] -> (s0 * 512)>

// CHECK-LABEL: func @symbolic_composition_a(%{{.*}}: index, %{{.*}}: index) -> index {
func @symbolic_composition_a(%arg0: index, %arg1: index) -> index {
  %0 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg0)
  %1 = affine.apply affine_map<()[s0, s1] -> (8 * s0)>()[%0, %arg0]
  %2 = affine.apply affine_map<()[s0, s1] -> (16 * s1)>()[%arg1, %1]
  // CHECK: %{{.*}} = affine.apply #[[$MAP_symbolic_composition_a]]()[%{{.*}}]
  return %2 : index
}

// -----

// CHECK-DAG: #[[$MAP_symbolic_composition_b:.*]] = affine_map<()[s0] -> (s0 * 4)>

// CHECK-LABEL: func @symbolic_composition_b(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> index {
func @symbolic_composition_b(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> index {
  %0 = affine.apply affine_map<(d0) -> (d0)>(%arg0)
  %1 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 + s3)>()[%0, %0, %0, %0]
  // CHECK: %{{.*}} = affine.apply #[[$MAP_symbolic_composition_b]]()[%{{.*}}]
  return %1 : index
}

// -----

// CHECK-DAG: #[[$MAP_symbolic_composition_c:.*]] = affine_map<()[s0, s1] -> (s0 * 3 + s1)>

// CHECK-LABEL: func @symbolic_composition_c(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> index {
func @symbolic_composition_c(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> index {
  %0 = affine.apply affine_map<(d0) -> (d0)>(%arg0)
  %1 = affine.apply affine_map<(d0) -> (d0)>(%arg1)
  %2 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 + s3)>()[%0, %0, %0, %1]
  // CHECK: %{{.*}} = affine.apply #[[$MAP_symbolic_composition_c]]()[%{{.*}}, %{{.*}}]
  return %2 : index
}

// -----

// CHECK-DAG: #[[$MAP_symbolic_composition_d:.*]] = affine_map<()[s0, s1] -> (s0 + s1 * 3)>

// CHECK-LABEL: func @symbolic_composition_d(
//  CHECK-SAME:   %[[ARG0:[0-9a-zA-Z]+]]: index
//  CHECK-SAME:   %[[ARG1:[0-9a-zA-Z]+]]: index
func @symbolic_composition_d(%arg0: index, %arg1: index, %arg2: index, %arg3: index) -> index {
  %0 = affine.apply affine_map<(d0) -> (d0)>(%arg0)
  %1 = affine.apply affine_map<()[s0] -> (s0)>()[%arg1]
  %2 = affine.apply affine_map<()[s0, s1, s2, s3] -> (s0 + s1 + s2 + s3)>()[%0, %0, %0, %1]
  // CHECK: %{{.*}} = affine.apply #[[$MAP_symbolic_composition_d]]()[%[[ARG1]], %[[ARG0]]]
  return %2 : index
}

// -----

// CHECK-DAG: #[[$MAP_mix_dims_and_symbols_b:.*]] = affine_map<()[s0, s1] -> (s0 * 42 + s1 + 6)>

// CHECK-LABEL: func @mix_dims_and_symbols_b(%arg0: index, %arg1: index) -> index {
func @mix_dims_and_symbols_b(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] -> (d0 - 1 + 42 * s0)> (%arg0)[%arg1]
  %b = affine.apply affine_map<(d0) -> (d0 + 7)> (%a)
  // CHECK: {{.*}} = affine.apply #[[$MAP_mix_dims_and_symbols_b]]()[%{{.*}}, %{{.*}}]

  return %b : index
}

// -----

// CHECK-DAG: #[[$MAP_mix_dims_and_symbols_c:.*]] = affine_map<()[s0, s1] -> (s0 * 168 + s1 * 4 - 4)>

// CHECK-LABEL: func @mix_dims_and_symbols_c(%arg0: index, %arg1: index) -> index {
func @mix_dims_and_symbols_c(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] -> (d0 - 1 + 42 * s0)> (%arg0)[%arg1]
  %b = affine.apply affine_map<(d0) -> (d0 + 7)> (%a)
  %c = affine.apply affine_map<(d0) -> (d0 * 4)> (%a)
  // CHECK: {{.*}} = affine.apply #[[$MAP_mix_dims_and_symbols_c]]()[%{{.*}}, %{{.*}}]
  return %c : index
}

// -----

// CHECK-DAG: #[[$MAP_mix_dims_and_symbols_d:.*]] = affine_map<()[s0, s1] -> ((s0 * 42 + s1 + 6) ceildiv 8)>

// CHECK-LABEL: func @mix_dims_and_symbols_d(%arg0: index, %arg1: index) -> index {
func @mix_dims_and_symbols_d(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] -> (d0 - 1 + 42 * s0)> (%arg0)[%arg1]
  %b = affine.apply affine_map<(d0) -> (d0 + 7)> (%a)
  %c = affine.apply affine_map<(d0) -> (d0 * 4)> (%a)
  %d = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)> ()[%b]
  // CHECK: {{.*}} = affine.apply #[[$MAP_mix_dims_and_symbols_d]]()[%{{.*}}, %{{.*}}]
  return %d : index
}

// -----

// CHECK-DAG: #[[$MAP_mix_dims_and_symbols_e:.*]] = affine_map<()[s0, s1] -> ((s0 * 168 + s1 * 4 - 4) floordiv 3)>

// CHECK-LABEL: func @mix_dims_and_symbols_e(%arg0: index, %arg1: index) -> index {
func @mix_dims_and_symbols_e(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] -> (d0 - 1 + 42 * s0)> (%arg0)[%arg1]
  %b = affine.apply affine_map<(d0) -> (d0 + 7)> (%a)
  %c = affine.apply affine_map<(d0) -> (d0 * 4)> (%a)
  %d = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)> ()[%b]
  %e = affine.apply affine_map<(d0) -> (d0 floordiv 3)> (%c)
  // CHECK: {{.*}} = affine.apply #[[$MAP_mix_dims_and_symbols_e]]()[%{{.*}}, %{{.*}}]
  return %e : index
}

// -----

// CHECK-LABEL: func @mix_dims_and_symbols_f(%arg0: index, %arg1: index) -> index {
func @mix_dims_and_symbols_f(%arg0: index, %arg1: index) -> index {
  %a = affine.apply affine_map<(d0)[s0] -> (d0 - 1 + 42 * s0)> (%arg0)[%arg1]
  %b = affine.apply affine_map<(d0) -> (d0 + 7)> (%a)
  %c = affine.apply affine_map<(d0) -> (d0 * 4)> (%a)
  %d = affine.apply affine_map<()[s0] -> (s0 ceildiv 8)> ()[%b]
  %e = affine.apply affine_map<(d0) -> (d0 floordiv 3)> (%c)
  %f = affine.apply affine_map<(d0, d1)[s0, s1] -> (d0 - s1 +  d1 - s0)> (%d, %e)[%e, %d]
  // CHECK: {{.*}} = constant 0 : index

  return %f : index
}

// -----

// CHECK-DAG: #[[$MAP_symbolic_composition_b:.*]] = affine_map<()[s0] -> (s0 * 4)>

// CHECK-LABEL: func @mix_dims_and_symbols_g(%arg0: index, %arg1: index) -> (index, index, index) {
func @mix_dims_and_symbols_g(%M: index, %N: index) -> (index, index, index) {
  %K = affine.apply affine_map<(d0) -> (4*d0)> (%M)
  %res1 = affine.apply affine_map<()[s0, s1] -> (4 * s0)>()[%N, %K]
  %res2 = affine.apply affine_map<()[s0, s1] -> (s1)>()[%N, %K]
  %res3 = affine.apply affine_map<()[s0, s1] -> (1024)>()[%N, %K]
  // CHECK-DAG: {{.*}} = constant 1024 : index
  // CHECK-DAG: {{.*}} = affine.apply #[[$MAP_symbolic_composition_b]]()[%{{.*}}]
  // CHECK-DAG: {{.*}} = affine.apply #[[$MAP_symbolic_composition_b]]()[%{{.*}}]
  return %res1, %res2, %res3 : index, index, index
}

// -----

// CHECK-DAG: #[[$symbolic_semi_affine:.*]] = affine_map<(d0)[s0] -> (d0 floordiv (s0 + 1))>

// CHECK-LABEL: func @symbolic_semi_affine(%arg0: index, %arg1: index, %arg2: memref<?xf32>) {
func @symbolic_semi_affine(%M: index, %N: index, %A: memref<?xf32>) {
  %f1 = constant 1.0 : f32
  affine.for %i0 = 1 to 100 {
    %1 = affine.apply affine_map<()[s0] -> (s0 + 1)> ()[%M]
    %2 = affine.apply affine_map<(d0)[s0] -> (d0 floordiv s0)> (%i0)[%1]
    // CHECK-DAG: {{.*}} = affine.apply #[[$symbolic_semi_affine]](%{{.*}})[%{{.*}}]
    memref.store %f1, %A[%2] : memref<?xf32>
  }
  return
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<()[s0] -> (0, s0)>
// CHECK: #[[$MAP1:.*]] = affine_map<()[s0] -> (100, s0)>

// CHECK-LABEL:  func @constant_fold_bounds(%arg0: index) {
func @constant_fold_bounds(%N : index) {
  // CHECK:      constant 3 : index
  // CHECK-NEXT: "foo"() : () -> index
  %c9 = constant 9 : index
  %c1 = constant 1 : index
  %c2 = constant 2 : index
  %c3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)> (%c1, %c2)
  %l = "foo"() : () -> index

  // CHECK:  affine.for %{{.*}} = 5 to 7 {
  affine.for %i = max affine_map<(d0, d1) -> (0, d0 + d1)> (%c2, %c3) to min affine_map<(d0, d1) -> (d0 - 2, 32*d1)> (%c9, %c1) {
    "foo"(%i, %c3) : (index, index) -> ()
  }

  // Bound takes a non-constant argument but can still be folded.
  // CHECK:  affine.for %{{.*}} = 1 to 7 {
  affine.for %j = max affine_map<(d0) -> (0, 1)> (%N) to min affine_map<(d0, d1) -> (7, 9)> (%N, %l) {
    "foo"(%j, %c3) : (index, index) -> ()
  }

  // None of the bounds can be folded.
  // CHECK: affine.for %{{.*}} = max #[[$MAP0]]()[%{{.*}}] to min #[[$MAP1]]()[%{{.*}}] {
  affine.for %k = max affine_map<()[s0] -> (0, s0)> ()[%l] to min affine_map<()[s0] -> (100, s0)> ()[%N] {
    "foo"(%k, %c3) : (index, index) -> ()
  }
  return
}

// -----

// CHECK-LABEL:  func @fold_empty_loop() {
func @fold_empty_loop() {
  // CHECK-NOT: affine.for
  affine.for %i = 0 to 10 {
  }
  return
}
// CHECK: return


// -----

// CHECK-DAG: #[[$SET:.*]] = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 1022 >= 0, d1 >= 0, -d1 + s0 - 2 >= 0)>

// CHECK-LABEL: func @canonicalize_affine_if
//  CHECK-SAME:   %[[M:[0-9a-zA-Z]*]]: index,
//  CHECK-SAME:   %[[N:[0-9a-zA-Z]*]]: index)
func @canonicalize_affine_if(%M : index, %N : index) {
  %c1022 = constant 1022 : index
  // Drop unused operand %M, propagate %c1022, and promote %N to symbolic.
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to %N {
      // CHECK: affine.if #[[$SET]](%{{.*}}, %{{.*}})[%[[N]]]
      affine.if affine_set<(d0, d1, d2, d3)[s0] : (d1 >= 0, d0 - d1 >= 0, d2 >= 0, d3 - d2 - 2 >= 0)>
          (%c1022, %i, %j, %N)[%M] {
        "foo"() : () -> ()
      }
      "bar"() : () -> ()
    }
  }
  return
}

// -----

// CHECK-DAG: #[[$LBMAP:.*]] = affine_map<()[s0] -> (0, s0)>
// CHECK-DAG: #[[$UBMAP:.*]] = affine_map<()[s0] -> (1024, s0 * 2)>

// CHECK-LABEL: func @canonicalize_bounds
// CHECK-SAME: %[[M:.*]]: index,
// CHECK-SAME: %[[N:.*]]: index)
func @canonicalize_bounds(%M : index, %N : index) {
  %c0 = constant 0 : index
  %c1024 = constant 1024 : index
  // Drop unused operand %N, drop duplicate operand %M, propagate %c1024, and
  // promote %M to a symbolic one.
  // CHECK: affine.for %{{.*}} = 0 to min #[[$UBMAP]]()[%[[M]]]
  affine.for %i = 0 to min affine_map<(d0, d1, d2, d3) -> (d0, d1 + d2)> (%c1024, %M, %M, %N) {
    "foo"() : () -> ()
  }
  // Promote %M to symbolic position.
  // CHECK: affine.for %{{.*}} = 0 to #{{.*}}()[%[[M]]]
  affine.for %i = 0 to affine_map<(d0) -> (4 * d0)> (%M) {
    "foo"() : () -> ()
  }
  // Lower bound canonicalize.
  // CHECK: affine.for %{{.*}} = max #[[$LBMAP]]()[%[[N]]] to %[[M]]
  affine.for %i = max affine_map<(d0, d1) -> (d0, d1)> (%c0, %N) to %M {
    "foo"() : () -> ()
  }
  return
}

// -----

// Compose maps into affine load and store ops.

// CHECK-LABEL: @compose_into_affine_load_store
func @compose_into_affine_load_store(%A : memref<1024xf32>, %u : index) {
  // CHECK: affine.for %[[IV:.*]] = 0 to 1024
  affine.for %i = 0 to 1024 {
    // Make sure the unused operand (%u below) gets dropped as well.
    %idx = affine.apply affine_map<(d0, d1) -> (d0 + 1)> (%i, %u)
    %0 = affine.load %A[%idx] : memref<1024xf32>
    affine.store %0, %A[%idx] : memref<1024xf32>
    // CHECK-NEXT: affine.load %{{.*}}[%[[IV]] + 1]
    // CHECK-NEXT: affine.store %{{.*}}, %{{.*}}[%[[IV]] + 1]

    // Map remains the same, but operand changes on composition.
    %copy = affine.apply affine_map<(d0) -> (d0)> (%i)
    %1 = affine.load %A[%copy] : memref<1024xf32>
    "prevent.dce"(%1) : (f32) -> ()
    // CHECK-NEXT: affine.load %{{.*}}[%[[IV]]]
  }
  return
}

// -----

func @affine_min(%arg0 : index, %arg1 : index, %arg2 : index) {
  %c511 = constant 511 : index
  %c1 = constant 0 : index
  %0 = affine.min affine_map<(d0)[s0] -> (1000, d0 + 512, s0 + 1)> (%c1)[%c511]
  "op0"(%0) : (index) -> ()
  // CHECK:       %[[CST:.*]] = constant 512 : index
  // CHECK-NEXT:  "op0"(%[[CST]]) : (index) -> ()
  // CHECK-NEXT:  return
  return
}

// -----

func @affine_min(%arg0 : index, %arg1 : index, %arg2 : index) {
  %c3 = constant 3 : index
  %c20 = constant 20 : index
  %0 = affine.min affine_map<(d0)[s0] -> (1000, d0 floordiv 4, (s0 mod 5) + 1)> (%c20)[%c3]
  "op0"(%0) : (index) -> ()
  // CHECK:       %[[CST:.*]] = constant 4 : index
  // CHECK-NEXT:  "op0"(%[[CST]]) : (index) -> ()
  // CHECK-NEXT:  return
  return
}

// -----

func @affine_max(%arg0 : index, %arg1 : index, %arg2 : index) {
  %c511 = constant 511 : index
  %c1 = constant 0 : index
  %0 = affine.max affine_map<(d0)[s0] -> (1000, d0 + 512, s0 + 1)> (%c1)[%c511]
  "op0"(%0) : (index) -> ()
  // CHECK:       %[[CST:.*]] = constant 1000 : index
  // CHECK-NEXT:  "op0"(%[[CST]]) : (index) -> ()
  // CHECK-NEXT:  return
  return
}

// -----

func @affine_max(%arg0 : index, %arg1 : index, %arg2 : index) {
  %c3 = constant 3 : index
  %c20 = constant 20 : index
  %0 = affine.max affine_map<(d0)[s0] -> (1000, d0 floordiv 4, (s0 mod 5) + 1)> (%c20)[%c3]
  "op0"(%0) : (index) -> ()
  // CHECK:       %[[CST:.*]] = constant 1000 : index
  // CHECK-NEXT:  "op0"(%[[CST]]) : (index) -> ()
  // CHECK-NEXT:  return
  return
}

// -----

// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (d0, d1 - 2)>

func @affine_min(%arg0: index) {
  affine.for %i = 0 to %arg0 {
    affine.for %j = 0 to %arg0 {
      %c2 = constant 2 : index
      // CHECK: affine.min #[[$MAP]]
      %0 = affine.min affine_map<(d0,d1,d2)->(d0, d1 - d2)>(%i, %j, %c2)
      "consumer"(%0) : (index) -> ()
    }
  }
  return
}

// -----

// Reproducer for PR45031. This used to fold into an incorrect map because
// symbols were concatenated in the wrong order during map folding. Map
// composition places the symbols of the original map before those of the map
// it is composed with, e.g. A.compose(B) will first have all symbols of A,
// then all symbols of B.

#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0)[s0] -> (1024, -d0 + s0)>

// CHECK: #[[$MAP:.*]] = affine_map<()[s0, s1] -> (1024, s0 - s1 * 1024)>

// CHECK: func @rep(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index)
func @rep(%arg0 : index, %arg1 : index) -> index {
  // CHECK-NOT: constant
  %c0 = constant 0 : index
  %c1024 = constant 1024 : index
  // CHECK-NOT: affine.apply
  %0 = affine.apply #map1(%arg0)[%c1024, %c0]

  // CHECK: affine.min #[[$MAP]]()[%[[ARG1]], %[[ARG0]]]
  %1 = affine.min #map2(%0)[%arg1]
  return %1 : index
}

// -----

// CHECK-DAG: #[[ub:.*]] = affine_map<()[s0] -> (s0 + 2)>

func @drop_duplicate_bounds(%N : index) {
  // affine.for %i = max #lb(%arg0) to min #ub(%arg0)
  affine.for %i = max affine_map<(d0) -> (d0, d0)>(%N) to min affine_map<(d0) -> (d0 + 2, d0 + 2)>(%N) {
    "foo"() : () -> ()
  }
  return
}

// -----

// Ensure affine.parallel bounds expressions are canonicalized.

#map3 = affine_map<(d0) -> (d0 * 5)>

// CHECK-LABEL: func @affine_parallel_const_bounds
func @affine_parallel_const_bounds() {
  %cst = constant 1.0 : f32
  %c0 = constant 0 : index
  %c4 = constant 4 : index
  %0 = memref.alloc() : memref<4xf32>
  // CHECK: affine.parallel (%{{.*}}) = (0) to (4)
  affine.parallel (%i) = (%c0) to (%c0 + %c4) {
    %1 = affine.apply #map3(%i)
    // CHECK: affine.parallel (%{{.*}}) = (0) to (%{{.*}} * 5)
    affine.parallel (%j) = (%c0) to (%1) {
      affine.store %cst, %0[%j] : memref<4xf32>
    }
  }
  return
}

// -----

func @compose_affine_maps_div_symbol(%A : memref<i64>, %i0 : index, %i1 : index) {
  %0 = affine.apply affine_map<()[s0] -> (2 * s0)> ()[%i0]
  %1 = affine.apply affine_map<()[s0] -> (3 * s0)> ()[%i0]
  %2 = affine.apply affine_map<(d0)[s0, s1] -> (d0 mod s1 + s0 * s1 + s0 * 4)> (%i1)[%0, %1]
  %3 = index_cast %2: index to i64
  memref.store %3, %A[]: memref<i64>
  affine.for %i2 = 0 to 3 {
    %4 = affine.apply affine_map<(d0)[s0, s1] -> (d0 ceildiv s1 + s0 + s0 * 3)> (%i2)[%0, %1]
    %5 = index_cast %4: index to i64
    memref.store %5, %A[]: memref<i64>
  }
  return
}

// -----

// CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1, s0 * s1)>

// CHECK: func @deduplicate_affine_min_expressions
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func @deduplicate_affine_min_expressions(%i0: index, %i1: index) -> index {
  // CHECK:  affine.min #[[MAP]]()[%[[I0]], %[[I1]]]
  %0 = affine.min affine_map<()[s0, s1] -> (s0 + s1, s0 * s1, s1 + s0, s0 * s1)> ()[%i0, %i1]
  return %0: index
}

// -----

// CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1, s0 * s1)>

// CHECK: func @deduplicate_affine_max_expressions
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func @deduplicate_affine_max_expressions(%i0: index, %i1: index) -> index {
  // CHECK:  affine.max #[[MAP]]()[%[[I0]], %[[I1]]]
  %0 = affine.max affine_map<()[s0, s1] -> (s0 + s1, s0 * s1, s1 + s0, s0 * s1)> ()[%i0, %i1]
  return %0: index
}

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1, s2] -> (s0 * 3, 16, -s1 + s2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> (-s1 + 5, 16, -s0 + s2)>

// CHECK: func @merge_affine_min_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index)
func @merge_affine_min_ops(%i0: index, %i1: index, %i2: index, %i3: index) -> (index, index) {
  %0 = affine.min affine_map<(d0)[s0] -> (16, d0 - s0)> (%i0)[%i1]

 // CHECK: affine.min #[[MAP0]]()[%[[I2]], %[[I1]], %[[I0]]]
  %1 = affine.min affine_map<(d0)[s0] -> (3 * s0, d0)> (%0)[%i2] // Use as dim
 // CHECK: affine.min #[[MAP1]]()[%[[I1]], %[[I3]], %[[I0]]]
  %2 = affine.min affine_map<(d0)[s0] -> (s0, 5 - d0)> (%i3)[%0] // Use as symbol

  return %1, %2: index, index
}

// -----

// CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 + 7, s1 + 16, s1 * 8, s2 + 8, s2 * 4)>

// CHECK: func @merge_multiple_affine_min_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
func @merge_multiple_affine_min_ops(%i0: index, %i1: index, %i2: index) -> index {
  %0 = affine.min affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  %1 = affine.min affine_map<()[s0] -> (s0 + 8, s0 * 4)> ()[%i1]
  // CHECK: affine.min #[[MAP]]()[%[[I2]], %[[I0]], %[[I1]]]
  %2 = affine.min affine_map<()[s0, s1, s2] -> (s0, 7 + s1, s2)> ()[%0, %i2, %1]
  return %2: index
}

// -----

// CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 2, s1 + 16, s1 * 8)>

// CHECK: func @merge_multiple_uses_of_affine_min_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func @merge_multiple_uses_of_affine_min_ops(%i0: index, %i1: index) -> index {
  %0 = affine.min affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  // CHECK: affine.min #[[MAP]]()[%[[I1]], %[[I0]]]
  %2 = affine.min affine_map<()[s0, s1, s2] -> (s0, s1, s2 * 2)> ()[%0, %0, %i1]
  return %2: index
}

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 + 16, s0 * 8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> (s0 + 1, s1 * 2, s2 + 16, s2 * 8)>

// CHECK: func @merge_mixed_uses_of_affine_min_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func @merge_mixed_uses_of_affine_min_ops(%i0: index, %i1: index) -> index {
  // CHECK: %[[AFFINE:.+]] = affine.min #[[MAP0]]()[%[[I0]]]
  %0 = affine.min affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  // %0 is bound to a symbol that is both a standalone expression and a part
  // of other expressions.
  // CHECK: affine.min #[[MAP1]]()[%[[AFFINE]], %[[I1]], %[[I0]]]
  %2 = affine.min affine_map<()[s0, s1, s2] -> (s0, s1 + 1, s2 * 2)> ()[%0, %0, %i1]
  return %2: index
}

// -----

// CHECK-LABEL: func @dont_merge_affine_min_if_not_single_dim
func @dont_merge_affine_min_if_not_single_dim(%i0: index, %i1: index, %i2: index) -> index {
  // CHECK-COUNT-2: affine.min
  %0 = affine.min affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  %1 = affine.min affine_map<(d0)[s0] -> (s0 + 4, 7 + d0)> (%0)[%i2]
  return %1: index
}


// -----

// CHECK-LABEL: func @dont_merge_affine_min_if_not_single_sym
func @dont_merge_affine_min_if_not_single_sym(%i0: index, %i1: index, %i2: index) -> index {
  // CHECK-COUNT-2: affine.min
  %0 = affine.min affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  %1 = affine.min affine_map<()[s0, s1] -> (s0 + 4, 7 + s1)> ()[%0, %i2]
  return %1: index
}

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0, s1, s2] -> (s0 * 3, 16, -s1 + s2)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> (-s1 + 5, 16, -s0 + s2)>

// CHECK: func @merge_affine_max_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index, %[[I3:.+]]: index)
func @merge_affine_max_ops(%i0: index, %i1: index, %i2: index, %i3: index) -> (index, index) {
  %0 = affine.max affine_map<(d0)[s0] -> (16, d0 - s0)> (%i0)[%i1]

 // CHECK: affine.max #[[MAP0]]()[%[[I2]], %[[I1]], %[[I0]]]
  %1 = affine.max affine_map<(d0)[s0] -> (3 * s0, d0)> (%0)[%i2] // Use as dim
 // CHECK: affine.max #[[MAP1]]()[%[[I1]], %[[I3]], %[[I0]]]
  %2 = affine.max affine_map<(d0)[s0] -> (s0, 5 - d0)> (%i3)[%0] // Use as symbol

  return %1, %2: index, index
}

// -----

// CHECK: #[[MAP:.+]] = affine_map<()[s0, s1, s2] -> (s0 + 7, s1 + 16, s1 * 8, s2 + 8, s2 * 4)>

// CHECK: func @merge_multiple_affine_max_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index, %[[I2:.+]]: index)
func @merge_multiple_affine_max_ops(%i0: index, %i1: index, %i2: index) -> index {
  %0 = affine.max affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  %1 = affine.max affine_map<()[s0] -> (s0 + 8, s0 * 4)> ()[%i1]
  // CHECK: affine.max #[[MAP]]()[%[[I2]], %[[I0]], %[[I1]]]
  %2 = affine.max affine_map<()[s0, s1, s2] -> (s0, 7 + s1, s2)> ()[%0, %i2, %1]
  return %2: index
}

// -----

// CHECK-DAG: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 * 2, s1 + 16, s1 * 8)>

// CHECK: func @merge_multiple_uses_of_affine_max_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func @merge_multiple_uses_of_affine_max_ops(%i0: index, %i1: index) -> index {
  %0 = affine.max affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  // CHECK: affine.max #[[MAP]]()[%[[I1]], %[[I0]]]
  %2 = affine.max affine_map<()[s0, s1, s2] -> (s0, s1, s2 * 2)> ()[%0, %0, %i1]
  return %2: index
}

// -----

// CHECK-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 + 16, s0 * 8)>
// CHECK-DAG: #[[MAP1:.+]] = affine_map<()[s0, s1, s2] -> (s0 + 1, s1 * 2, s2 + 16, s2 * 8)>

// CHECK: func @merge_mixed_uses_of_affine_max_ops
// CHECK-SAME: (%[[I0:.+]]: index, %[[I1:.+]]: index)
func @merge_mixed_uses_of_affine_max_ops(%i0: index, %i1: index) -> index {
  // CHECK: %[[AFFINE:.+]] = affine.max #[[MAP0]]()[%[[I0]]]
  %0 = affine.max affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  // %0 is bound to a symbol that is both a standalone expression and a part
  // of other expressions.
  // CHECK: affine.max #[[MAP1]]()[%[[AFFINE]], %[[I1]], %[[I0]]]
  %2 = affine.max affine_map<()[s0, s1, s2] -> (s0, s1 + 1, s2 * 2)> ()[%0, %0, %i1]
  return %2: index
}

// -----

// CHECK-LABEL: func @dont_merge_affine_max_if_not_single_dim
func @dont_merge_affine_max_if_not_single_dim(%i0: index, %i1: index, %i2: index) -> index {
  // CHECK-COUNT-2: affine.max
  %0 = affine.max affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  %1 = affine.max affine_map<(d0)[s0] -> (s0 + 4, 7 + d0)> (%0)[%i2]
  return %1: index
}


// -----

// CHECK-LABEL: func @dont_merge_affine_max_if_not_single_sym
func @dont_merge_affine_max_if_not_single_sym(%i0: index, %i1: index, %i2: index) -> index {
  // CHECK-COUNT-2: affine.max
  %0 = affine.max affine_map<()[s0] -> (s0 + 16, s0 * 8)> ()[%i0]
  %1 = affine.max affine_map<()[s0, s1] -> (s0 + 4, 7 + s1)> ()[%0, %i2]
  return %1: index
}
