//===- builder-api-test.cpp - Tests for Declarative Builder APIs ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: mlir-edsc-builder-api-test | FileCheck %s
// XFAIL: *

#include "mlir/Dialect/Affine/EDSC/Intrinsics.h"
#include "mlir/Dialect/Linalg/EDSC/Builders.h"
#include "mlir/Dialect/Linalg/EDSC/Intrinsics.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/EDSC/Intrinsics.h"
#include "mlir/Dialect/SCF/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/Vector/EDSC/Intrinsics.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

#include "APITest.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::edsc;
using namespace mlir::edsc::intrinsics;

static MLIRContext &globalContext() {
  static thread_local MLIRContext context;
  static thread_local bool initOnce = [&]() {
    // clang-format off
    context.loadDialect<AffineDialect,
                        scf::SCFDialect,
                        linalg::LinalgDialect,
                        math::MathDialect,
                        memref::MemRefDialect,
                        StandardOpsDialect,
                        vector::VectorDialect>();
    // clang-format on
    return true;
  }();
  (void)initOnce;
  context.allowUnregisteredDialects();
  return context;
}

static FuncOp makeFunction(StringRef name, ArrayRef<Type> results = {},
                           ArrayRef<Type> args = {}) {
  auto &ctx = globalContext();
  auto function = FuncOp::create(UnknownLoc::get(&ctx), name,
                                 FunctionType::get(&ctx, args, results));
  function.addEntryBlock();
  return function;
}

TEST_FUNC(builder_dynamic_for_func_args) {
  auto indexType = IndexType::get(&globalContext());
  auto f32Type = FloatType::getF32(&globalContext());
  auto f =
      makeFunction("builder_dynamic_for_func_args", {}, {indexType, indexType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value lb(f.getArgument(0)), ub(f.getArgument(1));
  Value f7(std_constant_float(llvm::APFloat(7.0f), f32Type));
  Value f13(std_constant_float(llvm::APFloat(13.0f), f32Type));
  Value i7(std_constant_int(7, 32));
  Value i13(std_constant_int(13, 32));
  affineLoopBuilder(lb, ub, 3, [&](Value i) {
    using namespace edsc::op;
    lb *std_constant_index(3) + ub;
    lb + std_constant_index(3);
    affineLoopBuilder(lb, ub, 2, [&](Value j) {
      ceilDiv(std_constant_index(31) * floorDiv(i + j * std_constant_index(3),
                                                std_constant_index(32)),
              std_constant_index(32));
      ((f7 + f13) / f7) % f13 - f7 *f13;
      ((i7 + i13) / i7) % i13 - i7 *i13;
    });
  });

  // clang-format off
  // CHECK-LABEL: func @builder_dynamic_for_func_args(%{{.*}}: index, %{{.*}}: index) {
  //     CHECK:  affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%{{.*}}) to affine_map<(d0) -> (d0)>(%{{.*}}) step 3 {
  //     CHECK:  {{.*}} = affine.apply affine_map<()[s0] -> (s0 * 3)>()[%{{.*}}]
  //     CHECK:  {{.*}} = affine.apply affine_map<()[s0, s1] -> (s0 * 3 + s1)>()[%{{.*}}, %{{.*}}]
  //     CHECK:  {{.*}} = affine.apply affine_map<()[s0] -> (s0 + 3)>()[%{{.*}}]
  //     CHECK:  affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%{{.*}}) to affine_map<(d0) -> (d0)>(%{{.*}}) step 2 {
  //     CHECK:    {{.*}} = affine.apply affine_map<(d0, d1) -> ((d0 + d1 * 3) floordiv 32)>(%{{.*}}, %{{.*}})
  //     CHECK:    {{.*}} = affine.apply affine_map<(d0, d1) -> (((d0 + d1 * 3) floordiv 32) * 31)>(%{{.*}}, %{{.*}})
  //     CHECK:    {{.*}} = affine.apply affine_map<(d0, d1) -> ((((d0 + d1 * 3) floordiv 32) * 31) ceildiv 32)>(%{{.*}}, %{{.*}})
  // CHECK-DAG:    [[rf1:%[0-9]+]] = addf {{.*}}, {{.*}} : f32
  // CHECK-DAG:    [[rf2:%[0-9]+]] = divf [[rf1]], {{.*}} : f32
  // CHECK-DAG:    [[rf3:%[0-9]+]] = remf [[rf2]], {{.*}} : f32
  // CHECK-DAG:    [[rf4:%[0-9]+]] = mulf {{.*}}, {{.*}} : f32
  //     CHECK:    {{.*}} = subf [[rf3]], [[rf4]] : f32
  // CHECK-DAG:    [[ri1:%[0-9]+]] = addi {{.*}}, {{.*}} : i32
  // CHECK-DAG:    [[ri2:%[0-9]+]] = divi_signed [[ri1]], {{.*}} : i32
  // CHECK-DAG:    [[ri3:%[0-9]+]] = remi_signed [[ri2]], {{.*}} : i32
  // CHECK-DAG:    [[ri4:%[0-9]+]] = muli {{.*}}, {{.*}} : i32
  //     CHECK:    {{.*}} = subi [[ri3]], [[ri4]] : i32
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(builder_dynamic_for) {
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("builder_dynamic_for", {},
                        {indexType, indexType, indexType, indexType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value i, a(f.getArgument(0)), b(f.getArgument(1)), c(f.getArgument(2)),
      d(f.getArgument(3));
  using namespace edsc::op;
  affineLoopBuilder(a - b, c + d, 2);

  // clang-format off
  // CHECK-LABEL: func @builder_dynamic_for(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
  // CHECK-DAG:    [[r0:%[0-9]+]] = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%{{.*}}, %{{.*}}]
  // CHECK-DAG:    [[r1:%[0-9]+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%{{.*}}, %{{.*}}]
  // CHECK-NEXT:   affine.for %{{.*}} = affine_map<(d0) -> (d0)>([[r0]]) to affine_map<(d0) -> (d0)>([[r1]]) step 2 {
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(builder_loop_for) {
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("builder_loop_for", {},
                        {indexType, indexType, indexType, indexType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value a(f.getArgument(0)), b(f.getArgument(1)), c(f.getArgument(2)),
      d(f.getArgument(3));
  using namespace edsc::op;
  loopNestBuilder(a - b, c + d, a);

  // clang-format off
  // CHECK-LABEL: func @builder_loop_for(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
  // CHECK-DAG:    [[r0:%[0-9]+]] = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%{{.*}}, %{{.*}}]
  // CHECK-DAG:    [[r1:%[0-9]+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%{{.*}}, %{{.*}}]
  // CHECK-NEXT:   scf.for %{{.*}} = [[r0]] to [[r1]] step {{.*}} {
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(builder_max_min_for) {
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("builder_max_min_for", {},
                        {indexType, indexType, indexType, indexType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value lb1(f.getArgument(0)), lb2(f.getArgument(1)), ub1(f.getArgument(2)),
      ub2(f.getArgument(3));
  affineLoopBuilder({lb1, lb2}, {ub1, ub2}, 1);
  std_ret();

  // clang-format off
  // CHECK-LABEL: func @builder_max_min_for(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
  // CHECK:  affine.for %{{.*}} = max affine_map<(d0, d1) -> (d0, d1)>(%{{.*}}, %{{.*}}) to min affine_map<(d0, d1) -> (d0, d1)>(%{{.*}}, %{{.*}}) {
  // CHECK:  return
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(builder_affine_for_iter_args) {
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("builder_affine_for_iter_args", {},
                        {indexType, indexType, indexType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value i, lb_1(f.getArgument(0)), ub_1(f.getArgument(1)),
      ub_2(f.getArgument(2));
  Value c32(std_constant_int(32, 32));
  Value c42(std_constant_int(42, 32));
  using namespace edsc::op;
  affineLoopBuilder(
      lb_1, {ub_1, ub_2}, 2, {c32, c42}, [&](Value iv, ValueRange args) {
        Value sum(args[0] + args[1]);
        builder.create<AffineYieldOp>(f.getLoc(), ValueRange({args[1], sum}));
      });

  // clang-format off
  // CHECK-LABEL: func @builder_affine_for_iter_args
  // CHECK:       (%[[lb_1:.*]]: index, %[[ub_1:.*]]: index, %[[ub_2:.*]]: index) {
  // CHECK-NEXT:    %[[c32:.*]] = constant 32 : i32
  // CHECK-NEXT:    %[[c42:.*]] = constant 42 : i32
  // CHECK-NEXT:    %{{.*}} = affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%{{.*}}) to min affine_map<(d0, d1) -> (d0, d1)>(%[[ub_1]], %[[ub_2]]) step 2 iter_args(%[[iarg_1:.*]] = %[[c32]], %[[iarg_2:.*]] = %[[c42]]) -> (i32, i32) {
  // CHECK-NEXT:      %[[sum:.*]] = addi %[[iarg_1]], %[[iarg_2]] : i32
  // CHECK-NEXT:      affine.yield %[[iarg_2]], %[[sum]] : i32, i32
  // CHECK-NEXT:    }
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(builder_block_append) {
  using namespace edsc::op;
  auto f = makeFunction("builder_blocks");

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  Block *b =
      buildInNewBlock(TypeRange(), [&](ValueRange) { std_constant_index(0); });
  appendToBlock(b, [&](ValueRange) { std_constant_index(1); });
  appendToBlock(b, [&](ValueRange) { std_ret(); });
  // Get back to entry block and add a branch into "b".
  appendToBlock(&f.front(), [&](ValueRange) { std_br(b, {}); });

  // clang-format off
  // CHECK-LABEL: @builder_blocks
  // CHECK-NEXT:   br ^bb1
  // CHECK-NEXT: ^bb1: // pred: ^bb0
  // CHECK-NEXT:   constant 0 : index
  // CHECK-NEXT:   constant 1 : index
  // CHECK-NEXT:   return
  // CHECK-NEXT: }
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(builder_blocks) {
  using namespace edsc::op;
  auto f = makeFunction("builder_blocks");

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value c1(std_constant_int(42, 32)), c2(std_constant_int(1234, 32));
  ReturnOp ret = std_ret();

  Block *b1 = createBlock({c1.getType(), c1.getType()});
  Block *b2 = buildInNewBlock({c1.getType(), c1.getType()},
                              [&](ValueRange args) { std_br(b1, args); });
  // The insertion point within the toplevel function is now past b2, we will
  // need to get back the entry block.
  // This is what happens with unstructured control-flow.
  appendToBlock(b1, [&](ValueRange args) {
    Value r = args[0] + args[1];
    std_br(b2, {args[0], r});
  });
  // Get back to entry block and add a branch into b1.
  appendToBlock(&f.front(), [&](ValueRange) { std_br(b1, {c1, c2}); });
  ret.erase();

  // clang-format off
  // CHECK-LABEL: @builder_blocks
  // CHECK:        %{{.*}} = constant 42 : i32
  // CHECK-NEXT:   %{{.*}} = constant 1234 : i32
  // CHECK-NEXT:   br ^bb1(%{{.*}}, %{{.*}} : i32, i32)
  // CHECK-NEXT: ^bb1(%{{.*}}: i32, %{{.*}}: i32):   // 2 preds: ^bb0, ^bb2
  // CHECK-NEXT:   %{{.*}} = addi %{{.*}}, %{{.*}} : i32
  // CHECK-NEXT:   br ^bb2(%{{.*}}, %{{.*}} : i32, i32)
  // CHECK-NEXT: ^bb2(%{{.*}}: i32, %{{.*}}: i32):   // pred: ^bb1
  // CHECK-NEXT:   br ^bb1(%{{.*}}, %{{.*}} : i32, i32)
  // CHECK-NEXT: }
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(builder_cond_branch) {
  auto f = makeFunction("builder_cond_branch", {},
                        {IntegerType::get(&globalContext(), 1)});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value c32(std_constant_int(32, 32)), c64(std_constant_int(64, 64)),
      c42(std_constant_int(42, 32));
  ReturnOp ret = std_ret();

  Block *b1 = buildInNewBlock(c32.getType(), [&](ValueRange) { std_ret(); });
  Block *b2 = buildInNewBlock({c64.getType(), c32.getType()},
                              [&](ValueRange) { std_ret(); });
  // Get back to entry block and add a conditional branch.
  appendToBlock(&f.front(), [&](ValueRange args) {
    std_cond_br(args[0], b1, {c32}, b2, {c64, c42});
  });
  ret.erase();

  // clang-format off
  // CHECK-LABEL: @builder_cond_branch
  // CHECK:   %{{.*}} = constant 32 : i32
  // CHECK-NEXT:   %{{.*}} = constant 64 : i64
  // CHECK-NEXT:   %{{.*}} = constant 42 : i32
  // CHECK-NEXT:   cond_br %{{.*}}, ^bb1(%{{.*}} : i32), ^bb2(%{{.*}}, %{{.*}} : i64, i32)
  // CHECK-NEXT: ^bb1(%{{.*}}: i32):   // pred: ^bb0
  // CHECK-NEXT:   return
  // CHECK-NEXT: ^bb2(%{{.*}}: i64, %{{.*}}: i32):  // pred: ^bb0
  // CHECK-NEXT:   return
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(builder_helpers) {
  using namespace edsc::op;
  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType =
      MemRefType::get({ShapedType::kDynamicSize, ShapedType::kDynamicSize,
                       ShapedType::kDynamicSize},
                      f32Type, {}, 0);
  auto f =
      makeFunction("builder_helpers", {}, {memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  // clang-format off
  Value f7 = std_constant_float(llvm::APFloat(7.0f), f32Type);
  MemRefBoundsCapture vA(f.getArgument(0)), vB(f.getArgument(1)),
      vC(f.getArgument(2));
  AffineIndexedValue A(f.getArgument(0)), B(f.getArgument(1)), C(f.getArgument(2));
  Value lb0, lb1, lb2, ub0, ub1, ub2;
  int64_t step0, step1, step2;
  std::tie(lb0, ub0, step0) = vA.range(0);
  std::tie(lb1, ub1, step1) = vA.range(1);
  lb2 = vA.lb(2);
  ub2 = vA.ub(2);
  step2 = vA.step(2);
  affineLoopNestBuilder({lb0, lb1}, {ub0, ub1}, {step0, step1}, [&](ValueRange ivs) {
    Value i = ivs[0];
    Value j = ivs[1];
    affineLoopBuilder(lb2, ub2, step2, [&](Value k1){
      C(i, j, k1) = f7 + A(i, j, k1) + B(i, j, k1);
    });
    affineLoopBuilder(lb2, ub2, step2, [&](Value k2){
      C(i, j, k2) += A(i, j, k2) + B(i, j, k2);
    });
  });

  // clang-format off
  // CHECK-LABEL: @builder_helpers
  //      CHECK:   affine.for %{{.*}} = affine_map<(d0) -> (d0)>({{.*}}) to affine_map<(d0) -> (d0)>({{.*}}) {
  // CHECK-NEXT:     affine.for %{{.*}} = affine_map<(d0) -> (d0)>({{.*}}) to affine_map<(d0) -> (d0)>({{.*}}) {
  // CHECK-NEXT:       affine.for %{{.*}} = affine_map<(d0) -> (d0)>({{.*}}) to affine_map<(d0) -> (d0)>({{.*}}) {
  //  CHECK-DAG:         [[a:%.*]] = affine.load %arg0[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-DAG:         [[b:%.*]] = addf {{.*}}, [[a]] : f32
  //  CHECK-DAG:         [[c:%.*]] = affine.load %arg1[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-DAG:         [[d:%.*]] = addf [[b]], [[c]] : f32
  // CHECK-NEXT:         affine.store [[d]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  // CHECK-NEXT:       }
  // CHECK-NEXT:       affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%{{.*}}) to affine_map<(d0) -> (d0)>(%{{.*}}) {
  //  CHECK-DAG:         [[a:%.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-DAG:         [[b:%.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-DAG:         [[c:%.*]] = addf [[b]], [[a]] : f32
  //  CHECK-DAG:         [[d:%.*]] = affine.load %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-DAG:         [[e:%.*]] = addf [[d]], [[c]] : f32
  // CHECK-NEXT:         affine.store [[e]], %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(insertion_in_block) {
  using namespace edsc::op;
  auto indexType = IndexType::get(&globalContext());
  auto f = makeFunction("insertion_in_block", {}, {indexType, indexType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  std_constant_int(0, 32);
  buildInNewBlock({}, [&](ValueRange) { std_constant_int(1, 32); });
  std_constant_int(2, 32);
  // clang-format off
  // CHECK-LABEL: @insertion_in_block
  // CHECK: {{.*}} = constant 0 : i32
  // CHECK: {{.*}} = constant 2 : i32
  // CHECK: ^bb1:   // no predecessors
  // CHECK: {{.*}} = constant 1 : i32
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(zero_and_std_sign_extendi_op_i1_to_i8) {
  using namespace edsc::op;
  auto i1Type = IntegerType::get(&globalContext(), 1);
  auto i8Type = IntegerType::get(&globalContext(), 8);
  auto memrefType = MemRefType::get({}, i1Type, {}, 0);
  auto f = makeFunction("zero_and_std_sign_extendi_op", {},
                        {memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  AffineIndexedValue A(f.getArgument(0));
  AffineIndexedValue B(f.getArgument(1));
  // clang-format off
  edsc::intrinsics::std_zero_extendi(A, i8Type);
  edsc::intrinsics::std_sign_extendi(B, i8Type);
  // CHECK-LABEL: @zero_and_std_sign_extendi_op
  //      CHECK:     %[[SRC1:.*]] = affine.load
  //      CHECK:     zexti %[[SRC1]] : i1 to i8
  //      CHECK:     %[[SRC2:.*]] = affine.load
  //      CHECK:     sexti %[[SRC2]] : i1 to i8
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(operator_or) {
  auto i1Type = IntegerType::get(&globalContext(), /*width=*/1);
  auto f = makeFunction("operator_or", {}, {i1Type, i1Type});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  using op::operator||;
  Value lhs(f.getArgument(0));
  Value rhs(f.getArgument(1));
  lhs || rhs;

  // clang-format off
  // CHECK-LABEL: @operator_or
  //       CHECK: [[ARG0:%.*]]: i1, [[ARG1:%.*]]: i1
  //       CHECK: or [[ARG0]], [[ARG1]]
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(operator_and) {
  auto i1Type = IntegerType::get(&globalContext(), /*width=*/1);
  auto f = makeFunction("operator_and", {}, {i1Type, i1Type});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  using op::operator&&;
  using op::negate;
  Value lhs(f.getArgument(0));
  Value rhs(f.getArgument(1));
  negate(lhs && rhs);

  // clang-format off
  // CHECK-LABEL: @operator_and
  //       CHECK: [[ARG0:%.*]]: i1, [[ARG1:%.*]]: i1
  //       CHECK: [[AND:%.*]] = and [[ARG0]], [[ARG1]]
  //       CHECK: [[TRUE:%.*]] = constant true
  //       CHECK: subi [[TRUE]], [[AND]] : i1
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(divis_op_i32) {
  using namespace edsc::op;
  auto f = makeFunction("divis_op", {}, {});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  auto i32Type = builder.getI32Type();
  std_divis(std_constant_int(10, i32Type), std_constant_int(2, i32Type));

  // clang-format off
  // CHECK-LABEL: @divis_op
  //   CHECK-DAG:     {{.*}} = constant 10
  //   CHECK-DAG:     {{.*}} = constant 2
  //  CHECK-NEXT:     {{.*}} = divi_signed
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(diviu_op_i32) {
  using namespace edsc::op;
  auto f = makeFunction("diviu_op", {}, {});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  auto i32Type = builder.getI32Type();
  std_diviu(std_constant_int(10, i32Type), std_constant_int(2, i32Type));

  // clang-format off
  // CHECK-LABEL: @diviu_op
  //   CHECK-DAG:     {{.*}} = constant 10
  //   CHECK-DAG:     {{.*}} = constant 2
  //  CHECK-NEXT:     {{.*}} = divi_unsigned
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(fpext_f32_f64) {
  using namespace edsc::op;
  auto f = makeFunction("fpext", {}, {});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  auto f32Type = builder.getF32Type();
  auto f64Type = builder.getF64Type();
  std_fpext(std_constant_float(llvm::APFloat(10.0f), f32Type), f64Type);

  // clang-format off
  // CHECK-LABEL: @fpext
  //       CHECK: {{.*}} = constant 1.0
  //  CHECK-NEXT: {{.*}} = fpext
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(fptrunc_f32_bf16) {
  using namespace edsc::op;
  auto f = makeFunction("fptrunc", {}, {});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  auto f32Type = builder.getF32Type();
  auto bf16Type = builder.getBF16Type();
  std_fptrunc(std_constant_float(llvm::APFloat(10.0f), f32Type), bf16Type);

  // clang-format off
  // CHECK-LABEL: @fptrunc
  //       CHECK: {{.*}} = constant 1.0
  //  CHECK-NEXT: {{.*}} = fptrunc
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(select_op_i32) {
  using namespace edsc::op;
  auto i32Type = IntegerType::get(&globalContext(), 32);
  auto memrefType = MemRefType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, i32Type, {}, 0);
  auto f = makeFunction("select_op", {}, {memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value zero = std_constant_index(0), one = std_constant_index(1);
  MemRefBoundsCapture vA(f.getArgument(0));
  AffineIndexedValue A(f.getArgument(0));
  affineLoopNestBuilder({zero, zero}, {one, one}, {1, 1}, [&](ValueRange ivs) {
    using namespace edsc::op;
    Value i = ivs[0], j = ivs[1];
    std_select(eq(i, zero), A(zero, zero), A(i, j));
    std_select(ne(i, zero), A(zero, zero), A(i, j));
    std_select(slt(i, zero), A(zero, zero), A(i, j));
    std_select(sle(i, zero), A(zero, zero), A(i, j));
    std_select(sgt(i, zero), A(zero, zero), A(i, j));
    std_select(sge(i, zero), A(zero, zero), A(i, j));
    std_select(ult(i, zero), A(zero, zero), A(i, j));
    std_select(ule(i, zero), A(zero, zero), A(i, j));
    std_select(ugt(i, zero), A(zero, zero), A(i, j));
    std_select(uge(i, zero), A(zero, zero), A(i, j));
  });

  // clang-format off
  // CHECK-LABEL: @select_op
  //      CHECK: affine.for %{{.*}} = 0 to 1 {
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 1 {
  //  CHECK-DAG:     {{.*}} = cmpi eq
  //  CHECK-DAG:     {{.*}} = affine.load
  //  CHECK-DAG:     {{.*}} = affine.load
  // CHECK-NEXT:     {{.*}} = select
  //  CHECK-DAG:     {{.*}} = cmpi ne
  //  CHECK-DAG:     {{.*}} = affine.load
  //  CHECK-DAG:     {{.*}} = affine.load
  // CHECK-NEXT:     {{.*}} = select
  //  CHECK-DAG:     {{.*}} = cmpi slt
  //  CHECK-DAG:     {{.*}} = affine.load
  //  CHECK-DAG:     {{.*}} = affine.load
  // CHECK-NEXT:     {{.*}} = select
  //  CHECK-DAG:     {{.*}} = cmpi sle
  //  CHECK-DAG:     {{.*}} = affine.load
  //  CHECK-DAG:     {{.*}} = affine.load
  // CHECK-NEXT:     {{.*}} = select
  //  CHECK-DAG:     {{.*}} = cmpi sgt
  //  CHECK-DAG:     {{.*}} = affine.load
  //  CHECK-DAG:     {{.*}} = affine.load
  // CHECK-NEXT:     {{.*}} = select
  //  CHECK-DAG:     {{.*}} = cmpi sge
  //  CHECK-DAG:     {{.*}} = affine.load
  //  CHECK-DAG:     {{.*}} = affine.load
  // CHECK-NEXT:     {{.*}} = select
  //  CHECK-DAG:     {{.*}} = cmpi ult
  //  CHECK-DAG:     {{.*}} = affine.load
  //  CHECK-DAG:     {{.*}} = affine.load
  // CHECK-NEXT:     {{.*}} = select
  //  CHECK-DAG:     {{.*}} = cmpi ule
  //  CHECK-DAG:     {{.*}} = affine.load
  //  CHECK-DAG:     {{.*}} = affine.load
  // CHECK-NEXT:     {{.*}} = select
  //  CHECK-DAG:     {{.*}} = cmpi ugt
  //  CHECK-DAG:     {{.*}} = affine.load
  //  CHECK-DAG:     {{.*}} = affine.load
  // CHECK-NEXT:     {{.*}} = select
  //  CHECK-DAG:     {{.*}} = cmpi uge
  //  CHECK-DAG:     {{.*}} = affine.load
  //  CHECK-DAG:     {{.*}} = affine.load
  // CHECK-NEXT:     {{.*}} = select
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(select_op_f32) {
  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, f32Type, {}, 0);
  auto f = makeFunction("select_op", {}, {memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  // clang-format off
  Value zero = std_constant_index(0), one = std_constant_index(1);
  MemRefBoundsCapture vA(f.getArgument(0)), vB(f.getArgument(1));
  AffineIndexedValue A(f.getArgument(0)), B(f.getArgument(1));
  affineLoopNestBuilder({zero, zero}, {one, one}, {1, 1}, [&](ValueRange ivs) {
    using namespace edsc::op;
    Value i = ivs[0], j = ivs[1];
    std_select(eq(B(i, j), B(i + one, j)), A(zero, zero), A(i, j));
    std_select(ne(B(i, j), B(i + one, j)), A(zero, zero), A(i, j));
    std_select(sge(B(i, j), B(i + one, j)), A(zero, zero), A(i, j));
    std_select(sle(B(i, j), B(i + one, j)), A(zero, zero), A(i, j));
    std_select(slt(B(i, j), B(i + one, j)), A(zero, zero), A(i, j));
    std_select(sgt(B(i, j), B(i + one, j)), A(zero, zero), A(i, j));
    std_select(uge(B(i, j), B(i + one, j)), A(zero, zero), A(i, j));
    std_select(ule(B(i, j), B(i + one, j)), A(zero, zero), A(i, j));
    std_select(ult(B(i, j), B(i + one, j)), A(zero, zero), A(i, j));
    std_select(ugt(B(i, j), B(i + one, j)), A(zero, zero), A(i, j));
  });

  // clang-format off
  // CHECK-LABEL: @select_op
  //      CHECK: affine.for %{{.*}} = 0 to 1 {
  // CHECK-NEXT:   affine.for %{{.*}} = 0 to 1 {
  //  CHECK-DAG:     cmpf oeq
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf one
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf oge
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf ole
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf olt
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf ogt
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf oge
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf ole
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf olt
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  //  CHECK-DAG:     cmpf ogt
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.load
  //  CHECK-DAG:     affine.apply
  // CHECK-NEXT:     select
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

// Inject an EDSC-constructed computation to exercise imperfectly nested 2-d
// tiling.
TEST_FUNC(tile_2d) {
  auto memrefType =
      MemRefType::get({ShapedType::kDynamicSize, ShapedType::kDynamicSize,
                       ShapedType::kDynamicSize},
                      FloatType::getF32(&globalContext()), {}, 0);
  auto f = makeFunction("tile_2d", {}, {memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value zero = std_constant_index(0);
  MemRefBoundsCapture vA(f.getArgument(0)), vB(f.getArgument(1)),
      vC(f.getArgument(2));
  AffineIndexedValue A(f.getArgument(0)), B(f.getArgument(1)),
      C(f.getArgument(2));
  Value i, j, k1, k2;
  Value M(vC.ub(0)), N(vC.ub(1)), O(vC.ub(2));

  // clang-format off
  using namespace edsc::op;
  affineLoopNestBuilder({zero, zero}, {M, N}, {1, 1}, [&](ValueRange ivs) {
    i = ivs[0];
    j = ivs[1];
    affineLoopBuilder(zero, O, 1, [&](Value k) {
      k1 = k;
      C(i, j, k1) = A(i, j, k1) + B(i, j, k1);
    });
    affineLoopBuilder(zero, O, 1, [&](Value k) {
      k2 = k;
      C(i, j, k2) = A(i, j, k2) + B(i, j, k2);
    });
  });
  // clang-format on

  auto li = getForInductionVarOwner(i), lj = getForInductionVarOwner(j),
       lk1 = getForInductionVarOwner(k1), lk2 = getForInductionVarOwner(k2);
  auto indicesL1 = mlir::tile({li, lj}, {512, 1024}, {lk1, lk2});
  auto lii1 = indicesL1[0][0], ljj1 = indicesL1[1][0];
  mlir::tile({ljj1, lii1}, {32, 16}, ljj1);

  // clang-format off
  // CHECK-LABEL: func @tile_2d
  //       CHECK: %[[ZERO:.*]] = constant 0 : index
  //       CHECK: %[[M:[0-9]+]] = memref.dim %arg2, %c0{{[_0-9]*}} : memref<?x?x?xf32>
  //       CHECK: %[[N:[0-9]+]] = memref.dim %arg2, %c1{{[_0-9]*}} : memref<?x?x?xf32>
  //       CHECK: %[[P:[0-9]+]] = memref.dim %arg2, %c2{{[_0-9]*}} : memref<?x?x?xf32>
  //       CHECK:   affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%[[ZERO]]) to affine_map<(d0) -> (d0)>(%[[M]]) step 512 {
  //  CHECK-NEXT:     affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%[[ZERO]]) to affine_map<(d0) -> (d0)>(%[[N]]) step 1024 {
  //  CHECK-NEXT:       affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%[[ZERO]]) to affine_map<(d0) -> (d0)>(%[[P]]) {
  //  CHECK-NEXT:         affine.for %{{.*}} = max affine_map<(d0) -> (0, d0)>(%{{.*}}) to min affine_map<(d0)[s0] -> (s0, d0 + 512)>(%{{.*}})[%[[M]]] step 16 {
  //  CHECK-NEXT:           affine.for %{{.*}} = max affine_map<(d0) -> (0, d0)>(%{{.*}}) to min affine_map<(d0)[s0] -> (s0, d0 + 1024)>(%{{.*}})[%[[N]]] step 32 {
  //  CHECK-NEXT:             affine.for %{{.*}} = max affine_map<(d0, d1) -> (0, d0, d1)>(%{{.*}}, %{{.*}}) to min affine_map<(d0, d1)[s0] -> (s0, d0 + 1024, d1 + 32)>(%{{.*}}, %{{.*}})[%[[N]]] {
  //  CHECK-NEXT:               affine.for %{{.*}} = max affine_map<(d0, d1) -> (0, d0, d1)>(%{{.*}}, %{{.*}}) to min affine_map<(d0, d1)[s0] -> (s0, d0 + 512, d1 + 16)>(%{{.*}}, %{{.*}})[%[[M]]] {
  //  CHECK-NEXT:                 {{.*}} = affine.load {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-NEXT:                 {{.*}} = affine.load {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-NEXT:                 {{.*}} = addf {{.*}}, {{.*}} : f32
  //  CHECK-NEXT:                 affine.store {{.*}}, {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //       CHECK:               }
  //  CHECK-NEXT:             }
  //  CHECK-NEXT:           }
  //  CHECK-NEXT:         }
  //  CHECK-NEXT:       }
  //  CHECK-NEXT:       affine.for %{{.*}} = affine_map<(d0) -> (d0)>(%[[ZERO]]) to affine_map<(d0) -> (d0)>(%[[P]]) {
  //  CHECK-NEXT:         affine.for %{{.*}} = max affine_map<(d0) -> (0, d0)>(%{{.*}}) to min affine_map<(d0)[s0] -> (s0, d0 + 512)>(%{{.*}})[%[[M]]] {
  //  CHECK-NEXT:           affine.for %{{.*}} = max affine_map<(d0) -> (0, d0)>(%{{.*}}) to min affine_map<(d0)[s0] -> (s0, d0 + 1024)>(%{{.*}})[%[[N]]] {
  //  CHECK-NEXT:             {{.*}} = affine.load {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-NEXT:             {{.*}} = affine.load {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  //  CHECK-NEXT:             {{.*}}= addf {{.*}}, {{.*}} : f32
  //  CHECK-NEXT:             affine.store {{.*}}, {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] : memref<?x?x?xf32>
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

// Exercise StdIndexedValue for loads and stores.
TEST_FUNC(indirect_access) {
  using namespace edsc::op;
  auto memrefType = MemRefType::get({ShapedType::kDynamicSize},
                                    FloatType::getF32(&globalContext()), {}, 0);
  auto f = makeFunction("indirect_access", {},
                        {memrefType, memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value zero = std_constant_index(0);
  MemRefBoundsCapture vC(f.getArgument(2));
  AffineIndexedValue B(f.getArgument(1)), D(f.getArgument(3));
  MemRefIndexedValue A(f.getArgument(0)), C(f.getArgument(2));
  Value N(vC.ub(0));

  // clang-format off
  affineLoopBuilder(zero, N, 1, [&](Value i) {
      C((Value)D(i)) = A((Value)B(i));
  });
  // clang-format on

  // clang-format off
  // CHECK-LABEL: func @indirect_access
  // CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: memref<?xf32>, %[[ARG3:.*]]: memref<?xf32>)
  // CHECK-DAG:  [[B:%.*]] = affine.load %[[ARG1]]
  // CHECK-DAG:  [[D:%.*]] = affine.load %[[ARG3]]
  // CHECK:  load %{{.*}}{{\[}}[[B]]{{\]}}
  // CHECK:  store %{{.*}}, %{{.*}}{{\[}}[[D]]{{\]}}
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

// Exercise affine loads and stores build with empty maps.
TEST_FUNC(empty_map_load_store) {
  using namespace edsc::op;
  auto memrefType =
      MemRefType::get({}, FloatType::getF32(&globalContext()), {}, 0);
  auto f = makeFunction("empty_map_load_store", {},
                        {memrefType, memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value zero = std_constant_index(0);
  Value one = std_constant_index(1);
  AffineIndexedValue input(f.getArgument(0)), res(f.getArgument(1));

  // clang-format off
  affineLoopBuilder(zero, one, 1, [&](Value) {
      res() = input();
  });
  // clang-format on

  // clang-format off
  // CHECK-LABEL: func @empty_map_load_store(
  // CHECK:  [[A:%.*]] = affine.load %{{.*}}[]
  // CHECK:  affine.store [[A]], %{{.*}}[]
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @affine_if_op
// CHECK:       affine.if affine_set<([[d0:.*]], [[d1:.*]]){{\[}}[[s0:.*]], [[s1:.*]]{{\]}}
// CHECK-NOT:   else
// CHECK:       affine.if affine_set<([[d0:.*]], [[d1:.*]]){{\[}}[[s0:.*]], [[s1:.*]]{{\]}}
// CHECK-NEXT: } else {
// clang-format on
TEST_FUNC(affine_if_op) {
  using namespace edsc::op;
  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, f32Type, {}, 0);
  auto f = makeFunction("affine_if_op", {}, {memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());

  Value zero = std_constant_index(0), ten = std_constant_index(10);

  SmallVector<bool, 4> isEq = {false, false, false, false};
  SmallVector<AffineExpr, 4> affineExprs = {
      builder.getAffineDimExpr(0),    // d0 >= 0
      builder.getAffineDimExpr(1),    // d1 >= 0
      builder.getAffineSymbolExpr(0), // s0 >= 0
      builder.getAffineSymbolExpr(1)  // s1 >= 0
  };
  auto intSet = IntegerSet::get(2, 2, affineExprs, isEq);

  SmallVector<Value, 4> affineIfArgs = {zero, zero, ten, ten};
  intrinsics::affine_if(intSet, affineIfArgs, /*withElseRegion=*/false);
  intrinsics::affine_if(intSet, affineIfArgs, /*withElseRegion=*/true);

  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @linalg_generic_pointwise
//       CHECK:   linalg.generic {
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins({{.*}}memref<?x?xf32>, memref<?x?xf32>)
// CHECK-SAME: outs({{.*}}memref<?x?xf32>)
//       CHECK:       addf
//       CHECK:   linalg.generic {
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins({{.*}}memref<?x?xf32>, memref<?x?xf32>)
// CHECK-SAME: outs({{.*}}memref<?x?xf32>)
//       CHECK:       cmpf ogt
//       CHECK:       select
//       CHECK:   linalg.generic {
// CHECK-SAME:      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%{{[a-z0-9]*}} : memref<?x?xf32>)
// CHECK-SAME: outs(%{{[a-z0-9]*}} : memref<?x?xf32>)
//       CHECK:     tanh
// clang-format on
TEST_FUNC(linalg_generic_pointwise_test) {
  using namespace edsc;
  using namespace edsc::ops;

  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, f32Type, {}, 0);
  auto f = makeFunction("linalg_generic_pointwise", {},
                        {memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value A(f.getArgument(0)), B(f.getArgument(1)), C(f.getArgument(2));
  AffineExpr i, j;
  bindDims(&globalContext(), i, j);
  StructuredIndexed SA(A), SB(B), SC(C);
  linalg_generic_pointwise_add(SA({i, j}), SB({i, j}), SC({i, j}));
  linalg_generic_pointwise_max(SA({i, j}), SB({i, j}), SC({i, j}));
  linalg_generic_pointwise_tanh(SA({i, j}), SC({i, j}));

  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @linalg_generic_matmul
//       CHECK:   linalg.generic {
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%{{[a-z0-9]*}}, %{{[a-z0-9]*}} : memref<?x?xf32>, memref<?x?xf32>)
// CHECK-SAME: outs(%{{[a-z0-9]*}} : memref<?x?xf32>)
///      CHECK:   ^bb0(%[[a0:.*]]: f32, %[[a1:.*]]: f32, %[[a2:.*]]: f32):
//       CHECK:     %[[a3:.*]] = mulf %[[a0]], %[[a1]] : f32
//       CHECK:     %[[a4:.*]] = addf %[[a2]], %[[a3]] : f32
//       CHECK:     linalg.yield %[[a4]] : f32
//       CHECK:   }
// clang-format on
TEST_FUNC(linalg_generic_matmul_test) {
  using namespace edsc;
  using namespace edsc::ops;

  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, f32Type, {}, 0);
  auto f = makeFunction("linalg_generic_matmul", {},
                        {memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  linalg_generic_matmul(f.getArguments());

  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @linalg_generic_conv_nhwc
//       CHECK:   linalg.generic {
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2 * 3 + d4 * 5, d3 * 4 + d5 * 6, d6)>,
// CHECK-SAME: affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d1)>,
// CHECK-SAME: affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d2, d3, d1)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
// CHECK-SAME: ins(%{{[a-z0-9]*}}, %{{[a-z0-9]*}} : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
// CHECK-SAME: outs(%{{[a-z0-9]*}} : memref<?x?x?x?xf32>)
///      CHECK:   ^bb0(%[[a0:.*]]: f32, %[[a1:.*]]: f32, %[[a2:.*]]: f32):
//       CHECK:     %[[a3:.*]] = mulf %[[a0]], %[[a1]] : f32
//       CHECK:     %[[a4:.*]] = addf %[[a2]], %[[a3]] : f32
//       CHECK:     linalg.yield %[[a4]] : f32
//       CHECK:   }
// clang-format on
TEST_FUNC(linalg_generic_conv_nhwc) {
  using namespace edsc;
  using namespace edsc::ops;

  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType =
      MemRefType::get({ShapedType::kDynamicSize, ShapedType::kDynamicSize,
                       ShapedType::kDynamicSize, ShapedType::kDynamicSize},
                      f32Type, {}, 0);
  auto f = makeFunction("linalg_generic_conv_nhwc", {},
                        {memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  linalg_generic_conv_nhwc(f.getArguments(),
                           /*strides=*/{3, 4}, /*dilations=*/{5, 6});

  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @linalg_generic_dilated_conv_nhwc
//       CHECK:   linalg.generic {
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d3 * 3 + d5 * 5, d4 * 4 + d6 * 6, d2)>,
// CHECK-SAME: affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5, d6, d2, d1)>,
// CHECK-SAME: affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d3, d4, d1 + d2 * 7)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]}
// CHECK-SAME: ins(%{{[a-z0-9]*}}, %{{[a-z0-9]*}} : memref<?x?x?x?xf32>, memref<?x?x?x?xf32>)
// CHECK-SAME: outs(%{{[a-z0-9]*}} : memref<?x?x?x?xf32>)
//       CHECK:   ^bb0(%[[a0:.*]]: f32, %[[a1:.*]]: f32, %[[a2:.*]]: f32):
//       CHECK:     %[[a3:.*]] = mulf %[[a0]], %[[a1]] : f32
//       CHECK:     %[[a4:.*]] = addf %[[a2]], %[[a3]] : f32
//       CHECK:     linalg.yield %[[a4]] : f32
//       CHECK:   }
// clang-format on
TEST_FUNC(linalg_generic_dilated_conv_nhwc) {
  using namespace edsc;
  using namespace edsc::ops;

  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType =
      MemRefType::get({ShapedType::kDynamicSize, ShapedType::kDynamicSize,
                       ShapedType::kDynamicSize, ShapedType::kDynamicSize},
                      f32Type, {}, 0);
  auto f = makeFunction("linalg_generic_dilated_conv_nhwc", {},
                        {memrefType, memrefType, memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  linalg_generic_dilated_conv_nhwc(f.getArguments(),
                                   /*depth_multiplier=*/7,
                                   /*strides=*/{3, 4}, /*dilations=*/{5, 6});

  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @linalg_metadata_ops
//       CHECK: linalg.reshape {{.*}} [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d2)>] : memref<4x8x16xf32> into memref<32x16xf32>
//       CHECK: linalg.reshape {{.*}} [affine_map<(d0, d1, d2) -> (d0, d1)>, affine_map<(d0, d1, d2) -> (d2)>] : memref<32x16xf32> into memref<4x8x16xf32>
// clang-format on
TEST_FUNC(linalg_metadata_ops) {
  using linalg::ReassociationExprs;

  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get({4, 8, 16}, f32Type, {}, 0);
  auto f = makeFunction("linalg_metadata_ops", {}, {memrefType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  AffineExpr i, j, k;
  bindDims(&globalContext(), i, j, k);
  Value v(f.getArgument(0));
  SmallVector<ReassociationExprs, 2> maps = {ReassociationExprs({i, j}),
                                             ReassociationExprs({k})};
  auto reshaped = linalg_reshape(v, maps);
  linalg_reshape(memrefType, reshaped, maps);

  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @linalg_tensors
//       CHECK:   linalg.generic {
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%{{[a-z0-9]*}}, %{{[a-z0-9]*}} : tensor<?x?xf32>, memref<?x?xf32>)
//       CHECK:       addf
//       CHECK:     } -> tensor<?x?xf32>
//       CHECK:   linalg.generic {
// CHECK-SAME: indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
// CHECK-SAME: iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%{{[a-z0-9]*}}, %{{[a-z0-9]*}} : tensor<?x?xf32>, tensor<?x?xf32>)
//       CHECK:       cmpf ogt
//       CHECK:       select
//       CHECK:   } -> tensor<?x?xf32>
//       CHECK:   linalg.generic {
// CHECK-SAME:      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]}
// CHECK-SAME: ins(%{{[a-z0-9]*}} : tensor<?x?xf32>)
//       CHECK:     tanh
//       CHECK:   } -> tensor<?x?xf32>
//       CHECK:   linalg.generic {
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d2, d1)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d0, d1)>],
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]}
// CHECK-SAME: ins(%{{[a-z0-9]*}}, %{{[a-z0-9]*}} : tensor<?x?xf32>, memref<?x?xf32>)
//       CHECK:     mulf
//       CHECK:   } -> tensor<?x?xf32>
//       CHECK:   linalg.generic {
//  CHECK-SAME:     indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d2, d1)>,
//  CHECK-SAME:                      affine_map<(d0, d1, d2) -> (d0, d1)>],
//  CHECK-SAME:     iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-SAME: ins(%{{[a-z0-9]*}}, %{{[a-z0-9]*}} : tensor<?x?xf32>, memref<?x?xf32>)
// CHECK-SAME: outs(%{{[a-z0-9]*}} : tensor<?x?xf32>)
//       CHECK:     mulf
//       CHECK:     addf
//       CHECK:   } -> tensor<?x?xf32>
// clang-format on
TEST_FUNC(linalg_tensors_test) {
  using namespace edsc;
  using namespace edsc::ops;

  auto f32Type = FloatType::getF32(&globalContext());
  auto memrefType = MemRefType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, f32Type, {}, 0);
  auto tensorType = RankedTensorType::get(
      {ShapedType::kDynamicSize, ShapedType::kDynamicSize}, f32Type);
  auto f =
      makeFunction("linalg_tensors", {}, {tensorType, memrefType, tensorType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value A(f.getArgument(0)), B(f.getArgument(1)), C(f.getArgument(2));
  AffineExpr i, j;
  bindDims(&globalContext(), i, j);
  StructuredIndexed SA(A), SB(B), SC(C);
  Value added = linalg_generic_pointwise_add(SA({i, j}), SB({i, j}), SC({i, j}))
                    ->getResult(0);
  Value maxed = linalg_generic_pointwise_max(
                    SA({i, j}), StructuredIndexed(added)({i, j}), SC({i, j}))
                    ->getResult(0);
  Value tanhed = linalg_generic_pointwise_tanh(StructuredIndexed(maxed)({i, j}),
                                               SC({i, j}))
                     ->getResult(0);
  Value o1 = linalg_generic_matmul(A, B, tanhed, tensorType)->getResult(0);
  linalg_generic_matmul(A, B, o1, tensorType);

  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(vector_extractelement_op_i32) {
  using namespace edsc::op;
  auto f = makeFunction("vector_extractelement_op", {}, {});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  auto i32Type = builder.getI32Type();
  auto vectorType = VectorType::get(/*shape=*/{8}, i32Type);
  vector_extract_element(
      i32Type, std_constant(vectorType, builder.getI32VectorAttr({10})),
      std_constant_int(0, i32Type));

  // clang-format off
  // CHECK-LABEL: @vector_extractelement_op
  //   CHECK-DAG:     {{.*}} = constant dense<10>
  //   CHECK-DAG:     {{.*}} = constant 0
  //  CHECK-NEXT:     {{.*}} = vector.extractelement
  // clang-format on
  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

// clang-format off
// CHECK-LABEL: func @memref_vector_matmul_test(
//  CHECK-SAME:   %[[A:.*]]: memref<?x?xvector<4x16xf32>>,
//  CHECK-SAME:   %[[B:.*]]: memref<?x?xvector<16x8xf32>>,
//  CHECK-SAME:   %[[C:.*]]: memref<?x?xvector<4x8xf32>>)
//       CHECK:   linalg.generic {{{.*}}}
// CHECK-SAME: ins(%[[A]], %[[B]] : memref<?x?xvector<4x16xf32>>, memref<?x?xvector<16x8xf32>>)
// CHECK-SAME: outs(%[[C]] : memref<?x?xvector<4x8xf32>>)
//       CHECK:     vector.contract{{.*}}[affine_map<(d0, d1, d2) -> (d0, d2)>,
//  CHECK-SAME:                       affine_map<(d0, d1, d2) -> (d2, d1)>,
//  CHECK-SAME:                       affine_map<(d0, d1, d2) -> (d0, d1)>],
//  CHECK-SAME:                {{.*}}["parallel", "parallel", "reduction"]
// clang-format on
TEST_FUNC(memref_vector_matmul_test) {
  using namespace edsc;
  using namespace edsc::ops;

  int64_t M = 4, N = 8, K = 16;
  auto f32Type = FloatType::getF32(&globalContext());
  auto mkVectorType = VectorType::get({M, K}, f32Type);
  auto knVectorType = VectorType::get({K, N}, f32Type);
  auto mnVectorType = VectorType::get({M, N}, f32Type);
  auto typeA =
      MemRefType::get({ShapedType::kDynamicSize, ShapedType::kDynamicSize},
                      mkVectorType, {}, 0);
  auto typeB =
      MemRefType::get({ShapedType::kDynamicSize, ShapedType::kDynamicSize},
                      knVectorType, {}, 0);
  auto typeC =
      MemRefType::get({ShapedType::kDynamicSize, ShapedType::kDynamicSize},
                      mnVectorType, {}, 0);
  auto f = makeFunction("memref_vector_matmul_test", {}, {typeA, typeB, typeC});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value A(f.getArgument(0)), B(f.getArgument(1)), C(f.getArgument(2));
  auto contractionBuilder = [](ValueRange args) {
    assert(args.size() == 3 && "expected 3 block arguments");
    (linalg_yield(vector_contraction_matmul(args[0], args[1], args[2])));
  };
  linalg_generic_matmul(A, B, C, contractionBuilder);

  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

TEST_FUNC(builder_loop_for_yield) {
  auto indexType = IndexType::get(&globalContext());
  auto f32Type = FloatType::getF32(&globalContext());
  auto f = makeFunction("builder_loop_for_yield", {},
                        {indexType, indexType, indexType, indexType});

  OpBuilder builder(f.getBody());
  ScopedContext scope(builder, f.getLoc());
  Value init0 = std_constant_float(llvm::APFloat(1.0f), f32Type);
  Value init1 = std_constant_float(llvm::APFloat(2.0f), f32Type);
  Value a(f.getArgument(0)), b(f.getArgument(1)), c(f.getArgument(2)),
      d(f.getArgument(3));
  using namespace edsc::op;
  auto results = loopNestBuilder(a - b, c + d, a, {init0, init1},
                                 [&](Value iv, ValueRange args) {
                                   Value sum = args[0] + args[1];
                                   return scf::ValueVector{args[1], sum};
                                 })
                     .getResults();
  results[0] + results[1];

  // clang-format off
  // CHECK-LABEL: func @builder_loop_for_yield(%{{.*}}: index, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index) {
  // CHECK:     [[init0:%.*]] = constant
  // CHECK:     [[init1:%.*]] = constant
  // CHECK-DAG:    [[r0:%[0-9]+]] = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%{{.*}}, %{{.*}}]
  // CHECK-DAG:    [[r1:%[0-9]+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1)>()[%{{.*}}, %{{.*}}]
  // CHECK-NEXT: [[res:%[0-9]+]]:2 = scf.for %{{.*}} = [[r0]] to [[r1]] step {{.*}} iter_args([[arg0:%.*]] = [[init0]], [[arg1:%.*]] = [[init1]]) -> (f32, f32) {
  // CHECK:     [[sum:%[0-9]+]] = addf [[arg0]], [[arg1]] : f32
  // CHECK:     scf.yield [[arg1]], [[sum]] : f32, f32
  // CHECK:     addf [[res]]#0, [[res]]#1 : f32
  // clang-format on

  f.print(llvm::outs(), OpPrintingFlags().useLocalScope());
  f.erase();
}

int main() {
  RUN_TESTS();
  return 0;
}
