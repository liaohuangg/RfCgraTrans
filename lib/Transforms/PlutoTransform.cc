// //===- PlutoTransform.cc - Transform MLIR code by PLUTO
// -------------------===//
//
// This file implements the transformation passes on MLIR using PLUTO.
//
//===----------------------------------------------------------------------===//

// #include "RfCgraTrans/Transforms/PlutoTransform.h"
// #include "RfCgraTrans/Support/OslScop.h"
// #include "RfCgraTrans/Support/OslScopStmtOpSet.h"
// #include "RfCgraTrans/Support/OslSymbolTable.h"
// #include "RfCgraTrans/Support/ScopStmt.h"
// #include "RfCgraTrans/Target/OpenScop.h"

// #include "pluto/internal/pluto.h"
// #include "pluto/matrix.h"
// #include "pluto/osl_pluto.h"
// #include "pluto/pluto.h"

// #include "mlir/Dialect/MemRef/IR/MemRef.h"

// #include "mlir/Analysis/AffineAnalysis.h"
// #include "mlir/Analysis/AffineStructures.h"
// #include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
// #include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Affine/IR/AffineValueMap.h"
// #include "mlir/Dialect/StandardOps/IR/Ops.h"
// #include "mlir/IR/BlockAndValueMapping.h"
// #include "mlir/IR/Builders.h"
// #include "mlir/IR/OpImplementation.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/IR/Types.h"
// #include "mlir/IR/Value.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Pass/PassManager.h"
// #include "mlir/Transforms/DialectConversion.h"
// #include "mlir/Transforms/Passes.h"
// #include "mlir/Transforms/Utils.h"
#include "RfCgraTrans/Transforms/SearchUnroll.h"
#include "llvm/Support/FormatVariadic.h"
#include <limits.h>
#include <stack>
// #include <sys/time.h>

using namespace mlir;
using namespace llvm;
using namespace RfCgraTrans;
using namespace memref;

namespace {
struct PlutoOptPipelineOptions
    : public mlir::PassPipelineOptions<PlutoOptPipelineOptions> {
  Option<std::string> dumpClastAfterPluto{
      *this, "dump-clast-after-pluto",
      llvm::cl::desc("File name for dumping the CLooG AST (clast) after Pluto "
                     "optimization.")};
  Option<bool> parallelize{*this, "parallelize",
                           llvm::cl::desc("Enable parallelization from Pluto."),
                           llvm::cl::init(false)};
  Option<bool> debug{*this, "debug",
                     llvm::cl::desc("Enable moredebug in Pluto."),
                     llvm::cl::init(false)};
  Option<bool> generateParallel{
      *this, "gen-parallel", llvm::cl::desc("Generate parallel affine loops."),
      llvm::cl::init(false)};
};

} // namespace

void CreatDFG(mlir::AffineForOp forOp, DFGList dfgList, mlir::FuncOp g,
              mlir::ModuleOp moduleop, ScopInformation *scopInformation,
              int &DFGStep, int &MII, int &unrollN,
              plutoCost_Matrix *pluto_trans_Cost, int unrollFlag, int &DFGNum,
              int &perfectLoopflag, int maxDim, int &unroll_transNum,
              singleTrans &singleT, scc_stmt_topSort &scc_stmt_topSort_map) {
  llvm::raw_ostream &os = llvm::outs();
  MapVector<int, SmallVector<bool, 8U>> reuseMap;
  SmallVector<FuncOp, 8U> calleeVec;
  InnerUnroll initial_unroll;
  std::vector<int> stmtIdV;
  // Schedule initial_schedule;
  for (std::pair<int, llvm::SmallVector<mlir::CallOp, 8U>> map : dfgList.list) {
    MlirDFG dfg(map.first, map.second, forOp, g, moduleop, scopInformation,
                DFGNum, pluto_trans_Cost->is_global);
    stmtIdV.push_back(dfg.getPnuStmtId());
    Schedule schedule(dfg);
    schedule.find_Simple_Schedule();

    if (perfectLoopflag && unrollFlag && dfg.RecMII == 0) {
      InnerUnroll unroll(dfg, scopInformation, pluto_trans_Cost->is_global,
                         unroll_transNum);
      initial_unroll = unroll;
      //unroll.print_unroll();
    }

    if (initial_unroll.need_unroll) {
      unrollN = initial_unroll.unrollNum;
      DFGStep += dfg.DfgStep;
      MII += initial_unroll.betterII;
    } else {
      DFGStep += dfg.DfgStep;
      MII += dfg.TheoreticalII;
    }
    if (pluto_trans_Cost->is_global != 0) {
      dfg.genFile();
      if (perfectLoopflag && schedule_Switch) {
        if (schedule.search_Schedule() == 1) {
        }
      }
      if (perfectLoopflag && dfg.dfg_dim == maxDim) {
        if (dfg.reuseDataNum != 0) {
          singleT.totalReuseData+=dfg.reuseDataNum;
        }
      }
    }
  }
  singleT.flagComp(stmtIdV, scc_stmt_topSort_map);
}

int isConsBound(mlir::Value value, ScopInformation *scopInformation) {
  int index1 = 0;
  int flag = 0;
  for (mlir::IndexCastOp op : scopInformation->BoundIndexCastVec) {
    if (op.getResult() == value) {
      flag = 1;
      break;
    }
    index1++;
  }

  if (flag) {
    return index1;
  } else {
    return -1;
  }
}

int findBound(mlir::AffineForOp forOp, mlir::AffineMap affinemap,
              mlir::AffineBound affinebound, ScopInformation *scopInformation,
              int UALflag) {
  llvm::raw_ostream &os = llvm::outs();
  int Bound;
  int index1, index2;
  //Case 1 constant
  if (affinemap.isSingleConstant()) {
    if (UALflag) { // 1 Upper bound
      Bound = forOp.getConstantUpperBound();
    } else { // 0 Lower bound
      Bound = forOp.getConstantLowerBound();
    }
    //Case 2 is not a constant but a single offset
  } else if (!affinemap.isSingleConstant()) {
    if (affinemap.getNumResults() == 1) { 
      mlir::Value value = affinebound.getOperand(0);
      if (isConsBound(value, scopInformation) >= 0) {
        Bound = scopInformation->getBoundMapConstant(affinemap, value);
      } else {
        Bound = scopInformation->BoundIntVec[0] /
                2;
      }
      // Case 3 is either max or min
    } else if (affinemap.getNumResults() > 1) {
      if (affinebound.getNumOperands() == 1) {
        mlir::Value value = affinebound.getOperand(0);
        index1 = isConsBound(value, scopInformation);
        if (index1 >= 0) {
          std::vector<mlir::Attribute, std::allocator<mlir::Attribute>> attrVec;
          llvm::SmallVector<int64_t> results;
          attrVec.push_back(scopInformation->BoundAttrVec[index1]);
          llvm::ArrayRef<mlir::Attribute> operandConstants(attrVec);
          affinemap.partialConstantFold(operandConstants, &results);
          if (UALflag) { // 1 上界 min
            Bound = results[0] < results[1] ? results[0] : results[1];
          } else { // 0 下界 max
            Bound = results[0] > results[1] ? results[0] : results[1];
          }
        } else {
          Bound = scopInformation->BoundIntVec[0] /
                  2; 
        }
      } else if (affinebound.getNumOperands() == 2) {
        mlir::Value value1 = affinebound.getOperand(0);
        index1 = isConsBound(value1, scopInformation);
        mlir::Value value2 = affinebound.getOperand(1);
        index2 = isConsBound(value2, scopInformation);
        if (index1 >= 0 && index2 >= 0) {
          std::vector<mlir::Attribute, std::allocator<mlir::Attribute>> attrVec;
          llvm::SmallVector<int64_t> results;
          attrVec.push_back(scopInformation->BoundAttrVec[index1]);
          attrVec.push_back(scopInformation->BoundAttrVec[index2]);
          llvm::ArrayRef<mlir::Attribute> operandConstants(attrVec);
          affinemap.partialConstantFold(operandConstants, &results);
          if (UALflag) {
            Bound = results[0] < results[1] ? results[0] : results[1];
          } else {
            Bound = results[0] > results[1] ? results[0] : results[1];
          }
        } else {
          Bound = scopInformation->BoundIntVec[0] /
                  2; 
        }
      }
    }
  }
  return Bound;
}
void subLoopCycleCompute(mlir::AffineForOp forOp, DFGList dfgList,
                         mlir::FuncOp g, mlir::MLIRContext *context,
                         mlir::ModuleOp moduleop,
                         ScopInformation *scopInformation,
                         std::stack<long long> &cycleStack, int sisterCountLoop,
                         plutoCost_Matrix *pluto_trans_Cost, int unrollFlag,
                         int &DFGNum, int maxDim, int &unroll_transNum,
                         singleTrans &singleT,
                         scc_stmt_topSort &scc_stmt_topSort_map) {
  MapVector<int, SmallVector<bool, 8U>> ReuseMap;
  llvm::raw_ostream &os = llvm::outs();
  int DFGStep = 0;
  int MII = 0;
  long long currentCycle = 0;
  // There are min or max boundaries
  int lowerBound;
  int upperBound;
  int unrollNum = -1;
  int perfectLoopflag = 0;
  lowerBound = findBound(forOp, forOp.getLowerBoundMap(), forOp.getLowerBound(),
                         scopInformation, 0);
  upperBound = findBound(forOp, forOp.getUpperBoundMap(), forOp.getUpperBound(),
                         scopInformation, 1);
  int TC = upperBound - lowerBound;
  if (TC < 0) {
    TC = 0;
  }

  if (dfgList.dfgNum == 0 && sisterCountLoop == 1) {
    // map clear is not empty after... This judgment is invalid, but does not affect the outcome of the program
    currentCycle = cycleStack.top() * TC;
    cycleStack.pop();
    cycleStack.push(currentCycle);

  } else if (dfgList.dfgNum == 0 && sisterCountLoop > 1) {
    for (int i = 0; i < sisterCountLoop; i++) {
      currentCycle += cycleStack.top() + Tf;
      cycleStack.pop();
    }
    currentCycle = currentCycle * TC;
    cycleStack.push(currentCycle);

  } else if (dfgList.dfgNum != 0 && sisterCountLoop > 0) {
    perfectLoopflag = 0;
    CreatDFG(forOp, dfgList, g, moduleop, scopInformation, DFGStep, MII,
             unrollNum, pluto_trans_Cost, unrollFlag, DFGNum, perfectLoopflag,
             maxDim, unroll_transNum, singleT, scc_stmt_topSort_map);
    for (int i = 0; i < sisterCountLoop; i++) {
      currentCycle += cycleStack.top();
      cycleStack.pop();
    }
    currentCycle = (DFGStep + currentCycle + Tf * MII) * TC;
    cycleStack.push(currentCycle);

  } else if (dfgList.dfgNum != 0 && sisterCountLoop == 0) {
    perfectLoopflag = 1;
    CreatDFG(forOp, dfgList, g, moduleop, scopInformation, DFGStep, MII,
             unrollNum, pluto_trans_Cost, unrollFlag, DFGNum, perfectLoopflag,
             maxDim, unroll_transNum, singleT, scc_stmt_topSort_map);

    if (unrollFlag == 1 && unrollNum != -1) {
      int NewTC = (TC / unrollNum);
      currentCycle = (MII * (NewTC - 1) + DFGStep) + Tf * MII;
      cycleStack.push(currentCycle);
    } else {
      currentCycle = (MII * (TC - 1) + DFGStep) + Tf * MII;
      cycleStack.push(currentCycle);
    }
  }
}

int getSisterLoopNum(mlir::AffineForOp forOp) {
  int countFor = 0;
  for (mlir::Block::iterator it = forOp.begin(); it != forOp.end(); it++) {
    if (mlir::AffineForOp forer = dyn_cast<mlir::AffineForOp>(*it)) {
      countFor++;
    }
  }
  return countFor;
}

//recursion
void LoopCycleCompute(mlir::AffineForOp forOp, mlir::FuncOp g,
                      mlir::MLIRContext *context, mlir::ModuleOp moduleop,
                      ScopInformation *scopInformation,
                      std::stack<long long> &cycleStack,
                      plutoCost_Matrix *pluto_trans_Cost, int unrollFlag,
                      int &DFGNum, int maxDim, int &unroll_transNum,
                      singleTrans &singleT,
                      scc_stmt_topSort &scc_stmt_topSort_map) {
  llvm::raw_ostream &os = llvm::outs();
  MapVector<int, SmallVector<mlir::CallOp, 8U>> callOpMapVec;
  DFGList dfgList(callOpMapVec);
  SmallVector<mlir::CallOp, 8U> callOpVec;
  int flag = 0;
  int count = 0;
  //Counts how many sister loops are wrapped in the current loop
  int sisterCountLoop = getSisterLoopNum(forOp);

  for (mlir::Block::iterator it = forOp.begin(); it != forOp.end(); it++) {
    if (mlir::AffineForOp forer = dyn_cast<mlir::AffineForOp>(*it)) {
      LoopCycleCompute(forer, g, context, moduleop, scopInformation, cycleStack,
                       pluto_trans_Cost, unrollFlag, DFGNum, maxDim,
                       unroll_transNum, singleT,
                       scc_stmt_topSort_map);
      flag = 1;
    }
    if (mlir::CallOp caller = dyn_cast<mlir::CallOp>(*it)) {
      if (flag == 1) {
        flag = 0;
        if (count > 0) {
          dfgList.insert(callOpVec);
          while (!callOpVec.empty()) {
            callOpVec.pop_back();
          }
          count = 0;
        }
      }
      callOpVec.push_back(caller);
      count++;
    }
  }
  if (count > 0) {
    dfgList.insert(callOpVec);
  }
  subLoopCycleCompute(forOp, dfgList, g, context, moduleop, scopInformation,
                      cycleStack, sisterCountLoop, pluto_trans_Cost, unrollFlag,
                      DFGNum, maxDim, unroll_transNum, singleT,
                      scc_stmt_topSort_map);
  callOpVec.clear();
  callOpMapVec.clear();
}

int getForNum(mlir::FuncOp g) {
  int countFor = 0;
  g.walk([&](mlir::Operation *op) {
    if (mlir::AffineForOp forOp = dyn_cast<mlir::AffineForOp>(op)) {
      if (mlir::AffineIfOp ifOp =
              dyn_cast<mlir::AffineIfOp>(forOp->getParentOp())) {
        if (mlir::FuncOp funcOp = dyn_cast<mlir::FuncOp>(ifOp->getParentOp())) {
          countFor++;
        }
      }
    }
  });
  return countFor;
}

void PerfProfiling(mlir::ModuleOp moduleop, mlir::FuncOp g,
                   mlir::MLIRContext *context,
                   plutoCost_Matrix *pluto_trans_Cost, int scc_num, int maxDim,
                   int unrollFlag, int &unroll_transNum, singleTrans &singleT,
                   scc_stmt_topSort &scc_stmt_topSort_map) {
  llvm::raw_ostream &os = llvm::outs();
  mlir::ModuleOp m = dyn_cast<mlir::ModuleOp>(g->getParentOp());
  long long totalCycle = 0;
  int i = 0;
  bool debug = false;
  ScopInformation scopInformation(g, moduleop);
  int countFor = getForNum(g);
  int DFGNum = 0;
  std::stack<long long> cycleStack;
  g.walk([&](mlir::Operation *op) {
    if (mlir::AffineForOp forOp = dyn_cast<mlir::AffineForOp>(op)) {
      if (mlir::AffineIfOp ifOp =
              dyn_cast<mlir::AffineIfOp>(forOp->getParentOp())) {
        if (mlir::FuncOp funcOp = dyn_cast<mlir::FuncOp>(ifOp->getParentOp())) {
          if (pluto_trans_Cost->is_global == 0 ||
              pluto_trans_Cost->is_global == 1 ||
              pluto_trans_Cost->is_global == 2 ||
              pluto_trans_Cost->is_global == 3 ||
              pluto_trans_Cost->is_global == 4 ||
              pluto_trans_Cost->is_global == 5) {
            debug = true;
            while (!cycleStack.empty()) {
              cycleStack.pop();
            }
            LoopCycleCompute(forOp, g, context, moduleop, &scopInformation,
                             cycleStack, pluto_trans_Cost, unrollFlag, DFGNum,
                             maxDim, unroll_transNum, singleT,
                             scc_stmt_topSort_map);
            totalCycle += cycleStack.top();
            if (singleT.currentCompIndex != -1) {
              singleT.CompV[singleT.currentCompIndex]->Cost += cycleStack.top();
            }
          } else {
            if (i >= pluto_trans_Cost->start_scc_id &&
                i < countFor - ((scc_num - 1) - pluto_trans_Cost->end_scc_id)) {
              while (!cycleStack.empty()) {
                cycleStack.pop();
              }
              LoopCycleCompute(forOp, g, context, moduleop, &scopInformation,
                               cycleStack, pluto_trans_Cost, unrollFlag, DFGNum,
                               maxDim, unroll_transNum, singleT,
                               scc_stmt_topSort_map);
              totalCycle += cycleStack.top();
              if (singleT.currentCompIndex != -1) {
                singleT.CompV[singleT.currentCompIndex]->Cost +=
                    cycleStack.top();
              }
            }
            i++;
          }
        }
      }
    }

    //  if (mlir::AffineIfOp subifOp = dyn_cast<mlir::AffineIfOp>(op)) {
    //   if (mlir::AffineIfOp ifOp =
    //   dyn_cast<mlir::AffineIfOp>(subifOp->getParentOp())) {
    //     if (mlir::FuncOp funcOp =
    //     dyn_cast<mlir::FuncOp>(ifOp->getParentOp())) {
    //        for(subifOp.getBody)
    //     }
    //   }
    //  }

    if (mlir::CallOp callOp = dyn_cast<mlir::CallOp>(op)) {
      if (mlir::AffineIfOp ifOp =
              dyn_cast<mlir::AffineIfOp>(callOp->getParentOp())) {
        if (mlir::FuncOp funcOp = dyn_cast<mlir::FuncOp>(ifOp->getParentOp())) {
          // LoopCycleCompute(ifOp,&cycle);
          totalCycle += Tf;
          totalCycle += 4;
          if (singleT.currentCompIndex != -1) {
            singleT.CompV[singleT.currentCompIndex]->Cost += Tf;
            singleT.CompV[singleT.currentCompIndex]->Cost += +4;
          }
        }
      }
    }
  });
}

/*

auto varOps = entryBlock.getOps<spirv::GlobalVariableOp>();
  for (spirv::GlobalVariableOp gvOp : varOps) {
     // process each GlobalVariable Operation in the block.
     ...
  }

  getFunction().walk([](LinalgOp linalgOp) {
    // process LinalgOp `linalgOp`.
  });

WalkResult result = getFunction().walk([&](AllocOp allocOp) {
    if (!isValid(allocOp))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    // One alloc wasn't matching.
    ...
*/
void initial_Pluto_Trans(plutoCost_Matrix *pluto_trans, PlutoProg *tempprog,
                         int n, partitionCut *parC) {
  pluto_trans->prog = tempprog;
  pluto_trans->prog_id = n;
  pluto_trans->Cost = 0;
  pluto_trans->start_scc_id = 0;
  pluto_trans->end_scc_id = 0;
  pluto_trans->iter = 0;
  pluto_trans->has_trans = false;
  pluto_trans->dim = 0;
  pluto_trans->fusionCondition = parC->fusionCondition;
  pluto_trans->interchCondition = parC->interCondition;
  pluto_trans->is_succeed = true;
  pluto_trans->is_global = 1;
}

void initial_Context(PlutoContext *context, bool debug, bool parallelize) {
  context->options->silent = !debug;
  context->options->moredebug = debug;
  context->options->debug = debug;
  context->options->isldep = 1;
  context->options->readscop = 1;
  context->options->identity = 0;
  context->options->parallel = parallelize;
  context->options->unrolljam = 0;
  context->options->prevector = 0;
}

unsigned int dim_Sum(plutoCost_Matrix *pluto_trans) {
  unsigned int dim_sum = 0;
  for (int i = 0; i < pluto_trans->prog->nstmts; i++) {
    dim_sum += pluto_trans->prog->stmts[i]->dim;
  }
  return dim_sum;
}
/// The main function that implements the Pluto based optimization.
/// TODO: transform options?
static mlir::FuncOp plutoTransform(mlir::FuncOp f, OpBuilder &rewriter,
                                   std::string dumpClastAfterPluto,
                                   bool parallelize = false,
                                   bool debug = false) {
  int search_space = 0;
  llvm::raw_ostream &os = llvm::outs();
  PlutoContext *contextParent = pluto_context_alloc();
  OslSymbolTable srcTable, dstTable;

  std::unique_ptr<OslScop> scopParent = createOpenScopFromFuncOp(f, srcTable);
  if (!scopParent)
    return nullptr;
  if (scopParent->getNumStatements() == 0)
    return nullptr;

  osl_scop_print(stderr, scopParent->get());

  // Should use isldep, candl cannot work well for this case.
  initial_Context(contextParent, debug, parallelize);
  PlutoProg *prog = osl_scop_to_pluto_prog(scopParent->get(), contextParent);

  //*******************************************************************
  PlutoOptions *options = prog->context->options;

  unsigned dim_sum = 0;
  for (int i = 0; i < prog->nstmts; i++) {
    dim_sum += prog->stmts[i]->dim;
  }

  // if (!options->silent) {
  //   fprintf(stdout, "[Pluto] Number of statements: %d\n", prog->nstmts);
  //   fprintf(stdout, "[Pluto] Total number of loops: %u\n", dim_sum);
  //   fprintf(stdout, "[Pluto] Number of deps: %d\n", prog->ndeps);
  //   fprintf(stdout, "[Pluto] Maximum domain dimensionality:
  //   %d\n",prog->nvar); fprintf(stdout, "[Pluto] Number of parameters:
  //   %d\n",prog->npar);
  // }

  Stmt **stmts = prog->stmts;
  int nstmts = prog->nstmts;

  for (int i = 0; i < prog->ndeps; i++) {
    prog->deps[i]->satisfied = false;
  }
  // os << "\n============进入pluto变换==========\n";
  /* Create the data dependence graph */
  prog->ddg = ddg_create(prog);
  ddg_compute_scc(prog);

  int scc_num = get_scc_num(prog);
  int scc_dim[scc_num];
  int maxDim = 0;
  get_scc_dim(prog, scc_num, scc_dim);
  int topSortList[scc_num];
  find_topSort(prog, topSortList, scc_dim);
  for (int i = 0; i < scc_num; i++) {
    if (maxDim < scc_dim[i]) {
      maxDim = scc_dim[i];
    }
  }
  // scc_stmt_topSort
  scc_stmt_topSort scc_stmt_topSort_map;
  scc_stmt_topSort_map.scc_num = scc_num;
  for (int i = 0; i < prog->nstmts; i++) {
    scc_stmt_topSort_map.stmt_scc_map.insert(
        std::make_pair(prog->stmts[i]->id, prog->stmts[i]->scc_id));
  }
  for (int i = 0; i < scc_num; i++) {
    // topSortList store scc order
    int index = 0;
    for (int j = 0; j < scc_num; j++) {
      if (topSortList[j] == i) {
        index = j;
      }
    }
    scc_stmt_topSort_map.scc_top_map.insert(std::make_pair(i, index));
  }
  // scc_stmt_topSort_map.print();
  int prog_num = 0;
  for (int n = 0; n < scc_num; n++) {
    for (int m = 0; m < n + 1; m++) {
      if (m_n_scc_dim_dif(prog, topSortList, scc_dim, m, n)) {
        prog_num += scc_dim[topSortList[m]];
      }
    }
  }

  fuseComponent *Comp =
      (fuseComponent *)malloc(prog_num * sizeof(fuseComponent));
  int compCount = 0;
  for (int n = 0; n < scc_num; n++) {
    for (int m = 0; m < n + 1; m++) {
      if (m_n_scc_dim_dif(prog, topSortList, scc_dim, m, n)) {
        for (int k = 0; k < scc_dim[topSortList[m]]; k++) {
          Comp[compCount].start_scc_id = m;
          Comp[compCount].end_scc_id = n;
          Comp[compCount].innerIndex = k;
          Comp[compCount].flag = 0;
          Comp[compCount].transId = -1;
          compCount++;
        }
      }
    }
  }

  fuseComponent temp;
  // end_id sort
  for (int n = 0; n < prog_num - 1; n++) {
    for (int m = n + 1; m < prog_num; m++) {
      if (Comp[n].end_scc_id == Comp[m].end_scc_id) {
        if (Comp[n].start_scc_id > Comp[m].start_scc_id) {
          temp = Comp[m];
          Comp[m] = Comp[n];
          Comp[n] = temp;
        }
      } else if (Comp[n].end_scc_id > Comp[m].end_scc_id) {
        temp = Comp[m];
        Comp[m] = Comp[n];
        Comp[n] = temp;
      }
    }
  }
  //========================greedy-minimum-transformation==========================
  std::vector<singleTrans> TransV;
  int mintransIds = 0;
  while (compCount > 0) {
    singleTrans singleT;
    singleT.compNumber = 0;
    int lastIndex = -1;
    for (int n = 0; n < prog_num; n++) {
      if (Comp[n].flag != 1 && Comp[n].start_scc_id > lastIndex) {
        Comp[n].flag = 1;
        compCount--;
        lastIndex = Comp[n].end_scc_id;
        Comp[n].transId = mintransIds;
        Comp[n].Cost = 0;
        singleT.CompV.push_back(&Comp[n]);
        singleT.compNumber++;
      }
    }
    mintransIds++;
    singleT.scc_num = scc_num;
    TransV.push_back(singleT);
  }

  std::vector<partitionCut> parV;
  for (int n = 0; n < mintransIds; n++) {
    partitionCut p;
    p.initial(scc_num);
    p.fusCut(TransV[n].CompV, scc_num);
    // p.print(scc_num);
    parV.push_back(p);
    TransV[n].parC = p;
  }

  timeval Ourtv, Ourtv1;
  gettimeofday(&Ourtv, 0);

  PlutoProg **tempprog =
      (PlutoProg **)malloc(mintransIds * sizeof(PlutoProg *));
  for (int n = 0; n < mintransIds; n++) {
    tempprog[n] = osl_scop_to_pluto_prog(scopParent->get(), contextParent);
    for (int i = 0; i < tempprog[n]->ndeps; i++) {
      tempprog[n]->deps[i]->satisfied = false;
    }
    tempprog[n]->ddg = ddg_create(tempprog[n]);
    ddg_compute_scc(tempprog[n]);
  }

  plutoCost_Matrix *pluto_trans;
  pluto_trans =
      (plutoCost_Matrix *)malloc(mintransIds * sizeof(plutoCost_Matrix));

  mlir::ModuleOp moduleop;
  mlir::FuncOp g;
  for (int n = 0; n < mintransIds; n++) {
    //======================step1=====================
    initial_Pluto_Trans(&pluto_trans[n], tempprog[n], n, &TransV[n].parC);
    //======================step2=====================
    PlutoContext *context = pluto_context_alloc();
    OslSymbolTable srcTable, dstTable;
    std::unique_ptr<OslScop> scop = createOpenScopFromFuncOp(f, srcTable);
    if (!scop)
      return nullptr;
    if (scop->getNumStatements() == 0)
      return nullptr;
    osl_scop_print(stderr, scop->get());

    // Should use isldep, candl cannot work well for this case.
    initial_Context(context, debug, parallelize);
    PlutoOptions *options = pluto_trans[n].prog->context->options;
    unsigned dim_sum = dim_Sum(&pluto_trans[n]);

    moduleop = dyn_cast<mlir::ModuleOp>(f->getParentOp());
    std::string funcName;
    std::string optFlag = "opt";
    moduleop.walk([&](mlir::FuncOp f) {
      llvm::StringRef sourceFuncName = f.getName();
      // os<<sourceFuncName;
      funcName = (std::string)sourceFuncName;
      if (funcName.length() > 4) {
        if (funcName.substr(funcName.length() - 3, funcName.length() - 1) ==
            optFlag) {
          f.erase();
        }
      }
    });

    //======================step3=====================
    pluto_our_schedule_prog(&pluto_trans[n], topSortList, scc_dim);
    pluto_populate_scop(scop->get(), pluto_trans[n].prog, context);

    if (debug) {
      fflush(stderr);
      fflush(stdout);
    }
    osl_scop_print(stderr, scop->get());
    const char *dumpClastAfterPlutoStr = nullptr;
    if (!dumpClastAfterPluto.empty())
      dumpClastAfterPlutoStr = dumpClastAfterPluto.c_str();

    g = cast<mlir::FuncOp>(judgeFuncOpFromOpenScop(
        std::move(scop), moduleop, dstTable, rewriter.getContext(),
        pluto_trans[n].prog, dumpClastAfterPlutoStr, pluto_trans[n].prog_id));
    pluto_trans[n].is_global = 0;
    int unroll_transNum = 0;
    if (pluto_trans[n].is_succeed) {
      PerfProfiling(moduleop, g, rewriter.getContext(), &pluto_trans[n],
                    scc_num, maxDim, final_unroll_Switch, unroll_transNum,
                    TransV[n], scc_stmt_topSort_map);
      pluto_context_free(context);
      //os << "\ntransformation success\n";
      // TransV[n].print();
    } else {
      //os << "\ntransformation fail\n";
      pluto_context_free(context);
      if (TransV[n].compNumber == 1) {
        TransV[n].CompV[0]->Cost = LLONG_MAX / 2;
        //TransV[n].print();
      } else {
        std::vector<singleTrans> TransV_sub;
        PlutoProg **tempprog_sub =
            (PlutoProg **)malloc(TransV[n].compNumber * sizeof(PlutoProg *));
        for (int n = 0; n < TransV[n].compNumber; n++) {
          tempprog_sub[n] =
              osl_scop_to_pluto_prog(scopParent->get(), contextParent);
          for (int i = 0; i < tempprog_sub[n]->ndeps; i++) {
            tempprog_sub[n]->deps[i]->satisfied = false;
          }
          tempprog_sub[n]->ddg = ddg_create(tempprog_sub[n]);
          ddg_compute_scc(tempprog_sub[n]);
        }

        plutoCost_Matrix *pluto_trans_sub;
        pluto_trans_sub = (plutoCost_Matrix *)malloc(TransV[n].compNumber *
                                                     sizeof(plutoCost_Matrix));

        for (int i = 0; i < TransV[n].compNumber; i++) {
          singleTrans sT;
          TransV[n].CompV[i]->Cost = 0;
          TransV[n].currentCompIndex = -1;
          sT.CompV.push_back(TransV[n].CompV[i]);
          sT.compNumber = 1;
          sT.parC.initial(scc_num);
          sT.parC.fusCut(sT.CompV, scc_num);
          TransV_sub.push_back(sT);
        }

        for (int m = 0; m < TransV[n].compNumber; m++) {
          //======================sub_step1=====================
          initial_Pluto_Trans(&pluto_trans_sub[m], tempprog_sub[m], n * 100 + m,
                              &TransV_sub[m].parC);
          //======================sub_step2=====================
          PlutoContext *context_sub = pluto_context_alloc();
          OslSymbolTable srcTable_sub, dstTable_sub;
          std::unique_ptr<OslScop> scop_sub =
              createOpenScopFromFuncOp(f, srcTable);
          if (!scop_sub)
            return nullptr;
          if (scop_sub->getNumStatements() == 0)
            return nullptr;
          osl_scop_print(stderr, scop_sub->get());

          // Should use isldep, candl cannot work well for this case.
          initial_Context(context_sub, debug, parallelize);
          PlutoOptions *options_sub = pluto_trans_sub[m].prog->context->options;
          unsigned dim_sum_sub = dim_Sum(&pluto_trans_sub[m]);

          moduleop = dyn_cast<mlir::ModuleOp>(f->getParentOp());
          std::string funcName;
          std::string optFlag = "opt";
          moduleop.walk([&](mlir::FuncOp f) {
            llvm::StringRef sourceFuncName = f.getName();
            // os<<sourceFuncName;
            funcName = (std::string)sourceFuncName;
            if (funcName.length() > 4) {
              if (funcName.substr(funcName.length() - 3,
                                  funcName.length() - 1) == optFlag) {
                f.erase();
              }
            }
          });
          //======================sub_step3=====================
          pluto_our_schedule_prog(&pluto_trans_sub[m], topSortList, scc_dim);
          pluto_populate_scop(scop_sub->get(), pluto_trans_sub[m].prog,
                              context_sub);

          if (debug) {
            fflush(stderr);
            fflush(stdout);
          }
          osl_scop_print(stderr, scop_sub->get());
          const char *dumpClastAfterPlutoStr_sub = nullptr;
          if (!dumpClastAfterPluto.empty())
            dumpClastAfterPlutoStr_sub = dumpClastAfterPluto.c_str();

          g = cast<mlir::FuncOp>(judgeFuncOpFromOpenScop(
              std::move(scop_sub), moduleop, dstTable_sub,
              rewriter.getContext(), pluto_trans_sub[m].prog,
              dumpClastAfterPlutoStr_sub, pluto_trans_sub[m].prog_id));
          pluto_trans_sub[m].is_global = 0;
          int unroll_transNum_sub = 0;
          if (pluto_trans_sub[m].is_succeed) {
            PerfProfiling(moduleop, g, rewriter.getContext(),
                          &pluto_trans_sub[m], scc_num, maxDim,
                          final_unroll_Switch, unroll_transNum, TransV_sub[m],
                          scc_stmt_topSort_map);
            // os << "\nsub transformation success\n";
            // TransV_sub[m].print();
            pluto_context_free(context_sub);
          } else {
            pluto_context_free(context_sub);
            if (TransV_sub[m].compNumber == 1) {
              TransV_sub[m].CompV[0]->Cost = LLONG_MAX / 2;
            }
            // os << "\nsub transformation fail\n";
            // TransV_sub[m].print();
          }
        }
      }
    }
  }
//  gettimeofday(&Ourtv1, 0);
//   os << "--------------------- Transformation Time -----------------------\n";
//   printf("Transformation time: %lf \n",
//          1000 * (Ourtv1.tv_sec - Ourtv.tv_sec +
//                  (double)(Ourtv1.tv_usec - Ourtv.tv_usec) / 1000000));

  //======================step4_search_optimal_solution=====================
  std::vector<std::vector<fuseComponent *>> P;
  std::vector<long long> minCost;
  for (int k = 0; k < scc_num; k++) {
    minCost.push_back(LLONG_MAX / 2);
  }
  int endNode = 0;
  int startNode = 0;
  std::vector<fuseComponent *> cc;
  for (int k = 0; k < prog_num; k++) {
    startNode = Comp[k].start_scc_id;
    if (Comp[k].end_scc_id > endNode) {
      P.push_back(cc);
      endNode = Comp[k].end_scc_id;
      for (fuseComponent *v : cc) {
        cc.pop_back();
      }
    }

    if (startNode == 0) {
      if (minCost[endNode] > Comp[k].Cost) {
        minCost[endNode] = Comp[k].Cost;
        for (fuseComponent *v : cc) {
          cc.pop_back();
        }
        cc.push_back(&Comp[k]);
      }
    } else {
      if (minCost[endNode] > Comp[k].Cost + minCost[startNode - 1]) {
        minCost[endNode] = Comp[k].Cost + minCost[startNode - 1];
        for (fuseComponent *v : cc) {
          cc.pop_back();
        }
        for (fuseComponent *v : P[startNode - 1]) {
          cc.push_back(v);
        }
        cc.push_back(&Comp[k]);
      }
    }
  }
  P.push_back(cc);
  //===========================step 4 final solution============================
  plutoCost_Matrix *finalSolution;
  singleTrans finalTrans;
  finalTrans.CompV = P[scc_num - 1];
  finalTrans.parC.initial(scc_num);
  finalTrans.parC.fusCut(finalTrans.CompV, scc_num);
  // finalTrans.parC.print(scc_num);

  finalSolution = (plutoCost_Matrix *)malloc(1 * sizeof(plutoCost_Matrix));
  initial_Pluto_Trans(finalSolution, prog, -1, &finalTrans.parC);
  finalSolution->is_global = Experimental_option;

  moduleop = dyn_cast<mlir::ModuleOp>(f->getParentOp());
  std::string funcName;
  std::string optFlag = "opt";
  moduleop.walk([&](mlir::FuncOp f) {
    llvm::StringRef sourceFuncName = f.getName();
    funcName = (std::string)sourceFuncName;
    if (funcName.length() > 4) {
      if (funcName.substr(funcName.length() - 3, funcName.length() - 1) ==
          optFlag) {
        f.erase();
      }
    }
  });

  pluto_our_schedule_prog(finalSolution, topSortList, scc_dim);
  pluto_populate_scop(scopParent->get(), finalSolution->prog, contextParent);

  if (debug) {
    fflush(stderr);
    fflush(stdout);
  }

  osl_scop_print(stderr, scopParent->get());
  const char *dumpClastAfterPlutoStr = nullptr;
  if (!dumpClastAfterPluto.empty())
    dumpClastAfterPlutoStr = dumpClastAfterPluto.c_str();

  int count = 0;
  mlir::FuncOp finalG = cast<mlir::FuncOp>(finalFuncOpFromOpenScop(
      std::move(scopParent), moduleop, dstTable, rewriter.getContext(),
      finalSolution->prog, dumpClastAfterPlutoStr, count));

  int unroll_transNum1 = 0;
  for (fuseComponent *v : finalTrans.CompV) {
    v->Cost = 0;
  }
  PerfProfiling(moduleop, finalG, rewriter.getContext(), finalSolution, scc_num,
                maxDim, final_unroll_Switch, unroll_transNum1, finalTrans,
                scc_stmt_topSort_map);
  long long finalCost = 0;

  pluto_context_free(contextParent);
  //===========================end final solution===========================
  return finalG;
}

namespace {
class PlutoTransformPass
    : public mlir::PassWrapper<PlutoTransformPass,
                               OperationPass<mlir::ModuleOp>> {
  std::string dumpClastAfterPluto = "";
  bool parallelize = false;
  bool debug = false;

public:
  PlutoTransformPass() = default;
  PlutoTransformPass(const PlutoTransformPass &pass) {}
  PlutoTransformPass(const PlutoOptPipelineOptions &options)
      : dumpClastAfterPluto(options.dumpClastAfterPluto),
        parallelize(options.parallelize), debug(options.debug) {}

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();
    mlir::OpBuilder b(m.getContext());
    SmallVector<mlir::FuncOp, 8> funcOps;
    llvm::DenseMap<mlir::FuncOp, mlir::FuncOp> funcMap;

    m.walk([&](mlir::FuncOp f) {
      if (!f->getAttr("scop.stmt"))
        funcOps.push_back(f);
    });
    for (mlir::FuncOp f : funcOps)
      if (mlir::FuncOp g =
              plutoTransform(f, b, dumpClastAfterPluto, parallelize, debug)) {

        funcMap[f] = g;
        g.setPrivate();
      }
    // Replacing the original scop top-level function with the pluto
    // transformed result, such that the whole end-to-end optimization is
    // complete.
    m.walk([&](mlir::FuncOp f) {
      for (const auto &it : funcMap) {
        mlir::FuncOp from = it.first;
        mlir::FuncOp to = it.second;
        if (f != from)
          f.walk([&](mlir::CallOp op) {
            if (op.getCallee() == from.getName())
              op->setAttr("callee", b.getSymbolRefAttr(to.getName()));
          });
      }
    });
  }
};
} // namespace

// -------------------------- PlutoParallelizePass
// ----------------------------

/// Find a single affine.for with scop.parallelizable attr.
static mlir::AffineForOp findParallelizableLoop(mlir::FuncOp f) {
  mlir::AffineForOp ret = nullptr;
  f.walk([&ret](mlir::AffineForOp forOp) {
    if (!ret && forOp->hasAttr("scop.parallelizable"))
      ret = forOp;
  });
  return ret;
}

/// Turns a single affine.for with scop.parallelizable into affine.parallel.
/// The design of this function is almost the same as affineParallelize. The
/// differences are:
///
/// 1. It is not necessary to check whether the parentOp of a parallelizable
/// affine.for has the AffineScop trait.
static void plutoParallelize(mlir::AffineForOp forOp, OpBuilder b) {
  assert(forOp->hasAttr("scop.parallelizable"));

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointAfter(forOp);

  Location loc = forOp.getLoc();

  // If a loop has a 'max' in the lower bound, emit it outside the parallel
  // loop as it does not have implicit 'max' behavior.
  AffineMap lowerBoundMap = forOp.getLowerBoundMap();
  ValueRange lowerBoundOperands = forOp.getLowerBoundOperands();
  AffineMap upperBoundMap = forOp.getUpperBoundMap();
  ValueRange upperBoundOperands = forOp.getUpperBoundOperands();

  // Creating empty 1-D affine.parallel op.
  mlir::AffineParallelOp newPloop = b.create<mlir::AffineParallelOp>(
      loc, llvm::None, llvm::None, lowerBoundMap, lowerBoundOperands,
      upperBoundMap, upperBoundOperands, 1);
  // Steal the body of the old affine for op and erase it.
  newPloop.region().takeBody(forOp.region());

  for (auto user : forOp->getUsers()) {
    user->dump();
  }
  forOp.erase();
}

/// Need to check whether the bounds of the for loop are using top-level
/// values as operands. If not, then the loop cannot be directly turned into
/// affine.parallel.
static bool isBoundParallelizable(mlir::AffineForOp forOp, bool isUpper) {
  llvm::SmallVector<mlir::Value, 4> mapOperands =
      isUpper ? forOp.getUpperBoundOperands() : forOp.getLowerBoundOperands();

  for (mlir::Value operand : mapOperands)
    if (!isTopLevelValue(operand))
      return false;
  return true;
}
static bool isBoundParallelizable(mlir::AffineForOp forOp) {
  return isBoundParallelizable(forOp, true) &&
         isBoundParallelizable(forOp, false);
}

/// Iteratively replace affine.for with scop.parallelizable with
/// affine.parallel.
static void plutoParallelize(mlir::FuncOp f, OpBuilder b) {
  mlir::AffineForOp forOp = nullptr;
  while ((forOp = findParallelizableLoop(f)) != nullptr) {
    if (!isBoundParallelizable(forOp))
      llvm_unreachable("Loops marked as parallelizable should have "
                       "parallelizable bounds.");
    plutoParallelize(forOp, b);
  }
}

namespace {
/// Turn affine.for marked as scop.parallelizable by Pluto into actual
/// affine.parallel operation.
struct PlutoParallelizePass
    : public mlir::PassWrapper<PlutoParallelizePass,
                               OperationPass<mlir::FuncOp>> {
  void runOnOperation() override {
    FuncOp f = getOperation();
    OpBuilder b(f.getContext());
    plutoParallelize(f, b);
  }
};
} // namespace

void RfCgraTrans::registerPlutoTransformPass() {
  PassPipelineRegistration<PlutoOptPipelineOptions>(
      "pluto-opt", "Optimization implemented by PLUTO.",
      [](OpPassManager &pm, const PlutoOptPipelineOptions &pipelineOptions) {
        pm.addPass(std::make_unique<PlutoTransformPass>(pipelineOptions));
        pm.addPass(createCanonicalizerPass());
        if (pipelineOptions.generateParallel) {
          pm.addPass(std::make_unique<PlutoParallelizePass>());
          pm.addPass(createCanonicalizerPass());
        }
      });
}
