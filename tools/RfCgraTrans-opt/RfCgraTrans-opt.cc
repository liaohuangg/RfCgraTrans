//===- RfCgraTrans-opt.cc - The RfCgraTrans optimisation tool -----------*- C++ -*-===//
//
// This file implements the RfCgraTrans optimisation tool, which is the RfCgraTrans
// analog of mlir-opt, used to drive compiler passes, e.g. for testing.
//
//===----------------------------------------------------------------------===//

#include "RfCgraTrans/Transforms/ExtractScopStmt.h"
#include "RfCgraTrans/Transforms/LoopAnnotate.h"
#include "RfCgraTrans/Transforms/LoopExtract.h"
#include "RfCgraTrans/Transforms/PlutoTransform.h"
#include "RfCgraTrans/Transforms/Reg2Mem.h"
#include "RfCgraTrans/Transforms/ScopStmtOpt.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include <sys/time.h>
using namespace llvm;
using namespace mlir;
using namespace RfCgraTrans;

static cl::opt<std::string>
    inputFilename(cl::Positional, cl::desc("<input file>"), cl::init("-"));

static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));

static cl::opt<bool>
    splitInputFile("split-input-file",
                   cl::desc("Split the input file into pieces and process each "
                            "chunk independently"),
                   cl::init(false));

static cl::opt<bool>
    verifyDiagnostics("verify-diagnostics",
                      cl::desc("Check that emitted diagnostics match "
                               "expected-* lines on the corresponding line"),
                      cl::init(false));

static cl::opt<bool>
    verifyPasses("verify-each",
                 cl::desc("Run the verifier after each transformation pass"),
                 cl::init(true));

static cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    cl::desc("Allow operation with no registered dialects"), cl::init(false));

int main(int argc, char *argv[]) {
 // timeval totaltv, totaltv1;    
 // gettimeofday(&totaltv, 0);
  InitLLVM y(argc, argv);

  DialectRegistry registry;

  // Register MLIR stuff
  registry.insert<StandardOpsDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::AffineDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

// Register the standard passes we want.
#include "mlir/Transforms/Passes.h.inc"
  registerCanonicalizerPass();
  registerCSEPass();
  registerInlinerPass();
  // Register RfCgraTrans specific passes.
  RfCgraTrans::registerPlutoTransformPass();
  RfCgraTrans::registerRegToMemPass();
  RfCgraTrans::registerExtractScopStmtPass();
  RfCgraTrans::registerScopStmtOptPasses();
  RfCgraTrans::registerLoopAnnotatePasses();
  RfCgraTrans::registerLoopExtractPasses();

  // Register any pass manager command line options.
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();

  // Register printer command line options.
  registerAsmPrinterCLOptions();

  PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, "RfCgraTrans pass driver\n");

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }
 //gettimeofday(&totaltv1, 0); 
  //os<<"---------------------for transformation-----------------------\n";
 // printf("total spand time: %lf \n" ,(totaltv1.tv_sec - totaltv.tv_sec + (double)(totaltv1.tv_usec - totaltv.tv_usec) / 1000000)); 

  return failed(MlirOptMain(output->os(), std::move(file), passPipeline,
                            registry, splitInputFile, verifyDiagnostics,
                            verifyPasses, allowUnregisteredDialects));
}
