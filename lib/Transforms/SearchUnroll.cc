#include "RfCgraTrans/Transforms/SearchUnroll.h"

using namespace mlir;
using namespace llvm;
using namespace RfCgraTrans;
using namespace memref;
namespace RfCgraTrans {
InnerUnroll::InnerUnroll(MlirDFG &dfg, ScopInformation *scopInformation,int is_global,int &unroll_transNum) {
  this->dfg = &dfg;
  this->scopInformation = scopInformation;
  this->is_global = is_global;
  //unroll因子搜索
  this->need_unroll = false;
  this->searchUnroll();
  unroll_transNum += this->unroll_transNum;
}

void InnerUnroll::searchUnroll(){
  llvm::raw_ostream &os = llvm::outs();
  this->dfg->saveOriginalInformation();
  int original_bankN = this->dfg->LSNodeNum;
  int original_peN = this->dfg->PeNodeNum;
  int bankNGap = 0;
  int peNGap = 0;
  this->original_II = this->dfg->ResMII;
  this->original_DFGStep = this->dfg->DfgStep;
  int original_reuseData = this->dfg->reuseDataNum;
   for(int i = 2;i<=PERow*PERow;i+=2){
    this->unroll_transNum++;
    this->dfg->UnrollNum = i;
    this->dfg->cleanDFGInformation();
    this->dfg->createUnrollDFG(i); 
   
    for (Node *node : this->dfg->inDFGNodesList) {
      if (node->nodeType == ArrayLoad || node->nodeType == ArrayStore ||
          node->nodeType == noFitIndexLoad || node->nodeType == noFitIndexStore) {
        this->bankN++;
      } else {
        this->peN++;
      }
    } 
    if(i == 2){
      bankNGap = this->bankN - original_bankN;
      peNGap = this->peN  - original_peN;
    }
    
    //unroll之后的 ResMII 需要与之前的相同 
    if(this->dfg->TheoreticalII > this->original_II){
      //os<<"\n unrollN = "<<i<<" 之后ResMII比之前大,所以不进行unroll \n";
      this->dfg->restoreOriginalInformation();
      this->dfg->TheoreticalII = this->original_II;
      this->dfg->DfgStep = this->original_DFGStep;
      this->dfg->reuseDataNum = original_reuseData;
      break;
    }
    
    //unroll之后的 RecMII 需要与之前的相同 
    Schedule schedule(*(this->dfg));
    schedule.find_Simple_Schedule();
    if(this->dfg->RecMII != 0 ){
        //os<<"\n unrollN = "<<i<<" 之后RecMII比之前大,所以不进行unroll \n";
        this->dfg->restoreOriginalInformation();
        this->dfg->TheoreticalII = this->original_II;
        this->dfg->DfgStep = this->original_DFGStep;
        this->dfg->reuseDataNum = original_reuseData;
        break;
    }

    //os<<"\n打印unroll之后的DFG看看 unroll "<<i<<"\n";
    this->unrollNum = i;
    //os<<"\n unrollNum :"<<this->unrollNum<<"\n";
    this->betterII = this->dfg->TheoreticalII;
    this->need_unroll = true;

    if((i+2)*bankNGap > PERow*this->original_II || (i+2)*peNGap > PERow*PERow * this->original_II){
      break;
    } 
  }
  //调度
  // this->betterII = this->original_II;
}

void InnerUnroll::print_unroll(){
  llvm::raw_ostream &fos = llvm::outs();
  fos << "\n\n============第 " << this->dfg->dfg_id
      << "个 PNU 的 unrollinformation==============\n";
  if (this->need_unroll) {
    fos <<"\n需要进行unroll\n";
    //fos << "\nbankN " << this->bankN << "\n peN " << this->peN << "\n";
    fos <<" unrollNum "<<this->unrollNum<<"\n";
  } else {
     fos <<"\n不需要进行unroll\n";
  }
}
void InnerUnroll::genFile() {
  std::ofstream fos;
  fos.open("unrollInformation.out", std::ios::app);
  fos << "\n\n============第 " << this->dfg->dfg_id
      << "个 PNU 的 unrollinformation==============\n";
  if (this->need_unroll) {
    fos <<"\n需要进行unroll\n";
    fos << "\nbankN " << this->bankN << "\n peN " << this->peN << "\n";
    fos << "\nbank_bottleneck " << std::fixed << std::setprecision(2)
        << this->bank_bottleneck << "\n pe_bottleneck " << std::fixed
        << std::setprecision(2) << this->pe_bottleneck << "\n";
    fos << "\nperformance_bottleneck " << performance_bottleneck;
    if (performance_bottleneck == 0) {
      // bank unroll的搜索空间1 - PERow
      fos << "\n求解bank 的 unroll\n";
      fos << "\nbank unroll betterII " << std::fixed << std::setprecision(2)
          << this->betterII << " unrollN " << this->unrollNum << " EachSII "
          << std::fixed << std::setprecision(2) << this->EachSII << "\n";
    } else if (performance_bottleneck == 1) {
      // peA unroll的搜索空间1 - PERow *PERow
      fos << "\n求解pe 的 unroll\n";
      fos << "\npe unroll betterII " << std::fixed << std::setprecision(2)
          << this->betterII << " unrollN " << this->unrollNum << " EachSII "
          << std::fixed << std::setprecision(2) << this->EachSII << "\n";
    }
  } else {
     fos <<"\n不需要进行unroll\n";
  }
  fos.close();
}

} // namespace RfCgraTrans