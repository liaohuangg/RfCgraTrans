#include "RfCgraTrans/Transforms/Schedule.h"
namespace RfCgraTrans {
class InnerUnroll {
public:
  InnerUnroll(MlirDFG &dfg, ScopInformation *scopInformation,int is_global,int &unroll_transNum);
  InnerUnroll(){this->need_unroll = false;};
  void searchUnroll();
  void genFile();
  void print_unroll();
  MlirDFG *dfg;
  ScopInformation *scopInformation;
  int betterII = 0;
  int original_II = 0;
  int original_DFGStep =0;
  int unrollNum = 0; 
  double EachSII = 0;
  int bankN = 0;
  int peN = 0;
  double bank_bottleneck;
  double pe_bottleneck;
  //0 is bank and 1 is pe
  int performance_bottleneck = -1;
  bool need_unroll = false;
  int is_global = 0;
  int unroll_transNum =0;
};
} // namespace RfCgraTrans