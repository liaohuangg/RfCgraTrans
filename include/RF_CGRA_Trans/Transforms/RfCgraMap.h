#include "DFG.h"
#include "CGRA.h"
#include "config.h"
#include "Register.h"
#include "tool.h"
#include "GraphRegister.h"
#include "Path.h"
#include "RfCgraTrans/Transforms/DfgCreate.h"
namespace RfCgraTrans {
// class Node_Sub{
//    public:
// //    std::vector<Node *> NodesList;
// //    std::vector<Edge *> EdgesList;
// }

class RF_CGRAMap{
  public:
  //构造函数
  RF_CGRAMap(MlirDFG &mlirdfg,int solutionNum);
  void preData(MlirDFG &mlirdfg);
  void print_Data(MlirDFG &mlirdfg);
  int find_Map(MlirDFG &mlirdfg,int solutionNum);
  void show_map(MlirDFG &mlirdfg);
  //成员
  vector<vector<int>> DFG_node;
  //<NodeID,NodeIndex>
  map<int,int> DFGNodeID_NodeIndex;
  //<NodeIndex,NodeID>
  map<int,int> DFGNodeIndex_NodeID;
  int childNum;
  //1 成功  0 不成功
  int is_success;
};
}// namespace RfCgraTrans