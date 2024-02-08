#include "RfCgraTrans/Transforms/DfgCreate.h"
using namespace mlir;
using namespace llvm;
using namespace RfCgraTrans;
using namespace memref;
using namespace math;
/*
int* ps = new int ; // 使用new 申明变量空间
...
...
delete ps ; '// delete 删除所申明的变量空间
*/
namespace RfCgraTrans {

void partitionCut::fusCut(std::vector<fuseComponent *> CompV, int scc_num) {
  //pair<start, end>
  std::vector<std::pair<int, int>> fuseNode;
  for (int i = 0; i < CompV.size(); i++) {
    fuseNode.push_back(std::make_pair(CompV[i]->start_scc_id,CompV[i]->end_scc_id)); 
  }
  int count = 0;
  int flag=0;
  for (int j = 0; j < scc_num; j++) {
    if(flag<CompV.size() && j>=fuseNode[flag].first && j<=fuseNode[flag].second){
      this->fusionCondition[j]=count;
      this->interCondition[j]=CompV[flag]->innerIndex;
      if(j==fuseNode[flag].second){
        flag++;
        count++;
      }
    }else{
      this->fusionCondition[j]=count;
      this->interCondition[j]=0;
      count++; 
    }
  }
}

void partitionCut::initial(int scc_num){
  this->fusionCondition = (int *)malloc(scc_num * sizeof(int));
  this->interCondition = (int *)malloc(scc_num * sizeof(int));
  for (int i = 0; i < scc_num; i++) {
    this->fusionCondition[i] = 0;
    this->interCondition[i] = 0;
  }
}

void partitionCut::print(int scc_num){
   llvm::raw_ostream &os = llvm::outs();
   os<<"\nfus\n";
  for(int j = 0; j < scc_num; j++){
    os<<this->fusionCondition[j]<<" ";
  }
  os<<"\ninter\n";
  for(int j = 0; j < scc_num; j++){
    os<<this->interCondition[j]<<" ";
  }
  // os<<"\n\n\n\n";
}

void singleTrans::print() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\ncomp";
  for (fuseComponent *c : CompV) {
    os << "\nstartScc " << c->start_scc_id << " endScc " << c->end_scc_id
       <<" inner "<<c->innerIndex<< " \n";
    os << "Cost " << c->Cost << "\n";
  }
}

void singleTrans::flagComp(std::vector<int> stmtIdV,
                           scc_stmt_topSort scc_stmt_topSort_map) {
  llvm::raw_ostream &os = llvm::outs();
  int index = 0;
  int stsid = 0;
  std::vector<int> topIDV;
  for (int i : stmtIdV) {
    stsid = i;
    int index = scc_stmt_topSort_map.stmt_scc_map.find(i)->second;
    topIDV.push_back(scc_stmt_topSort_map.scc_top_map.find(index)->second);
  }
  std::vector<int> a;
  int flag = -1;
  for (int i : topIDV) {
    int count = 0;
    for (fuseComponent *c : CompV) {
      if (i >= c->start_scc_id && i <= c->end_scc_id) {
        a.push_back(count);
        flag = count;
      }
      count++;
    }
  }

  for (int i : a) {
    if (flag != i) {
      os << "\n problem! \n";
    }
  }

  if (flag != -1) {
    this->currentCompIndex = flag;
  } else {
    this->currentCompIndex = -1;
  }
}

DFGList::DFGList(MapVector<int, SmallVector<mlir::CallOp, 8U>> map) {
  this->list = map;
  dfgNum = 0;
};

void DFGList::insert(SmallVector<mlir::CallOp, 8U> callOpVec) {
  this->list.insert(std::make_pair(this->dfgNum, callOpVec));
  this->dfgNum++;
}

void Node::print_node() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n Node";
  os << this->nodeID;
  os << ": label ";
  os << this->label;
  os << "\n";
}

void Node::print_node(raw_fd_ostream &os) {
  os << "\n Node";
  os << this->nodeID;
  os << ": label ";
  os << this->label;
  os << "\n";
}

LoadNode::LoadNode(const LoadNode *n) {
  this->nodeType = n->nodeType;
  this->label = n->label;
  this->nodePriority = n->nodePriority;
  this->SSA = n->SSA;
  this->timeStep = n->timeStep;
  this->earliestTimeStep = n->earliestTimeStep;
  this->latestTimeStep = n->latestTimeStep;
  this->nodeID = n->nodeID;
  this->reuseStep = n->reuseStep;
  this->belongS = n->belongS;
  for (int i = 0; i < n->NodeShift.size(); i++) {
    this->NodeShift.push_back(n->NodeShift[i]);
  }
  for (int i = 0; i < n->iterFlag.size(); i++) {
    this->iterFlag.push_back(n->iterFlag[i]);
  }
  for (int i = 0; i < n->iterOrder.size(); i++) {
    this->iterOrder.push_back(n->iterOrder[i]);
  }
  this->innerestIndex = n->innerestIndex;
  this->nodeInDFG = n->nodeInDFG;
  this->reuseflag = n->reuseflag;
  this->ArrayOp = n->ArrayOp;
  this->load = n->load;
  for (int i = 0; i < n->stmtParamLocationVec.size(); i++) {
    this->stmtParamLocationVec.push_back(n->stmtParamLocationVec[i]);
  }
}

StoreNode::StoreNode(const StoreNode *n) {
  this->nodeType = n->nodeType;
  this->label = n->label;
  this->nodePriority = n->nodePriority;
  this->SSA = n->SSA;
  this->timeStep = n->timeStep;
  this->earliestTimeStep = n->earliestTimeStep;
  this->latestTimeStep = n->latestTimeStep;
  this->nodeID = n->nodeID;
  this->reuseStep = n->reuseStep;
  this->belongS = n->belongS;
  for (int i = 0; i < n->NodeShift.size(); i++) {
    this->NodeShift.push_back(n->NodeShift[i]);
  }
  for (int i = 0; i < n->iterFlag.size(); i++) {
    this->iterFlag.push_back(n->iterFlag[i]);
  }
  for (int i = 0; i < n->iterOrder.size(); i++) {
    this->iterOrder.push_back(n->iterOrder[i]);
  }
  this->innerestIndex = n->innerestIndex;
  this->nodeInDFG = n->nodeInDFG;
  this->reuseflag = n->reuseflag;
  this->ArrayOp = n->ArrayOp;
  this->store = n->store;
  for (int i = 0; i < n->stmtParamLocationVec.size(); i++) {
    this->stmtParamLocationVec.push_back(n->stmtParamLocationVec[i]);
  }
}

CstNode::CstNode(const CstNode *n) {
  this->nodeType = n->nodeType;
  this->label = n->label;
  this->nodePriority = n->nodePriority;
  this->SSA = n->SSA;
  this->timeStep = n->timeStep;
  this->earliestTimeStep = n->earliestTimeStep;
  this->latestTimeStep = n->latestTimeStep;
  this->nodeID = n->nodeID;
  this->reuseStep = n->reuseStep;
  this->belongS = n->belongS;
  for (int i = 0; i < n->NodeShift.size(); i++) {
    this->NodeShift.push_back(n->NodeShift[i]);
  }
  for (int i = 0; i < n->iterFlag.size(); i++) {
    this->iterFlag.push_back(n->iterFlag[i]);
  }
  for (int i = 0; i < n->iterOrder.size(); i++) {
    this->iterOrder.push_back(n->iterOrder[i]);
  }
  this->innerestIndex = n->innerestIndex;
  this->nodeInDFG = n->nodeInDFG;
  this->reuseflag = n->reuseflag;
}

AddNode::AddNode(const AddNode *n) {
  this->nodeType = n->nodeType;
  this->label = n->label;
  this->nodePriority = n->nodePriority;
  this->SSA = n->SSA;
  this->timeStep = n->timeStep;
  this->earliestTimeStep = n->earliestTimeStep;
  this->latestTimeStep = n->latestTimeStep;
  this->nodeID = n->nodeID;
  this->reuseStep = n->reuseStep;
  this->belongS = n->belongS;
  for (int i = 0; i < n->NodeShift.size(); i++) {
    this->NodeShift.push_back(n->NodeShift[i]);
  }
  for (int i = 0; i < n->iterFlag.size(); i++) {
    this->iterFlag.push_back(n->iterFlag[i]);
  }
  for (int i = 0; i < n->iterOrder.size(); i++) {
    this->iterOrder.push_back(n->iterOrder[i]);
  }
  this->innerestIndex = n->innerestIndex;
  this->nodeInDFG = n->nodeInDFG;
  this->reuseflag = n->reuseflag;
}

SubNode::SubNode(const SubNode *n) {
  this->nodeType = n->nodeType;
  this->label = n->label;
  this->nodePriority = n->nodePriority;
  this->SSA = n->SSA;
  this->timeStep = n->timeStep;
  this->earliestTimeStep = n->earliestTimeStep;
  this->latestTimeStep = n->latestTimeStep;
  this->nodeID = n->nodeID;
  this->reuseStep = n->reuseStep;
  this->belongS = n->belongS;
  for (int i = 0; i < n->NodeShift.size(); i++) {
    this->NodeShift.push_back(n->NodeShift[i]);
  }
  for (int i = 0; i < n->iterFlag.size(); i++) {
    this->iterFlag.push_back(n->iterFlag[i]);
  }
  for (int i = 0; i < n->iterOrder.size(); i++) {
    this->iterOrder.push_back(n->iterOrder[i]);
  }
  this->innerestIndex = n->innerestIndex;
  this->nodeInDFG = n->nodeInDFG;
  this->reuseflag = n->reuseflag;
}

MulNode::MulNode(const MulNode *n) {
  this->nodeType = n->nodeType;
  this->label = n->label;
  this->nodePriority = n->nodePriority;
  this->SSA = n->SSA;
  this->timeStep = n->timeStep;
  this->earliestTimeStep = n->earliestTimeStep;
  this->latestTimeStep = n->latestTimeStep;
  this->nodeID = n->nodeID;
  this->reuseStep = n->reuseStep;
  this->belongS = n->belongS;
  for (int i = 0; i < n->NodeShift.size(); i++) {
    this->NodeShift.push_back(n->NodeShift[i]);
  }
  for (int i = 0; i < n->iterFlag.size(); i++) {
    this->iterFlag.push_back(n->iterFlag[i]);
  }
  for (int i = 0; i < n->iterOrder.size(); i++) {
    this->iterOrder.push_back(n->iterOrder[i]);
  }
  this->innerestIndex = n->innerestIndex;
  this->nodeInDFG = n->nodeInDFG;
  this->reuseflag = n->reuseflag;
}

DivNode::DivNode(const DivNode *n) {
  this->nodeType = n->nodeType;
  this->label = n->label;
  this->nodePriority = n->nodePriority;
  this->SSA = n->SSA;
  this->timeStep = n->timeStep;
  this->earliestTimeStep = n->earliestTimeStep;
  this->latestTimeStep = n->latestTimeStep;
  this->nodeID = n->nodeID;
  this->reuseStep = n->reuseStep;
  this->belongS = n->belongS;
  for (int i = 0; i < n->NodeShift.size(); i++) {
    this->NodeShift.push_back(n->NodeShift[i]);
  }
  for (int i = 0; i < n->iterFlag.size(); i++) {
    this->iterFlag.push_back(n->iterFlag[i]);
  }
  for (int i = 0; i < n->iterOrder.size(); i++) {
    this->iterOrder.push_back(n->iterOrder[i]);
  }
  this->innerestIndex = n->innerestIndex;
  this->nodeInDFG = n->nodeInDFG;
  this->reuseflag = n->reuseflag;
}

SqrtNode::SqrtNode(const SqrtNode *n) {
  this->nodeType = n->nodeType;
  this->label = n->label;
  this->nodePriority = n->nodePriority;
  this->SSA = n->SSA;
  this->timeStep = n->timeStep;
  this->earliestTimeStep = n->earliestTimeStep;
  this->latestTimeStep = n->latestTimeStep;
  this->nodeID = n->nodeID;
  this->reuseStep = n->reuseStep;
  this->belongS = n->belongS;
  for (int i = 0; i < n->NodeShift.size(); i++) {
    this->NodeShift.push_back(n->NodeShift[i]);
  }
  for (int i = 0; i < n->iterFlag.size(); i++) {
    this->iterFlag.push_back(n->iterFlag[i]);
  }
  for (int i = 0; i < n->iterOrder.size(); i++) {
    this->iterOrder.push_back(n->iterOrder[i]);
  }
  this->innerestIndex = n->innerestIndex;
  this->nodeInDFG = n->nodeInDFG;
  this->reuseflag = n->reuseflag;
}
Node::Node(const Node *n) {
  this->nodeType = n->nodeType;
  this->label = n->label;
  this->nodePriority = n->nodePriority;
  this->SSA = n->SSA;
  this->timeStep = n->timeStep;
  this->earliestTimeStep = n->earliestTimeStep;
  this->latestTimeStep = n->latestTimeStep;
  this->nodeID = n->nodeID;
  this->reuseStep = n->reuseStep;
  this->belongS = n->belongS;
  for (int i = 0; i < n->NodeShift.size(); i++) {
    this->NodeShift.push_back(n->NodeShift[i]);
  }
  for (int i = 0; i < n->iterFlag.size(); i++) {
    this->iterFlag.push_back(n->iterFlag[i]);
  }
  for (int i = 0; i < n->iterOrder.size(); i++) {
    this->iterOrder.push_back(n->iterOrder[i]);
  }
  this->innerestIndex = n->innerestIndex;
  this->nodeInDFG = n->nodeInDFG;
  this->reuseflag = n->reuseflag;
}

Edge::Edge(int begin, int end) {
  this->begin = begin;
  this->end = end;
  this->edgeType = Normal;
  this->edgeID = edgeID;
}

void Edge::print_edge() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\nEdge" << this->edgeID << ": begin " << this->begin << " end "
     << this->end << " min " << this->min << " dif " << this->dif << " unroll "
     << this->UnrollID << " type ";
  if (this->edgeType == Normal) {
    os << "Normal\n";
  } else if (this->edgeType == SLReuse) {
    os << "SLReuse\n";
  } else if (this->edgeType == LLReuse) {
    os << "LLReuse\n";
  } else if (this->edgeType == VarL) {
    os << "VarL\n";
  } else if (this->edgeType == VarS) {
    os << "VarS\n";
  } else if (this->edgeType == VarSLDepen) {
    os << "VarSLDepen\n";
  } else if (this->edgeType == RecSLDepen) {
    os << "RecSLDepen\n";
  } else {
    os << "Delete\n";
  }
}

Edge::Edge(const Edge *e) {
  this->begin = e->begin;
  this->dif = e->dif;
  this->min = e->min;
  this->end = e->end;
  this->edgeID = e->edgeID;
  this->edgeType = e->edgeType;
}

void Edge::print_edge(raw_fd_ostream &os) {
  os << "\nEdge" << this->edgeID << ": begin " << this->begin << " end "
     << this->end << " min " << this->min << " dif " << this->dif << " type ";
  if (this->edgeType == Normal) {
    os << "Normal\n";
  } else if (this->edgeType == SLReuse) {
    os << "SLReuse\n";
  } else if (this->edgeType == LLReuse) {
    os << "LLReuse\n";
  } else if (this->edgeType == VarL) {
    os << "VarL\n";
  } else if (this->edgeType == VarS) {
    os << "VarS\n";
  } else if (this->edgeType == VarSLDepen) {
    os << "VarSLDepen\n";
  } else if (this->edgeType == RecSLDepen) {
    os << "RecSLDepen\n";
  } else {
    os << "Delete\n";
  }
}

void AddNode::print_node() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n————————————AddNode————————————\n";
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\n";
  this->SSA.print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}

void AddNode::print_node(raw_fd_ostream &os) {
  os << "\n————————————AddNode————————————\n";
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\n";
  this->SSA.print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}

void SqrtNode::print_node() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n————————————SqrtNode————————————\n";
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\n";
  this->SSA.print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}

void SqrtNode::print_node(raw_fd_ostream &os) {
  os << "\n————————————SqrtNode————————————\n";
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\n";
  this->SSA.print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}

void SubNode::print_node() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n————————————SubNode————————————\n";
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\n";
  this->SSA.print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}

void SubNode::print_node(raw_fd_ostream &os) {
  os << "\n————————————SubNode————————————\n";
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\n";
  this->SSA.print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}
void DivNode::print_node() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n————————————DivNode————————————\n";
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\n";
  this->SSA.print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}

void DivNode::print_node(raw_fd_ostream &os) {
  os << "\n————————————DivNode————————————\n";
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\n";
  this->SSA.print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}

void MulNode::print_node() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n————————————MulNode————————————\n";
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\n";
  this->SSA.print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}

void MulNode::print_node(raw_fd_ostream &os) {
  os << "\n————————————MulNode————————————\n";
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\n";
  this->SSA.print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}

void CstNode::print_node() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n————————————CstNode————————————\n";
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\n";
  this->SSA.print(os);
  os << "\n————————————————————————————————\n";
}

void CstNode::print_node(raw_fd_ostream &os) {
  os << "\n————————————CstNode————————————\n";
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\n";
  this->SSA.print(os);
  os << "\n————————————————————————————————\n";
}
// new class LoadNode
//  Node
LoadNode::LoadNode(mlir::AffineLoadOp load, mlir::Operation *alloc,
                   int nodeIDCount, ScopInformation *scopInformation,
                   StmtInfomation *s) {
  llvm::raw_ostream &os = llvm::outs();
  this->ArrayOp = alloc;
  this->SSA = load.getResult();
  this->nodeID = nodeIDCount;
  if (mlir::memref::AllocaOp allocaOp =
          dyn_cast<mlir::memref::AllocaOp>(this->ArrayOp)) {
    this->nodeType = VarLoad;
    this->label = "VarLoad";
  } else if (mlir::memref::AllocOp allocOp =
                 dyn_cast<mlir::memref::AllocOp>(this->ArrayOp)) {
    this->nodeType = ArrayLoad;
    this->label = "ArrayLoad";
    this->getStmtParamLocationVec(load, s);
    this->getLoadShift(scopInformation, load, s);
  }
  this->load = load;
}

LoadNode::LoadNode(mlir::Operation *alloc, int nodeIDCount,
                   ScopInformation *scopInformation, StmtInfomation *s) {
  llvm::raw_ostream &os = llvm::outs();
  this->ArrayOp = alloc;
  this->SSA = load.getResult();
  this->nodeID = nodeIDCount;
  this->nodeType = VarLoad;
  this->label = "VarLoad";
}

void LoadNode::print_node() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n————————————loadNode————————————\n";
  os << "\nNodeShift\n";
  for (int i = 0; i < this->NodeShift.size(); i++) {
    os << this->NodeShift[i] << " ";
  }
  os << "\niterFlag\n";
  for (int i = 0; i < this->iterFlag.size(); i++) {
    os << this->iterFlag[i] << " ";
  }
  os << "\niterOrder\n";
  for (int i = 0; i < this->iterOrder.size(); i++) {
    os << this->iterOrder[i] << " ";
  }
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\nNodeType\n";
  if (this->nodeType == ArrayLoad) {
    os << "ArrayLoad";
  } else if (this->nodeType == VarLoad) {
    os << "VarLoad";
  } else if (this->nodeType == noFitIndexLoad) {
    os << "noFitIndexLoad";
  }
  os << "\nArray\n";
  this->ArrayOp->print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}

void LoadNode::print_node(raw_fd_ostream &os) {
  os << "\n————————————loadNode————————————\n";
  os << "\nNodeShift\n";
  for (int i = 0; i < this->NodeShift.size(); i++) {
    os << this->NodeShift[i] << " ";
  }
  os << "\niterFlag\n";
  for (int i = 0; i < this->iterFlag.size(); i++) {
    os << this->iterFlag[i] << " ";
  }
  os << "\niterOrder\n";
  for (int i = 0; i < this->iterOrder.size(); i++) {
    os << this->iterOrder[i] << " ";
  }
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\nNodeType\n";
  if (this->nodeType == ArrayLoad) {
    os << "ArrayLoad";
  } else if (this->nodeType == VarLoad) {
    os << "VarLoad";
  } else if (this->nodeType == noFitIndexLoad) {
    os << "noFitIndexLoad";
  }
  os << "\nArray\n";
  this->ArrayOp->print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}
// Update
// s Pass in the position of the AffineMap input starting from 0
// callee arg location stmtParamLocationVec(0 1 2 3 4 5) (X i A j k B)
// A[j][i][k] -1 1-1 0 2-1 (location of load access function)
// Form the above correspondence
void LoadNode::getStmtParamLocationVec(mlir::AffineLoadOp load,
                                       StmtInfomation *s) {
  SmallVector<int, 8U> interVector;
// step1 Determine whether A[i][i] this is the total situation
// Get the position of the s parameter corresponding to the parameter passed by AffineMap
  for (mlir::Value value : load.getMapOperands()) {
    mlir::BlockArgument *blockArgument = (mlir::BlockArgument *)&value;
    interVector.push_back(blockArgument->getArgNumber());
  }
  for (int i = 0; i < s->stmtCallee.getNumArguments();
       i++) { // For each position of the parameter s passed 0, 1, 2, 3, 4, 5
    int flag = 0;
    for (int j = 0; j < interVector.size();
         j++) { // Find the corresponding position in AffineMap: only the index position has it, meref does not
      if (i == interVector[j]) {
        this->stmtParamLocationVec.push_back(
            j); // Found the position of the i th parameter (starting with 0) corresponding to the AffineMap result (starting with 0)
        flag = 1;
        break;
      }
    }
    if (flag == 0)
      this->stmtParamLocationVec.push_back(-1); // Invalid index: meref does not have a corresponding position
  }
}

// Get offset values related
// Simple case ()[s0 s1 s2] -> (s0, s1+4, s2+2)
// (without) (s0 and s1) - > (s0, 4, s1) () (s0 and s1) - > (s0 and s1, 4)
// Last array_num > inputs_num
void LoadNode::getLoadShift(ScopInformation *scopInformation,
                            mlir::AffineLoadOp load, StmtInfomation *s) {
  int array_num = load.getAffineMap().getNumResults(); // 2
  int inputs_num = load.getAffineMap().getNumInputs(); // 1
  std::vector<mlir::Attribute, std::allocator<mlir::Attribute>> attrVec;
  llvm::SmallVector<int64_t>
      results; 
// And what you get in the results is
// The value after the map (can be subtracted from initial_results to get the offset result)
  llvm::SmallVector<int64_t> initial_results;
  for (int i = 0; i < inputs_num; i++) {
    attrVec.push_back(scopInformation->BoundAttrVec[0]);
  }
  llvm::ArrayRef<mlir::Attribute> operandConstants(attrVec);
  load.getAffineMap().partialConstantFold(operandConstants,
                                          &results);

  for (int i = 0; i < results.size(); i++) {
    initial_results.push_back(scopInformation->BoundIntVec[0]);
  }
  
// There are two situations
// 1. The load (s0) [4] [2] () (s0) - > (s0, 4, 2) the array subscript situation will be expanded
// 2. If S0 passed to %C4 is a constant, this case is not extended

// Allnode: Get variables s0,s1 sorted by AffineMap input
  std::vector<mlir::AffineExpr> Allnode;
  load.getAffineMap().walkExprs([&Allnode](mlir::AffineExpr expr) {
    if (expr.getKind() == mlir::AffineExprKind::SymbolId) {
      int flag = 1;
      for (auto node : Allnode) {
        if (node == expr) {
          flag = 0;
          break;
        }
      }
      if (flag)
        Allnode.push_back(expr);
    }
  // The int value of AllnodeMap is not used to determine whether A[i][i] condition exists. Therefore, the expr condition found by AllnodeMap is not processed
  });
// Extend initial_result and find after_index in the result with the passed argument
// Only for combinations between variables after affine_map, only for ()[s0] -> (s0, 4, 2)
// No ()[s0, s1] -> (s0+s1, 4, 2) os << "\ninitial_result and after_index\n";
  llvm::SmallVector<int64_t>
      after_index; 
  llvm::SmallVector<int64_t> after_results;                    
  if (inputs_num == array_num) { 
    for (int i = 0; i < inputs_num; i++) {
      after_results.push_back(initial_results[i]); 
      after_index.push_back(i);                    
    }
  } else if (inputs_num < array_num) {
// ()[s0, s1] -> (s0, s0, s1)
// Therefore, if type is not constant, we need to determine which is it, not directly array_index:
// Obtain the index before AffineMap corresponding to each variable of the array (post-affinemap)
    llvm::SmallVector<int64_t> array_index;
    for (int i = 0; i < array_num; i++) {
      mlir::AffineExpr expr = load.getAffineMap().getResult(i);
      if (expr.getKind() == mlir::AffineExprKind::Constant) {
        after_results.push_back(0);
        array_index.push_back(-1);
        continue;
      }
      int index = -1;
      expr.walk([&Allnode, &index](mlir::AffineExpr expr) {
        for (int j = 0; j < Allnode.size(); j++)
          if (Allnode[j] == expr) {
            index = j;
            return;
          }
      });
       llvm::raw_ostream &os = llvm::outs();
      if (index == -1)
        os << "\nnot find\n"; // 可能会segement default
      after_results.push_back(initial_results[index]);
      array_index.push_back(index);
    }
 
    for (int i = 0; i < inputs_num; i++) {
      for (int j = 0; j < array_index.size(); j++) {
        if (array_index[j] == i) {
          after_index.push_back(j);
          break; 
        }
      }
    }
  }
// start looking for fill the offset vector associated 
// Offset value of the constant: 1. + 2
// iter offset: the offset value passed in s + the offset value in affinemap
  for (int i = 0; i < results.size();
       i++) { // For each bit of the array (AffineMap result)
// cons: Two types
// 1. The result of mapping is cons
// 2.s is passed in as cons
// First find 1
    mlir::AffineExpr affine3 = load.getAffineMap().getResult(i);
    if (affine3.getKind() == mlir::AffineExprKind::Constant) {
      this->NodeShift.push_back(results[i]);
      this->iterFlag.push_back(0);          
      this->iterOrder.push_back(-1); 
    } else { 
      this->NodeShift.push_back(results[i] - after_results[i]);
      this->iterFlag.push_back(1);   
      this->iterOrder.push_back(-1); 
    }
  }

  for (int i = 0; i < s->ConsArgIndexVec.size(); i++) {
    int index =
        s->ConsArgIndexVec
            [i];
    index =
        this->stmtParamLocationVec
            [index]; 
    if (index == -1) { 
      continue;
    }
    index = after_index
        [index]; 
    this->iterFlag[index] = 0; 
    this->NodeShift[index] =
        this->NodeShift[index] +
        s->ConsValueVec[i]; 
  }

  std::map<int, int>
      stmt_iterShift; 
  int flag = 0;
  for (int i = 0; i < s->IterArgIndexVec.size();
       i++) { 
    int index =
        s->IterArgIndexVec
            [i]; 
    index =
        this->stmtParamLocationVec
            [index]; 
    if (index == -1) { 
      continue;
    }
    index =
        after_index[index]; 
    this->iterOrder[index] = i;
    this->NodeShift[index] =
        this->NodeShift[index] +
        s->IterShiftVec[i]; 
    stmt_iterShift.insert(std::make_pair(index, s->IterShiftVec[i]));
    if (!flag) { 
      this->innerestIndex = index;
      flag = 1;
    }
  }
  if (!flag)
    this->innerestIndex = -1; 

  //Handle ()[s0] -> [s0,s0]
  int iterFlagCount = 0;
  int iterOrderCount = 0;
  for (int i = 0; i < results.size(); i++) {
    if (this->iterFlag[i] == 1) {
      iterFlagCount++; 
    }
    if (this->iterOrder[i] != -1) {
      iterOrderCount++;
    }
   // This case is not found for A[i][i]
    if (this->iterFlag[i] == 1 && this->iterOrder[i] == -1) {
      mlir::AffineExpr affine3 = load.getAffineMap().getResult(i);
      int iterSame = -1;
      affine3.walk([&Allnode, &iterSame](mlir::AffineExpr expr) {
        llvm::raw_ostream &os = llvm::outs();
        for (int j = 0; j < Allnode.size(); j++)
          if (Allnode[j] == expr) {
            iterSame = j;
            return;
          }
      });
      iterSame = after_index[iterSame];
      iterOrder[i] = iterOrder[iterSame];
      NodeShift[i] = results[i] - after_results[i] + stmt_iterShift[iterSame];
    }
  }
  if (iterFlagCount > iterOrderCount) {
   // A[i][i] is marked and cannot be reused
    this->nodeType = noFitIndexLoad;
  }
}

StoreNode::StoreNode(mlir::AffineStoreOp store, mlir::Operation *alloc,
                     int nodeIDCount, ScopInformation *scopInformation,
                     StmtInfomation *s) {
  llvm::raw_ostream &os = llvm::outs();
  this->ArrayOp = alloc;
  // this->SSA = store.getODSResults(0);
  this->nodeID = nodeIDCount;
  if (mlir::memref::AllocaOp allocaOp =
          dyn_cast<mlir::memref::AllocaOp>(this->ArrayOp)) {
    this->nodeType = VarStore;
    this->label = "VarStore";
  } else if (mlir::memref::AllocOp allocOp =
                 dyn_cast<mlir::memref::AllocOp>(this->ArrayOp)) {
    this->nodeType = ArrayStore;
    this->label = "ArrayStore";
    this->getStmtParamLocationVec(store, s);
    this->getStoreShift(scopInformation, store, s);
  }
  this->store = store;
  // this->print_node();
}

void StoreNode::print_node() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n————————————storeNode————————————\n";
  os << "\nNodeShift\n";
  for (int i = 0; i < this->NodeShift.size(); i++) {
    os << this->NodeShift[i] << " ";
  }
  os << "\niterFlag\n";
  for (int i = 0; i < this->iterFlag.size(); i++) {
    os << this->iterFlag[i] << " ";
  }
  os << "\niterOrder\n";
  for (int i = 0; i < this->iterOrder.size(); i++) {
    os << this->iterOrder[i] << " ";
  }
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\nNodeType       \n";
  if (this->nodeType == ArrayStore) {
    os << "ArrayStore";
  } else if (this->nodeType == VarStore) {
    os << "VarStore";
  } else if (this->nodeType == noFitIndexStore) {
    os << "noFitIndexStore";
  }
  os << "\nArray\n";
  this->ArrayOp->print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}

void StoreNode::print_node(raw_fd_ostream &os) {
  os << "\n————————————storeNode————————————\n";
  os << "\nNodeShift\n";
  for (int i = 0; i < this->NodeShift.size(); i++) {
    os << this->NodeShift[i] << " ";
  }
  os << "\niterFlag\n";
  for (int i = 0; i < this->iterFlag.size(); i++) {
    os << this->iterFlag[i] << " ";
  }
  os << "\niterOrder\n";
  for (int i = 0; i < this->iterOrder.size(); i++) {
    os << this->iterOrder[i] << " ";
  }
  os << "\nnodeID\n";
  os << this->nodeID;
  os << "\nNodeType       \n";
  if (this->nodeType == ArrayStore) {
    os << "ArrayStore";
  } else if (this->nodeType == VarStore) {
    os << "VarStore";
  } else if (this->nodeType == noFitIndexStore) {
    os << "noFitIndexStore";
  }
  os << "\nArray\n";
  this->ArrayOp->print(os);
  os << "\nearlist " << this->earliestTimeStep;
  os << "\nlastest " << this->latestTimeStep;
  os << "\ntimeStep " << this->timeStep;
  os << "\n————————————————————————————————\n";
}

// Update
// s Pass in the position of the AffineMap input starting from 0
void StoreNode::getStmtParamLocationVec(mlir::AffineStoreOp store,
                                        StmtInfomation *s) {
  llvm::raw_ostream &os = llvm::outs();
  SmallVector<int, 8U> interVector;
  for (mlir::Value value : store.getMapOperands()) {
    mlir::BlockArgument *blockArgument = (mlir::BlockArgument *)&value;
    interVector.push_back(blockArgument->getArgNumber());
  }
  for (int i = 0; i < s->stmtCallee.getNumArguments();
       i++) { 
    int flag = 0;
    for (int j = 0; j < interVector.size();
         j++) { 
      if (i == interVector[j]) {
        this->stmtParamLocationVec.push_back(
            j); 
        flag = 1;
      }
    }
    if (flag == 0)
      this->stmtParamLocationVec.push_back(-1); 
  }
}

// Get offset values related
// Simple case ()[s0 s1 s2] -> (s0, s1+4, s2+2)
// (without) (s0 and s1) - > (s0, 4, s1) () (s0 and s1) - > (s0 and s1, 4)
// Last array_num > inputs_num
void StoreNode::getStoreShift(ScopInformation *scopInformation,
                              mlir::AffineStoreOp store, StmtInfomation *s) {
  llvm::raw_ostream &os = llvm::outs();
  int array_num = store.getAffineMap().getNumResults(); // 2
  int inputs_num = store.getAffineMap().getNumInputs(); // 1

  std::vector<mlir::Attribute, std::allocator<mlir::Attribute>> attrVec;
  llvm::SmallVector<int64_t>
      results; 
  llvm::SmallVector<int64_t> initial_results;
  for (int i = 0; i < inputs_num; i++) {
    attrVec.push_back(scopInformation->BoundAttrVec[0]);
  }
  llvm::ArrayRef<mlir::Attribute> operandConstants(attrVec);
  store.getAffineMap().partialConstantFold(operandConstants,
                                           &results); 

  for (int i = 0; i < results.size(); i++) {
    initial_results.push_back(scopInformation->BoundIntVec[0]);
  }

  std::vector<mlir::AffineExpr> Allnode;
  store.getAffineMap().walkExprs([&Allnode](mlir::AffineExpr expr) {
    if (expr.getKind() == mlir::AffineExprKind::SymbolId) {
      int flag = 1;
      for (auto node : Allnode) {
        if (node == expr) {
          flag = 0;
          break;
        }
      }
      if (flag)
        Allnode.push_back(expr);
    }

  });

  llvm::SmallVector<int64_t>
      after_index; 
  llvm::SmallVector<int64_t> after_results;
  if (inputs_num == array_num) { 
    for (int i = 0; i < inputs_num; i++) {
      after_results.push_back(initial_results[i]); 
      after_index.push_back(i);                   
    }
  } else if (inputs_num < array_num) { 
    llvm::SmallVector<int64_t> array_index;
    for (int i = 0; i < array_num; i++) {
      mlir::AffineExpr expr = store.getAffineMap().getResult(i);
      if (expr.getKind() == mlir::AffineExprKind::Constant) {
        after_results.push_back(0);
        array_index.push_back(-1);
        continue;
      }
      int index = -1;
      expr.walk([&Allnode, &index](mlir::AffineExpr expr) {
        llvm::raw_ostream &os = llvm::outs();
        for (int j = 0; j < Allnode.size(); j++)
          if (Allnode[j] == expr) {
            index = j;
            return;
          }
      });
      if (index == -1)
        os << "\nnot find\n"; // segement default
      after_results.push_back(initial_results[index]);
      array_index.push_back(index);
    }

    for (int i = 0; i < inputs_num; i++) {
      for (int j = 0; j < array_index.size(); j++) {
        if (array_index[j] == i) {
          after_index.push_back(j);
          break; 
        }
      }
    }
  }
  for (int i = 0; i < results.size();
       i++) { 
    mlir::AffineExpr affine3 = store.getAffineMap().getResult(i);
    if (affine3.getKind() == mlir::AffineExprKind::Constant) {
      this->NodeShift.push_back(results[i]); 
      this->iterFlag.push_back(0);         
      this->iterOrder.push_back(-1); 
    } else { 
      this->NodeShift.push_back(results[i] - after_results[i]);
      this->iterFlag.push_back(1);   
      this->iterOrder.push_back(-1); 
    }
  }

  for (int i = 0; i < s->ConsArgIndexVec.size(); i++) {
    int index =
        s->ConsArgIndexVec
            [i]; // The position of the constant parameter in the parameter passed to s (in s all parameters include memref, whose loaded value is 0 starting)
    index =
        this->stmtParamLocationVec
            [index]; // AffineMap entry position (in s all parameters include memref, whose loaded value is 0 starting)
    if (index == -1) { // Note The AffineMap is not passed to the iteration variable /s parameter of this layer
      continue;
    }
    index = after_index
        [index]; 
    this->iterFlag[index] = 0; 
    this->NodeShift[index] =
        this->NodeShift[index] +
        s->ConsValueVec[i]; 
  }

  std::map<int, int>
      stmt_iterShift; 
  int flag = 0;
  for (int i = 0; i < s->IterArgIndexVec.size();
       i++) { 
    int index =
        s->IterArgIndexVec
            [i]; 
    index =
        this->stmtParamLocationVec
            [index]; 
    if (index == -1) { 
      continue;
    }
    index =
        after_index[index]; 
    this->iterOrder[index] = i;
    this->NodeShift[index] =
        this->NodeShift[index] +
        s->IterShiftVec[i]; 
    stmt_iterShift.insert(std::make_pair(index, s->IterShiftVec[i]));
    if (!flag) {
      this->innerestIndex = index;
      flag = 1;
    }
  }
  if (!flag)
    this->innerestIndex = -1; 

  int iterFlagCount = 0;
  int iterOrderCount = 0;
  for (int i = 0; i < results.size(); i++) {
    if (this->iterFlag[i] == 1) {
      iterFlagCount++; 
    }
    if (this->iterOrder[i] != -1) {
      iterOrderCount++;
    }
    if (this->iterFlag[i] == 1 && this->iterOrder[i] == -1) {
      mlir::AffineExpr affine3 = store.getAffineMap().getResult(i);
      int iterSame = -1;
      affine3.walk([&Allnode, &iterSame](mlir::AffineExpr expr) {
        llvm::raw_ostream &os = llvm::outs();
        for (int j = 0; j < Allnode.size(); j++)
          if (Allnode[j] == expr) {
            iterSame = j;
            return;
          }
      });
      iterSame = after_index[iterSame];
      iterOrder[i] = iterOrder[iterSame];
      NodeShift[i] = results[i] - after_results[i] + stmt_iterShift[iterSame];
    }
  }
  if (iterFlagCount > iterOrderCount) {
    this->nodeType = noFitIndexStore;
  }
}

AddNode::AddNode(mlir::AddFOp AddOp, int nodeIDCount) {
  this->nodeID = nodeIDCount;
  this->nodeType = Add;
  this->label = "Add";
  this->SSA = AddOp.getResult();
}

SqrtNode::SqrtNode(mlir::math::SqrtOp sqrtOp, int nodeIDCount) {
  this->nodeID = nodeIDCount;
  this->nodeType = Sqrt;
  this->label = "Sqrt";
  this->SSA = sqrtOp.getResult();
}

SubNode::SubNode(mlir::SubFOp SubOp, int nodeIDCount) {
  this->nodeID = nodeIDCount;
  this->nodeType = Sub;
  this->label = "Sub";
  this->SSA = SubOp.getResult();
}

MulNode::MulNode(mlir::MulFOp MulOp, int nodeIDCount) {
  this->nodeID = nodeIDCount;
  this->nodeType = Mul;
  this->label = "Mul";
  this->SSA = MulOp.getResult();
}

DivNode::DivNode(mlir::DivFOp DivOp, int nodeIDCount) {
  this->nodeID = nodeIDCount;
  this->nodeType = Div;
  this->label = "Div";
  this->SSA = DivOp.getResult();
}

CstNode::CstNode(mlir::ConstantOp CstOp, int nodeIDCount) {
  this->nodeID = nodeIDCount;
  this->nodeType = Cst;
  this->label = "Cst";
  this->SSA = CstOp.getResult();
}

// class ReuseGraph
void ReuseGraph::print_ReuseGraph() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n==============print_ReuseGraph===============\n";
  os << "ReuseGrapgID : " << this->ReuseID << "\n";
  os << "Array : ";
  this->Array->print(os);
  os << "\n";
  for (Node *node : this->LoadStoreVec) {
    os << "\n";
    node->print_node();
    os << "\n";
  }
}

ReuseGraph::ReuseGraph(mlir::Operation *alloc,
                       std::vector<Node *> LoadStoreVec) {
  this->Array = alloc;
  this->LoadStoreVec = LoadStoreVec;
}

// class ScopInformation
ScopInformation::ScopInformation(mlir::FuncOp g, mlir::ModuleOp moduleop) {
  this->g = g;
  //Match the Bound value to IndexCastOp
  mlir::FuncOp FuncOpt;
  mlir::CallOp FuncOptCaller;
  for (mlir::Region::iterator it = g.begin(); it != g.end(); it++) {
    for (mlir::Block::iterator blockit = it->begin(); blockit != it->end();
         blockit++) {
      if (mlir::IndexCastOp index_cast =
              dyn_cast<mlir::IndexCastOp>(*blockit)) {
        BoundIndexCastVec.push_back(index_cast);
      }
    }
  }
  //Find the location of each IndexCastOp
  for (mlir::IndexCastOp funIndex : BoundIndexCastVec) {
    int countArg = 0;
    int IndexArg = 0;
    for (mlir::Block::args_iterator argit = g.getArguments().begin();
         argit != g.getArguments().end(); argit++) {
      if (g.getArgument(countArg) == funIndex.getOperand()) {
        IndexArg = countArg;
      }
      countArg++;
    }
    //You can get this is the number countArg passed in, and then you go to m and you look for the call
    FuncOpt = moduleop.lookupSymbol<mlir::FuncOp>(g.getName());
    llvm::StringRef kernelName = FuncOpt.getName();
    std::string Funcstr;
    for (int i = 0; i < 6; i++) {
      Funcstr.push_back(kernelName[i]);
    }
    mlir::Value callerOptArg;
    // callerArg
    mlir::Value callerArg = FuncOpt.getArgument(IndexArg);
    moduleop.walk([&](mlir::Operation *op) {
      if (mlir::CallOp callerOpt = dyn_cast<mlir::CallOp>(*op)) {
        llvm::StringRef callerName = callerOpt.callee();
        std::string callerstr;
        if (callerName.size() > 6) {
          for (int i = 0; i < 6; i++) {
            callerstr.push_back(callerName[i]);
          }
          //Locate the caller
          if (callerstr == Funcstr) {
            FuncOptCaller = callerOpt;
            callerOptArg = callerOpt.getOperand(IndexArg);
          }
        }
      }
    });

    double count = 0;
    int *intPtr;
    int BoundInt = 0;
    mlir::Attribute attr;
    moduleop.walk([&](mlir::Operation *op) {
      if (mlir::ConstantOp consOp = dyn_cast<mlir::ConstantOp>(*op)) {
        if (callerOptArg == consOp.getResult()) {
          if (auto intAttr = consOp.getValue().dyn_cast<IntegerAttr>()) {
            auto attrType = consOp.getValue().getType();
            bool isUnsigned =
                attrType.isUnsignedInteger() || attrType.isSignlessInteger(1);
            count = intAttr.getValue().bitsToDouble();
            intPtr = (int *)&count;
            BoundInt = *intPtr;
          }
          if (mlir::Attribute a = consOp.getValue().dyn_cast<Attribute>()) {
            attr = a;
          }
        }
      }
    });

    // I've got a bound bound, and I store it in BoundIntVec
    this->BoundIntVec.push_back(BoundInt);
    this->BoundAttrVec.push_back(attr);
    this->BoundMapSize++;
  }
  int ArrayArgIndexcount = 0;
  for (mlir::Block::args_iterator argit = g.getArguments().begin();
       argit != g.getArguments().end(); argit++) {
    if (argit->getType() != BoundIndexCastVec[0].getOperand().getType()) {
      ArrayValueVec.push_back(*argit);
      ArrayCount++;
      this->ArrayArgIndexVec.push_back(ArrayArgIndexcount);
    }
    ArrayArgIndexcount++;
  }

// 1111 Array passed in from outside Look for the requested array in caller but
// alloca is not an array, but a single number
  for (int i = 0; i < ArrayCount; i++) {
    // ArrayArgValueVec.push_back(FuncOptCaller.getOperand(ArrayArgIndexVec[i]));
    moduleop.walk([&](mlir::Operation *op) {
      if (memref::CastOp castOp = dyn_cast<memref::CastOp>(*op)) {
        if (castOp.getResult() ==
            FuncOptCaller.getOperand(this->ArrayArgIndexVec[i])) {
          moduleop.walk([&](mlir::Operation *op) {
            if (memref::AllocOp allocOp = dyn_cast<memref::AllocOp>(*op)) {
              if (allocOp.getResult() == castOp.getOperand()) {
                this->ArrayAllocVec.push_back(allocOp);
              }
            }
          });
        }
      }
      if (memref::AllocOp allocOp = dyn_cast<memref::AllocOp>(*op)) {
        if (allocOp.getResult() ==
            FuncOptCaller.getOperand(this->ArrayArgIndexVec[i])) {
          // alloc
          this->ArrayAllocVec.push_back(allocOp);
        }
      }

      if (mlir::AffineLoadOp loadOp = dyn_cast<mlir::AffineLoadOp>(*op)) {
        if (loadOp.getResult() ==
            FuncOptCaller.getOperand(this->ArrayArgIndexVec[i])) {
          moduleop.walk([&](mlir::Operation *op) {
            if (memref::AllocaOp allocaOp = dyn_cast<memref::AllocaOp>(*op)) {
              if (allocaOp.getResult() == loadOp.getOperand(0)) {
                this->ArrayAllocVec.push_back(allocaOp);
              }
            }
          });
        }
      }

      if (memref::AllocaOp allocaOp = dyn_cast<memref::AllocaOp>(*op)) {
        if (allocaOp.getResult() ==
            FuncOptCaller.getOperand(this->ArrayArgIndexVec[i])) {
          // alloc
          this->ArrayAllocVec.push_back(allocaOp);
        }
      }
    });
  }
// 222 not only comes in from the outside, but also temporary variables in g
  g.walk([&](mlir::Operation *op) {
    if (memref::AllocaOp allocaOp = dyn_cast<memref::AllocaOp>(*op)) {
      this->ArrayAllocVec.push_back(allocaOp); 
    }
  });
}

ScopInformation::~ScopInformation() {
  this->BoundIndexCastVec.clear();
  this->BoundIntVec.clear();
  this->ArrayAllocVec.clear();
  //The ssa value stored in the func array
  this->ArrayValueVec.clear();
  //The location of the array stored in func
  this->ArrayArgIndexVec.clear();
}

ScopInformation::ScopInformation(const ScopInformation &s) {
  for (int i = 0; i < s.ArrayAllocVec.size(); i++) {
    if (mlir::memref::AllocOp alloc =
            dyn_cast<mlir::memref::AllocOp>(s.ArrayAllocVec[i])) {
      this->ArrayAllocVec.push_back(alloc);
    }
  }
  for (int i = 0; i < s.ArrayArgIndexVec.size(); i++) {
    this->ArrayArgIndexVec.push_back(s.ArrayArgIndexVec[i]);
  }
  for (int i = 0; i < s.ArrayValueVec.size(); i++) {
    this->ArrayValueVec.push_back(s.ArrayValueVec[i]);
  }
  this->ArrayCount = s.ArrayCount;
  this->BoundMapSize = s.BoundMapSize;
  for (int i = 0; i < s.BoundAttrVec.size(); i++) {
    this->BoundAttrVec.push_back(s.BoundAttrVec[i]);
  }
  for (int i = 0; i < s.BoundIndexCastVec.size(); i++) {
    this->BoundIndexCastVec.push_back(s.BoundIndexCastVec[i]);
  }
}

int ScopInformation::getMapResult(mlir::AffineMap affineMap, int index) {
  int intReturn = 0;
  std::vector<mlir::Attribute, std::allocator<mlir::Attribute>> attrVec;
  llvm::SmallVector<int64_t> results;
  attrVec.push_back(this->BoundAttrVec[index]);
  llvm::ArrayRef<mlir::Attribute> operandConstants(attrVec);
  affineMap.partialConstantFold(operandConstants, &results);
  intReturn = results[0];
  return intReturn;
}

int ScopInformation::getBoundMapConstant(mlir::AffineMap affineMap,
                                         mlir::Value mapValue) {
  int index = 0;
  int intReturn = 0;
  // Get subscript
  for (mlir::IndexCastOp op : this->BoundIndexCastVec) {
    if (op.getResult() == mapValue) {
      break;
    }
    index++;
  }
  return getMapResult(affineMap, index);
}

void ScopInformation::print_Bound() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n BoundIndexCastVec \n";
  for (mlir::IndexCastOp op : this->BoundIndexCastVec) {
    os << "\n IndexCastOp \n";
    op.print(os);
  }
  os << "\n BoundIntVec \n";
  for (int op : this->BoundIntVec) {
    os << "\n int \n";
    os << op;
  }
  os << "\n BoundAttrVec \n";
  for (mlir::Attribute attr : this->BoundAttrVec) {
    os << "\n Attribute \n";
    attr.print(os);
  }
}
void ScopInformation::print_Array() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n ArrayArgIndexVec \n";
  for (int op : this->ArrayArgIndexVec) {
    os << "\n Value \n";
    os << op << "\n";
  }
  os << "\n ArrayAllocVec \n";
  for (mlir::Operation *op : this->ArrayAllocVec) {
    os << "\n AllocOp \n";
    op->print(os);
  }
}

// clsss  StmtInfomation
StmtInfomation::StmtInfomation(int dfg_dim,
                               SmallVector<mlir::AffineForOp, 8U> forIterVec,
                               mlir::AffineForOp forOp, mlir::CallOp callOp,
                               mlir::ModuleOp moduleop, mlir::FuncOp g,
                               ScopInformation *scopInformation) {
  llvm::raw_ostream &os = llvm::outs();
  this->stmtCaller = callOp;
  llvm::StringRef callerName = this->stmtCaller.callee();
  moduleop.walk([&](mlir::Operation *op) {
    if (mlir::FuncOp callee = dyn_cast<mlir::FuncOp>(*op)) {
      if (callee.getName() == callerName) {
        this->stmtCallee = callee;
      }
    }
  });

  this->find_ApplyConsOp(forOp, callOp, moduleop, g);

  this->find_IterArgIndex(dfg_dim, forIterVec, forOp, callOp, moduleop, g,
                          scopInformation);

  this->find_ConsArgIndex();

  // this->print_stmtInfo();
}

StmtInfomation::StmtInfomation(const StmtInfomation &s) {
  this->stmtCaller = s.stmtCaller;
  this->stmtCallee = s.stmtCallee;
  this->dim = s.dim;
  this->innerIter = s.innerIter;
  for (int i = 0; i < s.IterShiftVec.size(); i++) {
    this->IterShiftVec.push_back(s.IterShiftVec[i]);
  }
  for (int i = 0; i < s.IterArgIndexVec.size(); i++) {
    this->IterArgIndexVec.push_back(s.IterArgIndexVec[i]);
  }
  for (int i = 0; i < s.ApplyFlagVec.size(); i++) {
    this->ApplyFlagVec.push_back(s.ApplyFlagVec[i]);
  }
  for (int i = 0; i < s.ConsArgIndexVec.size(); i++) {
    this->ConsArgIndexVec.push_back(s.ConsArgIndexVec[i]);
  }
  for (int i = 0; i < s.ConsValueVec.size(); i++) {
    this->ConsValueVec.push_back(s.ConsValueVec[i]);
  }

  for (int i = 0; i < s.ApplyOrConsFlagVec.size(); i++) {
    this->ApplyOrConsFlagVec.push_back(s.ApplyOrConsFlagVec[i]);
  }
  for (int i = 0; i < s.ApplyOrConsOpVec.size(); i++) {
    if (mlir::AffineApplyOp Op =
            dyn_cast<mlir::AffineApplyOp>(s.ApplyOrConsOpVec[i])) {
      this->ApplyOrConsOpVec.push_back(Op);
    }
    if (mlir::ConstantOp Op =
            dyn_cast<mlir::ConstantOp>(s.ApplyOrConsOpVec[i])) {
      this->ApplyOrConsOpVec.push_back(Op);
    }
  }

  for (int i = 0; i < s.Argflag.size(); i++) {
    this->Argflag.push_back(s.Argflag[i]);
  }
}

// Find shift
void StmtInfomation::find_IterShift(mlir::AffineMap affineMap,
                                    ScopInformation *scopInformation) {
  llvm::raw_ostream &os = llvm::outs();
  int AfterMap_value = scopInformation->getMapResult(affineMap, 0);
  int Initial_value = scopInformation->BoundIntVec[0];
  this->IterShiftVec.push_back(AfterMap_value - Initial_value);

}
// Find an inside-out loop that iterates the position of the variable in the parameter
void StmtInfomation::find_IterArgIndex(
    int dfg_dim, SmallVector<mlir::AffineForOp, 8U> forIterVec,
    mlir::AffineForOp forOp, mlir::CallOp callOp, mlir::ModuleOp moduleop,
    mlir::FuncOp g, ScopInformation *scopInformation) {
  int count = 0;
  int applyMapFlag[this->dim];
  for (int i = 0; i < this->dim; i++) {
    applyMapFlag[i] = 0;
  }

  for (int i = 0; i < dfg_dim; i++) {
    for (int apc_i = 0; apc_i < this->dim; apc_i++) {
      if (this->ApplyOrConsFlagVec[apc_i] == 1) {
        // apply
        if (mlir::AffineApplyOp applyOp =
                dyn_cast<mlir::AffineApplyOp>(this->ApplyOrConsOpVec[apc_i])) {
          // iter
          if (applyOp.getOperand(0) ==
              forIterVec[i].getBody()->getArgument(0)) {
            applyMapFlag[apc_i] = 1; // applyOp
            for (int call_arg_i = 0;
                 call_arg_i < this->stmtCaller.getNumOperands(); call_arg_i++) {
              if (this->stmtCaller.getOperand(call_arg_i) ==
                  applyOp.getResult()) {
                this->IterArgIndexVec.push_back(call_arg_i);
              }
            }
            // // shift
            this->find_IterShift(applyOp.getAffineMap(), scopInformation);
          }
        }
      }
    }
  }
  for (int i = 0; i < this->dim; i++) {
    this->ApplyFlagVec.push_back(applyMapFlag[i]);
  }
  // Start working with constants and affine map constants
  int *intPtr;
  int consValue = 0;
  double temp;
  for (int apc_i = 0; apc_i < this->dim; apc_i++) {
    if (this->ApplyOrConsFlagVec[apc_i] == 1) {
      if (this->ApplyFlagVec[apc_i] == 0) {
        if (mlir::AffineApplyOp applyOp =
                dyn_cast<mlir::AffineApplyOp>(this->ApplyOrConsOpVec[apc_i])) {
          int applyConsValue = scopInformation->getBoundMapConstant(
              applyOp.getAffineMap(), applyOp.getOperand(0));
          this->ConsValueVec.push_back(applyConsValue);
          // Find the parameter location of ConsArg
          for (int call_arg_i = 0;
               call_arg_i < this->stmtCaller.getNumOperands(); call_arg_i++) {
            if (this->stmtCaller.getOperand(call_arg_i) ==
                applyOp.getResult()) {
              this->ConsArgIndexVec.push_back(call_arg_i);
            }
          }
        }
      }
    } else {
      if (mlir::ConstantOp consOp =
              dyn_cast<mlir::ConstantOp>(this->ApplyOrConsOpVec[apc_i])) {
        if (auto intAttr = consOp.getValue().dyn_cast<IntegerAttr>()) {
          auto attrType = consOp.getValue().getType();
          bool isUnsigned =
              attrType.isUnsignedInteger() || attrType.isSignlessInteger(1);
          temp = intAttr.getValue().bitsToDouble();
          intPtr = (int *)&temp;
          consValue = *intPtr;
          this->ConsValueVec.push_back(consValue);
        }
      }
    }
  }
}
// Find the statement depth and ApplyOp
void StmtInfomation::find_ApplyConsOp(mlir::AffineForOp forOp,
                                      mlir::CallOp callOp,
                                      mlir::ModuleOp moduleop, mlir::FuncOp g) {
  int begin = 0;
  int end = 0;
  bool flag1 = true;
  bool flag2 = true;
  // Walk through forOp
  for (mlir::Block::iterator it = forOp.getBody()->begin();
       it != forOp.getBody()->end(); it++) {
    if (flag1) {
      end++;
    }
    if (mlir::CallOp Op = dyn_cast<mlir::CallOp>(*it)) {
      if (callOp.getResult(0) == Op.getResult(0)) {
        flag1 = false;
      }
    }
  }
  // Iterate over the position of the call before the call, and set 0 if it does not appear
  int secondTime = 0;
  mlir::CallOp lastOp;
  bool hasLastOp = false;
  for (mlir::Block::iterator it = forOp.getBody()->begin();
       it != forOp.getBody()->end(); it++) {
    secondTime++;
    if (secondTime == end) {
      break;
    } else if (secondTime < end) {
      if (mlir::CallOp Op = dyn_cast<mlir::CallOp>(*it)) {
        lastOp = Op;
        hasLastOp = true;
      }
    }
  }

  for (mlir::Block::iterator it = forOp.getBody()->begin();
       it != forOp.getBody()->end(); it++) {
    if (hasLastOp) {
      if (flag2) {
        begin++;
      }
      if (mlir::CallOp Op = dyn_cast<mlir::CallOp>(*it)) {
        if (lastOp.getResult(0) == Op.getResult(0)) {
          flag2 = false;
        }
      }
    } else {
      begin = 0;
    }
  }

  int count = 0;
  for (mlir::Block::iterator it = forOp.getBody()->begin();
       it != forOp.getBody()->end(); it++) {
    count++;
    if (count > begin && count < end) {
      if (mlir::AffineApplyOp Op = dyn_cast<mlir::AffineApplyOp>(*it)) {
        this->ApplyOrConsOpVec.push_back(Op);
        this->dim++;
        this->ApplyOrConsFlagVec.push_back(1);
        //If it is constant
      } else if (mlir::ConstantOp Op = dyn_cast<mlir::ConstantOp>(*it)) {
        this->ApplyOrConsOpVec.push_back(Op);
        this->dim++;
        this->ApplyOrConsFlagVec.push_back(0);
      }
    }
  }
}

int StmtInfomation::ValueIndexInCallee(mlir::Value value) {
  int index = -1;
  for (int i = 0; i < this->stmtCallee.getNumArguments(); i++) {
    if (value == this->stmtCallee.getArgument(i)) {
      index = i;
    }
  }
  return index;
}

int ScopInformation::ValueIndexInFuncG(mlir::Value value) {
  int index = -1;
  for (int i = 0; i < this->g.getNumArguments(); i++) {
    if (value == this->g.getArgument(i)) {
      index = i;
    }
  }
  return index;
}

void StmtInfomation::print_stmtInfo() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n\n------print_stmtInfo--------\n";
  os << "\ncaller:\n";
  this->stmtCaller.print(os);
  os << "\nstmt dim: ";
  os << this->dim;
  os << "\nstmt ApplyOrConsFlagVec:\n";
  for (int i : this->ApplyOrConsFlagVec) {
    os << i << " ";
  }
  os << "\nstmt ApplyFlagVec:\n";
  for (int i : this->ApplyFlagVec) {
    os << i << " ";
  }
  os << "\nstmt IterArgIndexVec:\n";
  for (int i : this->IterArgIndexVec) {
    os << i << " ";
  }
  os << "\nstmt IterShiftVec:\n";
  for (int i : this->IterShiftVec) {
    os << i << " ";
  }
  os << "\nstmt ConsArgIndexVec:\n";
  for (int i : this->ConsArgIndexVec) {
    os << i << " ";
  }
  os << "\nstmt ConsValueVec:\n";
  for (int i : this->ConsValueVec) {
    os << i << " ";
  }
  os << "\nstmt ApplyOrConsOpVec:\n";
  for (mlir::Operation *Op : this->ApplyOrConsOpVec) {
    if (mlir::AffineApplyOp applyOp = dyn_cast<mlir::AffineApplyOp>(Op)) {
      applyOp.print(os);
      os << "\n";
    }
    if (mlir::ConstantOp consOp = dyn_cast<mlir::ConstantOp>(Op)) {
      consOp.print(os);
      os << "\n";
    }
  }
  os << "\ncallee\n";
  this->stmtCallee.print(os);
}

// Find constants and values
void StmtInfomation::find_ConsArgIndex() {
  for (int apc_i = 0; apc_i < this->dim; apc_i++) {
    if (this->ApplyOrConsFlagVec[apc_i] == 0) {
      if (mlir::ConstantOp consOp =
              dyn_cast<mlir::ConstantOp>(this->ApplyOrConsOpVec[apc_i])) {
        for (int call_arg_i = 0; call_arg_i < this->stmtCaller.getNumOperands();
             call_arg_i++) {
          if (this->stmtCaller.getOperand(call_arg_i) == consOp.getResult()) {
            this->ConsArgIndexVec.push_back(call_arg_i);
          }
        }
      }
    }
  }
}

// class MlirDFG
MlirDFG::MlirDFG(int dfg_id, SmallVector<mlir::CallOp, 8U> callOpVec,
                 mlir::AffineForOp forOp, mlir::FuncOp g,
                 mlir::ModuleOp moduleop, ScopInformation *scopInformation,
                 int &DFGNum, int methodflag) {

  //Loop all for around PNU into the forIterVec array from the inside out
  this->dfg_id = DFGNum;
  this->methodflag = methodflag;
  this->find_forIter(forOp, g, moduleop);
  for (mlir::CallOp caller : callOpVec) {
    StmtInfomation stmt(this->dfg_dim, this->forIterVec, forOp, caller,
                        moduleop, g, scopInformation);
    this->insert_Stmts(stmt);
  }
  this->scopInformation = scopInformation;
  // step1 a DFG graph is generated based on each operator
  this->createDfg(scopInformation);
  this->DFGDataReuse();
  DFGNum++;
}

void MlirDFG::DFGDataReuse() {
  // step2 Remove Cst operators and edges from DFG graphs
  this->deleteCstEdge();
  // step3 The Var relationship is handled in DFG
  this->createVarGraphs();
  this->addVarEdges();
  this->addRecEdges();
  if (this->UnrollNum < 0) {
    if (this->divideSubReuseGraphs() >= 1) {
      // step4 Add a reuse edge to the reuse diagram
      this->sortByIndexReuseStep();
      this->addReuseEdges();
    }
  }
  if (PBPMethod && this->methodflag == 0) {
    this->PBPNoRuseMethod();
  }
  this->InDFGNodeAndEdge();
  // step5 Got recourse constrained
  this->getRecourseII();
  // step6 Get the earliest and latest execution time ready for scheduling
  this->getEarliestStep();
  this->getLastestStep();
}

void MlirDFG::PBPNoRuseMethod() {
  int count = 0;
  Edge *eb;
  for (Node *n : this->NodesList) {
    if (n->nodeType == ArrayLoad || n->nodeType == noFitIndexLoad) {
      count = 0;
      for (Edge *e : this->EdgesList) {
        if (e->begin == n->nodeID) {
          count++;
          eb = e;
        }
        if (count > 1) {
          LoadNode *Ptr = (LoadNode *)n;
          Node *noReuseLoadNode = new LoadNode(Ptr);
          noReuseLoadNode->nodeID = this->NodesList.size();
          this->NodesList.push_back(noReuseLoadNode);
          eb->begin = noReuseLoadNode->nodeID;
        }
      }
    }
  }
}

int isNotInnerIter(Node *node1) {
  if (node1->nodeType == ArrayLoad || node1->nodeType == noFitIndexLoad) {
    LoadNode *node1 = (LoadNode *)node1;
  } else if (node1->nodeType == ArrayStore ||
             node1->nodeType == noFitIndexStore) {
    StoreNode *node1 = (StoreNode *)node1;
  } else
    return -1;

  for (int i = 0; i < node1->iterOrder.size(); i++) {
    if (node1->iterOrder[i] == 0) {
      return 0;
    }
  }
  return 1;
}

int isNotInnerIterEqul(Node *node1, Node *node2) {
  if (node1->nodeType == ArrayLoad || node1->nodeType == noFitIndexLoad) {
    LoadNode *node1 = (LoadNode *)node1;
  } else if (node1->nodeType == ArrayStore ||
             node1->nodeType == noFitIndexStore) {
    StoreNode *node1 = (StoreNode *)node1;
  } else
    return -1; 

  if (node2->nodeType == ArrayLoad || node2->nodeType == noFitIndexLoad) {
    LoadNode *node2 = (LoadNode *)node2;
  } else if (node2->nodeType == ArrayStore ||
             node2->nodeType == noFitIndexStore) {
    StoreNode *node2 = (StoreNode *)node2;
  } else
    return -1; 

  for (int i = 0; i < node1->iterOrder.size(); i++) {
    if (node1->iterOrder[i] != node2->iterOrder[i]) {
      return 0;
    }
  }

  for (int i = 0; i < node1->NodeShift.size(); i++) {
    if (node1->NodeShift[i] != node2->NodeShift[i]) {
      return 0;
    }
  }

  return 1;
}

void MlirDFG::noDataReuse() {
  llvm::raw_ostream &os = llvm::outs();
  std::map<std::pair<int, int>, int> SLreuseEdgeMap;
  // In the Reuse diagram, we sort sequentially, find Store, then look for a Load that has the same index, and find just one
  for (ReuseGraph *RG : this->reuseGraphs) {
    int StoreSrc = 0;
    int StoreDst = 0;
    bool StoreSrcflag = false;
    bool first = true;
    for (int i = 0; i < RG->LoadStoreVec.size(); i++) {
      if (RG->LoadStoreVec[i]->nodeType == ArrayStore ||
          RG->LoadStoreVec[i]->nodeType == noFitIndexStore) {
        StoreSrc = i;
        StoreSrcflag = true; 
        first = true;
      }

      if ((RG->LoadStoreVec[i]->nodeType == ArrayLoad ||
           RG->LoadStoreVec[i]->nodeType == noFitIndexLoad) &&
          StoreSrcflag && first) {
        StoreDst = i;
        if (isNotInnerIterEqul(RG->LoadStoreVec[StoreSrc],
                               RG->LoadStoreVec[StoreDst]) == 1) {
          RG->LoadStoreVec[StoreDst]->reuseflag = true;
          int dif = 0;
          SLreuseEdgeMap.insert(
              std::make_pair(std::make_pair(RG->LoadStoreVec[StoreDst]->nodeID,
                                            RG->LoadStoreVec[StoreSrc]->nodeID),
                             dif));
          first = false;
        }
      }
    }
  }

  int SLbegin = -1;
  int SLend = -1;

  for (auto reuseE : SLreuseEdgeMap) {
    for (Edge *e : this->EdgesList) {
      if (e->begin == reuseE.first.first) {
        SLend = e->end;
        e->edgeType = Delete;
      }
      if (e->end == reuseE.first.second) { 
        if (this->NodesList[e->begin]->nodeType == Cst) {
          SLbegin = -1;
        } else {
          SLbegin = e->begin;
        }
      }
    }
    if (SLbegin != -1 && SLend != -1) {
      Edge *edge = new Edge(SLbegin, SLend);
      edge->edgeID = this->EdgesList.size();
      edge->edgeType = SLReuse;
      edge->dif = reuseE.second;
      this->EdgesList.push_back(edge);
    }
  }
}

void MlirDFG::addRecEdges() {
  std::map<int, int> RecEdge; 
  for (ReuseGraph *RG : this->reuseGraphs) {
    int StoreSrc = 0;
    int StoreDst = 0;
    bool StoreSrcflag = false;
    for (int i = RG->LoadStoreVec.size() - 1; i >= 0; i--) {
      if (isNotInnerIter(RG->LoadStoreVec[i]) == 1) {
        if ((RG->LoadStoreVec[i]->nodeType == ArrayStore ||
             RG->LoadStoreVec[i]->nodeType == noFitIndexStore)) {
          StoreSrc = i;
          StoreSrcflag = true;
        }

        if ((RG->LoadStoreVec[i]->nodeType == ArrayLoad ||
             RG->LoadStoreVec[i]->nodeType == noFitIndexLoad) &&
            StoreSrcflag) {
          StoreDst = i;
          if (isNotInnerIterEqul(RG->LoadStoreVec[StoreSrc],
                                 RG->LoadStoreVec[StoreDst]) == 1) {
            RecEdge.insert(std::make_pair(RG->LoadStoreVec[StoreDst]->nodeID,
                                          RG->LoadStoreVec[StoreSrc]->nodeID));
          }
        }
      }
    }
  }

  for (auto reuseE : RecEdge) {
    Edge *edge = new Edge(reuseE.second, reuseE.first);
    edge->edgeID = this->EdgesList.size();
    edge->dif = 1;
    edge->edgeType = RecSLDepen;
    this->EdgesList.push_back(edge);
  }
}

void modifyShift(Node *n, int i) {
  if (n->nodeType == ArrayLoad || n->nodeType == noFitIndexLoad) {
    LoadNode *Ptr = (LoadNode *)n; 
    for (int j = 0; j < Ptr->NodeShift.size(); j++) {
      if (Ptr->iterOrder[j] == 0) {
        Ptr->NodeShift[j] += i;
      }
    }
  } else if (n->nodeType == ArrayStore || n->nodeType == noFitIndexStore) {
    StoreNode *Ptr = (StoreNode *)n;
    for (int j = 0; j < Ptr->NodeShift.size(); j++) {
      if (Ptr->iterOrder[j] == 0) {
        Ptr->NodeShift[j] += i;
      }
    }
  }
}

void MlirDFG::cleanDFGInformation() {
  this->NodesList.clear();
  this->EdgesList.clear();
  this->reuseGraphs.clear();
  this->varGraphs.clear();
  this->inDFGEdgesList.clear();
  this->inDFGNodesList.clear();
}

void MlirDFG::restoreOriginalInformation() {
  this->NodesList = this->original_NodesList;
  this->EdgesList = this->original_EdgesList;
  this->reuseGraphs = this->original_reuseGraphs;
  this->varGraphs = this->original_varGraphs;
  this->inDFGEdgesList = this->original_inDFGEdgesList;
  this->inDFGNodesList = this->original_inDFGNodesList;
}

void MlirDFG::saveOriginalInformation() {
  std::vector<RfCgraTrans::Node *> NList = this->original_NodesList;
  std::vector<RfCgraTrans::Edge *> EList = this->original_EdgesList;
  std::vector<RfCgraTrans::ReuseGraph *> rGraphs = this->original_reuseGraphs;
  std::vector<RfCgraTrans::ReuseGraph *> vGraphs = this->original_varGraphs;
  std::vector<RfCgraTrans::Node *> inDFGNList = this->original_inDFGNodesList;
  std::vector<RfCgraTrans::Edge *> inDFGEList = this->original_inDFGEdgesList;

  this->original_NodesList = this->NodesList;
  this->original_EdgesList = this->EdgesList;
  this->original_reuseGraphs = this->reuseGraphs;
  this->original_varGraphs = this->varGraphs;
  this->original_inDFGEdgesList = this->inDFGEdgesList;
  this->original_inDFGNodesList = this->inDFGNodesList;

  this->NodesList = NList;
  this->EdgesList = EList;
  this->reuseGraphs = rGraphs;
  this->varGraphs = vGraphs;
  this->inDFGEdgesList = inDFGEList;
  this->inDFGNodesList = inDFGNList;
}

void MlirDFG::createUnrollDFG(int UnrollNum) {
  //Replication node
  for (int i = 0; i < UnrollNum; i++) {
    for (int j = 0; j < this->original_NodesList.size(); j++) {
      // Cst
      if (this->original_NodesList[j]->nodeType == Cst) {
        CstNode *Ptr = (CstNode *)this->original_NodesList[j]; 
        Node *node = new CstNode(Ptr); 
        Ptr = (CstNode *)node;
        node->UnrollID = i;
        node->nodeID = this->NodesList.size();
        this->NodesList.push_back(Ptr);
      }

      // Load
      if (this->original_NodesList[j]->nodeType == ArrayLoad ||
          this->original_NodesList[j]->nodeType == noFitIndexLoad ||
          this->original_NodesList[j]->nodeType == VarLoad) {
        LoadNode *Ptr =
            (LoadNode *)this->original_NodesList[j]; 
        Node *node = new LoadNode(Ptr);              
        Ptr = (LoadNode *)node;
        node->UnrollID = i;
        modifyShift(node, i); //// Change the offset value of a node
        node->nodeID = this->NodesList.size();
        this->NodesList.push_back(Ptr);
      }

      // Store
      if (this->original_NodesList[j]->nodeType == ArrayStore ||
          this->original_NodesList[j]->nodeType == noFitIndexStore ||
          this->original_NodesList[j]->nodeType == VarStore) {
        StoreNode *Ptr =
            (StoreNode *)this->original_NodesList[j];
        Node *node = new StoreNode(Ptr);            
        Ptr = (StoreNode *)node;
        node->UnrollID = i;
        modifyShift(node, i); // Change the offset value of a node
        node->nodeID = this->NodesList.size();
        this->NodesList.push_back(Ptr);
      }

      // add
      if (this->original_NodesList[j]->nodeType == Add) {
        AddNode *Ptr = (AddNode *)this->original_NodesList[j];
        Node *node = new AddNode(Ptr); 
        Ptr = (AddNode *)node;
        node->UnrollID = i;
        node->nodeID = this->NodesList.size();
        this->NodesList.push_back(Ptr);
      }

      // sub
      if (this->original_NodesList[j]->nodeType == Sub) {
        SubNode *Ptr = (SubNode *)this->original_NodesList[j]; 
        Node *node = new SubNode(Ptr); 
        Ptr = (SubNode *)node;
        node->UnrollID = i;
        node->nodeID = this->NodesList.size();
        this->NodesList.push_back(Ptr);
      }

      // mul
      if (this->original_NodesList[j]->nodeType == Mul) {
        MulNode *Ptr = (MulNode *)this->original_NodesList[j]; 
        Node *node = new MulNode(Ptr); 
        Ptr = (MulNode *)node;
        node->UnrollID = i;
        node->nodeID = this->NodesList.size();
        this->NodesList.push_back(Ptr);
      }

      // Div
      if (this->original_NodesList[j]->nodeType == Div) {
        DivNode *Ptr = (DivNode *)this->original_NodesList[j]; 
        Node *node = new DivNode(Ptr); 
        Ptr = (DivNode *)node;
        node->UnrollID = i;
        node->nodeID = this->NodesList.size();
        this->NodesList.push_back(Ptr);
      }

      // math.sqrt
      if (this->original_NodesList[j]->nodeType == Sqrt) {
        SqrtNode *Ptr =
            (SqrtNode *)this->original_NodesList[j]; 
        Node *node = new SqrtNode(Ptr);              
        Ptr = (SqrtNode *)node;
        node->UnrollID = i;
        node->nodeID = this->NodesList.size();
        this->NodesList.push_back(Ptr);
      }
    }
  }
  // Copy edge
  for (int i = 0; i < UnrollNum; i++) {
    for (int j = 0; j < this->original_EdgesList.size(); j++) {
      Edge *edge = new Edge(this->original_EdgesList[j]);
      edge->UnrollID = i;
      edge->begin = this->original_NodesList.size() * i +
                    this->original_EdgesList[j]->begin;
      edge->end = this->original_NodesList.size() * i +
                  this->original_EdgesList[j]->end;
      edge->edgeID = this->EdgesList.size();
      this->EdgesList.push_back(edge);
    }
  }
  // Update reuse edge with dif
  int ENum = this->original_EdgesList.size();
  int NNum = this->original_NodesList.size();
  for (int i = 0; i < UnrollNum; i++) {
    for (int j = 0; j < this->original_EdgesList.size(); j++) {
      int index = i * ENum + j;
      int dif = this->EdgesList[index]->dif;
      if (dif > 0) {
        if ((this->EdgesList[index]->edgeType == LLReuse ||
             this->EdgesList[index]->edgeType == SLReuse) &&
            this->EdgesList[index]->dif > 0) {
          int k = (dif + i) % UnrollNum;
          int endID = this->EdgesList[index]->end % NNum;
          this->EdgesList[index]->end =
              this->NodesList[k * NNum + endID]->nodeID;
          this->EdgesList[index]->dif = (dif + i) / UnrollNum;
          this->EdgesList[index]->edgeType = SLReuse;
        }
      }
    }
  }
  this->InDFGNodeAndEdge();
  this->getRecourseII();
  this->getEarliestStep();
  this->getLastestStep();
}

void MlirDFG::genFile() {
  std::error_code ec;
  sys::fs::OpenFlags Flags = sys::fs::OF_Append;
  sys::fs::CreationDisposition Disp = sys::fs::CD_CreateAlways;
  sys::fs::FileAccess Access = sys::fs::FA_Write;
  raw_fd_ostream os("DFGInformation.out", ec, Disp, Access, Flags);
  os << "\n\n============DFG ID " << this->dfg_id << "==============";
  this->print_DFGInfo(os);
}

void MlirDFG::AfterScheduleGenFile() {
  std::error_code ec;
  sys::fs::OpenFlags Flags = sys::fs::OF_Append;
  sys::fs::CreationDisposition Disp = sys::fs::CD_CreateAlways;
  sys::fs::FileAccess Access = sys::fs::FA_Write;
  raw_fd_ostream os("AfterScheduleDFGInformation.out", ec, Disp, Access, Flags);
  os << "\n\n============DFG ID " << this->dfg_id << "==============";
  this->print_DFGInfo(os);
}

void MlirDFG::createVarGraphs() {
  // Scan var in DFG and add dependencies
  int VarGraphID = 0;
  std::map<mlir::Operation *, ReuseGraph *> varGraphs_map; 
  for (auto *node : this->NodesList) {
    if (node->nodeType == VarLoad) {
      LoadNode *Ptr = (LoadNode *)node;
      auto iter = varGraphs_map.find(Ptr->ArrayOp);
      if (iter != varGraphs_map.end()) {
        //A reuse graph of the var exists
        ReuseGraph *VG = iter->second;
        Ptr->reuseStep =
            VG->LoadStoreVec.size(); 
        VG->LoadStoreVec.push_back(Ptr);
      } else { // There is no reuse diagram for this array
// A varGraph is created
        std::vector<Node *> LoadStoreVec;
        Ptr->reuseStep = 0; 
        LoadStoreVec.push_back(Ptr);
        ReuseGraph *VG = new ReuseGraph(Ptr->ArrayOp, LoadStoreVec);
        VG->ReuseID = VarGraphID;
        varGraphs_map.insert(std::make_pair(Ptr->ArrayOp, VG));
        VarGraphID++;
      }
    } else if (node->nodeType == VarStore) {
      StoreNode *Ptr = (StoreNode *)node;
      auto iter = varGraphs_map.find(Ptr->ArrayOp);
      if (iter != varGraphs_map.end()) {
        ReuseGraph *VG = iter->second;
        Ptr->reuseStep =
            VG->LoadStoreVec.size(); 
        VG->LoadStoreVec.push_back(Ptr);
      } else { // There is no reuse diagram for this array
// A varGraph is created
        std::vector<Node *> LoadStoreVec;
        Ptr->reuseStep = 0; // 为第0个时间步
        LoadStoreVec.push_back(Ptr);
        ReuseGraph *VG = new ReuseGraph(Ptr->ArrayOp, LoadStoreVec);
        VG->ReuseID = VarGraphID;
        varGraphs_map.insert(std::make_pair(Ptr->ArrayOp, VG));
        VarGraphID++;
      }
    }
  }

  this->varGraphNum = 0;
  for (auto iter = varGraphs_map.begin(); iter != varGraphs_map.end(); iter++) {
    this->varGraphs.push_back(iter->second);
    this->varGraphNum += 1;
  }
}

void MlirDFG::addVarEdges() {
  std::map<std::pair<int, int>, int> SLVarEdgeMap;
  for (auto VG : this->varGraphs) {
    int lastVarStore = -1;
    int StoreSrc = 0;
    int StoreDst = 0;
    bool StoreSrcflag = false;
    for (int i = 0; i < VG->LoadStoreVec.size(); i++) {
      if (VG->LoadStoreVec[i]->nodeType == VarStore) {
        StoreSrc = i;
        StoreSrcflag = true;
        lastVarStore = i;
      }
      if (VG->LoadStoreVec[i]->nodeType == VarLoad && StoreSrcflag) {
        StoreDst = i;
        VG->LoadStoreVec[StoreDst]->reuseflag = true;
        SLVarEdgeMap.insert(
            std::make_pair(std::make_pair(VG->LoadStoreVec[StoreDst]->nodeID,
                                          VG->LoadStoreVec[StoreSrc]->nodeID),
                           0));
      }
    }

    if (lastVarStore != -1) {
      for (int i = VG->LoadStoreVec.size() - 1; i >= 0; i--) {
        if (VG->LoadStoreVec[i]->nodeType == VarLoad &&
            !VG->LoadStoreVec[i]->reuseflag) {
          VG->LoadStoreVec[i]->reuseflag = true;
          SLVarEdgeMap.insert(std::make_pair(
              std::make_pair(VG->LoadStoreVec[i]->nodeID,
                             VG->LoadStoreVec[lastVarStore]->nodeID),
              1));
        }
      }
    }
  }

  int SLbegin = -1;
  int SLend = -1;

  for (auto reuseE : SLVarEdgeMap) {
    for (Edge *e : this->EdgesList) {
      if (e->begin == reuseE.first.first) {
        SLend = e->end;
      }
      if (e->end == reuseE.first.second) { 
        SLbegin = e->begin;
      }
    }

    if (SLbegin != -1 && SLend != -1 &&
        this->NodesList[SLbegin]->nodeType != Cst &&
        this->NodesList[SLend]->nodeType != Cst) {
      Edge *edge = new Edge(SLbegin, SLend);
      edge->edgeID = this->EdgesList.size();
      edge->edgeType = VarSLDepen;
      edge->dif = reuseE.second;
      this->EdgesList.push_back(edge);
    }
  }
}

void MlirDFG::getNewEarliestStep(int UnrollNum) {
  int *inDegree;
  inDegree = (int *)malloc(
      (this->NodesList.size() + (UnrollNum * this->inDFGNodesList.size())) *
      sizeof(int));
  int *mark;
  mark = (int *)malloc(
      (this->NodesList.size() + (UnrollNum * this->inDFGNodesList.size())) *
      sizeof(int));

  for (int i = 0;
       i < this->NodesList.size() + (UnrollNum * this->inDFGNodesList.size());
       i++) {
    mark[i] = -1;
    inDegree[i] = -1;
  }
  for (int i = 0; i < this->inDFGNodesList.size(); i++) {
    inDegree[this->inDFGNodesList[i]->nodeID] = 0;
  }
  // The entry degree of a node is collected
  for (Edge *e : this->inDFGEdgesList) {
    if (!(e->edgeType == SLReuse && e->dif > 0) &&
        !(e->edgeType == VarSLDepen && e->dif > 0) &&
        !(e->edgeType == RecSLDepen)) {
      inDegree[e->end] += 1;
    }
  }

  int NumsCount = this->inDFGNodesList.size();
  int *toplist; //ID of the node that stores topology sorting
  toplist = (int *)malloc(this->inDFGNodesList.size() * sizeof(int));
  for (int i = 0; i < this->inDFGNodesList.size(); i++) {
    toplist[i] = -1;
  }
  int level = 0;
  int endcount = 0;
  int startcount = 0;

  while (endcount < NumsCount) {
    for (Node *node : this->inDFGNodesList) {
      if (inDegree[node->nodeID] == 0 && mark[node->nodeID] == -1) {
        toplist[endcount] = node->nodeID;
        node->earliestTimeStep = level;
        mark[node->nodeID] = level;
        endcount++;
      }
    }
    level++;
    for (int i = startcount; i < endcount; i++) {
      for (Edge *e : this->inDFGEdgesList) {
        if (!(e->edgeType == SLReuse && e->dif > 0) &&
            !(e->edgeType == VarSLDepen && e->dif > 0) &&
            !(e->edgeType == RecSLDepen)) {
          if (e->begin == toplist[i]) {
            inDegree[e->end]--;
          }
        }
      }
    }

    startcount = endcount;
  }

  if (endcount < this->inDFGNodesList.size()) {
  
  }
  int dfgStep = 0;
  for (int i = 0;
       i < this->NodesList.size() + (UnrollNum * this->inDFGNodesList.size());
       i++) {
    if (dfgStep < mark[i]) {
      dfgStep = mark[i];
    }
  }
  this->DfgStep = dfgStep + 1;
}

void MlirDFG::getEarliestStep() {
  int *inDegree;
  inDegree = (int *)malloc(this->NodesList.size() * sizeof(int));
  int *mark;
  mark = (int *)malloc(this->NodesList.size() * sizeof(int));

  for (int i = 0; i < this->NodesList.size(); i++) {
    mark[i] = -1;
    inDegree[i] = -1;
  }
  for (int i = 0; i < this->inDFGNodesList.size(); i++) {
    inDegree[this->inDFGNodesList[i]->nodeID] = 0;
  }
  //The entry degree of a node is collected
  for (Edge *e : this->inDFGEdgesList) {
    if (!(e->edgeType == SLReuse && e->dif > 0) &&
        !(e->edgeType == VarSLDepen && e->dif > 0) &&
        !(e->edgeType == RecSLDepen)) {
      inDegree[e->end] += 1;
    }
  }

  int NumsCount = this->inDFGNodesList.size();
  int *toplist; 
  toplist = (int *)malloc(this->inDFGNodesList.size() * sizeof(int));
  for (int i = 0; i < this->inDFGNodesList.size(); i++) {
    toplist[i] = -1;
  }
  int level = 0;
  int endcount = 0;
  int startcount = 0;

  while (endcount < NumsCount) {
    for (Node *node : this->inDFGNodesList) {
      if (inDegree[node->nodeID] == 0 && mark[node->nodeID] == -1) {
        toplist[endcount] = node->nodeID;
        node->earliestTimeStep = level;
        mark[node->nodeID] = level;
        endcount++;
      }
    }
    level++;
    for (int i = startcount; i < endcount; i++) {
      for (Edge *e : this->inDFGEdgesList) {
        if (!(e->edgeType == SLReuse && e->dif > 0) &&
            !(e->edgeType == VarSLDepen && e->dif > 0) &&
            !(e->edgeType == RecSLDepen)) {
          if (e->begin == toplist[i]) {
            inDegree[e->end]--;
          }
        }
      }
    }
    startcount = endcount;
  }

  if (endcount < this->inDFGNodesList.size()) {

  }
  int dfgStep = 0;
  for (int i = 0; i < this->NodesList.size(); i++) {
    if (dfgStep < mark[i]) {
      dfgStep = mark[i];
    }
  }
  this->DfgStep = dfgStep + 1;
}

int MlirDFG::getPnuStmtId() {
  int stmtId;
  if (!this->StmtsVec.empty()) {
    std::string s = this->StmtsVec[0].stmtCallee.getName().data();
    std::string stmtName = s.substr(1, s.length() - 1);
    stmtId = atoi(stmtName.c_str());
  }
  return stmtId;
}

void scc_stmt_topSort::print() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\nstmt_scc_map\n";
  for (std::map<int, int>::iterator iter = this->stmt_scc_map.begin();
       iter != this->stmt_scc_map.end(); iter++) {
    os << " stmt id " << iter->first << " scc id " << iter->second << "\n";
  }
  os << "\nscc_top_map\n";
  for (std::map<int, int>::iterator iter = this->scc_top_map.begin();
       iter != this->scc_top_map.end(); iter++) {
    os << " scc id " << iter->first << " top " << iter->second << "\n";
  }
}

void MlirDFG::getNewLastestStep(int UnrollNum) {
  // Inverse topology
  int *outDegree;
  outDegree = (int *)malloc(
      (this->NodesList.size() + (UnrollNum * this->inDFGNodesList.size())) *
      sizeof(int));
  int *mark;
  mark = (int *)malloc(
      (this->NodesList.size() + (UnrollNum * this->inDFGNodesList.size())) *
      sizeof(int));

  for (int i = 0;
       i < this->NodesList.size() + (UnrollNum * this->inDFGNodesList.size());
       i++) {
    mark[i] = -1;
    outDegree[i] = -1;
  }

  for (int i = 0; i < this->inDFGNodesList.size(); i++) {
    outDegree[this->inDFGNodesList[i]->nodeID] = 0;
  }

  //Statistics node output
  for (Edge *e : this->inDFGEdgesList) {
    if (!(e->edgeType == SLReuse && e->dif > 0) &&
        !(e->edgeType == VarSLDepen && e->dif > 0) &&
        !(e->edgeType == RecSLDepen)) {
      outDegree[e->begin] += 1;
    }
  }

  int NumsCount = this->inDFGNodesList.size();
  int *toplist; 
  toplist = (int *)malloc(this->inDFGNodesList.size() * sizeof(int));
  for (int i = 0; i < this->inDFGNodesList.size(); i++) {
    toplist[i] = -1;
  }
  int level = 0;
  int endcount = 0;
  int startcount = 0;

  //Elongate the activity range of the DFG operator
  int addRange = ((double)(this->LSNodeNum) / (double)PERow) >
                         ((double)(this->PeNodeNum) / (double)(PERow * PERow))
                     ? (this->LSNodeNum / PERow)
                     : (this->PeNodeNum / (PERow * PERow));
  this->DfgStep = this->DfgStep + AfterUnrollDFGLength;

  while (endcount < NumsCount) {
    for (Node *node : this->inDFGNodesList) {
      if (outDegree[node->nodeID] == 0 && mark[node->nodeID] == -1) {
        toplist[endcount] = node->nodeID;
        node->latestTimeStep = (this->DfgStep - 1) - level;
        mark[node->nodeID] = level;
        endcount++;
      }
    }
    level++;
    for (int i = startcount; i < endcount; i++) {
      for (Edge *e : this->inDFGEdgesList) {
        if (!(e->edgeType == SLReuse && e->dif > 0) &&
            !(e->edgeType == VarSLDepen && e->dif > 0) &&
            !(e->edgeType == RecSLDepen)) {
          if (e->end == toplist[i]) {
            outDegree[e->begin]--;
          }
        }
      }
    }
    startcount = endcount;
  }
}

void MlirDFG::getLastestStep() {
  //Inverse topology
  int *outDegree;
  outDegree = (int *)malloc(this->NodesList.size() * sizeof(int));
  int *mark;
  mark = (int *)malloc(this->NodesList.size() * sizeof(int));

  for (int i = 0; i < this->NodesList.size(); i++) {
    mark[i] = -1;
    outDegree[i] = -1;
  }

  for (int i = 0; i < this->inDFGNodesList.size(); i++) {
    outDegree[this->inDFGNodesList[i]->nodeID] = 0;
  }
  for (Edge *e : this->inDFGEdgesList) {
    if (!(e->edgeType == SLReuse && e->dif > 0) &&
        !(e->edgeType == VarSLDepen && e->dif > 0) &&
        !(e->edgeType == RecSLDepen)) {
      outDegree[e->begin] += 1;
    }
  }

  int NumsCount = this->inDFGNodesList.size();
  int *toplist; 
  toplist = (int *)malloc(this->inDFGNodesList.size() * sizeof(int));
  for (int i = 0; i < this->inDFGNodesList.size(); i++) {
    toplist[i] = -1;
  }
  int level = 0;
  int endcount = 0;
  int startcount = 0;

  this->DfgStep = this->DfgStep + DFGLength;
  while (endcount < NumsCount) {
    for (Node *node : this->inDFGNodesList) {
      if (outDegree[node->nodeID] == 0 && mark[node->nodeID] == -1) {
        toplist[endcount] = node->nodeID;
        node->latestTimeStep = (this->DfgStep - 1) - level;
        mark[node->nodeID] = level;
        endcount++;
      }
    }
    level++;
    for (int i = startcount; i < endcount; i++) {
      for (Edge *e : this->inDFGEdgesList) {
        if (!(e->edgeType == SLReuse && e->dif > 0) &&
            !(e->edgeType == VarSLDepen && e->dif > 0) &&
            !(e->edgeType == RecSLDepen)) {
          if (e->end == toplist[i]) {
            outDegree[e->begin]--;
          }
        }
      }
    }
    startcount = endcount;
  }
}

void MlirDFG::InDFGNodeAndEdge() {
  int countReuse = 0;
  for (Node *node : this->NodesList) {
    if (node->nodeInDFG && !node->reuseflag && node->nodeType != VarLoad &&
        node->nodeType != VarStore) {
      this->inDFGNodesList.push_back(node);
    }
    if (node->reuseflag) {
      countReuse++;
    }
  }
  for (Edge *e : this->EdgesList) {
    if (e->edgeType != Delete && e->edgeType != VarL && e->edgeType != VarS) {
      this->inDFGEdgesList.push_back(e);
    }
  }
}

int MlirDFG::divideSubReuseGraphs() {
  //The reuse graph for each array is divided into subgraphs by nodes that are only distant in the innermost layer
  std::vector<ReuseGraph *> reuseGraph_new;
  std::vector<RfCgraTrans::Node *> tempLoadStoreVec;
  ReuseGraph *subReuseGraph;
  for (ReuseGraph *RG : this->reuseGraphs) {
    if (RG->ReuseID == -1) {
      continue;
    }
    int *mark = (int *)calloc(RG->LoadStoreVec.size(), sizeof(int));
    for (int i = 0; i < RG->LoadStoreVec.size(); i++) {
      bool storeReuseG = false;
      //The first node that has not been marked needs to request a new graph
      if (mark[i] == 0) {
        tempLoadStoreVec.clear();
        subReuseGraph = new ReuseGraph(RG->Array, tempLoadStoreVec);
        subReuseGraph->ReuseID = reuseGraph_new.size();
        subReuseGraph->LoadStoreVec.push_back(RG->LoadStoreVec[i]);
        mark[i] = 1;
        storeReuseG = true;
      }

      for (int j = 0; j < RG->LoadStoreVec.size(); j++) {
        if (mark[j] == 0 && i != j) {
          if (isOnlyInnerestShift(RG->LoadStoreVec[i], RG->LoadStoreVec[j]) ==
              1) {
            subReuseGraph->LoadStoreVec.push_back(RG->LoadStoreVec[j]);
            mark[j] = 1;
          }
        }
      }

      if (storeReuseG) {
        if (subReuseGraph->LoadStoreVec.size() > 1) {
          reuseGraph_new.push_back(subReuseGraph);
        }
      }
    }
  }

  this->reuseGraphs.clear();
  this->reuseGraphs = reuseGraph_new;
  this->reuseGraphNum = reuseGraph_new.size();
  return this->reuseGraphNum;
}

void MlirDFG::sortByIndexReuseStep() {
  for (ReuseGraph *RG : this->reuseGraphs) {
    std::vector<Node *> LoadStoreVec_new;
    for (int i = 0; i < RG->LoadStoreVec.size() - 1; i++) {
      Node *temp;
      for (int j = i + 1; j < RG->LoadStoreVec.size(); j++) {
        if (this->compShift(RG->LoadStoreVec[i], RG->LoadStoreVec[j]) >= 0) {
          temp = RG->LoadStoreVec[j];
          RG->LoadStoreVec[j] = RG->LoadStoreVec[i];
          RG->LoadStoreVec[i] = temp;
        }
      }
    }
  }
}

void MlirDFG::addReuseEdges() {
// After sorting
// S is the interval, traversing from back to front to search for reusable L
// key is the end and value is the beginning. <<key,value>,dif>
  std::map<std::pair<int, int>, int> SLreuseEdgeMap;
  std::map<std::pair<int, int>, int> SLInSingleSEdgeMap;
  std::map<std::pair<int, int>, int> LLreuseEdgeMap;
  int first;
  for (ReuseGraph *RG : this->reuseGraphs) {
    int StoreSrc = 0;
    int StoreDst = 0;
    bool StoreSrcflag = false;
    for (int i = RG->LoadStoreVec.size() - 1; i >= 0; i--) {
      if (RG->LoadStoreVec[i]->nodeType == ArrayStore) {
        StoreSrc = i;
        StoreSrcflag = true; 
        first = 1;
      }
      if (PBPMethod && this->methodflag == 0) {
        if (RG->LoadStoreVec[i]->nodeType == ArrayLoad && StoreSrcflag &&
            first) {
          StoreDst = i;

          int dif = this->compShift(RG->LoadStoreVec[StoreSrc],
                                    RG->LoadStoreVec[StoreDst]);
          if (dif == 0 && RG->LoadStoreVec[StoreDst]->belongS !=
                              RG->LoadStoreVec[StoreSrc]->belongS) {
            RG->LoadStoreVec[StoreDst]->reuseflag = true;
            SLreuseEdgeMap.insert(std::make_pair(
                std::make_pair(RG->LoadStoreVec[StoreDst]->nodeID,
                               RG->LoadStoreVec[StoreSrc]->nodeID),
                dif));
            first = 0;
          } else if (dif != 0 && RG->LoadStoreVec[StoreDst]->belongS ==
                                     RG->LoadStoreVec[StoreSrc]->belongS) {
            SLInSingleSEdgeMap.insert(std::make_pair(
                std::make_pair(RG->LoadStoreVec[StoreDst]->nodeID,
                               RG->LoadStoreVec[StoreSrc]->nodeID),
                dif));
            first = 0;
          }
        }
      } else {
        if (RG->LoadStoreVec[i]->nodeType == ArrayLoad && StoreSrcflag) {
          StoreDst = i;
          RG->LoadStoreVec[StoreDst]->reuseflag = true;
          int dif = this->compShift(RG->LoadStoreVec[StoreSrc],
                                    RG->LoadStoreVec[StoreDst]);
          SLreuseEdgeMap.insert(
              std::make_pair(std::make_pair(RG->LoadStoreVec[StoreDst]->nodeID,
                                            RG->LoadStoreVec[StoreSrc]->nodeID),
                             dif));
        }
      }
    }
    if (PBPMethod && this->methodflag == 0) {
    } else {
      bool LoadSrcflag = false;
      int LoadSrc = 0;
      int LoadDst = 0;
      bool breakStore = false;
      for (int i = RG->LoadStoreVec.size() - 1; i >= 0; i--) {
        //Intermediate Load break
        if (RG->LoadStoreVec[i]->nodeType == ArrayStore) {
          breakStore = true;
          LoadSrcflag = false;
        }

        if (RG->LoadStoreVec[i]->nodeType == ArrayLoad &&
            !RG->LoadStoreVec[i]->reuseflag) {
          if (!LoadSrcflag) {
            LoadSrc = i;
            RG->LoadStoreVec[i]->reuseflag = false;
            LoadSrcflag = true;
            breakStore = false;
          } else if (LoadSrcflag && !breakStore) { 
            LoadDst = i;
            RG->LoadStoreVec[LoadDst]->reuseflag = true;
            int dif = this->compShift(RG->LoadStoreVec[LoadSrc],
                                      RG->LoadStoreVec[LoadDst]);
            LLreuseEdgeMap.insert(std::make_pair(
                std::make_pair(RG->LoadStoreVec[LoadDst]->nodeID,
                               RG->LoadStoreVec[LoadSrc]->nodeID),
                dif));
          }
        }
      }
    }
  }
  int begin;
  if (PBPMethod && this->methodflag == 0) {

    for (auto reuseE : SLInSingleSEdgeMap) {
      Edge *edge = new Edge(reuseE.first.second, reuseE.first.first);
      edge->edgeID = this->EdgesList.size();
      edge->edgeType = SLReuse;
      edge->dif = reuseE.second;
      this->EdgesList.push_back(edge);
    }
  }

  int SLbegin = -1;
  int SLend = -1;
// Find the child node of L and the parent node replacement of S
// Special handling cst
  for (auto reuseE : SLreuseEdgeMap) {
    for (Edge *e : this->EdgesList) {
      if (e->begin == reuseE.first.first) { 
        SLend = e->end;
        e->edgeType = Delete;
      }
      if (e->end == reuseE.first.second) { 
        if (this->NodesList[e->begin]->nodeType == Cst) {
          SLbegin = -1;
        } else {
          SLbegin = e->begin;
        }
      }
    }
    if (SLbegin != -1 && SLend != -1) {
      Edge *edge = new Edge(SLbegin, SLend);
      edge->edgeID = this->EdgesList.size();
      edge->edgeType = SLReuse;
      edge->dif = reuseE.second;
      this->EdgesList.push_back(edge);
    }
  }
  int original_dif = 0;
  for (auto reuseE : LLreuseEdgeMap) {
    for (Edge *e : this->EdgesList) {
      if (e->begin == reuseE.first.first) {
        Edge *edge = new Edge(reuseE.first.second, e->end);
        edge->edgeID = this->EdgesList.size();
        edge->edgeType = LLReuse;
        e->edgeType = Delete;
        edge->dif = reuseE.second + e->dif;
        this->EdgesList.push_back(edge);
      }
    }
  }
}

// Determine whether two nodes in the resueGraph have the same offset value after adjusting the subdivision (only offset in the innermost layer)
// 0: same -1: nodel1 < node2 1: node1 > node2
int MlirDFG::compShift(Node *node1, Node *node2) {
  if (node1->innerestIndex == -1 ||
      node2->innerestIndex == -1) 
    return 0;                    
  //The innermost (or closest) layer is the iteration variable
  if (node1->NodeShift[node1->innerestIndex] ==
      node2->NodeShift[node2->innerestIndex])
    return 0;
  else if (node1->NodeShift[node1->innerestIndex] <
           node2->NodeShift[node2->innerestIndex])
    return node1->NodeShift[node1->innerestIndex] -
           node2->NodeShift[node2->innerestIndex];

  return node1->NodeShift[node1->innerestIndex] -
         node2->NodeShift[node2->innerestIndex];
}

// Check whether the two nodes (Load Store type) are offset only in the innermost layer
// 1: Only the inner layer is offset
// 0: Not only the innermost layer is offset. -1: there is an invalid node input or other invalid node input
// [Currently not returned, use the isSameShift function to judge] 
// 2: Only the inner layer is offset and the offset value is the same
int MlirDFG::isOnlyInnerestShift(Node *node1, Node *node2) {
  if (node1->nodeType == ArrayLoad) {
    LoadNode *node1 = (LoadNode *)node1;
  } else if (node1->nodeType == ArrayStore) {
    StoreNode *node1 = (StoreNode *)node1;
  } else
    return -1; 
  if (node2->nodeType == ArrayLoad) {
    LoadNode *node2 = (LoadNode *)node2;
  } else if (node2->nodeType == ArrayStore) {
    StoreNode *node2 = (StoreNode *)node2;
  } else
    return -1; 

  if (node1->iterOrder.size() != node2->iterOrder.size())
    return -1;

  for (int i = 0; i < node1->iterOrder.size(); i++) {
    if (node1->iterOrder[i] != node2->iterOrder[i])
      return 0;
  }

  for (int i = 0; i < node1->NodeShift.size(); i++) {
    if (i != node1->innerestIndex &&
        (node1->NodeShift[i] - node2->NodeShift[i]) != 0) {
      return 0;
    }
  }
  return 1;
}

MlirDFG::~MlirDFG() {
  this->StmtsVec.clear();
  this->forIterVec.clear();
  for (Node *node : this->NodesList) {
    delete node;
  }
  for (Edge *edge : this->EdgesList) {
    delete edge;
  }
  this->EdgesList.clear();
  this->reuseGraphs.clear();
}

mlir::Operation *MlirDFG::getLoadStoreAlloc(ScopInformation *scopInformation,
                                            mlir::Value value,
                                            StmtInfomation *s) {
  int calleeIndex = 0;
  int funcGIndex = 0;
  mlir::Operation *allocOp = scopInformation->ArrayAllocVec[0];
  calleeIndex = s->ValueIndexInCallee(value);
  funcGIndex =
      scopInformation->ValueIndexInFuncG(s->stmtCaller.getOperand(calleeIndex));
  if (funcGIndex == -1) {
    for (mlir::Operation *a : scopInformation->ArrayAllocVec) {
      if (mlir::memref::AllocaOp allocaOp =
              dyn_cast<mlir::memref::AllocaOp>(a)) {
        if (s->stmtCaller.getOperand(calleeIndex) == allocaOp.getResult()) {
          return allocaOp;
        }
      }
    }
  } else {
    for (int i = 0; i < scopInformation->ArrayArgIndexVec.size(); i++) {
      if (funcGIndex == scopInformation->ArrayArgIndexVec[i]) {
        allocOp = scopInformation->ArrayAllocVec[i];
      }
    }
    return allocOp;
  }
}

void MlirDFG::getRecourseII() {
  double bankSrcC = 0;
  double peSrcC = 0;
  int reuseLoad = 0;
  for (Node *node : this->inDFGNodesList) {
    if (node->nodeType == ArrayStore || node->nodeType == ArrayLoad ||
        node->nodeType == noFitIndexLoad || node->nodeType == noFitIndexStore) {
      bankSrcC = bankSrcC + 1;
      this->LSNodeNum++;
    } else {
      peSrcC = peSrcC + 1;
      this->PeNodeNum++;
    }
  }

  for (Node *node : this->NodesList) {
    if (node->reuseflag == 1) {
      reuseLoad++;
    }
  }
  int bankII = ceil(bankSrcC / PERow);
  int peII = ceil(peSrcC / (PERow * PERow));
  this->TheoreticalII = bankII > peII ? bankII : peII;
  this->ResMII = this->TheoreticalII;
  this->reuseDataNum = reuseLoad;
}

int MlirDFG::find_NodeId(mlir::Value value) {
  for (Node *node : this->NodesList) {
    if (node->SSA == value) {
      return node->nodeID;
    }
  }
  return -1;
}

int MlirDFG::find_NodeId(Node *node) {
  for (Node *n : this->NodesList) {
    if (node->SSA == n->SSA) {
      return node->nodeID;
    }
  }
  return -1;
}

int MlirDFG::find_EdgeId(std::vector<Edge *> EdgesList, int begin, int end) {
  for (int i = 0; i < EdgesList.size(); i++) {
    if (EdgesList[i]->begin == begin && EdgesList[i]->end == end)
      return i;
  }
  return -1;
}

Edge *MlirDFG::create_edge(Node *start, Node *end) {
  int start_id = find_NodeId(start);
  int end_id = find_NodeId(end);
  Edge *edge = new Edge(start_id, end_id);
  return edge;
}

void MlirDFG::add_VarLoad(mlir::Value value, ScopInformation *scopInformation,
                          StmtInfomation *s, int end) {
  mlir::Operation *allocOp = this->getLoadStoreAlloc(scopInformation, value, s);
  LoadNode *Ptr =
      new LoadNode(allocOp, this->NodesList.size(), scopInformation, s);
  Ptr->belongS = s->stmtCaller;
  this->NodesList.push_back(Ptr);
  Edge *edge = new Edge(this->NodesList.size() - 1, end);
  edge->edgeType = VarL;
  this->EdgesList.push_back(edge);
}

void MlirDFG::judge_edge(ScopInformation *scopInformation, StmtInfomation *s,
                         int begin, mlir::Value value, int end) {
  if (begin != -1) {
    Edge *edge1 = new Edge(begin, end); 
    this->EdgesList.push_back(edge1);
    if (this->NodesList[begin]->nodeType == VarLoad) {
      edge1->edgeType = VarL;
    }
  } else if (begin == -1) {
    this->add_VarLoad(value, scopInformation, s, end);
  }
}

void MlirDFG::add_edges(mlir::Operation *op, ScopInformation *scopInformation,
                        StmtInfomation *s) {
  // Add
  if (mlir::AddFOp AddOp = dyn_cast<mlir::AddFOp>(op)) {
    int begin1 = find_NodeId(AddOp.rhs());
    int begin2 = find_NodeId(AddOp.lhs());
    int end = this->NodesList.size() - 1;
    this->judge_edge(scopInformation, s, begin1, AddOp.rhs(), end);
    this->judge_edge(scopInformation, s, begin2, AddOp.lhs(), end);
  }
  // sub
  if (mlir::SubFOp SubOp = dyn_cast<mlir::SubFOp>(op)) {
    int begin1 = find_NodeId(SubOp.rhs());
    int begin2 = find_NodeId(SubOp.lhs());
    int end = this->NodesList.size() - 1;
    this->judge_edge(scopInformation, s, begin1, SubOp.rhs(), end);
    this->judge_edge(scopInformation, s, begin2, SubOp.lhs(), end);
  }
  // mul
  if (mlir::MulFOp MulOp = dyn_cast<mlir::MulFOp>(op)) {
    int begin1 = find_NodeId(MulOp.rhs());
    int begin2 = find_NodeId(MulOp.lhs());
    int end = this->NodesList.size() - 1;
    this->judge_edge(scopInformation, s, begin1, MulOp.rhs(), end);
    this->judge_edge(scopInformation, s, begin2, MulOp.lhs(), end);
  }
  // Div
  if (mlir::DivFOp DivOp = dyn_cast<mlir::DivFOp>(op)) {
    int begin1 = find_NodeId(DivOp.rhs());
    int begin2 = find_NodeId(DivOp.lhs());
    int end = this->NodesList.size() - 1;
    this->judge_edge(scopInformation, s, begin1, DivOp.rhs(), end);
    this->judge_edge(scopInformation, s, begin2, DivOp.lhs(), end);
  }
  // Store
  if (mlir::AffineStoreOp StoreOp = dyn_cast<mlir::AffineStoreOp>(op)) {
    int begin = find_NodeId(StoreOp.getOperand(0));
    int end = this->NodesList.size() - 1;
    Edge *edge1 = new Edge(begin, end);
    if (this->NodesList[this->NodesList.size() - 1]->nodeType == VarStore) {
      edge1->edgeType = VarS;
    }
    this->EdgesList.push_back(edge1);
  }
  if (mlir::math::SqrtOp SqrtOp = dyn_cast<mlir::math::SqrtOp>(op)) {
    int begin = find_NodeId(SqrtOp.getOperand());
    int end = this->NodesList.size() - 1;
    this->judge_edge(scopInformation, s, begin, SqrtOp.getOperand(), end);
  }
}

void MlirDFG::createDfg(ScopInformation *scopInformation) {
  llvm::raw_ostream &os = llvm::outs();
  int ReuseGraphID = 0;
  std::map<mlir::Operation *, ReuseGraph *> reuseGraphs_map;
  for (StmtInfomation s : this->StmtsVec) {
    for (mlir::Region::OpIterator it = s.stmtCallee.body().getOps().begin();
         it != s.stmtCallee.body().getOps().end(); it++) {
      // Cst
      if (mlir::ConstantOp CstOp = dyn_cast<mlir::ConstantOp>(*it)) {
        Node *Ptr = new CstNode(CstOp, this->NodesList.size());
        Ptr->nodeInDFG = 0;
        Ptr->belongS = s.stmtCaller;
        this->NodesList.push_back(Ptr);
      }
      // Load
      if (mlir::AffineLoadOp LoadOp = dyn_cast<mlir::AffineLoadOp>(*it)) {
        mlir::Value value = LoadOp.getOperand(0);
        /// debug
        mlir::Operation *allocOp =
            this->getLoadStoreAlloc(scopInformation, value, &s);
        LoadNode *Ptr = new LoadNode(LoadOp, allocOp, this->NodesList.size(),
                                     scopInformation, &s);
        Ptr->nodeInDFG = 1;
        Ptr->belongS = s.stmtCaller;
        this->NodesList.push_back(Ptr);
        if (Ptr->nodeType == ArrayLoad) {
          auto iter = reuseGraphs_map.find(allocOp);
          if (iter != reuseGraphs_map.end()) {
            //There is a reuse graph for this array
            ReuseGraph *RG = iter->second;
            Ptr->reuseStep =
                RG->LoadStoreVec.size(); 
            RG->LoadStoreVec.push_back(Ptr);
          } else { // There is no reuse diagram for this array
            std::vector<Node *> LoadStoreVec;
            Ptr->reuseStep = 0;
            LoadStoreVec.push_back(Ptr);
            ReuseGraph *RG = new ReuseGraph(allocOp, LoadStoreVec);
            RG->ReuseID = ReuseGraphID;
            reuseGraphs_map.insert(std::make_pair(allocOp, RG));
            ReuseGraphID++;
          }
        }
      }
      // // Store
      if (mlir::AffineStoreOp StoreOp = dyn_cast<mlir::AffineStoreOp>(*it)) {
        mlir::Value value = StoreOp.getMemRef();
        mlir::Operation *allocOp =
            this->getLoadStoreAlloc(scopInformation, value, &s);
        StoreNode *Ptr = new StoreNode(StoreOp, allocOp, this->NodesList.size(),
                                       scopInformation, &s);
        Ptr->nodeInDFG = 1;
        Ptr->belongS = s.stmtCaller;
        this->NodesList.push_back(Ptr);
        this->add_edges(StoreOp, scopInformation, &s);
        if (Ptr->nodeType == ArrayStore) {
          auto iter = reuseGraphs_map.find(allocOp);
          if (iter != reuseGraphs_map.end()) {
            ReuseGraph *RG = iter->second;
            Ptr->reuseStep =
                RG->LoadStoreVec.size();
            RG->LoadStoreVec.push_back(Ptr);
          } else { 
            std::vector<Node *> LoadStoreVec;
            Ptr->reuseStep = 0; 
            LoadStoreVec.push_back(Ptr);
            ReuseGraph *RG = new ReuseGraph(allocOp, LoadStoreVec);
            RG->ReuseID = ReuseGraphID;
            reuseGraphs_map.insert(std::make_pair(allocOp, RG));
            ReuseGraphID++;
          }
        } else if (Ptr->nodeType == noFitIndexStore) {
          auto iter = reuseGraphs_map.find(allocOp);
          if (iter != reuseGraphs_map.end()) {
            // The reuse diagram exists. Mark the diagram
            ReuseGraph *RG = iter->second;
            RG->ReuseID = -1;
            Ptr->reuseStep =
                RG->LoadStoreVec.size(); // The time step is its current subscript in the array
            RG->LoadStoreVec.push_back(Ptr);
          } else {
            std::vector<Node *> LoadStoreVec;
            Ptr->reuseStep = 0; 
            LoadStoreVec.push_back(Ptr);
            ReuseGraph *RG = new ReuseGraph(allocOp, LoadStoreVec);
            RG->ReuseID = -1;
            reuseGraphs_map.insert(std::make_pair(allocOp, RG));
          }
        }
      }
      // add  
      if (mlir::AddFOp AddOp = dyn_cast<mlir::AddFOp>(*it)) {
        Node *Ptr = new AddNode(AddOp, this->NodesList.size());
        Ptr->nodeInDFG = 1;
        Ptr->belongS = s.stmtCaller;
        this->NodesList.push_back(Ptr);
        this->add_edges(AddOp, scopInformation, &s);
      }
      // sub
      if (mlir::SubFOp SubOp = dyn_cast<mlir::SubFOp>(*it)) {
        Node *Ptr = new SubNode(SubOp, this->NodesList.size());
        Ptr->nodeInDFG = 1;
        Ptr->belongS = s.stmtCaller;
        this->NodesList.push_back(Ptr);
        this->add_edges(SubOp, scopInformation, &s);
      }
      // mul
      if (mlir::MulFOp MulOp = dyn_cast<mlir::MulFOp>(*it)) {
        Node *Ptr = new MulNode(MulOp, this->NodesList.size());
        Ptr->nodeInDFG = 1;
        Ptr->belongS = s.stmtCaller;
        this->NodesList.push_back(Ptr);
        this->add_edges(MulOp, scopInformation, &s);
      }
      // Div
      if (mlir::DivFOp DivOp = dyn_cast<mlir::DivFOp>(*it)) {
        Node *Ptr = new DivNode(DivOp, this->NodesList.size());
        Ptr->nodeInDFG = 1;
        Ptr->belongS = s.stmtCaller;
        this->NodesList.push_back(Ptr);
        this->add_edges(DivOp, scopInformation, &s);
      }
      // math.sqrt
      if (mlir::math::SqrtOp sqrtOp = dyn_cast<mlir::math::SqrtOp>(*it)) {
        Node *Ptr = new SqrtNode(sqrtOp, this->NodesList.size());
        Ptr->nodeInDFG = 1;
        Ptr->belongS = s.stmtCaller;
        this->NodesList.push_back(Ptr);
        this->add_edges(sqrtOp, scopInformation, &s);
      }
    }
  }
  int edgeIDCount = 0;
  for (Edge *e : this->EdgesList) {
    e->edgeID = edgeIDCount;
    edgeIDCount++;
  }
  this->reuseGraphNum = 0; 
  for (auto iter = reuseGraphs_map.begin(); iter != reuseGraphs_map.end();
       iter++) {
    if (iter->second->ReuseID != -1) {
      this->reuseGraphs.push_back(iter->second);
      this->reuseGraphNum += 1;
    }
  }
}

// step2 Remove Cst operators and edges from DFG graphs
void MlirDFG::deleteCstEdge() {
  for (int i = 0; i < this->EdgesList.size(); i++) {
    int begin = this->EdgesList[i]->begin;
    if (this->NodesList[begin]->nodeType == Cst) {
      this->EdgesList[i]->edgeType = Delete;
    }
  }
}

void MlirDFG::print_DFGInfo(raw_fd_ostream &os) {
  os << "\n\n------print_DFGInfo file--------\n";
  os << "\ndfg_id\n";
  os << this->dfg_id;
  os << "\n该DFG的II\n";
  os << this->TheoreticalII;
  os << "\ndfg_dim\n";
  os << this->dfg_dim;
  os << "\ndfg_node_info\n";
  for (Node *node : this->inDFGNodesList) {
    node->print_node(os);
  }
  os << "\ndfg_edge_info\n";
  for (Edge *e : this->inDFGEdgesList) {
    e->print_edge(os);
  }
}

void MlirDFG::print_DFGInfo() {
  llvm::raw_ostream &os = llvm::outs();
  os << "\n\n------print_DFGInfo--------\n";
  os << "\ndfg_id\n";
  os << this->dfg_id;
  os << "\ndfg_dim\n";
  os << this->dfg_dim;
  os << "\nDFG Stmt\n";
  int i = 0;
  for (StmtInfomation s : this->StmtsVec) {
    os << "\nNo: " << i << " Stmt\n";
    s.print_stmtInfo();
    i++;
  }

  os << "\ndfg_node_info\n";
  for (Node *node : this->NodesList) {
    node->print_node();
  }
  os << "\ndfg_edge_info\n";
  for (Edge *e : this->EdgesList) {
    e->print_edge();
  }
}

void MlirDFG::insert_Stmts(StmtInfomation Stmt) {
  this->StmtsVec.push_back(Stmt);
}

void MlirDFG::find_forIter(mlir::AffineForOp forOp, mlir::FuncOp g,
                           mlir::ModuleOp moduleop) {
  mlir::AffineForOp tempforOp = forOp;
  this->forIterVec.push_back(tempforOp);
  bool flag = true;
  while (flag) {
    if (mlir::AffineForOp parentforOp = dyn_cast<mlir::AffineForOp>(
            tempforOp.getBodyRegion().getParentOp()->getParentOp())) {
      tempforOp = parentforOp;
      this->forIterVec.push_back(parentforOp);
      this->dfg_dim++;
    } else {
      flag = false;
    }
  }
}

void MlirDFG::print_forIter() {
  llvm::raw_ostream &os = llvm::outs();
  for (int i = 0; i < this->forIterVec.size(); i++) {
    os << "\n forIterVec \n";
    forIterVec[i].print(os);
  }
}
} // namespace RfCgraTrans
