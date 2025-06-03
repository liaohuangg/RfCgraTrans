#include "RfCgraTrans/Support/OslScop.h"
#include "RfCgraTrans/Support/OslScopStmtOpSet.h"
#include "RfCgraTrans/Support/OslSymbolTable.h"
#include "RfCgraTrans/Support/ScopStmt.h"
#include "RfCgraTrans/Target/OpenScop.h"
#include "RfCgraTrans/Transforms/PlutoTransform.h"

#include "pluto/internal/pluto.h"
#include "pluto/matrix.h"
#include "pluto/osl_pluto.h"
#include "pluto/pluto.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/AffineStructures.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"
#include <queue>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
namespace RfCgraTrans {
# define PERow 4
# define Tf 20
# define Experimental_option 1
# define PBPMethod 0
# define search_trans_unroll_Switch 1 
# define final_unroll_Switch 2
# define schedule_Switch 1
# define DFGLength 3
# define AfterUnrollDFGLength 3
# define searchScheduleNum 40
# define subScheduleNum 20
/*
0  local transformation
1  global transformation
3  nofuse transformation
4  original loop
*/
class ScopInformation;
class StmtInfomation;
class LoadNode;
class StoreNode;
class MlirDFG;

class plutoTrans{
  public:
  plutoCost_Matrix *plutoCost;
  mlir::FuncOp g;
};


//fuse component
class fuseComponent{
  public:
  int compId;
  int transId;
  //<cmpID, stmtID>
  int start_scc_id;
  int end_scc_id;
  int innerIndex;
  int flag;
  long long Cost=0;
};

class partitionCut{
  public:
  int partitionID;
  int *fusionCondition;
  int *interCondition;
  void fusCut(std::vector<fuseComponent*> CompV,int scc_num);
  void print(int scc_num);
  void initial(int scc_num);
};

class scc_stmt_topSort{
  public:
  void print();
  std::map<int,int> stmt_scc_map;
  std::map<int,int> scc_top_map;
  int scc_num;
};

//single transformation
class singleTrans{
  public:
  // void find_Comp(int stmtId,MlirDFG dfg);
  int transId;
  std::vector<fuseComponent*> CompV;
  partitionCut parC;
  int scc_num;
  int compNumber;
  int currentCompIndex=-1;
  int totalReuseData = 0;
  void flagComp(std::vector<int> stmtIdV,scc_stmt_topSort scc_stmt_topSort_map);
  void print();
};




//NodeType
enum NodeType
{
   ArrayLoad = 1,  
   ArrayStore = 2,
   Add = 3,
   Sub = 4,
   Mul = 5,
   Div = 6,
   Cst = 7,//constant
   VarLoad = 8,  
   VarStore = 9,
   Sqrt = 10,
   noFitIndexLoad = 11,  
   noFitIndexStore = 12
};

enum EdgeType
{
   LLReuse = 1, //Reused edge
   Delete = 2,//Edges that need to be deleted
   Normal = 3 ,
   SLReuse = 4, //The LLRuse of the last group after unroll is also modified to the SLRuse edge and does not participate in the input calculation
   VarL = 5,//VarLoad
   VarS = 6,//VarStore
   VarSLDepen = 7,//VarS L
   RecSLDepen = 8//Dependency constraints across the innermost iteration
};

//NodePriority
enum NodePriority
{
   level1 = 1,  
   level2 = 2,
   level3 = 3
};


//Node
class Node{
  public:
  Node(){};
  virtual void print_node();
  virtual void print_node(raw_fd_ostream &os);
  Node(const Node *node);
  NodeType nodeType;
  std::string label;
  NodePriority nodePriority;
  mlir::Value SSA;
  int timeStep = 0;
  //Earliest time step
  int earliestTimeStep = 0;
  //Latest time step
  int latestTimeStep = 0;
  int nodeID;
  //Reuse the time step of the data graph
  int reuseStep;
  int UnrollID = 0;
  //Which statement
  mlir::CallOp belongS;
  // load store Operator dependent use offsets
  SmallVector<int, 8U> NodeShift; // The offset value corresponding to each position (subscript) of the array (AffineMap result)
  SmallVector<int, 8U> iterFlag;  // Each position in the array corresponds to iter, which is 1; The corresponding cons (two kinds: cons of map, cons of s), is 0;
  SmallVector<int, 8U> iterOrder; // The iter corresponding to each position of the array is the iterated order of iter in the loop (0 is the innermost layer); The corresponding cons is an invalid bit and is set to -1.
  int innerestIndex; // The index where the innermost loop in the array (for each PNU, the innermost loop that wraps it) is located in the array; -1 is constant and has no innermost layer; If the innermost layer is not passed in, it is the index of the closest layer in the passed in.
  int nodeInDFG = 1;// 0 is not in the DFG and 1 is in the DFG
  bool reuseflag = false;
};

class LoadNode:public Node{
  public:
  LoadNode(mlir::AffineLoadOp load,mlir::Operation * alloc,int nodeIDCount, ScopInformation *scopInformation, StmtInfomation *s);
  LoadNode(mlir::Operation * alloc,int nodeIDCount, ScopInformation *scopInformation, StmtInfomation *s);
  LoadNode(const LoadNode *loadnode);
  mlir::Operation * ArrayOp;
  mlir::AffineLoadOp load;
  void print_node();
  void print_node(raw_fd_ostream &os);
  // Location mapping: s passes in the position of the input in AffineMap starting from 0
  SmallVector<int, 8U> stmtParamLocationVec; // vec installs an index that starts at 0
  void getStmtParamLocationVec(mlir::AffineLoadOp load, StmtInfomation *s);
  // Find the value of the offset correlation vector
  void getLoadShift(ScopInformation *scopInformation,mlir::AffineLoadOp load,StmtInfomation *s);
};

class StoreNode:public Node{
  public:
  StoreNode(mlir::AffineStoreOp store,mlir::Operation * alloc,int nodeIDCount, ScopInformation *scopInformation, StmtInfomation *s);
  StoreNode(const StoreNode *n);
  mlir::Operation * ArrayOp;
  mlir::AffineStoreOp store;
  void print_node();
  void print_node(raw_fd_ostream &os);
  SmallVector<int, 8U> stmtParamLocationVec;
  void getStmtParamLocationVec(mlir::AffineStoreOp store, StmtInfomation *s);
  void getStoreShift(ScopInformation *scopInformation,mlir::AffineStoreOp store,StmtInfomation *s);
};

//ADD
class AddNode:public Node{
  public:
  AddNode(const AddNode *node);
  AddNode(mlir::AddFOp AddOp,int nodeIDCount);
  void print_node();
  void print_node(raw_fd_ostream &os);
};
//Sub
class SubNode:public Node{
  public:
  SubNode(const SubNode *node);
  SubNode(mlir::SubFOp SubOp,int nodeIDCount);
  void print_node();
  void print_node(raw_fd_ostream &os);
};

//mul
class MulNode:public Node{
  public:
  MulNode(const  MulNode *node);
  MulNode(mlir::MulFOp MulOp,int nodeIDCount);
  void print_node();
  void print_node(raw_fd_ostream &os);
};

//div
class DivNode:public Node{
  public:
  DivNode(const  DivNode *node);
  DivNode(mlir::DivFOp DivOp,int nodeIDCount);
  void print_node();
  void print_node(raw_fd_ostream &os);
};
//cst
class CstNode:public Node{
  public:
  CstNode(const  CstNode *node);
  CstNode(mlir::ConstantOp CstOp,int nodeIDCount);
  void print_node();
  void print_node(raw_fd_ostream &os);
};

//Sqrt
class SqrtNode:public Node{
  public:
  SqrtNode(const  SqrtNode *node);
  SqrtNode(mlir::math::SqrtOp sqrtOp,int nodeIDCount);
  void print_node();
  void print_node(raw_fd_ostream &os);
};

class Edge{
  public:
  Edge(int begin,int	end);
  Edge(const Edge *e);
  int UnrollID = 0;
  void print_edge();
  void print_edge(raw_fd_ostream &os);
  int	begin;		//Side of the front drive
	int	end;		//The successor of the side
	int	min =0;		//Dependent delay
	int	dif =0;		//Dependent distance 
  int edgeID;
  EdgeType edgeType;
};

//reuse graph
class ReuseGraph{
  public:
  ReuseGraph(mlir::Operation * alloc, std::vector<Node *> LoadStoreVec);
  void print_ReuseGraph();
  mlir::Operation * Array;
  std::vector<Node *> LoadStoreVec;
  int ReuseID;//If ID = -1, the graph is not to be analyzed
};


//The array information and boundary information of scop are extracted and correspond to the following func one by one
class ScopInformation {
public:
  ScopInformation(mlir::FuncOp g, mlir::ModuleOp moduleop);
  ~ScopInformation();
  ScopInformation(const ScopInformation &s);
  int ValueIndexInFuncG(mlir::Value value);
  int getMapResult(mlir::AffineMap affineMap,int index);
  int getBoundMapConstant(mlir::AffineMap affineMap, mlir::Value mapValue);
  void print_Bound();
  void print_Array();
  //index_cast Op
  SmallVector<mlir::IndexCastOp, 8U> BoundIndexCastVec;
  //index_cast Op The corresponding int value
  SmallVector<int, 8U> BoundIntVec;
  // bound Value mlir::Attribute a = consOp.getValue()
  SmallVector<mlir::Attribute, 8U> BoundAttrVec;
  int BoundMapSize = 0;
  int ArrayCount = 0;
  //<mlir::memref,%0 = memref.alloc(),length,type>
  SmallVector<mlir::Operation *, 8U> ArrayAllocVec;
  //The ssa value stored in the func array
  SmallVector<mlir::Value, 8U> ArrayValueVec;
  //The location of the array stored in func
  SmallVector<int, 8U> ArrayArgIndexVec;
  mlir::FuncOp g;
};

class StmtInfomation {
public:
  StmtInfomation(int dfg_dim,SmallVector<mlir::AffineForOp,8U> forIterVec,mlir::AffineForOp forOp,mlir::CallOp callOp,mlir::ModuleOp moduleop,mlir::FuncOp g,ScopInformation *scopInformation);
  StmtInfomation(const StmtInfomation &s);
  int ValueIndexInCallee(mlir::Value value);
  //Find shift
  // If apply is shift and if consOp is the exact value
  void find_IterShift(mlir::AffineMap affineMap,ScopInformation *scopInformation);
  // Find the loop 0, 1, 2 (from inside out) around the PNU corresponding to the caller parameter position
  // The first is the innermost iteration variable
  void find_IterArgIndex(int dfg_dim,SmallVector<mlir::AffineForOp,8U> forIterVec,mlir::AffineForOp forOp,mlir::CallOp callOp,mlir::ModuleOp moduleop,mlir::FuncOp g,ScopInformation *scopInformation);
  //Find the ApplyOp above the statement
  void find_ApplyConsOp(mlir::AffineForOp forOp,mlir::CallOp callOp,mlir::ModuleOp moduleop,mlir::FuncOp g);
  //Find the parameter position of the constant
  void find_ConsArgIndex();
  void print_stmtInfo();
  int innerIter;
  int dim = 0;
  mlir::CallOp stmtCaller;
  mlir::FuncOp stmtCallee;
  //Offset value of the iterated variable
  //+1  -1
  SmallVector<int, 8U> IterShiftVec;
  // The iterating variable of the loop (from inside out) is on which argument
  // If it is a constant, don't worry
  SmallVector<int, 8U> IterArgIndexVec; 
  //If there is a constant, on which parameter
  SmallVector<int, 8U> ConsArgIndexVec; 
  //If there are constants, values
  SmallVector<int, 8U> ConsValueVec; 
  //According to the applyOp for passing parameter data
  SmallVector<mlir::Operation *, 8U> ApplyOrConsOpVec;
  //consOp:0   ApplyOp:1
  SmallVector<int, 8U> ApplyOrConsFlagVec;
  //ApplyOp cons:0 ApplyOp:1  
  SmallVector<int, 8U> ApplyFlagVec;
  //LoadOp
  std::vector<Node *> LoadStoreNodeVec;
  //StoreOp
  // SmallVector<mlir::AffineStoreOp, 8U> StoreOpVec;
  // Parameter markers, what is each parameter
  //0: constant 1: iteration variable 2: array
  SmallVector<int, 8U> Argflag;

};

class MlirDFG {
public:
  MlirDFG(int dfg_id,SmallVector<mlir::CallOp, 8U> callOpVec,mlir::AffineForOp forOp, mlir::FuncOp g,
      mlir::ModuleOp moduleop,ScopInformation *scopInformation,int &DFGNum,int methodflag);
  //MlirDFG(const MlirDFG &MlirDFG);
  ~MlirDFG();
  int getPnuStmtId();
  void add_VarLoad(mlir::Value value,ScopInformation *scopInformation,StmtInfomation *s,int end);
  int find_NodeId(mlir::Value value);
  int find_NodeId(Node *node);
  int find_EdgeId(std::vector<Edge *> EdgesList, int begin,int end);
  void judge_edge(ScopInformation *scopInformation,StmtInfomation *s,int begin,mlir::Value value,int end);
  void add_edges(mlir::Operation *op,ScopInformation *scopInformation,StmtInfomation *s);
  Edge* create_edge(Node *start, Node *end);
  //Get the access allocOp
  mlir::Operation * getLoadStoreAlloc(ScopInformation *scopInformation,mlir::Value value,StmtInfomation *s);
  //Insert a statement into StmtsVec in this DFG
  void insert_Stmts(StmtInfomation Stmt);
  //Outputs the order of the for loop that wraps the DFG
  void print_forIter();
  //Find the for loop that wraps the DFG and store it in forIterVec
  void find_forIter(mlir::AffineForOp forOp, mlir::FuncOp g, mlir::ModuleOp moduleop);
  //Move out all but the innermost offset access in reuseG
  void removeNoInnerShift(ScopInformation *scopInformation);
  //The DFG information is displayed
  void print_DFGInfo();
  void print_DFGInfo(raw_fd_ostream &os);
  //Helper function: Determines whether two nodes (Load Store type) are offset only in the innermost layer
  int isOnlyInnerestShift(Node *node1, Node *node2);
  //Auxiliary function: Determines whether two nodes in the resueGraph have the same innermost offset after adjusting the subdivision (only the innermost offset)
  int compShift(Node *node1, Node *node2);
  void sortByIndexReuseStep(); 
  void genFile();
  void AfterScheduleGenFile(); 
  void DFGDataReuse();

  // step1 First, a DFG graph is generated according to each operator, and a data reuse graph for each array is created
  void createDfg(ScopInformation *scopInformation);
  // step2 Remove Cst operators and edges from DFG graphs
  void deleteCstEdge(); 
  // step3 Subdividing reuses the graph for each array generated by createDfg
  // void adjustReuseGraphs();
  int divideSubReuseGraphs();
  // step4 Add a reuse edge to the reuse diagram
  void addReuseEdges();
  //Handle the relationship between var store and load
  void createVarGraphs();
  void addVarEdges();
  void addRecEdges();
  void InDFGNodeAndEdge();
  //step 5 Get the earliest and latest execution time ready for scheduling
  void getEarliestStep();
  void getLastestStep();
  //step6 Get recourse constrained
  void getRecourseII();
  void noDataReuse();
  //step7 
  // void getBestConcurrentII();
  // void theoreticalMII();

  //DFG after Unroll
  void getNewEarliestStep(int UnrollNum);
  void getNewLastestStep(int UnrollNum);
  void saveOriginalInformation();
  void cleanDFGInformation();
  void restoreOriginalInformation();
  void createUnrollDFG(int UnrollNum);
  void PBPNoRuseMethod();

  ScopInformation *scopInformation;
  std::vector<StmtInfomation> StmtsVec;
  int dfg_id;
  int dfg_dim = 1;
  int TheoreticalII;
  int ResMII = 0;//Resource MII
  int RecMII = 0;//Recurrent MII
  int unrolledII;
  double unrollII = -1;
  int UnrollNum = -1;//// If = -1, there is no unroll
  int UnrollbigDFGII = -1;//II of unrolled DFG
  SmallVector<mlir::AffineForOp,8U> forIterVec;
  int NodesNum;
  int EdgesNum;
  std::vector<Node *> NodesList;
  std::vector<Edge *> EdgesList;
  std::vector<Node *> original_NodesList;
  std::vector<Edge *> original_EdgesList;
  std::vector<Node *> inDFGNodesList;
  std::vector<Edge *> inDFGEdgesList;
  std::vector<Node *> original_inDFGNodesList;
  std::vector<Edge *> original_inDFGEdgesList;
  int LSNodeNum = 0;
  int PeNodeNum = 0;
  std::vector<ReuseGraph *> reuseGraphs;
  std::vector<ReuseGraph *> varGraphs;
  std::vector<ReuseGraph *> original_reuseGraphs;
  std::vector<ReuseGraph *> original_varGraphs;
  int reuseGraphNum = 0;
  int varGraphNum = 0;
  int DfgStep = 0;
  int methodflag;
  int reuseDataNum = 0;
};

class DFGList{
  public:
  DFGList(MapVector<int, SmallVector<mlir::CallOp, 8U>> map);
  void insert(SmallVector<mlir::CallOp, 8U> callOpVec);
  MapVector<int, SmallVector<mlir::CallOp, 8U>> list;
  int dfgNum;
};
} // namespace RfCgraTrans