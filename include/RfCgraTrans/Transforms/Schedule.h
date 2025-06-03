#include "glpk.h"
#include "RfCgraTrans/Transforms/RfCgraMap.h"
#include <sys/time.h>
namespace RfCgraTrans {
/*
Max 
z=10*x1+6*x2

Sub
x1+x2<=200
x1+2*x2>=10

var non-negative
x1>=0,x2>=0

Create a helper variable and transform the problem as follows:
Max 
z=10*x1+6*x2

Sub
x1+x2 = q
x1+2*x2 = p

var non-negative
x1>=0,x2>=0
200>=q>=0,p>=10

GLPK treats each auxiliary variable as a row and the original variable as a column    
constrant_matrix:
ia: number of rows 
ja: number of columns
ar: coefficient of this variable

ia[1] = 1, ja[1] = 1, ar[1] = 1;
ia[2] = 1, ja[2] = 2, ar[2] = 1; // q = x1 + x2

ia[3] = 2, ja[3] = 1, ar[3] = 1;
ia[4] = 2, ja[4] = 2, ar[4] = 2; // p = x1 + 2x2
*/
enum ConstraintType
{
   OnlyOneConstraint = 1,  
   peSrcConstraint = 2,
   bankSrcConstraint = 3,
   interBeforeAfterConstraint = 4,
   innerBeforeAfterConstraint = 5,
   NarrowConstraint = 6,
   SubOptimalConstraint = 8,
   MaxOpConstraint = 9,
   RegisterMaxDisConstraint = 10,
   MaxOpLowerConstraint = 11
};

enum VarType
{
   peVar = 0,
   bankVar = 1,
   maxOpVar = 2,  
  maxDisVar = 3
};

struct Var
{
    Var(){};
    std::string name;
    int nodeID;
    int col;
    double value = -1;
    //Earliest time step
    int earliestTimeStep = 0;
    //Latest time step
    int latestTimeStep = 0;
    int TimeStep;
    NodeType nodetype;
    VarType vartype;
    int obj_cof = 0;
};
struct Constraint
{
    Constraint(){};
    std::string auxVarName;
    int row;
    std::vector<std::pair<Var*,int>> varArVector;
    double upperBound = 0;
    double lowerBound = 0;
    ConstraintType consType;
};

class Schedule{
  public:
  //调度
  Schedule(MlirDFG &dfg);
  Schedule(){
    //this->is_global= false;
  };
  int find_Simple_Schedule();
  void genFile();
  void genScheduleFile();
  int search_Schedule();
  int preData();
  void print_cons();
  void print_vars();
  void print_schedule_output();
  //Add constraint
  void add_OnlyOneConstraint();
  void add_SrcConstraint();
  void add_BeforeAfterConstraint();
  void add_NarrowConstraint();
  void add_SubOptimalConstraint();
  void add_MaxOpConstraint(double maxOp);
  void add_RegisterMaxDisConstraint();
  void add_MaxOpLowerConstraint(double maxOp);
  //Add variables and constraints to lp
  void add_var_cons_lp(glp_prob *lp);
  bool fail_Schedule();
  MlirDFG *dfg;
  int sche_id;
  ScopInformation *scopInformation;
  std::vector<Var*> vars;
  //map<nodeID_vars>
  std::map<int,Var*> nodeID_Var_map;
  std::vector<Constraint*> Constraints;
  Constraint* Object;
  int *ia;
  int *ja;
  double *ar;
  int countSum = 0;
  bool is_global;
  int registerCount = 0;
  double maxOpValue = 0;
  int SolutionNum  = 0;
};

}  // namespace RfCgraTrans