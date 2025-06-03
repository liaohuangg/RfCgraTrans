#include "RfCgraTrans/Transforms/Schedule.h"
#include <stdexcept>
#include <stdio.h>
#include <time.h>
using namespace mlir;
using namespace llvm;
using namespace RfCgraTrans;
using namespace memref;

namespace RfCgraTrans {
Schedule::Schedule(MlirDFG &dfg) {
  this->dfg = &dfg;
  if (this->preData() == 1) {
    // if (this->find_Schedule_Map() == -2) {
    //   dfg.TheoreticalII += 1;
    // }
  }
  this->genScheduleFile();
}
void Schedule::genScheduleFile() {
  this->print_vars();
  this->print_cons();
  this->print_schedule_output();
}
void Schedule::add_OnlyOneConstraint() {
  /*
  Prepare the variable N0T0 and add a uniqueness constraint, which can also constrain operators with unique time steps
  N*Tj + ...  = 1
  */
  // maxop
  std::string varName = "MaxOp";
  Var *var = new Var();
  var->name = varName;
  var->nodeID = -1;
  this->vars.push_back(var);
  var->col = this->vars.size();
  var->value = -1;
  var->TimeStep = -1;
  var->vartype = maxOpVar;
  this->nodeID_Var_map.insert(std::make_pair(-1, var));

  std::string varName1 = "maxDisVar";
  Var *var1 = new Var();
  var1->name = varName1;
  var1->nodeID = -2;
  this->vars.push_back(var1);
  var1->col = this->vars.size();
  var1->value = -1;
  var1->TimeStep = -1;
  var1->vartype = maxDisVar;
  this->nodeID_Var_map.insert(std::make_pair(-2, var1));

  int endTimeStep = this->dfg->DfgStep - 1;
  for (auto *node : this->dfg->inDFGNodesList) {
    std::string ConstraintName =
        "OnlyOneConstraint_" + std::to_string(node->nodeID);
    Constraint *cons = new Constraint();
    cons->auxVarName = ConstraintName;

    for (int i = node->earliestTimeStep; i <= node->latestTimeStep; i++) {
      std::string varName = "N" + std::to_string(node->nodeID) + "T" +
                            std::to_string(i) + "_" +
                            std::to_string(this->vars.size() + 1);
      Var *var = new Var();
      var->name = varName;
      var->nodeID = node->nodeID;
      var->earliestTimeStep = node->earliestTimeStep;
      var->latestTimeStep = node->latestTimeStep;
      var->TimeStep = i;
      var->nodetype = node->nodeType;
      this->nodeID_Var_map.insert(std::make_pair(var->nodeID, var));

      if (var->nodetype == ArrayLoad || var->nodetype == ArrayStore ||
          var->nodetype == noFitIndexLoad || var->nodetype == noFitIndexStore) {
        var->vartype = bankVar;
      } else {
        var->vartype = peVar;
      }
      this->vars.push_back(var);
      var->col = this->vars.size();
      var->value = -1;
      cons->varArVector.push_back(std::make_pair(var, 1));
      cons->consType = OnlyOneConstraint;
    }
    this->Constraints.push_back(cons);
    cons->row = this->Constraints.size();
    cons->upperBound = 1;
    cons->lowerBound = 1;
  }
}
void Schedule::add_MaxOpConstraint(double maxOp) {
  std::string ConstraintName = "MaxOpConstraint";
  Constraint *cons = new Constraint();
  cons->auxVarName = ConstraintName;
  cons->consType = MaxOpConstraint;
  for (int j = 0; j < this->vars.size(); j++) {
    if (this->vars[j]->vartype == maxOpVar) {
      cons->varArVector.push_back(std::make_pair(this->vars[j], 1));
    }
  }
  this->Constraints.push_back(cons);
  cons->row = this->Constraints.size();
  cons->upperBound = maxOp;
  cons->lowerBound = 0;
}
void Schedule::add_SrcConstraint() {
  /*
  These vars are folded according to II, adding resource constraints
  */
  for (int II = 0; II < this->dfg->TheoreticalII; II++) {

    std::string ConstraintName1 = "peSrcConstraint" + std::to_string(II + 1);
    Constraint *cons1 = new Constraint();
    cons1->auxVarName = ConstraintName1;

    std::string ConstraintName2 = "bankSrcConstraint" + std::to_string(II + 1);
    Constraint *cons2 = new Constraint();
    cons2->auxVarName = ConstraintName2;

    for (int i = 0; i < this->vars.size(); i++) {
      if (this->vars[i]->TimeStep % this->dfg->TheoreticalII == II) {
        if (this->vars[i]->vartype == bankVar) {
          cons2->varArVector.push_back(std::make_pair(this->vars[i], 1));
        } else if (this->vars[i]->vartype == peVar) {
          cons1->varArVector.push_back(std::make_pair(this->vars[i], 1));
        }
      }
    }
    this->Constraints.push_back(cons1);
    cons1->row = this->Constraints.size();
    cons1->upperBound = (double)(PERow * PERow);
    cons1->lowerBound = 0;
    cons1->consType = peSrcConstraint;

    this->Constraints.push_back(cons2);
    cons2->row = this->Constraints.size();
    cons2->upperBound = (double)(PERow);
    cons2->lowerBound = 0;
    cons2->consType = bankSrcConstraint;
  }
}
void Schedule::add_BeforeAfterConstraint() {
  /*
 Add before and after constraints to var, such before and after constraints are constraints within iteration
j*NiTj(successor) - j*NiTj(predecessor)... >= 1 // Within iteration
 */
  int count_Inner = 1;
  for (auto *e : this->dfg->inDFGEdgesList) {
    // dif == 0 is required to maintain inner pre - and post-dependency

    bool differentNode = true;
    if (e->begin == e->end)
      differentNode = false;

    if (differentNode && e->dif == 0) {
      std::string ConstraintName =
          "InnerBeforeAndAfterConstraint_" + std::to_string(count_Inner);
      Constraint *cons = new Constraint();
      cons->auxVarName = ConstraintName;
      cons->consType = innerBeforeAfterConstraint;
      for (int i = 0; i < this->vars.size(); i++) {
        if (this->vars[i]->nodeID == e->begin) {
          cons->varArVector.push_back(std::make_pair(
              this->vars[i], -1 * (this->vars[i]->TimeStep + 1)));
        }
        if (this->vars[i]->nodeID == e->end) {
          cons->varArVector.push_back(
              std::make_pair(this->vars[i], 1 * (this->vars[i]->TimeStep + 1)));
        }
      }

      this->Constraints.push_back(cons);
      cons->row = this->Constraints.size();
      cons->upperBound = (double)(this->dfg->DfgStep);
      cons->lowerBound = 1;

      count_Inner++;
    }
  }

  /*
Before and after constraints are added to var, which are constraints between iterations
(dif * II + this->vars[i]->TimeStep +1)*NiTj(successor) - (this->vars[i]->TimeStep
+1)*NiTj(Front drive)... >= 1 // Between iterations
*/
  for (auto *e : this->dfg->inDFGEdgesList) {
    //dif > 0 satisfies the inverse side of the constraint between inter iterations

    bool differentNode = true;
    if (e->begin == e->end)
      differentNode = false;

    if (differentNode && e->dif > 0) {
      std::string ConstraintName =
          "InterBeforeAndAfterConstraint_" + std::to_string(count_Inner);
      Constraint *cons = new Constraint();
      cons->auxVarName = ConstraintName;
      cons->consType = interBeforeAfterConstraint;
      for (int i = 0; i < this->vars.size(); i++) {
        if (this->vars[i]->nodeID == e->begin) {
          cons->varArVector.push_back(std::make_pair(
              this->vars[i], -1 * (this->vars[i]->TimeStep + 1)));
        }
        if (this->vars[i]->nodeID == e->end) {
          cons->varArVector.push_back(std::make_pair(
              this->vars[i], 1 * (e->dif * this->dfg->TheoreticalII +
                                  this->vars[i]->TimeStep + 1)));
        }
      }

      this->Constraints.push_back(cons);
      cons->row = this->Constraints.size();
      cons->upperBound = e->dif * this->dfg->TheoreticalII + this->dfg->DfgStep;
      cons->lowerBound = 1;

      count_Inner++;
    }
  }
}
void Schedule::add_NarrowConstraint() {

  for (int k = 0; k < this->dfg->TheoreticalII; k++) {
    std::string ConstraintName = "NarrowConstraint_" + std::to_string(k + 1);
    Constraint *cons = new Constraint();
    cons->auxVarName = ConstraintName;
    cons->consType = NarrowConstraint;
    for (int j = 0; j < this->vars.size(); j++) {
      if ((this->vars[j]->TimeStep % this->dfg->TheoreticalII) == k &&
          (this->vars[j]->vartype == bankVar ||
           this->vars[j]->vartype == peVar)) {
        cons->varArVector.push_back(std::make_pair(this->vars[j], 1));
      } else if (this->vars[j]->vartype == maxOpVar) {
        cons->varArVector.push_back(std::make_pair(this->vars[j], -1));
      }
    }
    this->Constraints.push_back(cons);
    cons->row = this->Constraints.size();
    cons->upperBound = 0;
    cons->lowerBound = -1;
  }
}
void Schedule::add_RegisterMaxDisConstraint() {
  std::vector<int> VarsAr;
  for (int i = 0; i < this->vars.size(); i++) {
    VarsAr.push_back(0);
  }

  std::string ConstraintName = "RegisterMaxDisConstraint";
  Constraint *cons = new Constraint();
  cons->auxVarName = ConstraintName;
  cons->consType = RegisterMaxDisConstraint;

  for (auto *e : this->dfg->inDFGEdgesList) {
    for (int i = 0; i < this->vars.size(); i++) {
      if (this->vars[i]->nodeID == e->begin) {
        VarsAr[i] += -1 * (this->vars[i]->TimeStep);
        // cons->varArVector.push_back(std::make_pair(this->vars[i],-1 *
        // (this->vars[i]->TimeStep)));
      }
    }
    for (int i = 0; i < this->vars.size(); i++) {
      if (this->vars[i]->nodeID == e->end && e->dif == 0) {
        // cons->varArVector.push_back(std::make_pair(this->vars[i],this->vars[i]->TimeStep));
        VarsAr[i] += this->vars[i]->TimeStep;
      } else if (this->vars[i]->nodeID == e->end && e->dif > 0) {
        VarsAr[i] +=
            this->vars[i]->TimeStep + e->dif * this->dfg->TheoreticalII;
        // cons->varArVector.push_back(std::make_pair(this->vars[i],
        // this->vars[i]->TimeStep + e->dif * this->dfg->TheoreticalII));
      }
    }
  }
  for (int i = 0; i < this->vars.size(); i++) {
    if (this->vars[i]->vartype == peVar || this->vars[i]->vartype == bankVar) {
      cons->varArVector.push_back(std::make_pair(this->vars[i], VarsAr[i]));
    }
  }
  for (Var *v : this->vars) {
    if (v->vartype == maxDisVar) {
      cons->varArVector.push_back(std::make_pair(v, -1));
    }
  }
  this->Constraints.push_back(cons);
  cons->row = this->Constraints.size();
  cons->upperBound = 0;
  cons->lowerBound = -1;
}

void Schedule::add_SubOptimalConstraint() {
  std::string ConstraintName =
      "SubOptimalConstraint" + std::to_string(this->SolutionNum);
  Constraint *cons = new Constraint();
  cons->auxVarName = ConstraintName;
  cons->consType = SubOptimalConstraint;
  for (int i = 0; i < this->vars.size(); i++) {
    if ((this->vars[i]->vartype == peVar ||
         this->vars[i]->vartype == bankVar) &&
        this->vars[i]->value == 1) {
      cons->varArVector.push_back(std::make_pair(this->vars[i], 1));
    }
    if ((this->vars[i]->vartype == peVar ||
         this->vars[i]->vartype == bankVar) &&
        this->vars[i]->value == 0) {
      cons->varArVector.push_back(std::make_pair(this->vars[i], -1));
    }
  }
  this->Constraints.push_back(cons);
  cons->row = this->Constraints.size();
  cons->upperBound = (double)(this->dfg->inDFGNodesList.size() - 1);
  cons->lowerBound = -(double)(this->vars.size());

  for (int i = 0; i < this->vars.size(); i++) {
    this->vars[i]->value = -1;
  }
}
int Schedule::preData() {
  std::ofstream os;
  os.open("Scheduledebug.out", std::ios::app);
  /*
  Once all the constraints are in place, build the ILP matrix
  */
  int sum = 0; // 由1开始，已经+1
  if (this->vars.size() > 0 && this->Constraints.size() > 0) {

    for (int i = 0; i < this->Constraints.size(); i++) {
      for (auto v : this->Constraints[i]->varArVector) {
        sum++;
      }
    }

    this->countSum = sum;
    this->ia = (int *)malloc((sum + 1) * sizeof(int));
    this->ja = (int *)malloc((sum + 1) * sizeof(int));
    this->ar = (double *)malloc((sum + 1) * sizeof(double));
    // 初始化
    for (int i = 0; i <= sum; i++) {
      this->ia[i] = 0;
      this->ja[i] = 0;
      this->ar[i] = 0;
    }

    //For each constraint
    int countSum = 1;
    for (int i = 0; i < this->Constraints.size(); i++) {
      for (auto v : this->Constraints[i]->varArVector) {
        this->ia[countSum] = this->Constraints[i]->row;
        this->ja[countSum] = v.first->col;
        this->ar[countSum] = v.second;
        countSum++;
      }
    }
    // llvm::raw_ostream &os = llvm::outs();
    //  os<<"\n=============print_matrix===========\n";
    // for(int i = 1;i<this->countSum+1;i++){
    //   os<<" ia["<<i<<"] = "<<this->ia[i];
    //   os<<" ja["<<i<<"] = "<<this->ja[i];
    //   os<<" ar["<<i<<"] = "<<this->ar[i]<<"\n";
    // }
    return 1;
  } else {
    return 0;
  }
}
void Schedule::print_schedule_output() {
  std::ofstream os;
  // os.open("ScheduleInformation.out", std::ios::app);
  os.open("Schedule" + std::to_string(this->dfg->dfg_id) + "_solu" +
              std::to_string(this->SolutionNum) + ".out",
          std::ios::app);
  os << "\n=======schedule=====\n";
  // os << "\n Z = "<<this->dfg->TheoreticalII * PERow * PERow * 5 <<" *
  // "<<this->maxOpValue <<" + "<<this->registerCount;
  os << "\n II =  " << this->dfg->TheoreticalII << "\n";
  os << "\n";
  for (int j = 0; j < this->dfg->DfgStep; j++) {
    os << "timeStep" << j << " ";
    os << " LSU ";
    for (int i = 0; i < this->vars.size(); i++) {
      if (this->vars[i]->TimeStep == j && this->vars[i]->value == 1) {
        if (this->vars[i]->nodetype == ArrayLoad ||
            this->vars[i]->nodetype == ArrayStore ||
            this->vars[i]->nodetype == noFitIndexLoad ||
            this->vars[i]->nodetype == noFitIndexStore) {
          os << this->vars[i]->name << " ";
        }
      }
    }
    os << " PE ";
    for (int i = 0; i < this->vars.size(); i++) {
      if (this->vars[i]->TimeStep == j && this->vars[i]->value == 1) {
        if (!(this->vars[i]->nodetype == ArrayLoad ||
              this->vars[i]->nodetype == ArrayStore ||
              this->vars[i]->nodetype == noFitIndexLoad ||
              this->vars[i]->nodetype == noFitIndexStore)) {
          os << this->vars[i]->name << " ";
        }
      }
    }
    os << "\n";
  }

  os << "\nThe total number of register\n";
  int registerCount = 0;
  int startTime = 0;
  int endTime = 0;
  for (Edge *e : this->dfg->inDFGEdgesList) {
    os << "begin " << e->begin << " end " << e->end << " dif " << e->dif;
    for (Var *v : this->vars) {
      if (v->nodeID == e->begin && v->value == 1) {
        startTime = v->TimeStep;
      }
    }
    for (Var *v : this->vars) {
      if (v->nodeID == e->end && v->value == 1) {
        endTime = v->TimeStep;
      }
    }
    os << " startTime " << startTime << " endTime " << endTime;
    registerCount += endTime - startTime;
    os << " useRe " << endTime - startTime + e->dif * this->dfg->TheoreticalII
       << "\n";
    if (e->dif > 0) {
      registerCount += e->dif * this->dfg->TheoreticalII;
    }
  }
  this->registerCount = registerCount;
  os << "  " << this->registerCount << "\n";

  // os << "\n=============print_matrix===========\n";
  // for (int i = 1; i < this->countSum + 1; i++) {
  //   os << " ia[" << i << "] = " << this->ia[i];
  //   os << " ja[" << i << "] = " << this->ja[i];
  //   os << " ar[" << i << "] = " << this->ar[i] << "\n";
  // }
  int countNode = 0;
  os.close();
}
void Schedule::print_vars() {
  std::ofstream os;
  // os.open("ScheduleInformation.out", std::ios::app);
  os.open("Schedule" + std::to_string(this->dfg->dfg_id) + "_solu" +
              std::to_string(this->SolutionNum) + ".out",
          std::ios::app);

  os << "\n\n============ " << this->dfg->dfg_id<< " PNU  Schedule==============\n";
  os << "\n=======print_vars=====\n";
  for (Var *v : this->vars) {
    if (v->vartype == maxOpVar) {
      os << " maxOpVar " << v->name << " " << v->col << "\n";
    }
    if (v->vartype == maxDisVar) {
      os << " maxDisVar " << v->name << " " << v->col << "\n";
    }
  }
  os << "\n";
  for (int i = 0; i < this->dfg->inDFGNodesList.size(); i++) {
    os << "\n nodeID " << this->dfg->inDFGNodesList[i]->nodeID << "\n";
    for (Var *v : this->vars) {
      if (v->nodeID == this->dfg->inDFGNodesList[i]->nodeID) {
        os << " " << v->name << " " << v->col << "\n";
      }
    }
  }
  os.close();
}
void Schedule::print_cons() {
  std::ofstream os;
  // os.open("ScheduleInformation.out", std::ios::app);
  os.open("Schedule" + std::to_string(this->dfg->dfg_id) + "_solu" +
              std::to_string(this->SolutionNum) + ".out",
          std::ios::app);

  os << "\n=======print_cons=====\n";

  for (int i = 0; i < this->Constraints.size(); i++) {
    os << "\n " << this->Constraints[i]->row << " Constraint\n";
    for (auto v : this->Constraints[i]->varArVector) {
      os << v.second << "*" << v.first->name << " + ";
    }
    os << " = " << this->Constraints[i]->auxVarName << "\n";

    if (this->Constraints[i]->lowerBound == -1 &&
        this->Constraints[i]->upperBound != -1) {
      os << this->Constraints[i]->auxVarName
         << " <= " << this->Constraints[i]->upperBound << "\n\n";
    } else if (this->Constraints[i]->lowerBound != -1 &&
               this->Constraints[i]->upperBound == -1) {
      os << this->Constraints[i]->lowerBound
         << " <= " << this->Constraints[i]->auxVarName << "\n\n";
    } else {
      os << this->Constraints[i]->lowerBound
         << " <= " << this->Constraints[i]->auxVarName
         << " <= " << this->Constraints[i]->upperBound << "\n\n";
    }
  }
  if (this->vars.size() > 0 && this->Constraints.size() > 0) {
    int sum = this->vars[this->vars.size() - 1]->col *
                  this->Constraints[this->Constraints.size() - 1]->row +
              1;
    os << "\n vars " << this->vars.size() << " Constraints "
       << this->Constraints.size() << "\n";
    os << "\n sum " << sum << "\n";
  }

  // os<<"\n=============print_matrix===========\n";
  // for(int i;i<this->countSum+1;i++){
  //   os<<" ia["<<i<<"] = "<<this->ia[i];
  //   os<<" ja["<<i<<"] = "<<this->ja[i];
  //   os<<" ar["<<i<<"] = "<<this->ar[i]<<"\n";
  // }
  os.close();
}
void Schedule::add_var_cons_lp(glp_prob *lp) {

  glp_add_rows(lp, this->Constraints.size());
  for (int i = 0; i < this->Constraints.size(); i++) {
    glp_set_row_name(lp, this->Constraints[i]->row,
                     this->Constraints[i]->auxVarName.c_str());
    if (this->Constraints[i]->consType == SubOptimalConstraint) {
      glp_set_row_bnds(lp, this->Constraints[i]->row, GLP_UP, 0,
                       (double)(this->dfg->inDFGNodesList.size() - 1));
    } else if (this->Constraints[i]->consType == MaxOpLowerConstraint) {
      glp_set_row_bnds(lp, this->Constraints[i]->row, GLP_LO,
                       this->Constraints[i]->lowerBound, 0);
    } else if (this->Constraints[i]->consType == NarrowConstraint ||
               this->Constraints[i]->consType == RegisterMaxDisConstraint) {
      glp_set_row_bnds(lp, this->Constraints[i]->row, GLP_UP, 0, 0);
    } else if (this->Constraints[i]->lowerBound ==
               this->Constraints[i]->upperBound) {
      glp_set_row_bnds(lp, this->Constraints[i]->row, GLP_FX,
                       this->Constraints[i]->lowerBound,
                       this->Constraints[i]->upperBound);
    } else {
      glp_set_row_bnds(lp, this->Constraints[i]->row, GLP_DB,
                       this->Constraints[i]->lowerBound,
                       this->Constraints[i]->upperBound);
    }
  }

  glp_add_cols(lp, this->vars.size());
  for (int i = 0; i < this->vars.size(); i++) {
    if (this->vars[i]->vartype == maxOpVar ||
        this->vars[i]->vartype == maxDisVar) {
      glp_set_col_name(lp, this->vars[i]->col, this->vars[i]->name.c_str());
      glp_set_col_bnds(lp, this->vars[i]->col, GLP_LO, 0, 0);
      glp_set_col_kind(lp, this->vars[i]->col, GLP_IV);
    } else {
      glp_set_col_name(lp, this->vars[i]->col, this->vars[i]->name.c_str());
      glp_set_col_bnds(lp, this->vars[i]->col, GLP_DB, 0,
                       1); /* Double-bounded variable */
      glp_set_col_kind(lp, this->vars[i]->col, GLP_IV); /* integer variable */
    }
  }
}
void Schedule::add_MaxOpLowerConstraint(double maxOp) {
  std::string ConstraintName = "MaxOpLowerConstraint";
  Constraint *cons = new Constraint();
  cons->auxVarName = ConstraintName;
  cons->consType = MaxOpLowerConstraint;

  for (Var *v : this->vars) {
    if (v->vartype == maxOpVar) {
      cons->varArVector.push_back(std::make_pair(v, 1));
    }
  }

  this->Constraints.push_back(cons);
  cons->row = this->Constraints.size();
  cons->upperBound = -1;
  cons->lowerBound = maxOp;
}
int Schedule::search_Schedule() {
  // std::ofstream os;
  // os.open("Scheduledebug.out", std::ios::app);
  bool find_schedule = true;
  bool firstTime = true;
  bool onlyOneMaxOpConstriant = true;
  this->maxOpValue = 0;
  int lastMaxOp = 0;
  this->SolutionNum = 0;
  int maxOpsolutionNum;
  llvm::raw_ostream &os = llvm::outs();
  int count = 0;
  while (this->SolutionNum < searchScheduleNum) {
    maxOpsolutionNum = 0;
    this->add_OnlyOneConstraint();
    this->add_SrcConstraint();
    this->add_BeforeAfterConstraint();
    this->add_NarrowConstraint();
    // this->add_RegisterMaxDisConstraint();
    this->add_MaxOpLowerConstraint(this->maxOpValue);
    while (find_schedule) {
      if (!firstTime) {
        if (onlyOneMaxOpConstriant) {
          this->add_MaxOpConstraint(this->maxOpValue);
          onlyOneMaxOpConstriant = false;
        }
        this->add_SubOptimalConstraint();
      }
      firstTime = false;

      if (this->preData() != 1)
        return 0;
      // std::ofstream fos;
      // fos.open("Schedule"+std::to_string(this->dfg->dfg_id)+"_"+std::to_string(this->SolutionNum)+".out",
      // std::ios::app); fos << "\n\n============search schedule 第 " <<
      // this->dfg->dfg_id << "====================";
      /*initialize*/
      glp_prob *lp;
      lp = glp_create_prob();
      glp_set_prob_name(lp, "schedule");
      /* The target is the minimum*/
      glp_set_obj_dir(lp, GLP_MIN);
      /*auxiliary_variables_rows*/
      this->add_var_cons_lp(lp);

      /*min_dependence_distance_schedule
      The narrower the DFG, the better, and based on that, the search for schedules that use fewer registers
      Z =(this->dfg->TheoreticalII * PERow *PERow *5) * maxOp + registerCount
      */
      for (int i = 0; i < this->vars.size(); i++) {
        if (this->vars[i]->vartype == maxOpVar) {
          // fos << " \nmaxOp " << i << "\n";
          glp_set_obj_coef(lp, this->vars[i]->col,
                           1 * (this->dfg->TheoreticalII * PERow * PERow * 5));
        }
        // if(this->vars[i]->vartype == maxDisVar){
        //   glp_set_obj_coef(lp, this->vars[i]->col, 1);
        // }
      }
      /*constrant_matrix*/
      glp_load_matrix(lp, this->countSum, this->ia, this->ja, this->ar);

      /*calculate*/
      glp_smcp simplexParm;
      glp_init_smcp(&simplexParm);
      simplexParm.msg_lev = GLP_MSG_OFF;
      glp_simplex(lp, &simplexParm);

      glp_iocp MIPparm;
      glp_init_iocp(&MIPparm);
      MIPparm.msg_lev = GLP_MSG_OFF;
      glp_intopt(lp, &MIPparm);

      // glp_simplex(lp, NULL);
      // glp_intopt(lp, NULL); 

      /*output*/
      double solu = glp_mip_obj_val(lp);
      double max = 0;

      // os<<"\nsolution \n"<<solu;
      for (int i = 0; i < this->vars.size(); i++) {
        // os<<"\n glp_mip_col_val varname "<<this->vars[i]->name<<"
        // "<<glp_mip_col_val(lp, this->vars[i]->col);
        if (glp_mip_col_val(lp, this->vars[i]->col) == 1) {
          this->vars[i]->value = 1;
        } else if (glp_mip_col_val(lp, this->vars[i]->col) == 0) {
          this->vars[i]->value = 0;
        }
        if (this->vars[i]->vartype == maxOpVar) {
          max = glp_mip_col_val(lp, this->vars[i]->col);
          this->maxOpValue = max;
        }
        if (this->vars[i]->vartype == maxDisVar) {
          this->registerCount = glp_mip_col_val(lp, this->vars[i]->col);
          // os<<"\nregisterCount "<<registerCount<<"\n";
        }
      }
      if (this->maxOpValue == 0) {
        this->maxOpValue = lastMaxOp + 1;
      }
      if (fail_Schedule() || maxOpsolutionNum > subScheduleNum) {
        find_schedule = false;
        break;
      }

      lastMaxOp = this->maxOpValue;
      int countNode = 0;
      for (auto *v : this->vars) {
        for (auto *n : this->dfg->inDFGNodesList) {
          if (v->nodeID == n->nodeID && v->value == 1) {
            n->timeStep = v->TimeStep;
            countNode++;
          }
        }
      }
      RF_CGRAMap map(*this->dfg, this->SolutionNum);
      this->print_schedule_output();
      // this->print_vars();
      // this->print_cons();
      //   /*cleanup*/
      // fos.close();
      glp_delete_prob(lp);
      delete this->ia;
      delete this->ja;
      delete this->ar;
      this->SolutionNum++;
      maxOpsolutionNum++;
      count++;
    }
    this->maxOpValue++;
    find_schedule = true;
    this->vars.clear();
    this->Constraints.clear();
  }
  return 1;
}

int Schedule::find_Simple_Schedule() {
  std::ofstream os;
  os.open("simpleSchedule" + std::to_string(this->SolutionNum) + ".out",
          std::ios::app);
  os << "\n\n============DFG ID " << this->dfg->dfg_id << "====================\n";
  bool find_schedule = false;
  int count_Times = 0;
  while (!find_schedule && count_Times < 3) {
    this->add_OnlyOneConstraint();
    this->add_SrcConstraint();
    this->add_BeforeAfterConstraint();
    this->add_NarrowConstraint();
    count_Times++;
    if (this->preData() != 1)
      return 0;
    glp_prob *lp;
    lp = glp_create_prob();
    glp_set_prob_name(lp, "min_dependence_distance_schedule");
    glp_set_obj_dir(lp, GLP_MIN);

    /*auxiliary_variables_rows*/
    glp_add_rows(lp, this->Constraints.size());
    for (int i = 0; i < this->Constraints.size(); i++) {
      glp_set_row_name(lp, this->Constraints[i]->row,
                       this->Constraints[i]->auxVarName.c_str());
      if (this->Constraints[i]->consType == NarrowConstraint) {
        glp_set_row_bnds(lp, this->Constraints[i]->row, GLP_UP, 0, 0);
      }
      if (this->Constraints[i]->lowerBound ==
          this->Constraints[i]->upperBound) {
        glp_set_row_bnds(lp, this->Constraints[i]->row, GLP_FX,
                         this->Constraints[i]->lowerBound,
                         this->Constraints[i]->upperBound);
      } else {
        glp_set_row_bnds(lp, this->Constraints[i]->row, GLP_DB,
                         this->Constraints[i]->lowerBound,
                         this->Constraints[i]->upperBound);
      }
    }
    /*variables_columns*/
    glp_add_cols(lp, this->vars.size());
    for (int i = 0; i < this->vars.size(); i++) {
      if (this->vars[i]->vartype == maxOpVar) {
        glp_set_col_name(lp, this->vars[i]->col, this->vars[i]->name.c_str());
        glp_set_col_bnds(lp, this->vars[i]->col, GLP_LO, 0, 0);
        glp_set_col_kind(lp, this->vars[i]->col, GLP_IV);
      } else {
        glp_set_col_name(lp, this->vars[i]->col, this->vars[i]->name.c_str());
        glp_set_col_bnds(lp, this->vars[i]->col, GLP_DB, 0,
                         1); /* Double-bounded variable */
        glp_set_col_kind(lp, this->vars[i]->col, GLP_IV); /* integer variable */
      }
    }

    for (int i = 0; i < this->vars.size(); i++) {
      if (this->vars[i]->vartype == maxOpVar) {
        glp_set_obj_coef(lp, this->vars[i]->col, 1);
      }
    }
    // glp_set_obj_coef(lp, this->vars[this->vars.size() - 1]->col, 1);
    // glp_set_obj_coef(lp, this->vars[this->vars.size() - 2]->col, 1);

    /*constrant_matrix*/
    glp_load_matrix(lp, this->countSum, this->ia, this->ja, this->ar);

    /*calculate*/
    glp_smcp simplexParm;
    glp_init_smcp(&simplexParm);
    simplexParm.msg_lev = GLP_MSG_OFF;
    glp_simplex(lp, &simplexParm);

    glp_iocp MIPparm;
    glp_init_iocp(&MIPparm);
    MIPparm.msg_lev = GLP_MSG_OFF;
    glp_intopt(lp, &MIPparm);

    // glp_simplex(lp, NULL);
    // glp_intopt(lp, NULL); 

    /*output*/

    double z = glp_mip_obj_val(lp);
    this->maxOpValue = z;
    for (int i = 0; i < this->vars.size(); i++) {
      if (glp_mip_col_val(lp, this->vars[i]->col) == 1) {
        this->vars[i]->value = 1;
      }
    }

    int countNode = 0;
    for (auto *v : this->vars) {
      for (auto *n : this->dfg->inDFGNodesList) {
        if (v->nodeID == n->nodeID && v->value == 1) {
          n->timeStep = v->TimeStep;
          countNode++;
        }
      }
    }

    for (int j = 0; j < this->dfg->DfgStep; j++) {
      os << "timeStep" << j << " ";
      os << " bank ";
      for (int i = 0; i < this->vars.size(); i++) {
        if (this->vars[i]->TimeStep == j && this->vars[i]->value == 1) {
          if (this->vars[i]->nodetype == ArrayLoad ||
              this->vars[i]->nodetype == ArrayStore ||
              this->vars[i]->nodetype == noFitIndexLoad ||
              this->vars[i]->nodetype == noFitIndexStore) {
            os << this->vars[i]->name << " ";
          }
        }
      }
      os << " PE ";
      for (int i = 0; i < this->vars.size(); i++) {
        if (this->vars[i]->TimeStep == j && this->vars[i]->value == 1) {
          if (!(this->vars[i]->nodetype == ArrayLoad ||
                this->vars[i]->nodetype == ArrayStore ||
                this->vars[i]->nodetype == noFitIndexLoad ||
                this->vars[i]->nodetype == noFitIndexStore)) {
            os << this->vars[i]->name << " ";
          }
        }
      }
      os << "\n";
    }

    /*cleanup*/
    glp_delete_prob(lp);
    delete this->ia;
    delete this->ja;
    delete this->ar;
    // return true;
    if (fail_Schedule()) {
      find_schedule = false;
      this->dfg->TheoreticalII += 1;
      this->dfg->RecMII = this->dfg->TheoreticalII;
      os << "\nCan't find the schedule\n";
    } else {
      find_schedule = true;
    }
    this->vars.clear();
    this->Constraints.clear();
  }
}

bool Schedule::fail_Schedule() {

  int countNode = 0;
  for (int i = 0; i < this->vars.size(); i++) {
    if (this->vars[i]->value == 1) {
      countNode++;
    }
  }

  if (countNode < this->dfg->inDFGNodesList.size()) {
    return true;
  } else {
    return false;
  }
}
} // namespace RfCgraTrans