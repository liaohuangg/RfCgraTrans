#include "RfCgraTrans/Transforms/RfCgraMap.h"
using namespace mlir;
using namespace llvm;
using namespace RfCgraTrans;
using namespace memref;

/*
DEFINE_string(dfg_file, "data/test.txt", "dfg file.");
DEFINE_int32(II, 3, "initiation interval");
DEFINE_int32(pea_column, 4, "CGRA columns");//PE的列数
DEFINE_int32(pea_row, 4, "CGRA rows");//行数
DEFINE_int32(childNum, 4, "childNum");//DFG算子的最大子节点数量
DEFINE_int32(Rnum, 5, "Register Num in PE");//PE内的寄存器数量，默认为5
*/
//DEFINE_int32(Rnum, 5, "Register Num in PE");//PE内的寄存器数量，默认为5
#define Rnum 5
namespace RfCgraTrans {
void RF_CGRAMap::print_Data(MlirDFG &mlirdfg){
  std::ofstream fos;
  fos.open("MapInformation.out", std::ios::app);
  fos << "\n\n============第 " << mlirdfg.dfg_id  << "个 PNU 的 MapInformation==============\n";
   fos<<"\n 算子编号 时间步  算子种类 ";
   for(int i =0;i<this->childNum;i++){
     fos<<"子节点"<<i<<" ";
   }
   for(int i =0;i<this->childNum;i++){
     fos<<" dif"<<i<<"   ";
   }
   fos<<"\n";

  for(int i =0;i<this->DFG_node.size();i++){
    for(int j =0;j<this->DFG_node[i].size();j++){
      if(this->DFG_node[i][j] == -1 || this->DFG_node[i][j] >=10){
          fos<<"   "<<this->DFG_node[i][j]<<"   ";
      }else{
          fos<<"    "<<this->DFG_node[i][j]<<"   ";
      }
        
    }
    fos<<"\n";
  }

  fos.close();
}
void RF_CGRAMap::preData(MlirDFG &mlirdfg){
//   vector<vector<int>> DFG_node;
//   vector<int> DFGNodeID;
//   this->DFGNodeID = DFGNodeID;
//   this->DFG_node = DFG_node;
  vector<int> NodeInfo;
  int NodeNum= 0;
  //先按照timeStep排序来sort一遍
  Node *temp;
  std::vector<Node *> List;
  for(int i = 0;i<mlirdfg.inDFGNodesList.size();i++){
    List.push_back(mlirdfg.inDFGNodesList[i]);
  }

  for(int i = 0;i<List.size()-1;i++){
    for(int j = i+1;j<List.size();j++){
        if(List[i]->timeStep > List[j]->timeStep){
            temp = List[j];
            List[j] = List[i];
            List[i] = temp;
        }
    }
  }

//   for(int i = 0;i<mlirdfg.inDFGNodesList.size()-1;i++){
//     for(int j = i+1;j<mlirdfg.inDFGNodesList.size();j++){
//         if(mlirdfg.inDFGNodesList[i]->timeStep > mlirdfg.inDFGNodesList[j]->timeStep){
//             temp = mlirdfg.inDFGNodesList[j];
//             mlirdfg.inDFGNodesList[j] =mlirdfg.inDFGNodesList[i];
//             mlirdfg.inDFGNodesList[i] = temp;
//         }
//     }
//   }


  for(auto *n:List){
    vector<int> NodeInfo;
    this->DFGNodeID_NodeIndex.insert(std::make_pair(n->nodeID,NodeNum));//记录DFG与mlirDFG的关联
    this->DFGNodeIndex_NodeID.insert(std::make_pair(NodeNum,n->nodeID));
    NodeInfo.push_back(NodeNum);
    NodeInfo.push_back(n->timeStep);
    //算子种类
    if(n->nodeType == ArrayStore || n->nodeType == noFitIndexStore){
        NodeInfo.push_back(1);
    }else if(n->nodeType == ArrayLoad || n->nodeType == noFitIndexLoad){
        NodeInfo.push_back(1);
    }else{
        NodeInfo.push_back(0);
    }
    this->DFG_node.push_back(NodeInfo);
    NodeNum++;
  }


  vector<int> difInfo;
   for(int i =0;i<this->DFG_node.size();i++){
     //子节点
        vector<int> difInfo;
        int countSub = 0;
        for (auto *e:mlirdfg.inDFGEdgesList)
        {
            if(e->begin == this->DFGNodeIndex_NodeID.find(i)->second){
                this->DFG_node[i].push_back(this->DFGNodeID_NodeIndex.find(e->end)->second);
                difInfo.push_back(e->dif);
                countSub++;
            }
        }
        
        if(countSub<this->childNum){
            for(int j = countSub;j<this->childNum;j++){
            this->DFG_node[i].push_back(-1);
            difInfo.push_back(-1);
            }
        }
        
        //dif
        for(int j = 0;j<this->childNum;j++){
            this->DFG_node[i].push_back(difInfo[j]);
        }
   }
   

}
int RF_CGRAMap::find_Map(MlirDFG &mlirdfg ,int solutionNum){
    int level_flag;		
    int *flag;	
    flag = & level_flag;			  
    clock_t start, end;/* 记录时间 */
    timeval tv, tv1;    
    gettimeofday(&tv, 0);
    std::ofstream fos;
    fos.open("MapInformation.out", std::ios::app);
    //fos << "\n\n============第 " << mlirdfg.dfg_id  << "个 PNU 的 MapInformation==============\n";
    int childArrayNum = (mlirdfg.inDFGNodesList.size() + mlirdfg.NodesList.size());
    //得到最大子节点数量
    int *childArray =(int *)malloc(sizeof(int) * childArrayNum);
    for(int i = 0;i<childArrayNum;i++){
        childArray[i]  = 0;
    }
  //最大字节点数量
    for(int j = 0;j<mlirdfg.inDFGNodesList.size();j++){
        for(int i = 0;i<mlirdfg.inDFGEdgesList.size();i++){
            if(mlirdfg.inDFGNodesList[j]->nodeID == mlirdfg.inDFGEdgesList[i]->begin){
                childArray[mlirdfg.inDFGNodesList[j]->nodeID]++;
            }
        }
    }

  this->childNum = 0;
  for(int i = 0;i<childArrayNum;i++){
    if(this->childNum < childArray[i]){
        this->childNum = childArray[i];
    }
  }

    this->preData(mlirdfg);
    this->print_Data(mlirdfg);
    // fos<<"\n算子数量\n"<<mlirdfg.inDFGNodesList.size()<<"\n";
    std::ofstream fffos;
    fffos.open("map"+to_string(mlirdfg.dfg_id)+"_solu"+to_string(solutionNum)+".txt", std::ios::app);
    //排序输出..
    int NodeCount = 0;
    // for(int time = 0;time<=mlirdfg.DfgStep;time++){
        for(int i =0;i<this->DFG_node.size();i++){
            // if(this->DFG_node[i][1] == time){
                for(int j =0;j<this->DFG_node[i].size();j++){
                if(j !=  DFG_node[i].size()-1){
                    fffos << DFG_node[i][j] << ",";
                }else{
                    fffos << DFG_node[i][j];
                }
                }  
                if(i !=  DFG_node.size()-1) {
                    fffos << "\n";
                }else{
                    fffos.close();
                }
            // }else{
            //     continue;
            // }
        } 
    // }
    //DFG *D = new DFG(mlirdfg.TheoreticalII,childNum,mlirdfg.inDFGNodesList.size(),this->DFG_node,mlirdfg.dfg_id,solutionNum); /* 创建DFG */
    // CGRA *C = new CGRA(PERow,PERow);/* 创建CGRA */

    // /* 维护一个TEC 中各个PE的输入输出端口数量数组 */

    // int *inportNum;
    // int *outportNum;
    // int TECPENUM = C->ElmNum *mlirdfg.TheoreticalII;
    // inportNum = new int[TECPENUM];
    // outportNum = new int[TECPENUM];
    // memset(inportNum, 0, TECPENUM * sizeof(int));
    // memset(outportNum, 0, TECPENUM * sizeof(int));
    // for(int i = 0; i < TECPENUM; i++)
    // {
    //     inportNum[i] = 5;
    //     outportNum[i] = 6;
    // }

    
    // Register *R = new Register(Rnum,C);/* 创建平面寄存器 */
    // R->CreatTER(mlirdfg.TheoreticalII); /*  创建时间扩展的TEC寄存器 */
    
    // // /* 输出信息 */
    // // fos << "\nthe number of DFG's nodes:" << D->numDFGnodes;
    // // fos << "\nthe number of DFG's edge:" << D->numDFGedges;

    // //fos << "II:" << mlirdfg.TheoreticalII << endl;

    // /* DFG折叠 */
    // D->CreatMDFG(mlirdfg.TheoreticalII);
   
    // *flag = D->Constraint_Level(mlirdfg.TheoreticalII);/* 资源判断 LSU和load,PE和普通算子,store和SU */
    // //CHECK_EQ(*flag, -1) << "Modulo resource constraints are not satisffied!";/* 不满足时输出 */
    // if(*flag ==  -1){
    //     //fos<<"Modulo resource constraints are not satisffied!\n";
    // }
    // AllPath *allPathClass = new AllPath();/* 存放最终所有边的路径对象 */
      
   
    // //fos<<"D->numDFGedgesc<="<<D->numDFGedges<<endl;
    // int is_success = 1;
    /* 先布线非地址计算，再布线地址计算的 */
    // for(int i = 0; i < D->numDFGedges; i++)
    // {
    //     int prenodeIndex  = D->getIndex(D->DFGedgesList[i]->prenode);
    //      fos<<"\nprenodeIndex :"<<prenodeIndex<<"\n";
    //     int srcR;
    //     int posnodeIndex  = D->getIndex(D->DFGedgesList[i]->posnode);
    //     fos<<"\nposnodeIndex :"<<posnodeIndex<<"\n";
    //     int time = D->DFGnodesList[prenodeIndex]->nodelevel;
    //     fos<<"\n time  :"<< time <<"\n";

    //     fos<<"\n D->DFGnodesList.size() :"<< D->DFGnodesList.size() <<"\n";

    //     // if(prenodeIndex > D->DFGnodesList.size()-1){
    //     //     fos<<"\nprenodeIndex > D->DFGnodesList.size()-1\n";
    //     //     return 0;
    //     // }
    //     fos<<"pre:"<<D->DFGnodesList[prenodeIndex]->nodelabel<<" ------- pos:"<<D->DFGnodesList[posnodeIndex]->nodelabel<<"  ------- preIsRouted:"<<D->DFGnodesList[prenodeIndex]->isRoute<<endl;
        
       
    //     // if(prenodeIndex > D->DFGnodesList.size()-1){
    //     //     fos<<"\nprenodeIndex > D->DFGnodesList.size()-1\n";
    //     //     return 0;
    //     // }
        
    //     fos<<"\nD->DFGnodesList[prenodeIndex]->kind "<<D->DFGnodesList[prenodeIndex]->kind;
    //     fos<<"\nD->DFGnodesList[prenodeIndex]->isRoute "<<D->DFGnodesList[prenodeIndex]->isRoute;

    //     if(D->DFGnodesList[prenodeIndex]->kind == 1 && D->DFGnodesList[prenodeIndex]->isRoute == false)/* 边的前驱是load算子,且没有被布线过*/
    //     {
    //          fos<<"\nif(D->DFGnodesList[prenodeIndex]->kind == 1\n";
    //         /* 获得一个没有被占用的LU,后面有相应的Bank，容量处理。
    //         isOccupied：1.如果Bank容量为0不可再使用;2如果某一层（II层）的bank已经使用过，那么该层不能再被使用  */
    //         /* 获得首结点的时间步 */
    //         int LUR = R->getLU(time,D->DFGedgesList[i]->prenode, D->numDFGnodes);
    //         // fos<<"time="<<time<<"  "<<"LUR="<<LUR<<endl;
    //         srcR = LUR ;/* 该LU就是真实的源头寄存器 */
    //         D->DFGnodesList[prenodeIndex]->bindResource = srcR;
    //     }else if(D->DFGnodesList[prenodeIndex]->kind == 0 && D->DFGnodesList[prenodeIndex]->isRoute == false) /* 依赖边的前端是普通算子，且没有被访问过 */
    //     {
    //          fos<<"D->DFGnodesList[prenodeIndex]->kind == 0\n";
    //         if(D->DFGnodesList[prenodeIndex]->isBind == true)
    //         {
    //             fos<<"if(D->DFGnodesList[prenodeIndex]->isBind == true)\n";
    //             srcR = D->DFGnodesList[prenodeIndex]->bindResource;/* 普通算子肯定已经布局了 */
    //         } else{
    //             fos<<"if(D->DFGnodesList[prenodeIndex]->isBind == true)  else\n";
    //             int nodewithoutIN = D->DFGedgesList[i]->prenode;
    //             int loadNo = D->DFGnodesList[nodewithoutIN]->loadNo;
    //             int bindResOfLoad = D->DFGnodesList[loadNo]->bindResource;
    //             int peresult = R->getpe(time,nodewithoutIN,bindResOfLoad);
    //             srcR = peresult ;
    //         }
            
    //         // fos<<"srcR = "<<srcR<<endl;
        
    //     }else/* 剩下的就是前驱结点已经被访问过了，同源边已经处理了 */
    //     {
    //         fos<<" continue;\n";
    //         continue;
    //     }
    //     fos<<" shareRoute \n";
    //     is_success = shareRoute(D, R, prenodeIndex,srcR,mlirdfg.TheoreticalII,allPathClass,inportNum,outportNum);
    //     if(is_success == 0){
    //         fos<<i<<" Failed to find the path \n";
    //         return 0;
    //     }
    //     D->DFGnodesList[prenodeIndex]->isRoute = true;/* 将边的前驱设为已经访问 */
    //     D->DFGnodesList[prenodeIndex]->bindResource = srcR;/* 为边的前驱绑定好资源 */
    //     D->DFGnodesList[prenodeIndex]->isBind = true;

       
    // }

    // for(int i = 0; i < D->DFGnodesList.size(); i++ )
    // {
    //     int bindR = D->DFGnodesList[i]->bindResource;
    //     fos<<i<<"   "<<bindR<<" "<<R->TERnodesList[bindR]->PE<<endl;
    // }
     /* to do:node的位置用界面展示 */
  
    //show(D, R,mlirdfg.TheoreticalII,PERow,PERow);
    // fos<<"--------------------------------------------"<<endl;
    
    // for(int i = 0; i < allPathClass->PathsList.size(); i++ )
    // {
    //     fos<<"Path["<<i<<"]:"<<endl;
    //     fos<<"pre:"<<allPathClass->PathsList[i]->DFGpre<<"  ------  pos:"<<allPathClass->PathsList[i]->DFGpos<<"  -----  latency:"<<allPathClass->PathsList[i]->latency<<endl;
    //     //showPath(R,allPathClass->PathsList[i]->point,mlirdfg.TheoreticalII); 
    // }
    
    //====================
    // int column = PERow + 2;
    // // fos<<"CGRARow="<<CGRARow<<"  "<<"column="<<column<<endl;
    // fos<<"---------------------show the map result-----------------------"<<endl;
    // int count = 0;
    //     count = 0;
    //     for (int k = 0; k < mlirdfg.TheoreticalII; k++)
    //     {
    //         for(int z = 0; z < column; z++)
    //         {
    //             fos << "--------";
    //         }
    //         fos << "["<< k <<"]";
    //         for(int z = 0; z < column; z++)
    //         {
    //             fos << "--------" ;
    //         }
    //         fos<<endl;                          
    //         for (int i = 0; i < PERow; i++) 
    //         {//行
    //             for (int j = 0; j < column; j++)
    //             {//列
    //                 for(int m = 0; m < D->DFGnodesList.size(); m++ )
    //                 {
    //                     int bindR = D->DFGnodesList[m]->bindResource;/* 绑定的寄存器 */
    //                     int postion = R->TERnodesList[bindR]->PE;/* 寄存器所在的PE */
    //                     int op_order = D->DFGnodesList[m]->nodelabel; 
                       
    //                     int time_step = D->DFGnodesList[op_order]->nodelevel;
    //                     if (time_step == k && postion / column % PERow == i && postion % column == j)
    //                     {
                        
    //                         count ++;
    //                         fos << "|\t  "<< op_order << "\t  ";
    //                     }

    //                 }
    //                 if (count == 0){
    //                     fos << "|\t\t  ";
    //                 }
    //                 count = 0;
    //             }
    //             fos << "|  " << endl;
    //         }
    //     }
    //     // fos << endl;
    //     for(int z = 0; z < column; z++)
    //     {
    //         fos << "---------" ;
    //     }
       
    
    //====================
    // for(int i = 0; i < allPathClass->PathsList.size(); i++ )
    // {
    //     vector<int> PathPoint = allPathClass->PathsList[i]->point;
    //     for(int j = 0; j < PathPoint.size(); j++ )
    //     {
            
    //         int node = PathPoint[j];
    //         // if(node>=264 && node<=267)
    //         // {
    //         //     // fos<<node<<endl;
    //         //     int node1PE = R->TERnodesList[node]->PE;
    //         //     int node2PE = R->TERnodesList[PathPoint[j+1]]->PE;
    //         //     // if(j-1!=0 && (node1PE % 32 != node2PE % 32))
    //         //     if(j-1!=0)
    //         //     {
    //         //         fos<<PathPoint[j-1]<<" "<<node<<endl;

    //         //     }

    //         // }
            
    //     }
        
        
    // }

    

    // for(int i = 0; i < TECPENUM; i++)
    // {
    //     if(C->CGRAnodesList[i%C->ElmNum]->ElmKind == 0)
    //     {
    //         fos<<i<<" "<<inportNum[i]<<" "<<outportNum[i]<<endl;

    //     }
    // }

   
    // gettimeofday(&tv1, 0); 
    // fos<<"--------------------------------------------"<<endl;
    // fos<< "spand time:" <<(tv1.tv_sec - tv.tv_sec + (double)(tv1.tv_usec - tv.tv_usec) / CLOCKS_PER_SEC)<<endl; 
     
    // fos.close();
    // delete C;
    // // delete T;
    // delete D;
    // delete allPathClass;
    return 1;

}
RF_CGRAMap::RF_CGRAMap(MlirDFG &mlirdfg,int solutionNum){
    this->is_success = this->find_Map(mlirdfg,solutionNum);  
}
}
