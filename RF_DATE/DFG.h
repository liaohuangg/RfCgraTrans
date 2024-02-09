#ifndef DFG_H
#define DFG_H

#include <vector>
using namespace std;

class DFGnode{
public:
    DFGnode();
    int nodelabel;//DFG算子的编号，就是它的名字
    int oldlevel;//DFG原始时间步
    int nodelevel;//DFG折叠后的时间步
    int kind;//DFG算子的种类,0是普通算子，1是Load，2是store算子
    int isAddress;//是否是地址
    int bindResource;//绑定的硬件资源
    bool isBind;
    bool isRoute;
    int loadNo;
    
    
};

class DFGedge{
public:
    DFGedge();
    DFGedge(const DFGedge* edge);
    int edgeorder;//
    int prenode;//
    int posnode;//
    int dif;
    //bool isLong;
    bool isRoute;
    int latency;
   
};

class DFG{
public:
    int numDFGnodes;
    int numDFGedges;
    vector<DFGnode*> DFGnodesList;//
    vector<DFGedge*> DFGedgesList;//DFG边集合
    DFG(int II, int childNum);
    DFG* CreatDFG();//通过文件创建一个DFG
    ~DFG();
    bool DFGgraphHasEdge(size_t begin, size_t end);
    int getNodeTime(int nodeLabel);
    int getNodeModuleTime(int nodeLabel);
    int getNodeKind(int nodeLabel);
    int getIndex(int nodeLabel);
    void CreatMDFG(int II);
    int Constraint_Level(int II);
};

#endif // DFG_H

