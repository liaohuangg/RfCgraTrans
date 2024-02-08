#ifndef GRAPHREGISTER_H
#define GRAPHREGISTER_H
#include "config.h"
#include "CGRA.h"
#include "Register.h"
using namespace std;

class GraphNode{      
public:
    GraphNode();    
    int GNodeID; /* 寄存器图顶点的ID */  
    int RegisterKind;
    vector<int> GNodeNeighbors;
};

class GraphEdge{      
public:
    GraphEdge();    
    int GEdgeId; /* 边的ID */   
   	int pre; /*边的前端*/
	  int pos; /*边的后端*/
    int value;
};

class GraphRegister
{
public:
    GraphRegister(int length, Register *R, int SrcTrueTime,int II);
    int GraphnodesNums;
		int GraphedgesNums;
    
		vector<GraphNode*> GraphnodesList;
		vector<GraphEdge*> GraphedgesList;
   
    bool graphHasEdge(int begin, int end);
    int getEdgeCost(int begin, int end);
    void setEdgeValueFromPre(int pre,int value);
    void setEdgeValueToPos(int pos,int value);
    ~GraphRegister();
};
#endif