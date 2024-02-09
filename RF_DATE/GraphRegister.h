#ifndef GRAPHREGISTER_H
#define GRAPHREGISTER_H
#include "config.h"
#include "CGRA.h"
#include "Register.h"
using namespace std;

class GraphNode{      
public:
    GraphNode();    
    int GNodeID; /* �Ĵ���ͼ�����ID */  
    int RegisterKind;
    vector<int> GNodeNeighbors;
};

class GraphEdge{      
public:
    GraphEdge();    
    int GEdgeId; /* �ߵ�ID */   
   	int pre; /*�ߵ�ǰ��*/
	  int pos; /*�ߵĺ��*/
    int value;
};

class GraphRegister
{
public:
    GraphRegister(int CGRAElmNum,int length, Register *R, int SrcTrueTime,int II);
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