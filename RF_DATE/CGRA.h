#ifndef CGRA_H
#define CGRA_H
#include "config.h"

using namespace std;

class CGRAnode{      
public:
    CGRAnode();    
    int ElmID; /* 元素的ID */   
    int inportNum;
    int outportNum;
    int ElmKind;/* 元素的种类,0代表计算单元PE，1代表LU单元，3代表Bank */
    vector<int> ElmNeighbors;
};

class CGRA
{
public:
    CGRA();
	int ElmNum;/* 元素的总数 */
    int ColNum;//CGRA阵列的列数
    CGRA(int CGRARow, int CGRAColumn);
    vector<CGRAnode*> CGRAnodesList;
    ~CGRA();
};
#endif