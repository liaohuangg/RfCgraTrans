#ifndef REGISTER_H
#define REGISTER_H
#include "config.h"
#include "CGRA.h"
using namespace std;

class Registernode{      
public:
    Registernode();    
    int RegisterID; /* 寄存器的ID */  
    int PE;  /*所属的PE*/ 
    int PEinportNum;
    int PEoutportNum;
    int RegisterKind;/* 0 R，1，load,2，store，FIFO,4，FU，5*/
    bool isOccupied;
    bool usedBypass;
    bool usedTimeTrans;
    int time;
    bool inPort;//输入端口是否已经被占。
    
    vector<int> RegisterNeighbors;
    string info;
};

class Registeredge{      
public:
    Registeredge();    
    int edgeId; /* �ߵ�ID */   
   	int srcReg; /*Դ�Ĵ���*/
	  int tgtReg; /*Ŀ��Ĵ���*/
};

class Register
{
public:
    Register(int Rnum,CGRA *C);
    int RnodesNums;
		int RedgesNums;
    int II;
    int TERnodesNums;
    int TERedgesNums;
   
    int Rnum;//ÿ��PE�ļĴ�������
		vector<Registernode*> RnodesList;
		vector<Registeredge*> RedgesList;
   
    vector<Registernode*> TERnodesList;
    vector<Registeredge*> TERedgesList;
    
   
   
    int getLU(int time,int,int);
    int getpe(int time,int,int);
    int getNodeKind(int nodeLabel);
    int getSUBank(int lu_su_index);
    int getIndex(int R);
    int getLUR(int LU);
    int getLUorResultSU(int SRC);
    void getLURSet(int time,vector<int> &candidataR);
    int getBankLU(int su_bank);
    void getSUSet(int time,vector<int> &candidateR);
    void getResultRSet(int time,vector<int> &candidataR);
    void CreatTER(int II,int CGRAElmNum);
    ~Register();
};
#endif