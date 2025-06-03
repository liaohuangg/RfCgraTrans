#ifndef PATH_H
#define PATH_H
#include "config.h"

using namespace std;

class Path{      
public:
    Path();    
    //int PathID;  
    int DFGpre;  
    int DFGpos;  
    int latency;
    vector<int> point;
};


class AllPath
{
public:
    AllPath();       
		vector<Path*> PathsList;
    ~AllPath();
};

class Dijk
{
public:
    Dijk();    
    int latency;
    int virtualSrc;
    int trueTRNum;
    int srcTrueTime;
    int nodeDesKind;
    
};
#endif