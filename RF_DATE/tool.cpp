#include "config.h"
#include "DFG.h"
#include "tool.h"
#define LIMIT 64

using namespace std;
DECLARE_string(dfg_file);

bool comp(DFGedge* &a,DFGedge* &b)
{
    return a->latency < b->latency;
}


int calculateFU(int neiPE,int colNum,int Rnum)
{
    int lsuColNum = 1;
    
    return ((neiPE % colNum) - 1 + neiPE / colNum * (colNum-lsuColNum)) * (Rnum+1) + (neiPE / colNum + 1) * lsuColNum;
}
			

vector<DFGedge*> getEdgesFromBegin(vector<DFGedge*> edgesList, int beginNo)
{
	vector<DFGedge*> edges;/* 新建一个 */
	for (auto e : edgesList)
	{
		if (e->prenode == beginNo)
		{
			edges.push_back(new DFGedge(e));
		}
	}
	return edges;
}


void strtok(string& str, vector<string>& cont, string defstr = " ")
{ 
    cont.clear();
    size_t size = 0;
    size_t begpos = 0;
    size_t endpos = 0;
    begpos = str.find_first_not_of(defstr);
    while (begpos != std::string::npos)
    {
        size++;
        endpos = str.find_first_of(defstr, begpos);

        if (endpos == std::string::npos)
        {
            endpos = str.size();
        }

        string ssubstr = str.substr(begpos, endpos - begpos);

        cont.push_back(ssubstr);
        begpos = str.find_first_not_of(defstr, endpos+1);
    }
}

int ReadDataFromFileLBLIntoString(vector<vector<int>> &DFG_node)
{
    ifstream fin(FLAGS_dfg_file); 
    cout << "********************************" << endl;
    cout << "the name of File:"  << FLAGS_dfg_file << endl;
    cout << "********************************" << endl;
    string s;
    int nodes = 0;
    vector<int> linedata;
    int line = 0;
    while(getline(fin,s))
    {
        vector<string> vec;
        strtok(s,vec,",");
        linedata.clear();
        for(int i = 0; i < vec.size(); i++)
        {
            
           //atoi:ascii to integer
        
           int e = atoi(vec[i].c_str());
            linedata.push_back(e);

        }
        DFG_node.push_back(linedata);
        line++;
    }
    
    fin.close();
    return line;
}



void show(DFG *D, Register *R,  int II,int CGRARow, int PEColumn)
{
    int lsuColNum = 1;
    int column = PEColumn + lsuColNum;
    // cout<<"CGRARow="<<CGRARow<<"  "<<"column="<<column<<endl;
    cout<<"\n========================map成功===========================\n";
    int count = 0;
    {
        count = 0;
        for (int k = 0; k < II; k++)
        {
            for(int z = 0; z < column; z++)
            {
                cout << "--------";
            }
            cout << "["<< k <<"]";
            for(int z = 0; z < column; z++)
            {
                cout << "--------" ;
            }
            cout<<endl;                          
            for (int i = 0; i < CGRARow; i++) 
            {//行
                for (int j = 0; j < column; j++)
                {//列
                    for(int m = 0; m < D->DFGnodesList.size(); m++ )
                    {
                        int bindR = D->DFGnodesList[m]->bindResource;/* 绑定的寄存器 */
                        int postion = R->TERnodesList[bindR]->PE;/* 寄存器所在的PE */
                        int op_order = D->DFGnodesList[m]->nodelabel; 
                       
                        int time_step = D->DFGnodesList[op_order]->nodelevel;
                        if (time_step == k && postion / column % CGRARow == i && postion % column == j)
                        {
                        
                            count ++;
                            cout << "|\t  "<< op_order << "\t  ";
                        }

                    }
                    if (count == 0){
                        cout << "|\t\t  ";
                    }
                    count = 0;
                }
                cout << "|  " << endl;
            }
        }
        // cout << endl;
        if(column == 5)
        {
            for(int z = 0; z < column; z++)
            {
                cout << "---------" ;
            
            }

        }
        else  if(column == 7)
        {
            for(int z = 0; z < column; z++)
            {
                cout << "----------" ;
            
            }

        }
        else  if(column == 9)
        {
            for(int z = 0; z < column; z++)
            {
                cout << "-----------" ;
            
            }

        }
        
       
        
       
    } 
} 



bool isConcludeFU(GraphRegister *GR,int path[], int v)
{
    bool flag = false;
    stack<int> q;/* 建立一个栈 */
    int p = v;  /* p用来遍历path数组获得路径点 */
    
    while(path[p]!=-1)
    {
        q.push(path[p]);
        p = path[p];
    }   
    while(!q.empty())
    {
        int temp = q.top();
        
        if(GR->GraphnodesList[temp]->RegisterKind == 4)/* 走过的路包含了FU */
        {
            flag = true;
            break;
        }
        q.pop();
    
    }
    return flag;

}
                    


bool isUsedV(GraphRegister *GR,int path[], int v, int trueTRNum)
{
    
    bool flag = false;
    stack<int> q;/* 建立一个栈 */
    int p = v;  /* p用来遍历path数组获得路径点 */
    // cout<<"trueTRNum"<<trueTRNum<<endl;
    while(path[p]!=-1)
    {
        q.push(path[p]);
        p = path[p];
    }   
    // cout<<"tempPath:"
    while(!q.empty())
    {
        int temp = q.top();
        // cout<<temp<<" ";
    
        if(temp % trueTRNum == v % trueTRNum)/* 相同不包含bank */
        {
            flag = true;
            break;
        }
        q.pop();
    
    }
    // cout<<endl;

   


    return flag;
}




bool isIncludeCandidate(int n,vector<int> &virtualCandidate)
{
    bool flag = false;
    for(int i = 0;i < virtualCandidate.size(); i++)
    {
        if(virtualCandidate[i] == n)
        {
            flag = true;
            // cout<<"flag"<<flag<<endl;
            break;
        }
    }
    return flag;
}

void getPath(int path[], int minReg, vector<int> &pathNode)
{
    
    stack<int> q;/* 建立一个栈 */
    int p = minReg;  /* p用来遍历path数组获得路径点 */
    while(path[p]!=-1)
    {
        q.push(path[p]);
        p = path[p];
    }   
    while(!q.empty())
    {
        int temp = q.top();
        pathNode.push_back(temp);
        q.pop();
    }

}


// void processPEPort(node1,node2,Register *R);



int getPreLastLU(vector<int> pathNode, Register *R)
{

    for(int i = pathNode.size() - 1; i >= 0 ; i--)
    {
        int node = pathNode[i];
        int RegisterKind = R->TERnodesList[node]->RegisterKind;
        if(RegisterKind == 1)
        {
            return node;
            
        }

    }
    return -1;
}



 void deletePartPath(vector<int> &pathNode, int srcR)
 {
    /* 先找到迭代器 */
    for(std::vector<int>::iterator it = pathNode.begin();it != pathNode.end();it++)
    {
        if( *it == srcR)
        {
            pathNode.erase(it, pathNode.end());
            break;
        }

    }
}

/* 将共享的路径设为已经访问，除bank外*/
void setVisited(Register *R,vector<int> SingleTruePath)
{
    for(int i = 0; i < SingleTruePath.size(); i++)
    {
        int RegisterKind = R->TERnodesList[SingleTruePath[i]]->RegisterKind;
        if(RegisterKind == 1 || RegisterKind == 5 || RegisterKind == 4)/* 只有Lu要彻底设置已经访问 */
        {
            R->TERnodesList[SingleTruePath[i]]->inPort = true;
            R->TERnodesList[SingleTruePath[i]]->isOccupied = true;
        }      
    }
}



bool compLegal(int CGRAElmNum,Register *R,vector<int> sameSrcEdge, vector<int> tempTruePath)
{
    int longlength;
    int shortlength;
    int length1 = sameSrcEdge.size();
    int length2 = tempTruePath.size();

    // cout<<"sameSrcEdge.size()="<<length1<<"  "<<"tempTruePath.size()="<<length2<<endl;
    if(length1 > length2)
    {
        longlength = length1;
        shortlength = length2;
    }
    else
    {
        longlength = length2;
        shortlength = length1;
    }
    
    // cout<<" i am here"<<endl;
    // cout<<endl;
    // cout<<"trueNode";
    // for(int i = 0; i < tempTruePath.size(); i++ )
    // {
    //     cout<<tempTruePath[i]<<" ";
    // }
    // cout<<endl;
    
    // cout<<endl;
    // cout<<"sameSrcEdge";
    // for(int i = 0; i < sameSrcEdge.size(); i++ )
    // {
    //     cout<<sameSrcEdge[i]<<" ";
    // }
    // cout<<endl;

    /* 看该条已经布线的路径是否含Bank */
    int cgraPE =CGRAElmNum;
    int label =0;
    // cout<<"tempTruePath[length2-1]="<<tempTruePath[length2-1]<<endl;
    //  cout<<"shortlength="<<shortlength<<endl;
    for(int i = 0; i < shortlength ; i++ )
    {
        if(sameSrcEdge[i] != tempTruePath[i])
        {
            
            label = i;
            // cout<<"i="<<i<<endl;
            // cout<<"sameSrcEdge[i]="<<sameSrcEdge[i]<<" "<<"tempTruePath[i]="<<tempTruePath[i]<<endl;

            break;

        }
    }
   
   

    // cout<<"label"<<label<<endl;
    // cout<<"length1-2="<<length1-2<<endl;
    // cout<<"length2="<<length2<<endl
    if(label != 0)
    {

        if(length1-2 > 0)
        {
            for(int i = label; i < length1-2 ; i++ )/* 已经布好的 */
            {
                for(int j = label; j < length2 ; j++ )
                {
                    if(sameSrcEdge[i] == tempTruePath[j])
                    {
                        
                        if(i-1 >=0)
                        {
                            // cout<<"sameSrcEdge[i]="<<sameSrcEdge[i]<<" "<<"tempTruePath[j]="<<tempTruePath[j]<<endl;
                            // cout<<"sameSrcEdge[i]="<<sameSrcEdge[i-1]<<" "<<"tempTruePath[j]="<<tempTruePath[j]<<endl;
                            /* 平面的PE不同 */
                            if((R->TERnodesList[sameSrcEdge[i-1]]->PE % cgraPE != R->TERnodesList[sameSrcEdge[i]]->PE%cgraPE) && (R->TERnodesList[tempTruePath[j-1]]->PE%cgraPE != R->TERnodesList[tempTruePath[j]]->PE%cgraPE ))
                            {
                                
                                // cout<<"R->TERnodesList[sameSrcEdge[i-1]]->PE="<<R->TERnodesList[sameSrcEdge[i-1]]->PE<<"  "<<" R->TERnodesList[tempTruePath[j-1]]->PE2="<< R->TERnodesList[sameSrcEdge[i]]->PE<<endl;
                                return false;
                            }
                            

                        }
                        
                    }
                }
            }

        }
    }


    
    return true;
}
// bool shareisLegal(Register *R, int i,int *path,int srcTrueTime,vector<vector<int>> sameSrcEdge,vector<int> tempTruePath,int v ) 
// {

// }

bool isLegalShare(int CGRAElmNum,Register *R, int i,int *path,int srcTrueTime,vector<vector<int>> sameSrcEdge,vector<int> tempTruePath,int *inportNum,int *outportNum)
{

    stack<int> q;/* 建立一个栈 */
    int p = i;  /* p用来遍历path数组获得路径点 */
    // cout<<"p"<<p<<endl;
   
    while(path[p]!=-1)
    {
        q.push(path[p]);
        p = path[p];
    }  
    
    // cout<<"temp"<<endl;
    while(!q.empty())
    {
        int temp = q.top();
        int tempTime = temp / R->RnodesNums;
        /* temp是虚拟的点，要将其转换为真实的点 */
        int trueNode   = ( temp - (tempTime* R->RnodesNums) )+ ( (srcTrueTime + tempTime)  * R->RnodesNums ) % (R->II * R->RnodesNums);
        // cout<<temp<<" ";
        // cout<<trueNode<<" ";
        tempTruePath.push_back(trueNode);/*tempTruePath包含了所有路径了  */
        q.pop();
    }
    int iTime = i / R->RnodesNums;
    
    int truei   = ( i - (iTime* R->RnodesNums) )+ ( (srcTrueTime + iTime)  * R->RnodesNums ) % (R->II * R->RnodesNums);
    
    tempTruePath.push_back(truei);


    // cout<<"tempTruePath,.size()="<<tempTruePath.size()<<"  "<<"truei="<<truei<<endl;
    // cout<<endl;

    // for(int i = 0; i < tempTruePath.size(); i++)
    // {
    //     cout<<tempTruePath[i]<<" ";
    // }
    // cout<<endl;


    


    /* tempTruePath放的就是当前真实的路径，少最后一个结果寄存器 */
    /* 遍历前面已经布线的路径 */
    bool isLegal = true;
    // cout<<"sameSrcEdge.size()="<<sameSrcEdge.size()<<endl;
    for(int i = 0; i < sameSrcEdge.size(); i++ )
    {
        
        isLegal = compLegal(CGRAElmNum,R,sameSrcEdge[i],tempTruePath);/* 会被覆盖 */
        if(isLegal == false)
        {
            break;
        }
        // cout<<"isLegal"<<isLegal<<endl;
        
    }
    // cout<<"-----------------"<<endl;

    /* 借楼写一个：判断当前v和前一个路径点之间的关系 */
    

    /* 获得最后一个点的PE */
    int vR = tempTruePath[tempTruePath.size()-1];
    int vRpre = tempTruePath[tempTruePath.size()-2];

    int vPE = R->TERnodesList[vR]->PE;
    int vPEpre = R->TERnodesList[vRpre]->PE;
    if(inportNum[vPE] <= 0 || outportNum[vPEpre]<=0 )
    // if(inportNum[vPE] <= 0 )
    {
        isLegal = false;
    }
    // if( R->TERnodesList[vR]->time == R->TERnodesList[vPEpre]->time )
    // {
    //     if(R->TERnodesList[vR]->usedBypass == true)
    //     {
    //         isLegal = false;
    //     }
        
    // }


    tempTruePath.clear();

    return isLegal;

}
/* i之前的路径，v是是否可以添加的结点 */
/* TO DO :行不通 */
// void alterValue(GraphRegister *GR,int path[], int v, int i, int **&Graph)
// {
//     stack<int> q;/* 建立一个栈 */
//     int p = i;  /* p用来遍历path数组获得路径点 */
//     while(path[p]!=-1)compLegal
//     {
//         q.push(path[p]);
//         p = path[p];
//     }  

//     vector<int> tempPath;
//     while(!q.empty())
//     {
//         int temp = q.top();
//         tempPath.push_back(temp);
//         q.pop();
//     }

//     int count = 0;
//     for(int j = 0; j< tempPath.size();j++)
//     {
//         int temp = tempPath[j];
//         int RegisterKind = GR->GraphnodesList[temp]->RegisterKind; 
//         if(RegisterKind == 3)/* 第一次出现的 */
//         {
//            count ++;
//         }

//     }
//     if(count % 2 == 0 && Graph[v][i] != INF )/* 偶数 并且v,i间有边*/
//     {
//         Graph[v][i] = 100;
//     }



    
// }





bool dijkstra(int CGRAElmNum,Dijk *dijk,Register *R,GraphRegister *GR,vector<int> &virtualCandidate,vector<int> &pathNode,vector<vector<int>> sameSrcEdge,vector<int> SingleTruePath,int *inportNum,int *outportNum)
// void dijkstra(Register *R,GraphRegister *GR,int latency,int virtualSrc,vector<int> &virtualCandidate,vector<int> &pathNode,int trueTRNum,int srcTrueTime,vector<vector<int>> sameSrcEdge,vector<int> SingleTruePath)
{

    // cout<<"-----------------"<<endl;
    /*定义 */
    bool *known;
    int *dist;
    int *path;
    int nodeNum = GR->GraphnodesNums;
    // cout<<"nodeNum"<<nodeNum<<endl;
    // cout<<"virtualCandidate.size()="<<virtualCandidate.size()<<endl;
    // cout<<"nodeNum"<<nodeNum<<endl;
    known = new bool[nodeNum];/* 是否已经访问，确定了最短路径的标志 */
    memset(known, 0, nodeNum * sizeof(bool));
    dist = new int[nodeNum];
    memset(dist, 0, nodeNum * sizeof(int));/* 源点到当前点的距离 */
    path = new int[nodeNum];
    memset(path, 0, nodeNum * sizeof(int));/* 前驱节点 */

    int **Graph = new int*[nodeNum]; //开辟行  
    for (int i = 0; i < nodeNum; i++)
        Graph[i] = new int[nodeNum]; //开辟列 
    
    for (int i = 0; i < nodeNum; i++)/* 初始化为无穷大 */
        for(int j = 0;j < nodeNum; j++)
            Graph[i][j] = INF;
     
    for (int i = 0; i < GR->GraphedgesList.size(); i++)
    {
        
        Graph[GR->GraphedgesList[i]->pre][GR->GraphedgesList[i]->pos] = GR->GraphedgesList[i]->value;

    }
    

    /* 初始化 */
    // cout<<"dijk->latency="<<dijk->latency<<endl;
    // cout<<"dijk->virtualSrc="<<dijk->virtualSrc<<endl;
    for(int i = 0; i < nodeNum; ++i)
    {
        
        
        known[i] = false;/* 最开始所有点都没有被访问 */
        /* 获得源点到所有点的代价 */
        //dist[i]  = GR->getEdgeCost(virtualSrc,i);
        dist[i]  = Graph[dijk->virtualSrc][i];
        //path[i]  = GR->graphHasEdge(virtualSrc,i) ==  true ? virtualSrc:-1;
        path[i]  = Graph[dijk->virtualSrc][i] <  INF ? dijk->virtualSrc:-1;
    }
    
   
    known[dijk->virtualSrc] = true;
    dist[dijk->virtualSrc]  = 0;
    path[dijk->virtualSrc]  = -1;
    
    // cout<<"dijk->virtualSrc="<<dijk->virtualSrc<<"  "<<"dijk->virtualSrc="<<dijk->virtualSrc<<endl;
    for(int j = 0; j < nodeNum-1 ;j++)
    {
        //找到unknown的dist最小的顶点 
        int v = 0;
        int min = INF;
        for(int i = 0; i < nodeNum; ++i){
            if(!known[i] && (min > dist[i]))/* i没有知道最小的dist */
            {
                min = dist[i];
                v = i;
            }
        }
        known[v] = true;


        //更新与v相邻所有顶点w的dist,path
        for(int i = 0; i < nodeNum;i++){
            if(!known[i] ){     
                if(dist[i] > dist[v] + Graph[v][i] && isUsedV(GR, path, v, dijk->trueTRNum) == false && isLegalShare(CGRAElmNum,R,v,path,dijk->srcTrueTime,sameSrcEdge,SingleTruePath,inportNum, outportNum) == true)              
                // if(dist[i] > dist[v] + Graph[v][i] && isUsedV(GR, path, v, dijk->trueTRNum) == false )              
                // if(dist[i] > dist[v] + Graph[v][i] )              
                // if(dist[i] > dist[v] + Graph[v][i]  && isLegalShare(R,v,path,dijk->srcTrueTime,sameSrcEdge,SingleTruePath) == true)              
                          
                {                  
                    dist[i] = dist[v] + Graph[v][i];
                    path[i] = v;                  
                }
            }
        }
    }
     

    int minRegcost = INF;
    int minReg = dijk->virtualSrc;
    // cout<<"virtualCandidate="<<virtualCandidate[0]<<endl;
    
    for(int i = 0; i < nodeNum; ++i)
    {      
        if(known[i])/* 访问过的 */
        {
           
            if (dist[i] < minRegcost)
            {
                if(isIncludeCandidate(i,virtualCandidate) == true )
                {
                  
                        /* mini,找出mini*/
                        // cout<<"minReg"<<minReg<<endl;
                        minRegcost = dist[i];
                        minReg = i;
            
                }
            }            
        }
            
    } 

   
    /* 将mini的路径读出，记在pathNode中 */

    getPath(path, minReg,pathNode);
    
    if(pathNode.size() == 0)
    {
        return false;

    }
    else
    {
        pathNode.push_back(minReg);
        
        delete [] known; 
        delete [] dist; 
        delete [] path;
        
        for (int m = 0; m < nodeNum; m++)
        {
            delete[] Graph[m];
            // cout<<"i am here"<<endl; 

        }
        delete[] Graph;
        
        return true;
        
    }
    
    
    
}





void setSrcRandRestLantency(Register *R, vector<int>  &SingleTruePath,int &srcR,int latency,int &restLatency,int II)
{
 
    SingleTruePath.clear();/* 不能共享的全部清除置0 */
    restLatency = latency  - SingleTruePath.size();/* 除掉共享的，剩余的长度 */   
}

/* 同源的边一起布线
prenodeIndex：所有同源边的前驱
srcR:该前驱绑定的资源 */
void shareRoute(int CGRAElmNum,DFG *D, Register *R, int prenodeIndex, int srcR, int II, AllPath *allPathClass,int *inportNum,int *outportNum)
{

    vector<DFGedge*>  DFGedgeFromLoad = getEdgesFromBegin(D->DFGedgesList,prenodeIndex);/* 获得以该Node为源头的边 */
    // cout<<"DFGedgeFromLoad.size()"<<DFGedgeFromLoad.size()<<endl;
    sort(DFGedgeFromLoad.begin(),DFGedgeFromLoad.end(),comp); /* 将同源边根据latency的大小排一个序,小的在前 */
    
    vector<int> SingleTruePath;/* 单条真实的路径 */
    vector<vector<int>> sameSrcEdge; /* 已经布线了同源边 集合 */
   
    for(int i = 0; i < DFGedgeFromLoad.size(); i++) /* 从小到大一条条布线 */
    {
        // cout<<"prenode="<<DFGedgeFromLoad[i]->prenode<<"      posenode="<<DFGedgeFromLoad[i]->posnode<<"       latency"<<DFGedgeFromLoad[i]->latency<<endl;
        
        int posNode = DFGedgeFromLoad[i]->posnode;/* 边的后驱 */
        int posnodeIndex  = D->getIndex(posNode);/* 边的后驱索引 */
        int latency = DFGedgeFromLoad[i]->latency;/* 当前边的总latency */
        int restLatency = 0;

        int desKind = D->getNodeKind(posNode);    /* 获得DFG目标算子的种类 */  
        int desTime = D->getNodeModuleTime(posNode);/* 获得目的算子的折叠后的时间步 */
       
        /* ----------------------确定在dijkstr算法的 srcR, restLatency（构建图，虚拟层数），目标点，已经存在的共享路径点 */
        
        // cout<<"srcR = "<<srcR<<endl;
        /* ------------ srcR, restLatency-------------- */
        setSrcRandRestLantency(R, SingleTruePath,srcR,latency,restLatency,II);
        // cout<<"srcR = "<<srcR<<endl;


        /* ------------目标点-------------------------- */
        vector<int> candidataR;/* 存放真实的目的点集合 */

        if(D->DFGnodesList[posnodeIndex]->isBind == false)/* 没有被绑定 */
        {
           if(desKind == 0)/* 否则就是普通的计算算子 */ /* 需要放在真实结果寄存器中 */
            {
                // cout<<"desTime="<<desTime<<endl;
                R->getResultRSet(desTime,candidataR);
                // cout<<"candidataR.size()="<<candidataR.size()<<endl;
                // cout<<"candidataR[0]="<<candidataR[0]<<endl;
            }
            else if(desKind == 1)/* Load算子 */
            {
                R->getLURSet(desTime,candidataR);
                // cout<<"candidataR.size()="<<candidataR.size()<<endl;

            }
            
        }
        else/* 绑定过了的 */
        {
            int des = D->DFGnodesList[posnodeIndex]->bindResource;
            // cout<<"des="<<des<<endl;
            
            R->TERnodesList[des]->inPort = false;/* 防止因为设置过已经访问，而无法再到达Result */
            if(des-1 > 0)
            {
                R->TERnodesList[des-1]->inPort = false;/* 防止因为设置过已经访问，而无法再到达FU */
            }
            candidataR.push_back(des);/* des是结果寄存器 */
            
        }
        // cout<<"srcR="<<srcR<<endl;
        



        int srcTrueTime = R->TERnodesList[srcR]->time;/* 当前源点所在的真实层数 */
       
        /* -------------构建虚拟图来寻路------------------------ */
        GraphRegister *GR = new GraphRegister(CGRAElmNum,restLatency + 1, R, srcTrueTime, II);/* latency+1层 */
       

        /* -----------------------为dijkstr准备参数-------------------------- */
       
        /* 将srcR真实的，转为虚拟的,虚拟的在第一层 */
        int virtualSrc = srcR % R->RnodesNums; 
        // cout<<"virtualSrc="<<virtualSrc<<endl;

        /* 将candidataR真实的，也转为虚拟的，层数在虚拟的最后一层 */
        vector<int> virtualCandidate;
        for(int m = 0; m < candidataR.size(); m++)
        {
            int virtualDes = candidataR[m] % R->RnodesNums + restLatency * R->RnodesNums;
            virtualCandidate.push_back(virtualDes);
            // cout<<"candidatashareRouteR[m]="<<candidataR[m]<<endl;
        }
        
        

        vector<int> SingleVirtualPath;/* 存放在单条虚拟的在dijkstr算法的路径，可能是不完整路径 */ 
        Dijk *dijk = new Dijk();
        /* 创建dijstra传入参数的类 */
        dijk->latency = restLatency;
        dijk->virtualSrc = virtualSrc;
        dijk->trueTRNum = R->TERnodesNums;
        dijk->srcTrueTime = srcTrueTime;
        dijk->nodeDesKind = desKind;
        bool dijSuccess = false;
        
        // cout<<"restLatency="<<restLatency<<endl;
        dijSuccess = dijkstra(CGRAElmNum,dijk,R,GR,virtualCandidate,SingleVirtualPath,sameSrcEdge,SingleTruePath,inportNum,outportNum);/* 将candidataR传入，i/92 == latency，代表最后一层，并且它在candidaeR里就比较*/            
        // cout<<"dijSuccess="<<dijSuccess<<endl;
        if(dijSuccess == false)
        {
            cout<<"#################################"<<endl;
            cout<<"dijkstra short can't find path"<<endl;
            cout<<"#################################"<<endl;
        }
        
       
        delete dijk;

       

        // cout<<"virtual:"<<endl;
        for(int m = 0; m < SingleVirtualPath.size(); m++)
        {
            int count = 0;
            int virtualTime = SingleVirtualPath[m] / R->RnodesNums;
            // cout<<SingleVirtualPath[m]<<" ";
            int TruePath = ( SingleVirtualPath[m] - (virtualTime * R->RnodesNums) )+ ( (srcTrueTime + virtualTime)  * R->RnodesNums ) % (II * R->RnodesNums);
            int kind = R->TERnodesList[TruePath]->RegisterKind;

          
           
            SingleTruePath.push_back(TruePath);/* 将在dijkstra算法中找的部分路径转为真实的路径添加到全部路径中去 */
           
        }
        // cout<<endl;
        
       


        /* 单条真实的布好的线 */
        // showPath(R,SingleTruePath,II); 
       
        /* 创建一个Path对象，将vector的数据复制 */
        Path* path = new Path();
		path->DFGpre = prenodeIndex;
        path->DFGpos = posNode;
        path->point = SingleTruePath;
        path->latency = latency;
        
        allPathClass->PathsList.push_back(path);/* 总路线 */

        sameSrcEdge.push_back(SingleTruePath);/* 只包含同源路线 */

        /* if SingleTruePath是最后一条了，在这里设置访问，*/

        /* to do:设置为已经访问，主要是ld，su.  bank，PE内的寄存器都暂时不需要 */
        if( i == DFGedgeFromLoad.size() - 1)
        {
            setVisited(R,SingleTruePath);
        }

        delete GR;/* 某条边已经布好线了 */

        int lastnodeInPath = SingleTruePath[SingleTruePath.size() - 1];/* 这个是结果寄存器或SU */
        // cout<<"lastnodeInPath="<<lastnodeInPath<<endl;
        // cout<<"posnodeIndex="<<posnodeIndex<<endl;
        if(desKind == 0)
        {
            R->TERnodesList[lastnodeInPath-1]->inPort = true;
        }
        R->TERnodesList[lastnodeInPath]->inPort = true;
        R->TERnodesList[lastnodeInPath]->isOccupied = true;
        
        D->DFGnodesList[posnodeIndex]->bindResource = lastnodeInPath;
        // cout<<"posnodeIndex="<<posnodeIndex<<"  "<<"D->DFGnodesList[posnodeIndex]->bindResource ="<<D->DFGnodesList[posnodeIndex]->bindResource <<endl;
        D->DFGnodesList[posnodeIndex]->isBind = true;
  
        SingleVirtualPath.clear();
        candidataR.clear();
        SingleTruePath.clear();
 
    }//for:同源的边
   
    

    /* 同源的布完线后，重新设置一下TERegister的访问。pe1->pe2的边，pe2经历过从外面得到数据，因此入PE2的外界边的权值都要设为infine */
    /* 遍历每一条同源的路径，如果是PE-PE,就要将 TERegister对应的点的属性设置一下*/
    // sameSrcEdge vector<vector<int>> sameSrcEdge;
    // cout<<"-----***********------------------"<<endl;
    /* 不是粗暴的将某个寄存器设为不能访问，因为它可能还能旁路 */

    int CGRAnum = CGRAElmNum;
    map<int, int> m1;
    for(int i = 0; i < sameSrcEdge.size(); i++)/* 同源的所有长依赖 */
    {
        for(int j = 0; j < sameSrcEdge[i].size() - 1; j++)/*  */
        {
            int node1 = sameSrcEdge[i][j];
            int node2 = sameSrcEdge[i][j+1];
            int node1PE = R->TERnodesList[node1]->PE;
            int node2PE = R->TERnodesList[node2]->PE;
            /* to do:将node1和node2同时设置为已经访问，map */
            
            if(m1.count(node1)==0 || m1.count(node2)==0)
            {
                m1.insert ( pair <int, int> ( node1, node2 ) );
                if(R->TERnodesList[node1]->RegisterKind == 0 && R->TERnodesList[node2]->RegisterKind == 0 )/* 普通寄存器-》普通寄存器 */
                {
                    if(node1PE % CGRAnum != node2PE % CGRAnum )/* 数据传出去了 */
                    {
                        /* 获得当前的PE */
                        inportNum[node2PE]--;/* 输入端口减1 */
                        outportNum[node1PE]--;/* 输出端口减1 */
                    }

                }
                if(R->TERnodesList[node1]->RegisterKind == 5 && R->TERnodesList[node2]->RegisterKind == 0 )/* 普通寄存器-》普通寄存器 */
                {
                    if(node1PE % CGRAnum != node2PE % CGRAnum )/* 数据传出去了 */
                    {
                        /* 获得当前的PE */
                        inportNum[node2PE]--;/* 输入端口减1 */
                        outportNum[node1PE]--;/* 输出端口减1 */
                    }

                }
                if(R->TERnodesList[node1]->RegisterKind == 1 && R->TERnodesList[node2]->RegisterKind == 0 )/* LU-》普通寄存器 */
                {
                    
                    inportNum[node2PE]--;/* 输入端口减1 */          
                }
                if(R->TERnodesList[node1]->RegisterKind == 0 && R->TERnodesList[node2]->RegisterKind == 2 )/* 普通寄存器-->SU */
                {
                    
                    outportNum[node1PE]--;/* 输出端口减1 */          
                }
                if(R->TERnodesList[node1]->RegisterKind == 0 && R->TERnodesList[node2]->RegisterKind == 1 )/* 普通寄存器-->LU */
                {
                    
                    outportNum[node1PE]--;/* 输出端口减1 */          
                }

            }
            else
            {
                continue;
            }
            // processPEPort(node1,node2,R);
            /* 可能会被重复的减 */

            


            /* lu->pe,其实这个Pe还可以用
            PE内部流动也可以不能用了。 */
            /*  */
            if((R->TERnodesList[node1]->time == R->TERnodesList[node2]->time)  && R->TERnodesList[node2]->RegisterKind == 0)/* PE内寄存器 */
            {
                // cout<<"node1=" <<node1 <<"  " <<"node2=" <<node2 <<endl;
                R->TERnodesList[node2]->usedBypass = true;
                
               
            }
            else if((R->TERnodesList[node1]->time != R->TERnodesList[node2]->time)  && R->TERnodesList[node2]->RegisterKind == 0)
            {
                R->TERnodesList[node2]->usedTimeTrans = true;
            }

            if(R->TERnodesList[node2]->usedBypass && R->TERnodesList[node2]->usedTimeTrans)
            {
                R->TERnodesList[node2]->inPort = true;/* 已经访问了，不能外来数据了 */

            }
           

            
        }
     
    }
    for (auto edge : DFGedgeFromLoad)
    {
        delete edge;
    }
    sameSrcEdge.clear();
 

}

void showPath(Register *R, vector<int> PathPoint,int II)
{
   
    for(int j = 0; j < PathPoint.size(); j++ )
    {
        
        int node = PathPoint[j];
        cout<<node<<" ";
        
    }
    cout<<endl;

    


}



