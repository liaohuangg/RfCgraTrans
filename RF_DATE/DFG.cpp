#include "CGRA.h"
#include "DFG.h"
#include "config.h"
#include "tool.h"

//#define childNum 4/* 待改为google形式 */
using namespace std;

DECLARE_string(dfg_file);
DFG::~DFG()
{
	for (auto node : DFGnodesList)
	{
		delete node;
	}
    for (auto edge : DFGedgesList)
	{
		delete edge;
	}
}



DFGnode::DFGnode()
{

}

DFGedge::DFGedge(const DFGedge* edge)
{
	this->prenode = edge->prenode;
	this->posnode = edge->posnode;
	this->dif = edge->dif;
	this->latency = edge->latency;
 
	this->isRoute = edge->isRoute;
	this->edgeorder = edge->edgeorder;
}

DFGedge::DFGedge()
{

}

DFG::DFG(int II, int childNum)
{
    int i, j, k;
    vector<vector<int>> DFG_node;
    int lines = ReadDataFromFileLBLIntoString(DFG_node);
    
    numDFGnodes = lines;
    int count_nodes = 0;
    // for (int i = 0; i < DFG_node.size(); ++i) {
    //    for (int j = 0; j < DFG_node[i].size(); j++){
    //        cout << DFG_node[i][j] << ",";
    //    }   
    //    cout << endl;
    // } 

    for (int i = 0; i < DFG_node.size(); i++){
        DFGnode *node = new DFGnode();

        node->nodelabel = DFG_node[i][0];/* DFG编号，从0开始 */
        node->nodelevel = DFG_node[i][1];/* DFG算子折叠后的时间步，从0开始 */
        node->oldlevel =  DFG_node[i][1];/* DFG算子的时间步，从0开始 */
        node->kind = DFG_node[i][2];/* DFG算子的种类，0为普通算子，1为load算子，2为store算子 */
        
        node->bindResource = -1;
        node->isBind = false;
        node->isRoute = false;
        DFGnodesList.push_back(node);   
        // cout<<node->loadNo<< " ";
        count_nodes ++;
    }
    // cout<<endl;

    int edge_count = 0;
    for (int i = 0; i < DFG_node.size(); i++){
        int current_node_order = DFG_node[i][0];
        int child[childNum]={0}; 
        int childDif[childNum]={0};
        for (int j = 0; j < childNum; j++)
        {         
            child[j] = DFG_node[i][3 + j];
            childDif[j] = DFG_node[i][3 + childNum + j];
        }
        for (int l = 0; l < childNum; l++){
            if(child[l] != -1){/* 因为编号从0开始，所以-1表示不是孩子 */
                DFGedge *edge = new DFGedge();
                edge->edgeorder = edge_count;
                edge->prenode = current_node_order;
                edge->posnode = child[l];
                edge->dif = childDif[l];

                /* 为每条边赋值 latency */
                int la =  edge->dif * II + abs(getNodeTime(edge->posnode) - getNodeTime(edge->prenode));
                edge->latency = la;
                DFGedgesList.push_back(edge);   
                edge_count ++;
            }
        }
    }
    numDFGedges = edge_count;
    int numedges = numDFGedges; 
    int longcount = 0;
    for (int i = 0; i < numDFGedges; i++)
    {
        if(DFGedgesList[i]->latency > 1)
        {

            longcount ++;
        }
        //cout<<DFGedgesList[i]->edgeorder<<" "<<DFGedgesList[i]->prenode<<" "<<DFGedgesList[i]->posnode<<" "<<DFGedgesList[i]->dif<<" "<<endl;
    }
    cout<<"the number of long latency:"<<longcount<<endl;


   
}

int DFG:: getIndex(int nodeLabel)
{
    vector<DFGnode*> nodesList = this->DFGnodesList;
	for (int i = 0;i<nodesList.size();i++)
	{
		if (nodesList[i]->nodelabel == nodeLabel)
		{
			return i;
		}
	}
	return -1;

}


bool DFG::DFGgraphHasEdge(size_t begin, size_t end)
{
	bool flag = false;
	vector<DFGedge*> edges = this->DFGedgesList;
	for (auto edge : edges)
	{
		if (edge->prenode == begin && edge->posnode == end)
		{
			flag = true;
			break;
		}
	}
	return flag;
}

int DFG:: getNodeTime(int nodeLabel)
{
	vector<DFGnode*> nodesList = this->DFGnodesList;
	for (auto node : nodesList)
	{
		if (node->nodelabel == nodeLabel)
		{
			return node->oldlevel;
		}
	}
	return -1;
}

int DFG:: getNodeModuleTime(int nodeLabel)
{
	vector<DFGnode*> nodesList = this->DFGnodesList;
	for (auto node : nodesList)
	{
		if (node->nodelabel == nodeLabel)
		{
			return node->nodelevel;
		}
	}
	return -1;
}

int DFG::getNodeKind(int nodeLabel)
{
    vector<DFGnode*> nodesList = this->DFGnodesList;
	for (auto node : nodesList)
	{
		if (node->nodelabel == nodeLabel)
		{
			return node->kind;
		}
	}
	return -1;

}




void DFG::CreatMDFG(int II){

    for(int i = 0; i <this->numDFGnodes; i++)
    { 
        this->DFGnodesList[i]->nodelevel = this->DFGnodesList[i]->nodelevel % II;    
    }
}

int DFG::Constraint_Level(int II){
    int i;

    int LoadNum[II]={0};/* 每一层的Load都为0 */
    int SUNum[II]={0};/* 每一层的store算子都为0 */
    int PENum[II]={0};/* 每一层的普通算子都为0 */
    int flag;   
    flag = -1;
    for(i = 0; i < this->numDFGnodes; i++)/* 遍历所有的DFG节点 */
    {
        if(this->DFGnodesList[i]->kind == 0)/* 普通算子 */
        {
            PENum[this->DFGnodesList[i]->nodelevel] ++;

        }
        else if(this->DFGnodesList[i]->kind == 1)/* load算子 */
        {
            LoadNum[this->DFGnodesList[i]->nodelevel] ++;
        }
        else if(this->DFGnodesList[i]->kind == 2)/* store算子 */
        {
            SUNum[this->DFGnodesList[i]->nodelevel] ++;
        }
    }
    // for(i = 0; i < II; i++)
    // {
    //     if(PENum[i] > 16 || LoadNum[i] > 4  || SUNum[i]>4 )
    //     {
    //         flag = i;
    //         printf("The %d Level of the DFG cannot meet the constrait! \n", flag);
    //         printf("*****************************************************************************\n");
    //         break;
    //     }
    // }

    return flag;
}

