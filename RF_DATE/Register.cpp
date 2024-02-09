#include "Register.h"
#include "tool.h"

Registernode::Registernode()
{

}
Registeredge::Registeredge()
{

}

Register::~Register()
{
	for (auto node : RnodesList)
	{
		delete node;
	}
 	for (auto edge : RedgesList)
	{
		delete edge;
	}
	for (auto tenode : TERnodesList)
	{
		delete tenode;
	}
 	for (auto teedge : TERedgesList)
	{
		delete teedge;
	}
	

}

int Register::getIndex(int R)
{
	for (int i = 0; i < this->TERnodesList.size();i++)
	{
		if (this->TERnodesList[i]->RegisterID == R )
		{
			return i;

		}
	}
}

int Register::getLU(int time, int pre,int DFGnodeNum)
{
	// 感觉放在第一个LU不好跑出来
    for (int i = 1; i < this->TERnodesList.size();i++)
	{
		// if (this->TERnodesList[i]->RegisterKind == 1 && this->TERnodesList[i]->time == time && this->TERnodesList[i]->isOccupied == false)/* LU寄存器 */
		if (this->TERnodesList[i]->RegisterKind == 1 && this->TERnodesList[i]->time == time )/* LU寄存器 */
		{
			// cout<<"this->TERnodesList[i]->RegisterID="<<this->TERnodesList[i]->RegisterID<<"this->TERnodesList[i]->isOccupied="<<this->TERnodesList[i]->isOccupied<<endl;
			if(this->TERnodesList[i]->isOccupied == false )
			{
				int lu = this->TERnodesList[i]->RegisterID;
				return lu;

                   
			}
		}
	}


	return -1;/* 没有LU可使用了 */
}

int Register::getpe(int time,int pre,int bindResOfLoad)
{
	for (int i = 0; i < this->TERnodesList.size();i++)
	{

		
		if (this->TERnodesList[i]->RegisterKind == 5 && this->TERnodesList[i]->time == time )
		{
			// cout<<"this->TERnodesList[i]->RegisterID="<<this->TERnodesList[i]->RegisterID<<"this->TERnodesList[i]->isOccupied="<<this->TERnodesList[i]->isOccupied<<endl;
			if(this->TERnodesList[i]->isOccupied == false)
			{
                int PEr = this->TERnodesList[i]->RegisterID;
                return PEr;
			}
		}
	}

 

	return -1;/* 没有PE可使用了 */
}




/* 获得与LU或结果寄存器相连的相连的SU寄存器 */
int Register::getLUorResultSU(int SRC)
{
	for(int i = 0; i < this->TERnodesList[SRC]->RegisterNeighbors.size(); i++)
	{
		
		int neighboreIndex = this->getIndex(this->TERnodesList[SRC]->RegisterNeighbors[i]);
		/* 这个邻居是SU,并且它没有被访问过 */
		if( this->TERnodesList[neighboreIndex]->isOccupied == false)
		{
			return this->TERnodesList[SRC]->RegisterNeighbors[i];
		}

	}
	/* 遍历完了，都没有 */
	return -1;
	
}










/* 获得寄存器的种类 */
int Register::getNodeKind(int nodeLabel)
{
    vector<Registernode*> nodesList = this->TERnodesList;
	for (auto node : nodesList)
	{
		if (node->RegisterID == nodeLabel)
		{
			return node->RegisterKind;
		}
	}
	return -1;

}

/* 获得与LU相连的PE寄存器 */
int Register::getLUR(int LU)
{
	for (int i = 0; i < this->TERnodesList.size();i++)
	{
		if (this->TERnodesList[i]->RegisterID == LU )
		{
			for(int j = 0; j<this->TERnodesList[i]->RegisterNeighbors.size(); j++)
			{
				
				if(this->getNodeKind(this->TERnodesList[i]->RegisterNeighbors[j]) == 0)
				return this->TERnodesList[i]->RegisterNeighbors[j];

			}
		}
	}
	
}
	






void Register::getResultRSet(int time,vector<int> &candidataR)
{
	
	for (auto node : this->TERnodesList)
	{
		if (node->RegisterKind == 5 && node->time == time && node->isOccupied == false)/* 结果寄存器 */
		{
			candidataR.push_back(node->RegisterID);
		}

	}
}

void Register::getLURSet(int time,vector<int> &candidataR)
{
	
	for (auto node : this->TERnodesList)
	{
		if (node->RegisterKind == 1 && node->time == time && node->isOccupied == false)
		{
			candidataR.push_back(node->RegisterID);
		}

	}
}





Register::Register(int Rnum, CGRA *C)
{   
    // cout<<"size "<<C->CGRAnodesList.size()<<endl;
    int count = 0;
	int colNum = C->ColNum;
	
	for(int i = 0; i < C->CGRAnodesList.size(); i++)
	{
		
		if(C->CGRAnodesList[i]->ElmKind == 1)/* LSU */
		{
			Registernode* node = new Registernode();
			node->RegisterID = count;
			node->PE = C->CGRAnodesList[i]->ElmID;
			node->RegisterKind = 1;
			node->isOccupied = false;

			for(int m = 0; m < C->CGRAnodesList[i]->ElmNeighbors.size(); m++)
			{
				int neiPE =  C->CGRAnodesList[i]->ElmNeighbors[m];
				int baseFU = calculateFU(neiPE,colNum,Rnum);
				for(int m = 0; m < Rnum - 1; m++)
				{
					node->RegisterNeighbors.push_back(baseFU + 2 + m);
				}

			}
			RnodesList.push_back(node); 
			count++;
		}

		else if(C->CGRAnodesList[i]->ElmKind == 0)/* PE */
		{
			/* 先FU */
			Registernode* node = new Registernode();
			node->RegisterID = count;
			node->PE = C->CGRAnodesList[i]->ElmID;
			node->RegisterKind = 4;/* fu */
			node->isOccupied = false;
			node->RegisterNeighbors.push_back(node->RegisterID + 1);/* 结果寄存器 */
			RnodesList.push_back(node); 
			count++;

			


			/* 再其他寄存器 */
			for(int j = 0; j < Rnum ; j++)
			{
				Registernode* node = new Registernode();
				node->RegisterID = count;
				node->PE = C->CGRAnodesList[i]->ElmID;
				//baseFU是FU
				int baseFU = calculateFU(node->PE,colNum,Rnum);
				if(node->RegisterID == baseFU + 1 )
				{
					node->RegisterKind = 5;/* 结果寄存器 */
				}
				else
				{
					node->RegisterKind = 0;/* 普通寄存器 */			
				}

				node->isOccupied = false;
				
				int neighNum = C->CGRAnodesList[i]->ElmNeighbors.size();
				
						
				if( node->RegisterID + 1 < baseFU + Rnum + 1)
				{
					node->RegisterNeighbors.push_back(node->RegisterID + 1 );/* 同一PE的下一个寄存器 */
				}
					
				for(int m = 0; m < neighNum; m++)
				{
					int neighbor = C->CGRAnodesList[i]->ElmNeighbors[m];
					if(C->CGRAnodesList[neighbor]->ElmKind == 0)/* 如果邻居是PE */
					{
						/* 该邻居的前四个寄存器是 邻居 */
						/* 该PE邻居 */
						int baseFU = calculateFU(neighbor,colNum,Rnum);
						
						for(int n = 1; n < Rnum; n++)
						{
							//不可以直接进邻居PE的结果寄存器
							node->RegisterNeighbors.push_back(baseFU + 1 + n);
						}
				
					}
					else if(C->CGRAnodesList[neighbor]->ElmKind == 1)/* 如果邻居是LSU */
					{
				
						int lsuColNum = 1;
						int TolRegInRow = (colNum-lsuColNum)* (Rnum+1) + lsuColNum;
						
						int LSURNo = neighbor/colNum * TolRegInRow;
						
						
						
						node->RegisterNeighbors.push_back(LSURNo);
						
				
					}
					

				}

				/* 邻居还有FU,结果寄存器的邻居也有FU */
				node->RegisterNeighbors.push_back(baseFU);
				
				RnodesList.push_back(node); 
				count++;
			}

			
			
		}
		else
		{


		}
		
	}
	RnodesNums = count;
	/* 寄存器点的输出 */
	// for(int x = 0; x < RnodesList.size(); x++)
	// {
	// 	cout<<RnodesList[x]->RegisterID<<" pe"<<RnodesList[x]->PE<<" kind"<<RnodesList[x]->RegisterKind<<endl;
	// 	cout<<" neghbor:";
	// 	for(int y = 0; y < RnodesList[x]->RegisterNeighbors.size(); y++)
	// 	{
	// 		cout<< RnodesList[x]->RegisterNeighbors[y]<<" ";

	// 	}
	// 	cout<<endl;
	// }


	/* 寄存器边 */
	count = 0;
	for(int x = 0; x < RnodesList.size(); x++)
	{	
		for(int y = 0; y < RnodesList[x]->RegisterNeighbors.size(); y++)
		{
			Registeredge* edge = new Registeredge();

			edge->edgeId = count;
			edge->srcReg = RnodesList[x]->RegisterID;
			edge->tgtReg = RnodesList[x]->RegisterNeighbors[y];
			RedgesList.push_back(edge); 
			count ++;

		}
	}
	RedgesNums = count;
	// for(int x = 0; x < RedgesList.size(); x++)
	// {
	// 	cout<<RedgesList[x]->edgeId<<" "<<RedgesList[x]->srcReg<<" "<<RedgesList[x]->tgtReg<<endl;
	// }
	
}

void Register::CreatTER(int II,int CGRAElmNum)
{
	int count = 0;/* 记录扩展后的寄存器的数量 */
	this->II = II;
	int CGRANum = CGRAElmNum;/* 平面CGRA的PE总数 */

	for(int i = 0; i < II; i++)/* 遍历所有的寄存器节点，平面的 */
	{
		for(int j = 0; j < this->RnodesNums; j++)/* 遍历所有的平面寄存器 */
		{
			Registernode* tenode = new Registernode();
			tenode->RegisterID = count;
			tenode->inPort = false;/* 默认输入端口还能用 */
			tenode->usedBypass = false;
    		tenode->usedTimeTrans = false;
			tenode->isOccupied = false;	
			
			tenode->time = i;
			
			if(this->RnodesList[j]->RegisterKind == 1)/* LSU */
			{
				tenode->PE = RnodesList[j]->PE + i * CGRANum;
				
				tenode->RegisterKind = 1;
					
				
				for(int n = 0; n < this->RnodesList[j]->RegisterNeighbors.size(); n++)
				{
					/* 当前层 */
					tenode->RegisterNeighbors.push_back(this->RnodesList[j]->RegisterNeighbors[n]% this->RnodesNums + (i%II) * this->RnodesNums % this->RnodesNums + (i%II) * this->RnodesNums);/*  */
				} 	
				
				/* 自己到自己的 *//* LU它可以连接到下一层的自己，代表下一层不动,LSU也可以旁路 */ 
				if(II != 1)
					tenode->RegisterNeighbors.push_back(tenode->RegisterID %this->RnodesNums + ((i+1)%II) * this->RnodesNums);
				
				
				for(int n = 0; n < this->RnodesList[j]->RegisterNeighbors.size(); n++)
				{
					tenode->RegisterNeighbors.push_back(this->RnodesList[j]->RegisterNeighbors[n]% this->RnodesNums + ((i+1)%II) * this->RnodesNums % this->RnodesNums + ((i+1)%II) * this->RnodesNums);/* 下一层 */
				} 			
			}

			
			
			else if(this->RnodesList[j]->RegisterKind == 4)/* 先FU */
			{
				tenode->PE = RnodesList[j]->PE + i * CGRANum;
				
				tenode->RegisterKind = 4;
				
				tenode->time = i;
				/* 邻居不能自己到自己，可以被旁路进来，不可以被旁路出去 */
				/* 只能是同层的结果寄存器 */			
				tenode->RegisterNeighbors.push_back(tenode->RegisterID + 1);/*  */


			}
			else if(this->RnodesList[j]->RegisterKind == 5)/* 再结果寄存器 */
			{
				tenode->PE = RnodesList[j]->PE + i * CGRANum;
				
				tenode->RegisterKind = 5;
				
				

				if(II != 1)
				{
					/* 旁路只能邻居间的旁路，:不能自己内部间的旁路。只能PE->PE ; */
					for(int n = 0; n < this->RnodesList[j]->RegisterNeighbors.size(); n++)
					{
						int neighborIndex = this->RnodesList[j]->RegisterNeighbors[n];
					
						/* 结果寄存器不能旁路到FU */
						if( this->RnodesList[neighborIndex]->RegisterKind != 4 && this->RnodesList[neighborIndex]->RegisterKind != 2 && this->RnodesList[neighborIndex]->RegisterKind != 1)/* 结果寄存器只能旁路到邻居寄存器 */
						// if( this->RnodesList[neighborIndex]->RegisterKind != 4 )/* 结果寄存器不能旁路到FU */
						{
							
							
							if((this->RnodesList[neighborIndex]->PE + i* CGRANum)!= tenode->PE )/* 不能旁路到下一个寄存器 */
							{
								tenode->RegisterNeighbors.push_back(this->RnodesList[j]->RegisterNeighbors[n]% this->RnodesNums + (i%II) * this->RnodesNums % this->RnodesNums + (i%II) * this->RnodesNums);/* 平面的同一层 */
								
							}
						}
						
					} 
					/* 最好不要让结果寄存器停留？因为当前的FU就不能使用了 */ 
					// if(II != 1)
					// /* 自己到自己 */
					// 	tenode->RegisterNeighbors.push_back(tenode->RegisterID % this->RnodesNums + ((i+1)%II) * this->RnodesNums % this->RnodesNums + ((i+1)%II) * this->RnodesNums);
					/* 下一层 */
					for(int n = 0; n < this->RnodesList[j]->RegisterNeighbors.size(); n++)
					{
						tenode->RegisterNeighbors.push_back(this->RnodesList[j]->RegisterNeighbors[n]% this->RnodesNums + ((i+1)%II) * this->RnodesNums % this->RnodesNums + ((i+1)%II) * this->RnodesNums);/* 下一层 */

					} 

				}
				else
				{
					for(int n = 0; n < this->RnodesList[j]->RegisterNeighbors.size(); n++)
					{
						int neighborIndex = this->getIndex(this->RnodesList[j]->RegisterNeighbors[n]);
						if(this->RnodesList[neighborIndex]->RegisterKind != 4)/* 当II=1的时候，不可能到同层的FU */
							tenode->RegisterNeighbors.push_back(this->RnodesList[j]->RegisterNeighbors[n]);
					}

				}


			}
			else if(this->RnodesList[j]->RegisterKind == 0)/* PE内的寄存器 */
			{
				
				tenode->PE = RnodesList[j]->PE + i * CGRANum;
				
				tenode->RegisterKind = 0;
					
				
				if(II != 1)
				{
					
					/* 旁路只能邻居间的旁路，TO DO:不能自己内部间的旁路。PE->PE 以及 到FU; */
					for(int n = 0; n < this->RnodesList[j]->RegisterNeighbors.size(); n++)
					{
						
						int neighborIndex = this->RnodesList[j]->RegisterNeighbors[n];
						
						
						
						
						
						/* 普通寄存器可以旁路到FU和邻居寄存器；不能旁路到FU好吧 */
						
						if( (this->RnodesList[neighborIndex]->PE + i* CGRANum)!= tenode->PE && this->RnodesList[neighborIndex]->RegisterKind != 2 && this->RnodesList[neighborIndex]->RegisterKind != 1)
						{
							if(this->RnodesList[neighborIndex]->RegisterKind != 4)/* 不能旁路到FU */
							{
								tenode->RegisterNeighbors.push_back(this->RnodesList[j]->RegisterNeighbors[n]% this->RnodesNums + (i%II) * this->RnodesNums % this->RnodesNums + (i%II) * this->RnodesNums);/* 平面的同一层 */
							}

						}
						
					} 
					if(II != 1)
					/* 自己到自己 */
						tenode->RegisterNeighbors.push_back(tenode->RegisterID % this->RnodesNums + ((i+1)%II) * this->RnodesNums % this->RnodesNums + ((i+1)%II) * this->RnodesNums);
					/* 下一层 */
					for(int n = 0; n < this->RnodesList[j]->RegisterNeighbors.size(); n++)
					{
						tenode->RegisterNeighbors.push_back(this->RnodesList[j]->RegisterNeighbors[n]% this->RnodesNums + ((i+1)%II) * this->RnodesNums % this->RnodesNums + ((i+1)%II) * this->RnodesNums);/* 下一层 */

					} 
				}
				else/* 如果II等于1 */
				{
					// cout<<"RegisterrID="<<this->RnodesList[j]->RegisterID<<endl;
					for(int n = 0; n < this->RnodesList[j]->RegisterNeighbors.size(); n++)
					{
						// cout<<"neighborIndex="<<this->RnodesList[j]->RegisterNeighbors[n]<<endl;
						tenode->RegisterNeighbors.push_back(this->RnodesList[j]->RegisterNeighbors[n]);
					}

				}
			}
			
			TERnodesList.push_back(tenode);
			count++;

			
		}
			
	}
	this->TERnodesNums = count;


	/* 扩展寄存器的输出 */
	// for(int x = 0; x < TERnodesList.size(); x++)
	// {
	// 	cout<<"ID:"<<TERnodesList[x]->RegisterID<<"   PE: "<<TERnodesList[x]->PE<<"   kind: "<<TERnodesList[x]->RegisterKind<<"   time: "<<TERnodesList[x]->time<<"   inPort: "<<TERnodesList[x]->inPort<<endl;
	// 	cout<<" neghbor:";
	// 	for(int y = 0; y < TERnodesList[x]->RegisterNeighbors.size(); y++)
	// 	{
	// 		cout<< TERnodesList[x]->RegisterNeighbors[y]<<" ";

	// 	}
	// 	cout<<endl;
	// }



	/* 寄存器边 */
	count = 0;
	for(int x = 0; x < TERnodesList.size(); x++)
	{	
		for(int y = 0; y < TERnodesList[x]->RegisterNeighbors.size(); y++)
		{
			Registeredge* edge = new Registeredge();

			edge->edgeId = count;
			edge->srcReg = TERnodesList[x]->RegisterID;
			edge->tgtReg = TERnodesList[x]->RegisterNeighbors[y];
			TERedgesList.push_back(edge); 
			count ++;

		}
	}
	this->TERedgesNums = count;
	// for(int x = 0; x < TERedgesList.size(); x++)
	// {
	// 	cout<<TERedgesList[x]->edgeId<<" "<<TERedgesList[x]->srcReg<<" "<<TERedgesList[x]->tgtReg<<endl;
	// }

}




