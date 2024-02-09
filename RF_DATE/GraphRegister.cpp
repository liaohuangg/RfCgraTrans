#include "GraphRegister.h"
#include "Register.h"


GraphNode::GraphNode()
{

}
GraphEdge::GraphEdge()
{

}

GraphRegister::~GraphRegister()
{
	for (auto node : GraphnodesList)
	{
		delete node;
	}
 	for (auto edge : GraphedgesList)
	{
		delete edge;
	}
}


bool GraphRegister::graphHasEdge(int begin, int end)
{
	bool flag = false;
	vector<GraphEdge*> edges = this->GraphedgesList;
	for (auto edge : edges)
	{
		if (edge->pre == begin && edge->pos == end)
		{
			flag = true;
			break;
		}
	}
	return flag;
}


int GraphRegister::getEdgeCost(int begin, int end)
{
	vector<GraphEdge*> edges = this->GraphedgesList;
	if(this->graphHasEdge(begin,end))
	{
		for (auto edge : edges)
		{
			if (edge->pre == begin && edge->pos == end)
			{
				return edge->value;
				
			}
		}

	}
	else
	{
		return INF;
	}
}




void GraphRegister::setEdgeValueFromPre(int pre,int value)
{
		
	for(int i = 0; i < this->GraphedgesList.size(); i++)
	{
		if(this->GraphedgesList[i]->pre == pre )
		{
			this->GraphedgesList[i]->value = value;
		}

	
	}
	


}


void GraphRegister::setEdgeValueToPos(int pos,int value)
{
		
	for(int i = 0; i < this->GraphedgesList.size(); i++)
	{
		if(this->GraphedgesList[i]->pos == pos)
		{
			this->GraphedgesList[i]->value = value;
		}

	
	}
	


}


/* to do:最后一层不必要要返回来 */
GraphRegister::GraphRegister(int CGRAElmNum, int length, Register *R,int SrcTrueTime,int II)
{

	int CGRANum = CGRAElmNum;/* 平面CGRA的元素总数(包含LSU) */
	int RnodesNums = R->RnodesNums;
	int count = 0;
	for(int i = 0; i < length; i++)/* length为层数 */
	{
		for(int j = 0; j < R->RnodesNums; j++)
		{
			GraphNode* node = new GraphNode();
			node->GNodeID = count;
			
			if(R->RnodesList[j]->RegisterKind == 1)/* LSU */
			{
				node->RegisterKind = 1;
				for(int n = 0; n < R->RnodesList[j]->RegisterNeighbors.size(); n++)
				{
					/* 当前层就是同层的寄存器 */
					node->GNodeNeighbors.push_back(R->RnodesList[j]->RegisterNeighbors[n]% RnodesNums + (i%length) * RnodesNums % RnodesNums + (i%length) * RnodesNums);/*  */
				} 	
				
				if(i != (length-1))/* 不是最后一层，就要连接下一层。如果是最后一层，就只需要平面的就行*/
				{
					node->GNodeNeighbors.push_back(node->GNodeID % R->RnodesNums + ((i+1)%length) * R->RnodesNums);/* 下一层的自己 */
					
					for(int n = 0; n < R->RnodesList[j]->RegisterNeighbors.size(); n++)/* 下一层的邻居 */
					{
						node->GNodeNeighbors.push_back(R->RnodesList[j]->RegisterNeighbors[n]% R->RnodesNums + ((i+1)%length) * R->RnodesNums % R->RnodesNums + ((i+1)%length) * R->RnodesNums);/* ��һ�� */
					} 

				}
				
			}

			
			else if(R->RnodesList[j]->RegisterKind == 4)/* 先FU */
			{	
				node->RegisterKind = 4;	
				/* 只能同层的结果寄存器 */
				node->GNodeNeighbors.push_back(node->GNodeID + 1);/*同一层的结果寄存器  */
			}

			else if(R->RnodesList[j]->RegisterKind == 5)/* 再结果寄存器 */
			{		
				node->RegisterKind = 5;
				/* 旁路只能邻居间的旁路，TO DO:不能自己内部间的旁路。PE->PE 以及 到FU; */
				for(int n = 0; n < R->RnodesList[j]->RegisterNeighbors.size(); n++)
				{
					
					int neighborIndex = R->getIndex(R->RnodesList[j]->RegisterNeighbors[n]);
					
					if( R->RnodesList[neighborIndex]->RegisterKind != 2 &&  R->RnodesList[neighborIndex]->RegisterKind != 4 &&  R->RnodesList[neighborIndex]->RegisterKind != 1)/* 旁路不能到lsu和FU */
					{
						if((R->RnodesList[neighborIndex]->PE + i* CGRANum)!= (R->RnodesList[j]->PE + i* CGRANum))
						{
							node->GNodeNeighbors.push_back(R->RnodesList[j]->RegisterNeighbors[n]% R->RnodesNums + (i%length) * R->RnodesNums % R->RnodesNums + (i%length) * R->RnodesNums);/* 平面的同一层 */

						}
					}
					
				} 

				if(i != (length-1))
				{
					// /* 自己到下一层自己 */
					// node->GNodeNeighbors.push_back(node->GNodeID % R->RnodesNums + ((i+1)%length) * R->RnodesNums % R->RnodesNums + ((i+1)%length) * R->RnodesNums);
					/* 下一层 */
					
					for(int n = 0; n < R->RnodesList[j]->RegisterNeighbors.size(); n++)
					{
						
						node->GNodeNeighbors.push_back(R->RnodesList[j]->RegisterNeighbors[n]% R->RnodesNums  + ((i+1)%length) * R->RnodesNums % R->RnodesNums + ((i+1)%length) * R->RnodesNums);/* 下一层 */
						
					} 
				}
			}
			else if(R->RnodesList[j]->RegisterKind == 0)/* PE内的寄存器 */
			{
				node->RegisterKind = 0;
				/* 旁路只能邻居间的旁路，TO DO:不能自己内部间的旁路。PE->PE 以及 到FU; */
				for(int n = 0; n < R->RnodesList[j]->RegisterNeighbors.size(); n++)
				{
					
					int neighborIndex = R->getIndex(R->RnodesList[j]->RegisterNeighbors[n]);
					/* 不能旁路到相同的pe内的寄存器和FU,还有LSU */
					if( ((R->RnodesList[neighborIndex]->PE + i* CGRANum) != (R->RnodesList[j]->PE + i* CGRANum)) && R->RnodesList[neighborIndex]->RegisterKind != 4 )
					{
						if(R->RnodesList[neighborIndex]->RegisterKind != 2 && R->RnodesList[neighborIndex]->RegisterKind != 1 )/* 不能旁路到SU */
						
						node->GNodeNeighbors.push_back(R->RnodesList[j]->RegisterNeighbors[n]% R->RnodesNums + (i%length) * R->RnodesNums % R->RnodesNums + (i%length) * R->RnodesNums);/* 平面的同一层 */
					}
					
				} 
				if(i != (length-1))
				{
					/* 自己到下一层自己 */
					node->GNodeNeighbors.push_back(node->GNodeID % R->RnodesNums + ((i+1)%length) * R->RnodesNums % R->RnodesNums + ((i+1)%length) * R->RnodesNums);
					
					/* 下一层 */
					for(int n = 0; n < R->RnodesList[j]->RegisterNeighbors.size(); n++)
					{
						
						node->GNodeNeighbors.push_back(R->RnodesList[j]->RegisterNeighbors[n]% R->RnodesNums + ((i+1)%length) * R->RnodesNums % R->RnodesNums + ((i+1)%length) * R->RnodesNums);/* 下一层 */

					} 
				}
				


			}
			GraphnodesList.push_back(node); 
			count++;
			
		}
			
	}
	GraphnodesNums = count;

	
	// for(int x = 0; x < GraphnodesList.size(); x++)
	// {
	// 	cout<<"ID:"<<GraphnodesList[x]->GNodeID<<endl;
	// 	cout<<" neghbor:";
	// 	for(int y = 0; y < GraphnodesList[x]->GNodeNeighbors.size(); y++)
	// 	{
	// 		cout<< GraphnodesList[x]->GNodeNeighbors[y]<<" ";

	// 	}
	// 	cout<<endl;
	// }
	

	

	/*  */
	count = 0;
	for(int x = 0; x < GraphnodesList.size(); x++)
	{	
		for(int y = 0; y < GraphnodesList[x]->GNodeNeighbors.size(); y++)
		{
			GraphEdge* edge = new GraphEdge();

			edge->GEdgeId = count;
			edge->pre = GraphnodesList[x]->GNodeID;
			edge->pos = GraphnodesList[x]->GNodeNeighbors[y];

			edge->value = 2;/* 都有一个初值 */
			GraphedgesList.push_back(edge); 
			count ++;

		}
	}
	GraphedgesNums = count;


	/* 为GraphRegister的边赋值 */
	// cout<<"R->RnodesNums="<<R->RnodesNums<<endl;
	for(int x = 0; x < GraphedgesList.size(); x++)
	{

		int preNode = GraphedgesList[x]->pre;
		int posNode = GraphedgesList[x]->pos;
		int preTime = preNode  / R->RnodesNums;
		int posTime = posNode  / R->RnodesNums;
		

		/* 从虚拟的点到真实的点 */
		int firstR   = ( preNode - (preTime* R->RnodesNums) )+ ( (SrcTrueTime + preTime)  * R->RnodesNums ) % (II * R->RnodesNums);
		int secondR  = ( posNode - (posTime* R->RnodesNums) )+ ( (SrcTrueTime + posTime)  * R->RnodesNums ) % (II * R->RnodesNums);
		// cout<<"preNode="<<preNode<<"   posNode="<<posNode<<"   firstR="<<firstR<<"   secondR="<<secondR<<endl;
		

		/* 先正常的设置，普通寄存器，结果寄存器，FU,LU,SU
		1.结果寄存器，普通寄存器，LU  ->普通寄存器
		2. ->fu
		2.		FU		->结果寄存器
		2.bank->load,2
		3.load->store,2
		4.load-pe,2
		5.store->bank,2
		6.pe->pe2(同一PE)，PE->PE(不同PE)3,pe->store
		 */
		
		if( R->TERnodesList[firstR]->RegisterKind == 5 && R->TERnodesList[secondR]->RegisterKind == 0 )/* 如果是结果寄存器到普通寄存器 */
		{


			if(R->TERnodesList[firstR]->PE % CGRANum != R->TERnodesList[secondR]->PE % CGRANum )/* 不同PE */
			{
				
				if(R->TERnodesList[secondR]->inPort == true)/* 末端的输入端口被占 */
				{
	
					GraphedgesList[x]->value = INF;
				}
				else
				{
					GraphedgesList[x]->value = 30;/* 传出去 */
				}

			}
			else/* 相同PE（不同层次） */
			{
				if(II != 1)/* 如果II！=1,自己到自己更好 */
				{
					/* II层下的 */
					if(firstR % R->RnodesNums == secondR % R->RnodesNums)
					{
						GraphedgesList[x]->value = 2;
					}
					else
					{
						GraphedgesList[x]->value = 5;
					}
					

				}
				else
				{
					GraphedgesList[x]->value = 5;
				}
			}
			if( R->TERnodesList[secondR]->usedTimeTrans == true)/* 立体传输已经占了 */
			{
				
				if(R->TERnodesList[firstR]->time!= R->TERnodesList[secondR]->time)/* 两个正好在不同的时间 */
				{
					GraphedgesList[x]->value = INF;
				}

			}
			if( R->TERnodesList[secondR]->usedBypass == true)/* 平面传输已经占了 */
			{
				
				if(R->TERnodesList[firstR]->time == R->TERnodesList[secondR]->time)/* 两个正好在不同的时间 */
				{
					GraphedgesList[x]->value = INF;
				}

			}


		}

		else if(R->TERnodesList[firstR]->RegisterKind == 0 && R->TERnodesList[secondR]->RegisterKind == 0 )/* 如果是普通寄存器到普通寄存器 */
		{
			
			if(R->TERnodesList[firstR]->PE % CGRANum  != R->TERnodesList[secondR]->PE % CGRANum  )/* 不同PE */
			{
				
				if(R->TERnodesList[secondR]->inPort == true)/* 末端的输入端口被占 */
				{
					
					GraphedgesList[x]->value = INF;
				}
				else
				{
					GraphedgesList[x]->value = 10;
				}

			}
			else/* 相同PE（不同层次） */
			{
				if(II != 1)/* 如果II！=1,自己到自己更好 */
				{
					/* II层下的 */
					if(firstR % R->RnodesNums == secondR % R->RnodesNums)
					{
						GraphedgesList[x]->value = 2;
					}
					else
					{
						GraphedgesList[x]->value = 5;
					}
					

				}
				else
				{
					GraphedgesList[x]->value = 5;
				}
			}

			if( R->TERnodesList[secondR]->usedTimeTrans == true)/* 立体传输已经占了 */
			{
				
				if(R->TERnodesList[firstR]->time!= R->TERnodesList[secondR]->time)/* 两个正好在不同的时间 */
				{
					GraphedgesList[x]->value = INF;
				}

			}
			if( R->TERnodesList[secondR]->usedBypass == true)/* 平面传输已经占了 */
			{
				
				if(R->TERnodesList[firstR]->time == R->TERnodesList[secondR]->time)/* 两个正好在不同的时间 */
				{
					GraphedgesList[x]->value = INF;
				}

			}




		}
		else if(R->TERnodesList[firstR]->RegisterKind == 1 && R->TERnodesList[secondR]->RegisterKind == 0)/* LD->普通寄存器 */
		{
			
			// cout<<"R->TERnodesList[secondR]->inPort="<<R->TERnodesList[secondR]->inPort<<endl;
			
			if(R->TERnodesList[secondR]->inPort == true)/* 末端的输入端口被占 */
			{
				
				GraphedgesList[x]->value = INF;
			}
			else{
				// cout<<"firstR="<<firstR<<"  "<<"secondR="<<secondR<<endl;
				GraphedgesList[x]->value = 4;
			}
			if( R->TERnodesList[secondR]->usedTimeTrans == true)/* 立体传输已经占了 */
			{
				
				if(R->TERnodesList[firstR]->time!= R->TERnodesList[secondR]->time)/* 两个正好在不同的时间 */
				{
					GraphedgesList[x]->value = INF;
				}

			}
			if( R->TERnodesList[secondR]->usedBypass == true)/* 平面传输已经占了 */
			{
				
				if(R->TERnodesList[firstR]->time == R->TERnodesList[secondR]->time)/* 两个正好在不同的时间 */
				{
					GraphedgesList[x]->value = INF;
				}

			}


			
		}

		else if(R->TERnodesList[firstR]->RegisterKind == 1 && R->TERnodesList[secondR]->RegisterKind == 1)/* LD->LD */
		{
			GraphedgesList[x]->value = 20;
		}
		
		
		else if(R->TERnodesList[firstR]->RegisterKind == 0 && R->TERnodesList[secondR]->RegisterKind == 1)/* 普通寄存器->LU */
		{

			GraphedgesList[x]->value = 50;
		}
		else if(R->TERnodesList[firstR]->RegisterKind == 5 && R->TERnodesList[secondR]->RegisterKind == 1)/* 结果寄存器->LU */
		{
			GraphedgesList[x]->value = 100;
		}


		else if(R->TERnodesList[firstR]->RegisterKind == 0 && R->TERnodesList[secondR]->RegisterKind == 4)/*普通寄存器到->FU */
		{
			if(R->TERnodesList[secondR]->inPort == true)/* 末端的输入端口被占 */
			{
				// cout<<"firstR="<<firstR<<"  "<<"secondR="<<secondR<<endl;
				GraphedgesList[x]->value = INF;
			}
			else
			{
				GraphedgesList[x]->value = 30;

			}

		}
	
		
		else if(R->TERnodesList[firstR]->RegisterKind == 5 && R->TERnodesList[secondR]->RegisterKind == 4)/* Result到FU*/
		{
			if(R->TERnodesList[secondR]->inPort == true)/* 末端的输入端口被占 */
			{
				// cout<<"firstR="<<firstR<<"  "<<"secondR="<<secondR<<endl;
				GraphedgesList[x]->value = INF;
			}
			else
			{
				GraphedgesList[x]->value = 20;

			}
		}

		else if(R->TERnodesList[firstR]->RegisterKind == 5 && R->TERnodesList[secondR]->RegisterKind == 4)/* Result到FU*/
		{
			if(R->TERnodesList[secondR]->inPort == true)/* 末端的输入端口被占 */
			{
				// cout<<"firstR="<<firstR<<"  "<<"secondR="<<secondR<<endl;
				GraphedgesList[x]->value = INF;
			}
			else
			{
				GraphedgesList[x]->value = 20;

			}
		}



		else
		{
			GraphedgesList[x]->value = 5;
		}

		/* 根据时间扩展的R实时调整权值 */

		// if(R->TERnodesList[firstR]->isOccupied == true)
		// {
		// 	// cout<<"firstR="<<firstR<<endl;
		// 	/* 将以edge->pre为源点的 边设置为 无穷大 */
		// 	// this->setEdgeValueFromPre(preNode,INF);
		// 	// /* 将以edge->pre为目标点的 边设置为 无穷大 */
		// 	// this->setEdgeValueToPos(preNode,INF);
		// }
		// if(R->TERnodesList[secondR]->isOccupied == true)
		// {
		// 	// cout<<"secondR="<<secondR<<endl;
		// 	/* 将以edge->pos为源点的 边设置为 无穷大 */
		// 	// this->setEdgeValueFromPre(posNode,INF);
		// 	// /* 将以edge->pos为目标点的 边设置为 无穷大 */
		// 	// this->setEdgeValueToPos(posNode,INF);
		// }
		// cout<<GraphedgesList[x]->GEdgeId<<" "<<GraphedgesList[x]->pre<<" "<<GraphedgesList[x]->pos<<" "<<GraphedgesList[x]->value<<endl;
	}

}




