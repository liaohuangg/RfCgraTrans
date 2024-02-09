#include "CGRA.h"

CGRAnode::CGRAnode()
{

}

CGRA::~CGRA()
{
	for (auto node : CGRAnodesList)
	{
		delete node;
	}
}


CGRA::CGRA(int CGRARow, int CGRAColumn)
{   
	int column = CGRAColumn + 1;
	ColNum = column;
    ElmNum = CGRARow * column;//PEA的总数，包括LSU单元	
	
    int i, j, k;           
    for( j = 0; j < CGRARow; j++)
	{
      	for(i = 0; i < column;i++)
		{	
           	CGRAnode* node = new CGRAnode();
			int ElmCurrent = i + j * column;
			node->ElmID = ElmCurrent;/* ElmID从0开始编号 */
			node->inportNum = 5;
			node->outportNum  = 5;
			int ElmN = i + ((j - 1 + CGRARow) % CGRARow ) * column;
			int ElmS = i + ((j + 1) % CGRARow) * column;
			int ElmW = ((i - 1) + column) % column + j * column;
			int ElmE = (i + 1) % column + j * column;

			
			if(i == 0)//LSU
			{
        		node->ElmKind = 1;
				for(int m = 0; m < CGRAColumn; m++)
				{
					node->ElmNeighbors.push_back(ElmCurrent + 1 + m); 
				}

	
			}

		
			else//PE
			{
        		node->ElmKind = 0;
				
				if(j == 0 || j == CGRARow-1)/* 第一行和最后一行 */
				{
					if(i == 1)/* PE的第一列 */
					{
						node->ElmNeighbors.push_back(ElmE);  

						
					}
					else if(i == column - 1)//PE最后一列
					{

						node->ElmNeighbors.push_back(ElmW);  
                        
					}
					else//PE的中间列
					{
						node->ElmNeighbors.push_back(ElmW); 
						node->ElmNeighbors.push_back(ElmE); 
					}


					if(j == 0)
					{
						node->ElmNeighbors.push_back(ElmS);

					}
					else if(j == CGRARow-1)
					{
						node->ElmNeighbors.push_back(ElmN);
					}
					 
				}

				else/* 中间行 */
				{
					node->ElmNeighbors.push_back(ElmN); 
					node->ElmNeighbors.push_back(ElmS); 
					if(i == 1)/* PE的第一列 */
					{
						node->ElmNeighbors.push_back(ElmE); 
						

					}
					else if(i == column - 1)/* PE的最后一列 */
					{
						node->ElmNeighbors.push_back(ElmW);  
                       

					}
					else//PE的中间列
					{
						node->ElmNeighbors.push_back(ElmW);  
						node->ElmNeighbors.push_back(ElmE); 
						
					}
				
				}
				node->ElmNeighbors.push_back(j * column); 


			}

			CGRAnodesList.push_back(node);  
		}
	}
	// cout<<"ElmNum"<<ElmNum<<endl;
	// for(int Z = 0; Z < ElmNum;Z++)
	// {
	// 	cout<<CGRAnodesList[Z]->ElmID<<endl;
	// 	int num = CGRAnodesList[Z]->ElmNeighbors.size();
	// 	cout<<"neighbor:";
	// 	for(int y = 0; y<num;y++)
	// 	{
	// 		cout<< CGRAnodesList[Z]->ElmNeighbors[y]<<" ";

	// 	}
	// 	cout<<endl;

	// }


	
}

