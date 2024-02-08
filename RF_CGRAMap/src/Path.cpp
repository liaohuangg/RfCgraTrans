#include "Path.h"

Path::Path()
{

}
AllPath::AllPath()
{

}
Dijk::Dijk()
{

}

AllPath::~AllPath()
{
	for (auto node : PathsList)
	{
		delete node;
	}
}