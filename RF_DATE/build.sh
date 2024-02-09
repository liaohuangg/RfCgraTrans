FLAG=-DEXE
export LD_LIBRARY_PATH=/usr/local/lib
g++ -o rf_tcad $FLAG -g  CGRA.cpp    main.cpp   Register.cpp  DFG.cpp tool.cpp GraphRegister.cpp  Path.cpp\
  -lm -std=c++11 -lgflags  -lglog -lpthread 

  