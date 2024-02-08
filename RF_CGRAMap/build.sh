cmake .
make
make install
echo "cp include file"
cp -rf ./include ../build/RF_CGRAMap/.
echo "cp lib"
cp -rf ./lib ../build/RF_CGRAMap/.