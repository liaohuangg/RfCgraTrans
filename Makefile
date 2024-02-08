user=9001
group=1000
RfCgraTrans=/workspace

build-docker: test-docker
	docker run -it -v $(shell pwd):/workspace RfCgraTrans20:latest /bin/bash \
	-c "make build"
	echo "RfCgraTrans has been installed successfully!"

test-docker:
	(cd docker; docker build --build-arg UID=$(user) --build-arg GID=$(group) . --tag RfCgraTrans20)

shell:
	docker run -it -v $(shell pwd):/workspace RfCgraTrans20:latest /bin/bash

build_:
	set -e # Abort if one of the commands fail
	# build LLVM
	# mkdir -p $(RfCgraTrans)/llvm/build
	# (cd $(RfCgraTrans)/llvm/build; \
	#  cmake ../llvm \
	#  -DLLVM_ENABLE_PROJECTS="llvm;clang;mlir" \
	#  -DLLVM_TARGETS_TO_BUILD="host" \
	#  -DLLVM_ENABLE_ASSERTIONS=ON \
	#  -DCMAKE_BUILD_TYPE=DEBUG \
	#  -DLLVM_BUILD_EXAMPLES=OFF \
	#  -DLLVM_ENABLE_RTTI=OFF \
	#  -DLLVM_INSTALL_UTILS=ON \
	#  -DCMAKE_C_COMPILER=clang-9 \
	#  -DCMAKE_CXX_COMPILER=clang++-9 \
	#  -G Ninja || exit 1; \
	#  ninja || exit 1; \
	#  ninja check-mlir || exit 1)

	# build RfCgraTrans
	mkdir -p $(RfCgraTrans)/build
	(cd $(RfCgraTrans)/build; \
	 cmake .. \
	 -DCMAKE_BUILD_TYPE=DEBUG \
	 -DMLIR_DIR=$(RfCgraTrans)/llvm/build/lib/cmake/mlir \
	 -DLLVM_DIR=$(RfCgraTrans)/llvm/build/lib/cmake/llvm \
	 -DLLVM_ENABLE_ASSERTIONS=ON \
	 -DCMAKE_C_COMPILER=clang-9 \
	 -DCMAKE_CXX_COMPILER=clang++-9 \
	 -DLLVM_EXTERNAL_LIT=$(RfCgraTrans)/llvm/build/bin/llvm-lit \
	 -G Ninja || exit 1; \
	 ninja -j4 || exit 1)
	(cd $(RfCgraTrans)/build; LD_LIBRARY_PATH=$(RfCgraTrans)/build/pluto/lib:$LD_LIBRARY_PATH ninja check-RfCgraTrans)

clean: clean_RfCgraTrans
	rm -rf $(RfCgraTrans)/llvm/build

clean_RfCgraTrans:
	rm -rf $(RfCgraTrans)/build
