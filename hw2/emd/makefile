nvcc = /usr/local/cuda-8.0/bin/nvcc
cudalib = /usr/local/cuda-8.0/lib64/
tensorflow = /home/fanjin/.local/lib/python3.5/site-packages/tensorflow/include
tensorflow_lib = /home/fanjin/.local/lib/python3.5/site-packages/tensorflow
LDFLAGS="-Wl,-rpath,/home/fanjin/.local/lib/python3.5/site-packages/tensorflow"

all: tf_emddistance_so.so 
.PHONY : all

tf_emddistance_so.so:  tf_emddistance.cpp
	g++ -std=c++11 tf_emddistance.cpp $(LDFLAGS) -o tf_emddistance_so.so -shared -fPIC -I $(tensorflow) -I $(tensorflow)/external/nsync/public -ltensorflow_framework -L $(tensorflow_lib) -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-invalid-partial-specialization 
