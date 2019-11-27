_core.so: core.cpp core.h core_wrap.cxx
		g++ -shared -fPIC -I/usr/include/python2.7 -lpython2.7 -o _core.so core.cpp core_wrap.cxx
core_wrap.cxx: core.i
		swig -c++ -python core.i
clean: 
		rm _core.so core_wrap.cxx core.py