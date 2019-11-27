
set(RVO_HEADERS
	MyRVO.h
	MyRVOSim.h
	Vector2.h)

set(RVO_SOURCES
	Definitions.h
	KdTree.cpp
	KdTree.h
	MyAgent.cpp
	MyAgent.h
	MyRVO.cpp
	MyRVOSim.cpp)


add_library(RVO SHARED ${RVO_HEADERS} ${RVO_SOURCES})

if(WIN32)
    set_target_properties(RVO PROPERTIES COMPILE_DEFINITIONS NOMINMAX)
endif()

install(FILES ${RVO_HEADERS} DESTINATION include)
install(TARGETS RVO DESTINATION lib)