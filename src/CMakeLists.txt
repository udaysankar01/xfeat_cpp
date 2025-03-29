add_library(xfeat
    XFModel.cc
    XFeat.cc
    InterpolateSparse2d.cc
)

set_property(TARGET xfeat PROPERTY CXX_STANDARD 17)

target_link_libraries(
    xfeat
    ${THIRD_PARTY_LIBS}
)


add_library(realsense SHARED 
    realsense.cc
)
target_link_libraries(realsense PUBLIC 
    realsense2 
    ${OpenCV_LIBS}
)
target_include_directories(realsense PUBLIC 
    include
)
