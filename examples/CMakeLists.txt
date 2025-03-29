find_package(Threads REQUIRED)

add_executable(match match.cc)
target_link_libraries(match
    xfeat
    ${THIRD_PARTY_LIBS}
)
set_property(TARGET match PROPERTY CXX_STANDARD 17)

add_executable(realtime_demo realtime_demo.cc)
target_link_libraries(realtime_demo
    xfeat
    ${THIRD_PARTY_LIBS}
    Threads::Threads
)
set_property(TARGET realtime_demo PROPERTY CXX_STANDARD 17)


add_executable(realsense_demo realsense_demo.cc)
target_link_libraries(realsense_demo
    xfeat
    realsense
    ${THIRD_PARTY_LIBS}
    Threads::Threads
)
set_property(TARGET realsense_demo PROPERTY CXX_STANDARD 17)



