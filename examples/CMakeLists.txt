add_executable(example example.cc)
target_link_libraries(example
    xfeat
    ${THIRD_PARTY_LIBS}
)
set_property(TARGET example PROPERTY CXX_STANDARD 17)

add_executable(realtime_demo realtime_demo.cc)
target_link_libraries(realtime_demo
    xfeat
    ${THIRD_PARTY_LIBS}
)
set_property(TARGET realtime_demo PROPERTY CXX_STANDARD 17)