add_executable(example example.cc)
target_link_libraries(example
    xfeat
    ${THIRD_PARTY_LIBS}
)
set_property(TARGET example PROPERTY CXX_STANDARD 17)