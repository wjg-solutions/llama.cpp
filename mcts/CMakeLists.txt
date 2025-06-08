set(TARGET mcts)

set(TARGET_SRCS
    mcts.cpp
    mcts.h
)

add_library(${TARGET} STATIC ${TARGET_SRCS})

target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})

# Link with llama and common libraries
target_link_libraries(${TARGET} PRIVATE llama common)

target_compile_features(${TARGET} PRIVATE cxx_std_17)