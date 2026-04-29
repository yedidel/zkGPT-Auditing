aux_source_directory(. gpt_src)
list(FILTER gpt_src EXCLUDE REGEX "main*")
# lasso-fork: explicitly enumerate all newly added source files so that
# re-running `make` alone (without re-running `cmake`) still picks them up
# after the directory listing changes. aux_source_directory only refreshes
# at CMake-configure time, which is a frequent source of "undefined
# reference" link errors during iterative development.
list(APPEND gpt_src
    lasso_logup.cpp
    lasso_grand_product.cpp
    lasso_memcheck.cpp
    lasso_surge.cpp
    lasso_unstructured.cpp
)
list(REMOVE_DUPLICATES gpt_src)
option(ENABLE_MY_ASSERT "Turn on MY_ASSERT checks" ON)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
add_library(gpt_lib ${gpt_src})


add_executable(demo_llm_run main_demo_llm.cpp)
target_link_libraries(demo_llm_run gpt_lib mcl)