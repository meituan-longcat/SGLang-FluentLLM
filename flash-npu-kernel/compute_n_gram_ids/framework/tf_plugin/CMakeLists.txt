
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} plugin_srcs)
add_library(cust_tf_parsers SHARED ${plugin_srcs})
target_compile_definitions(cust_tf_parsers PRIVATE google=ascend_private)
if(ENABLE_CROSS_COMPILE)
    target_link_directories(cust_tf_parsers PRIVATE
                            ${CMAKE_COMPILE_COMPILER_LIBRARY}
                            ${CMAKE_COMPILE_RUNTIME_LIBRARY}
    )
endif()
target_link_libraries(cust_tf_parsers PRIVATE intf_pub graph)
install(TARGETS cust_tf_parsers
        LIBRARY DESTINATION packages/vendors/${vendor_name}/framework/tensorflow
)
