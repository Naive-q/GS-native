add_executable(train train.cpp)
target_link_libraries(train PRIVATE gs-native)

add_executable(render render.cpp)
target_link_libraries(render PRIVATE gs-native)

if(MSVC)
    target_compile_options(train PRIVATE /utf-8)
    target_compile_options(render PRIVATE /utf-8)
endif()
if(WIN32)
    foreach(target train render)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS)
    endforeach()
endif()

set_target_properties(train render PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS train render
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(WIN32)
    find_file(cudart_runtime_dll
        NAMES "cudart64_${CUDAToolkit_VERSION_MAJOR}.dll" "cudart64_${CUDAToolkit_VERSION_MAJOR}0.dll"
        PATHS "${CUDAToolkit_BIN_DIR}"
        NO_DEFAULT_PATH NO_CACHE REQUIRED)
    install(FILES $<TARGET_RUNTIME_DLLS:train> "${cudart_runtime_dll}"
        DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES "${CUDAToolkit_BIN_DIR}/../EULA.txt"
        DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/gs-native/cuda OPTIONAL)
    if(EXISTS "${OpenCV_INSTALL_PATH}/LICENSE")
        install(FILES "${OpenCV_INSTALL_PATH}/LICENSE"
            DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/gs-native/opencv)
    endif()
    if(EXISTS "${OpenCV_INSTALL_PATH}/etc/licenses")
        install(DIRECTORY "${OpenCV_INSTALL_PATH}/etc/licenses/"
            DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/gs-native/opencv/components)
    endif()
endif()
