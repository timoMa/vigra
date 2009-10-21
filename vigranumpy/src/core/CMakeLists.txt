INCLUDE_DIRECTORIES(${VIGRANUMPY_INCLUDE_DIR})

IF (CMAKE_GENERATOR MATCHES "Visual Studio")
  ADD_DEFINITIONS(-DVIGRA_DLL)
ENDIF (CMAKE_GENERATOR MATCHES "Visual Studio")

IF(HDF5_FOUND)
    ADD_DEFINITIONS(-DHasHDF5 -D_HDF5USEDLL_ -DHDF5CPP_USEDLL)
    INCLUDE_DIRECTORIES(${HDF5_INCLUDE_DIR})
ENDIF(HDF5_FOUND)

ADD_LIBRARY(vigranumpycmodule SHARED
    vigranumpycmodule.cxx
    vigranumpyanalysis.cxx
    vigranumpyimpex.cxx
    vigranumpykernel.cxx
    vigranumpyconverters.cxx
    vigranumpymultianalysis.cxx
    vigranumpyconvolution.cxx
    vigranumpymulticonvolution.cxx
    vigranumpyfilters.cxx
    vigranumpyrf.cxx)

IF(PYTHON_PLATFORM MATCHES "^win32$")
    SET_TARGET_PROPERTIES(vigranumpycmodule PROPERTIES PREFIX "" SUFFIX  ".pyd")
    INSTALL(TARGETS vigranumpycmodule
            RUNTIME DESTINATION ${VIGRA_NUMPY_INSTALL_DIR})
ELSE()
    SET_TARGET_PROPERTIES(vigranumpycmodule PROPERTIES PREFIX "")
    INSTALL(TARGETS vigranumpycmodule
            LIBRARY DESTINATION ${VIGRA_NUMPY_INSTALL_DIR}/vigra)
ENDIF()

TARGET_LINK_LIBRARIES(vigranumpycmodule ${VIGRANUMPY_LIBRARIES} vigraimpex)
ADD_DEPENDENCIES(vigranumpy vigranumpycmodule)

IF(HDF5_FOUND)
    TARGET_LINK_LIBRARIES(vigranumpycmodule ${HDF5_LIBRARIES})
ENDIF(HDF5_FOUND)