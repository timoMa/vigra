INCLUDE_DIRECTORIES(${VIGRANUMPY_INCLUDE_DIRS} ${FFTW3_INCLUDE_DIR})

VIGRA_CONFIGURE_THREADING()

VIGRA_ADD_NUMPY_MODULE(fourier 
  SOURCES
    fourier.cxx
  LIBRARIES   
    ${FFTW3_LIBRARIES}
    ${FFTW3F_LIBRARIES}
  VIGRANUMPY)   
