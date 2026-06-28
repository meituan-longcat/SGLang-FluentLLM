# find python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Found Python3: ${Python3_EXECUTABLE} (found version ${Python3_VERSION})")
message(STATUS "Python3 include dir: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
