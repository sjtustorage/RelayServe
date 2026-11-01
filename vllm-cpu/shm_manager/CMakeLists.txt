cmake_minimum_required(VERSION 3.26)

# Set the project name
project(shm_manager LANGUAGES CXX)

# Set c++ 17
set(CMAKE_CXX_STANDARD 17)

set(FETCHCONTENT_QUIET OFF)

# Adapted from https://github.com/vllm-project/vllm/blob/a1242324c99ff8b1e29981006dfb504da198c7c3/CMakeLists.txt
include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)

#
# Supported python versions.  These versions will be searched in order, the
# first match will be selected.  These should be kept in sync with setup.py.
#
set(PYTHON_SUPPORTED_VERSIONS "3.9" "3.10" "3.11" "3.12")

#
# Try to find python package with an executable that exactly matches
# `SHM_MANAGER_PYTHON_EXECUTABLE` and is one of the supported versions.
#
if (SHM_MANAGER_PYTHON_EXECUTABLE)
  find_python_from_executable(${SHM_MANAGER_PYTHON_EXECUTABLE} "${PYTHON_SUPPORTED_VERSIONS}")
else()
  message(FATAL_ERROR
    "Please set SHM_MANAGER_PYTHON_EXECUTABLE to the path of the desired python version"
    " before running cmake configure.")
endif()


#
# Update cmake's `CMAKE_PREFIX_PATH` with torch location.
#
append_cmake_prefix_path("torch" "torch.utils.cmake_prefix_path")

#
# Import torch cmake configuration.
# Torch also imports CUDA (and partially HIP) languages with some customizations,
# so there is no need to do this explicitly with check_language/enable_language,
# etc.
#
find_package(Torch REQUIRED)
find_library(numa_LIBRARY numa REQUIRED)

#
# Normally `torch.utils.cpp_extension.CUDAExtension` would add
# `libtorch_python.so` for linking against an extension. Torch's cmake
# configuration does not include this library (presumably since the cmake
# config is used for standalone C++ binaries that link against torch).
# The `libtorch_python.so` library defines some of the glue code between
# torch/python via pybind and is required by SHM_MANAGER extensions for this
# reason. So, add it by manually with `find_library` using torch's
# installed library path.
#
find_library(torch_python_LIBRARY torch_python PATHS
  "${TORCH_INSTALL_PREFIX}/lib")


#
# Define extension targets
#

#
# _C extension
#

set(TENSOR_EXTENSION_SRC
    "csrc/tensor_writer.cpp"
    "csrc/aligned_buffer.cpp"
    "csrc/model_manager.cpp"
    "csrc/kvcache_manager.cpp"
    "csrc/hidden_states_manager.cpp"
    "csrc/shm_extension.cpp"
    "csrc/python_bindings.cpp"
)

define_gpu_extension_target(
  _C
  DESTINATION shm_manager
  LANGUAGE CXX
  SOURCES ${TENSOR_EXTENSION_SRC}
  LIBRARIES ${SHM_MANAGER_LIBRARIES} ${numa_LIBRARY}
  WITH_SOABI)


