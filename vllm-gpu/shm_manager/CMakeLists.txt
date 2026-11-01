cmake_minimum_required(VERSION 3.26)

# Set the project name
project(shm_manager LANGUAGES CXX)

# Set c++ 17
set(CMAKE_CXX_STANDARD 17)

set(FETCHCONTENT_QUIET OFF)

find_package(CUDAToolkit QUIET)

if (CUDAToolkit_FOUND)
  message(STATUS "CUDA found")
  set(CUDA_SUPPORTED_ARCHS "7.0;7.5;8.0;8.6;8.9;9.0")
  set(SHM_MANAGER_GPU_LANG "CUDA")
  enable_language(CUDA)
else()
  message(FATAL_ERROR "CUDA not found")
endif()


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
# Override the GPU architectures detected by cmake/torch and filter them by
# the supported versions for the current language.
# The final set of arches is stored in `SHM_MANAGER_GPU_ARCHES`.
#
override_gpu_arches(SHM_MANAGER_GPU_ARCHES
  ${SHM_MANAGER_GPU_LANG}
  "${${SHM_MANAGER_GPU_LANG}_SUPPORTED_ARCHS}")

#
# Define extension targets
#

#
# _C extension
#


#
# NOTE: The following (pure CXX) source files will be excluded from the input
# of the hipify tool. This is a temporary solution to avoid the recursive
# inclusion of the hipify tool (#206).
#
# NOTE: If you are making changes to this section or adding new source files,
# please make sure to update the list of excluded files here.
#

set(TENSOR_EXTENSION_SRC
    "csrc/model_manager.cpp"
    "csrc/unpack.cu"
    "csrc/tensor_writer.cpp"
    "csrc/aligned_buffer.cpp"
    "csrc/python_bindings.cpp"
    "csrc/shm_extension.cpp"
    "csrc/kvcache_manager.cpp"
    "csrc/hidden_states_manager.cpp"
)

define_gpu_extension_target(
  _C
  DESTINATION shm_manager
  LANGUAGE ${SHM_MANAGER_GPU_ARCHES}
  SOURCES ${TENSOR_EXTENSION_SRC}
  ARCHITECTURES ${SHM_MANAGER_GPU_ARCHES}
  LIBRARIES ${SHM_MANAGER_LIBRARIES}
  WITH_SOABI)


