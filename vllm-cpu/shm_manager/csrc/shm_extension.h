#include <string>
#include <cstddef>

void DestroySharedMemory(const std::string& name, void* mem_ptr, size_t size);

void* GetSharedMemoryPtr(const std::string& name, size_t size, bool create);