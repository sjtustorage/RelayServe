#include <vector>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <string>
#include <future>
#include <queue>
#include <torch/torch.h>
#include <c10/core/ScalarType.h>


void DestroySharedMemory(const std::string& name, uint64_t mem_ptr, size_t size);
// Detach a consumer mapping; the CPU producer retains the SHM name.
void ReleasePinnedSharedMemory(uint64_t mem_ptr, size_t size);

std::unordered_map<int, uint64_t> GetPinnedSharedMemoryPtrs(
    const std::unordered_map<int, std::string>& name_map, 
    const std::unordered_map<int, size_t>& device_memory, bool cacheable = false);

// Only idle weight registrations count toward this retention budget.
void SetPinnedWeightCacheBudget(size_t bytes);
std::unordered_map<std::string, size_t> PinnedWeightCacheStats();
