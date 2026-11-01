#include "shm_extension.h"
#include <cstring>
#include <chrono>
#include <cstdlib>
#include <limits>
#include <filesystem>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <future>
#include <sched.h>


namespace fs = std::filesystem;

static size_t page_align_size(size_t size) {
    long psz = sysconf(_SC_PAGESIZE);
    if (psz <= 0) {
        psz = 4096;
    }
    const size_t page_mask = static_cast<size_t>(psz) - 1;
    return (size + page_mask) & ~page_mask;
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error(std::string("CUDA Error in ") + #call + ": " + cudaGetErrorString(err)); \
        } \
    } while (0)

namespace {
void populate_source_mapping(void* ptr, size_t size) {
    // The producer's physical SHM pages exist, but this consumer's page tables
    // are empty. Otherwise cudaHostRegister faults them serially under driver
    // locks. Read one byte per page without copying or changing source data.
    if (size < 64 * 1024 * 1024) return;
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
        throw std::runtime_error("Cannot query SHM prefault CPU affinity");
    const size_t workers = std::min<size_t>(8, CPU_COUNT(&allowed));
    if (workers < 2) return;
    const long configured_page_size = sysconf(_SC_PAGESIZE);
    const size_t page_size = configured_page_size > 0 ? configured_page_size : 4096;
    const size_t pages = size / page_size;
    std::vector<std::future<void>> tasks;
    for (size_t worker = 0; worker < workers; ++worker) {
        tasks.push_back(std::async(std::launch::async, [=]() {
            const auto* source = static_cast<volatile unsigned char*>(ptr);
            for (size_t page = pages * worker / workers;
                 page < pages * (worker + 1) / workers; ++page) {
                // Volatile read prevents the compiler removing page touches.
                // Mutable KV/hidden states may be concurrently written by
                // the producer; the value is discarded and never cached.
                const unsigned char value = source[page * page_size];
                (void)value;
            }
        }));
    }
    for (auto& task : tasks) task.get();
}

struct Registration {
    uint64_t address;
    size_t size;
    size_t references;
    uint64_t last_release;
};
std::mutex cache_mutex;
std::unordered_map<std::string, Registration> registrations;
size_t cache_budget = 0;
size_t idle_bytes = 0;
uint64_t release_clock = 0;
size_t cache_hits = 0;

void detach(uint64_t address, size_t size) {
    void* ptr = reinterpret_cast<void*>(address);
    CUDA_CHECK(cudaHostUnregister(ptr));
    if (munmap(ptr, size) != 0)
        throw std::runtime_error(std::string("Failed to detach SHM: ") + strerror(errno));
}

void trim_cache() {
    while (idle_bytes > cache_budget) {
        auto oldest = registrations.end();
        for (auto it = registrations.begin(); it != registrations.end(); ++it) {
            if (!it->second.references && (oldest == registrations.end() ||
                    it->second.last_release < oldest->second.last_release)) oldest = it;
        }
        if (oldest == registrations.end()) break;
        detach(oldest->second.address, oldest->second.size);
        idle_bytes -= oldest->second.size;
        registrations.erase(oldest);
    }
}
} // namespace

void SetPinnedWeightCacheBudget(size_t bytes) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    cache_budget = bytes;
    trim_cache();
}

std::unordered_map<std::string, size_t> PinnedWeightCacheStats() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    size_t active = 0;
    for (const auto& item : registrations)
        if (item.second.references) active += item.second.size;
    return {{"budget_bytes", cache_budget}, {"idle_bytes", idle_bytes},
            {"active_bytes", active}, {"entries", registrations.size()},
            {"hits", cache_hits}};
}

std::string format_shm_name_from_path(const std::string& full_path_str) {
    if (full_path_str.empty()) {
        return "/";
    }

    fs::path p(full_path_str);

    fs::path last_part = p.filename();
    if (last_part.empty()) {
        p = p.parent_path();
        last_part = p.filename();
    }

    fs::path parent_part = p.parent_path().filename();
    std::string middle = parent_part.string() == "" ? "" : "-";
    std::string combined = parent_part.string() + middle + last_part.string();
    
    return "/" + combined;
}

std::unordered_map<int, uint64_t> GetPinnedSharedMemoryPtrs(
    const std::unordered_map<int, std::string>& name_map, 
    const std::unordered_map<int, size_t>& device_memory, bool cacheable) {
        
    std::unordered_map<int, std::string> shm_name_map;
    for (const auto& [device, name] : name_map) {
        std::string shm_name = format_shm_name_from_path(name);
        shm_name_map[device] = shm_name;
    }
    std::unordered_map<int, uint64_t> memory_ptrs;
    try {
    for (const auto& [device, size] : device_memory) {
        int shm_fd = shm_open(shm_name_map[device].c_str(), O_RDWR, 0666);
        if (shm_fd < 0) {
            throw std::runtime_error("Failed to open shared pinned memory " + shm_name_map[device]);
        }
        if (!size || size > std::numeric_limits<size_t>::max() - 4095) {
            close(shm_fd);
            throw std::invalid_argument("Invalid SHM mapping size");
        }
        size_t aligned_size = page_align_size(size);
        if (aligned_size == 0) {
            std::cerr << "Invalid zero aligned size for shared pinned memory "
                      << shm_name_map[device] << std::endl;
            close(shm_fd);
            throw std::invalid_argument("Shared pinned memory size must be positive");
        }
        struct stat info;
        if (fstat(shm_fd, &info) != 0 || info.st_size < size) {
            close(shm_fd);
            throw std::runtime_error("Shared pinned memory is smaller than requested mapping");
        }
        std::unique_lock<std::mutex> cache_lock(cache_mutex);
        int cuda_device = 0;
        CUDA_CHECK(cudaGetDevice(&cuda_device));
        // Holding a mapping keeps its unlinked inode alive. A replacement at
        // the same name cannot match this key. This retains a registration,
        // not contents: producer writes to KV/hidden mappings stay visible.
        const std::string key = std::to_string(info.st_dev) + ":" +
            std::to_string(info.st_ino) + ":" + std::to_string(info.st_size) + ":" +
            std::to_string(aligned_size) + ":" + std::to_string(cuda_device);
        auto existing = registrations.find(key);
        if (cacheable && existing != registrations.end()) {
            auto& entry = existing->second;
            close(shm_fd);
            memory_ptrs.emplace(device, entry.address);
            if (!entry.references) idle_bytes -= entry.size;
            ++entry.references;
            ++cache_hits;
            continue;
        }
        printf("Mapped shared pinned memory %s, size: %ld\n", shm_name_map[device].c_str(), aligned_size);
        const auto mapping_start = std::chrono::steady_clock::now();
        void* ptr =
            mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (ptr == MAP_FAILED) {
            std::cerr << "Failed to mmap shared pinned memory " << shm_name_map[device]
                      << ": " << strerror(errno) << std::endl;
            close(shm_fd);
            throw std::runtime_error("Failed to mmap shared pinned memory " + shm_name_map[device]);
        }
        close(shm_fd);
        const auto prefault_start = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point registration_start;
        try {
            populate_source_mapping(ptr, aligned_size);
            registration_start = std::chrono::steady_clock::now();
            CUDA_CHECK(cudaHostRegister(ptr, aligned_size, cudaHostRegisterDefault));
        } catch (...) {
            munmap(ptr, aligned_size);
            throw;
        }
        const auto registration_end = std::chrono::steady_clock::now();
        if (std::getenv("VLLM_PROFILE_SHM_ATTACH")) {
            printf("SHM source attachment %s: bytes=%zu mmap=%.6fs prefault=%.6fs register=%.6fs\n",
                   shm_name_map[device].c_str(), aligned_size,
                   std::chrono::duration<double>(prefault_start - mapping_start).count(),
                   std::chrono::duration<double>(registration_start - prefault_start).count(),
                   std::chrono::duration<double>(registration_end - registration_start).count());
        }
        const auto address = reinterpret_cast<uint64_t>(ptr);
        try {
            memory_ptrs.emplace(device, address);
            if (cacheable) registrations.emplace(key, Registration{address, aligned_size, 1, 0});
        } catch (...) {
            memory_ptrs.erase(device);
            detach(address, aligned_size);
            throw;
        }
    }
    } catch (...) {
        for (const auto& [rank, address] : memory_ptrs)
            ReleasePinnedSharedMemory(address, device_memory.at(rank));
        throw;
    }
    return memory_ptrs;
}

void ReleasePinnedSharedMemory(uint64_t mem_ptr, size_t size) {
    if (!mem_ptr) return;
    std::lock_guard<std::mutex> lock(cache_mutex);
    for (auto& item : registrations) {
        auto& entry = item.second;
        if (entry.address != mem_ptr) continue;
        if (entry.size != page_align_size(size) || !entry.references)
            throw std::invalid_argument("Invalid cached SHM release");
        if (!--entry.references) {
            idle_bytes += entry.size;
            entry.last_release = ++release_clock;
            trim_cache();
        }
        return;
    }
    detach(mem_ptr, page_align_size(size));
}

void DestroySharedMemory(const std::string& name, uint64_t mem_ptr, size_t size) {
    munmap(reinterpret_cast<void*>(mem_ptr), size);
    shm_unlink(name.c_str());
    printf("Destroyed shared memory %s\n", name.c_str());
}
