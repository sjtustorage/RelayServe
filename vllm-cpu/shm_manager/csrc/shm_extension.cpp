#include <string>
#include <cstddef>
#include <chrono>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

#define PAGE_SIZE 4096

void* GetSharedMemoryPtr(const std::string& name, size_t size, bool create) {
    std::cout << name << " " << size << ".\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    int oflag = create ? O_CREAT | O_RDWR : O_RDWR;
    int shm_fd = shm_open(name.c_str(), oflag, 0666);
    if (shm_fd < 0) {
        std::cerr << "Failed to open shared memory " << name << ".\n";
        return nullptr;
    }
    // Round up to a multiple of PAGE_SIZE. Must mask with ~(PAGE_SIZE - 1), not
    // ~PAGE_SIZE: the latter only clears bit 12 and leaves the low 12 bits
    // wrong, so e.g. ftruncate can become (size + 4095) when size is already
    // 4K-aligned — matching off-by-4095 vs the intended logical size.
    const size_t page_mask = static_cast<size_t>(PAGE_SIZE) - 1;
    size_t aligned_size = (size + page_mask) & ~page_mask;
    if (ftruncate(shm_fd, aligned_size) != 0) {
        shm_unlink(name.c_str());
        std::cerr << "Failed to truncate shared memory " << name << ".\n";
        return nullptr;
    }
    void* mem_ptr = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    char* mem_ptr_char = static_cast<char*>(mem_ptr);
    for (int i = 0; i < aligned_size; i += PAGE_SIZE) {
        memset(mem_ptr_char + i, 0, PAGE_SIZE);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken to create shared memory '" 
        << name << "': " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() << "ms" << std::endl;
    return mem_ptr;
}

void DestroySharedMemory(const std::string& name, void* mem_ptr, size_t size) {
    munmap(mem_ptr, size);
    shm_unlink(name.c_str());
    std::cout << "Shared memory '" << name << "' is destroyed" << std::endl;
}