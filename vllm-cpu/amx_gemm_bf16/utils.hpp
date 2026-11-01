#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <cstdint>
#include <cstring>  // for strerror
#include <cerrno>   // for errno
#include <fcntl.h>  // for open
#include <unistd.h> // for pread, pwrite, close
#include <sched.h>
#include <numa.h>       // libnuma
#include <system_error> // C++ 标准错误处理
#include <stdexcept>

#define OFFSET2D(x, y, ld) ((x) * (ld) + (y))
#define OFFSET3D(x, y, z, ld1, ld2) ((x) * (ld1) * (ld2) + (y) * (ld2) + (z))

#if !defined(likely)
#define likely(cond) __builtin_expect(cond, 1)
#define unlikely(cond) __builtin_expect(cond, 0)
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CEIL(x, y) (((x) + (y) - 1) / (y))
#define ROUNDUP(x, y) (CEIL(x, y) * (y))
#define ROUNDDOWN(x, y) ((x) / (y) * (y))

/**
 * @brief 将当前线程绑定到指定的 CPU 核心
 * @param cpu_id 逻辑核心 ID
 * @throws std::system_error 如果绑定失败
 */
inline void bind_thread_to_cpu(int cpu_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);

    // 使用 system_error 包装 errno，这是最标准的 C++ 系统错误抛出方式
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        throw std::system_error(errno, std::generic_category(), 
            "Failed to bind thread to CPU " + std::to_string(cpu_id));
    }
}

inline std::vector<int> parse_cpu_list_string(const std::string& cpu_list) {
    std::vector<int> cpus;
    std::stringstream ss(cpu_list);
    std::string token;

    while (std::getline(ss, token, ',')) {
        if (token.empty()) {
            continue;
        }

        size_t dash = token.find('-');
        if (dash == std::string::npos) {
            cpus.push_back(std::stoi(token));
            continue;
        }

        const int begin = std::stoi(token.substr(0, dash));
        const int end = std::stoi(token.substr(dash + 1));
        if (end < begin) {
            continue;
        }
        for (int cpu = begin; cpu <= end; ++cpu) {
            cpus.push_back(cpu);
        }
    }

    return cpus;
}

inline int get_thread_sibling_cpu(int cpu_id) {
    const std::string sibling_path =
        "/sys/devices/system/cpu/cpu" + std::to_string(cpu_id) + "/topology/thread_siblings_list";
    std::ifstream sibling_file(sibling_path);
    if (!sibling_file.is_open()) {
        return -1;
    }

    std::string sibling_list;
    std::getline(sibling_file, sibling_list);
    const std::vector<int> siblings = parse_cpu_list_string(sibling_list);
    for (int sibling_cpu : siblings) {
        if (sibling_cpu != cpu_id) {
            return sibling_cpu;
        }
    }

    return -1;
}

/**
 * @brief 检查并初始化 NUMA 环境
 * @param required_nodes 需要使用的 NUMA 节点数量
 * @throws std::runtime_error 如果 NUMA 不可用或节点数不足
 */
inline void init_numa(int required_nodes, std::vector<int> &core_list) {
    // 检查 NUMA 可用性
    if (numa_available() < 0) {
        throw std::runtime_error("NUMA support is not available on this system.");
    }

    int available_nodes = numa_max_node() + 1;

    if (required_nodes > available_nodes) {
        throw std::range_error(
            "Requested NUMA nodes (" + std::to_string(required_nodes) + 
            ") exceed system availability (" + std::to_string(available_nodes) + ")."
        );
    }

    core_list.clear();
    struct bitmask* mask = numa_allocate_cpumask();
    // 遍历 0 ... required_nodes - 1 NUMA Node, 获取每个节点的 CPU 列表
    for (int node_id = 0; node_id < required_nodes; ++node_id) {
        if (numa_node_to_cpus(node_id, mask) != 0) { // 获取该 node 下的所有 CPU 到 mask 中
            numa_free_cpumask(mask);
            throw std::runtime_error("Failed to get CPUs for NUMA node " + std::to_string(node_id));
        }

        for (unsigned int cpu_id = 0; cpu_id < mask->size; ++cpu_id) {
            if (numa_bitmask_isbitset(mask, cpu_id)) {
                core_list.push_back(cpu_id);
            }
        }
    }
    numa_free_cpumask(mask);
}


/**
 * @brief 将当前线程绑定到指定 NUMA 节点的所有核心上
 * @note 使用 libnuma 查询真实的 CPU 拓扑，而不是假设 ID 是连续的
 */
inline void bind_thread_to_numa_node(int node_id) {
    struct bitmask* mask = numa_allocate_cpumask();
    
    // 获取该 NUMA 节点实际包含哪些 CPU (解决核心 ID 不连续的问题)
    if (numa_node_to_cpus(node_id, mask) != 0) {
        numa_free_cpumask(mask);
        throw std::system_error(errno, std::generic_category(), 
            "Failed to query CPUs for NUMA node " + std::to_string(node_id));
    }

    // 将 bitmask 转换为 cpu_set_t
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (unsigned int i = 0; i < mask->size; i++) {
        if (numa_bitmask_isbitset(mask, i)) {
            CPU_SET(i, &cpuset);
        }
    }

    // 绑定
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0) {
        numa_free_cpumask(mask);
        throw std::system_error(errno, std::generic_category(), 
            "Failed to bind thread to NUMA node " + std::to_string(node_id));
    }

    numa_free_cpumask(mask);
}

inline std::string init_cpu_threads_env(const std::string& cpu_ids) {
    bitmask* omp_cpu_mask = numa_parse_cpustring_all(cpu_ids.c_str());
    TORCH_CHECK(omp_cpu_mask->size > 0);
    std::vector<int> omp_cpu_ids;
    omp_cpu_ids.reserve(omp_cpu_mask->size);
  
    constexpr int group_size = 8 * sizeof(*omp_cpu_mask->maskp);
  
    for (int offset = 0; offset < omp_cpu_mask->size; offset += group_size) {
      unsigned long group_mask = omp_cpu_mask->maskp[offset / group_size];
      int i = 0;
      while (group_mask) {
        if (group_mask & 1) {
          omp_cpu_ids.emplace_back(offset + i);
        }
        ++i;
        group_mask >>= 1;
      }
    }
  
    // Memory node binding
    if (numa_available() != -1) {
      std::set<int> node_ids;
      for (const auto& cpu_id : omp_cpu_ids) {
        int node_id = numa_node_of_cpu(cpu_id);
        if (node_id != -1) {
          node_ids.insert(node_id);
        }
      }
      // Concatenate all node_ids into a single comma-separated string
      if (!node_ids.empty()) {
        std::string node_ids_str;
        for (const int node_id : node_ids) {
          if (!node_ids_str.empty()) {
            node_ids_str += ",";
          }
          node_ids_str += std::to_string(node_id);
        }
  
        bitmask* mask = numa_parse_nodestring(node_ids_str.c_str());
        bitmask* src_mask = numa_get_membind();
        
        int pid = getpid();
        
        if (mask && src_mask) {
          // move all existing pages to the specified numa node.
          *(src_mask->maskp) = *(src_mask->maskp) ^ *(mask->maskp);
          int page_num = numa_migrate_pages(pid, src_mask, mask);
          if (page_num == -1) {
            TORCH_WARN("numa_migrate_pages failed. errno: " + 
                       std::to_string(errno));
          }
   
          // restrict memory allocation node.
          numa_set_membind(mask);
          numa_set_strict(1);
  
          numa_free_nodemask(mask);
          numa_free_nodemask(src_mask);
        } else {
          TORCH_WARN("numa_parse_nodestring or numa_get_membind failed. errno: " + 
                     std::to_string(errno));
        }
      }
    }
  
    // OMP threads binding
    omp_set_num_threads((int)omp_cpu_ids.size());
    torch::set_num_threads((int)omp_cpu_ids.size());
    TORCH_CHECK_EQ(omp_cpu_ids.size(), torch::get_num_threads());
    TORCH_CHECK_EQ(omp_cpu_ids.size(), omp_get_max_threads());

  
    std::vector<std::pair<int, int>> thread_core_mapping;
    thread_core_mapping.reserve(omp_cpu_ids.size());
    omp_lock_t writelock;
    omp_init_lock(&writelock);
  
    #pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < omp_cpu_ids.size(); ++i) {
      cpu_set_t mask;
      CPU_ZERO(&mask);
      CPU_SET(omp_cpu_ids[i], &mask);
      int ret = sched_setaffinity(0, sizeof(cpu_set_t), &mask);
      if (ret == -1) {
        TORCH_CHECK(false,
                    "sched_setaffinity failed. errno: " + std::to_string(errno));
      }
  
      omp_set_lock(&writelock);
      thread_core_mapping.emplace_back(gettid(), omp_cpu_ids[i]);
      omp_unset_lock(&writelock);
    }
  
    omp_destroy_lock(&writelock);
  
    numa_free_nodemask(omp_cpu_mask);
  
    std::stringstream ss;
    ss << "new OMP threads binding of Process " << getpid() << ":\n";
    std::sort(thread_core_mapping.begin(), thread_core_mapping.end(),
              [](auto&& a, auto&& b) { return a.second < b.second; });
    for (auto&& item : thread_core_mapping) {
      ss << "\t"
         << "OMP tid: " << item.first << ", core " << item.second << "\n";
    }
  
    return ss.str();
}


#include <type_traits>
#include <iomanip>
#include <sstream>
#include <algorithm>

// Matrix data output format
template <typename T>
void print_matrix(const std::string& name,
                  const T* data,
                  int rows,
                  int cols,
                  int ld,
                  int max_print_rows = 8,
                  int max_print_cols = 8,
                  bool hex = true) {
    int pr = std::min(rows, max_print_rows);
    int pc = std::min(cols, max_print_cols);

    // 将元素转为字符串（统一处理格式问题）
    auto format_elem = [&](T v) {
        std::ostringstream oss;

        if constexpr (std::is_floating_point_v<T>) {
            oss << std::fixed << std::setprecision(2) << v;
        } 
        else if constexpr (std::is_same_v<T, int8_t>) {
            if (hex)
                oss << "0x" << std::hex << static_cast<int>(v);
            else
                oss << static_cast<int>(v);
        }
        else if constexpr (std::is_same_v<T, uint8_t>) {
            if (hex)
                oss << "0x" << std::hex << static_cast<unsigned int>(v);
            else
                oss << static_cast<unsigned int>(v);
        }
        else if constexpr (std::is_integral_v<T>) {
            if (hex)
                oss << "0x" << std::hex << v;
            else
                oss << v;
        }
        else {
            oss << v;
        }

        return oss.str();
    };

    // 预扫描，计算列宽
    int width = 0;
    for (int i = 0; i < pr; ++i) {
        for (int j = 0; j < pc; ++j) {
            width = std::max(width, static_cast<int>(format_elem(data[i * ld + j]).size()));
        }
    }
    width += 1;  // 留一点间距

    // 打印矩阵
    std::cout << name << " = \n";
    for (int i = 0; i < pr; ++i) {
        std::cout << name << "[ ";
        for (int j = 0; j < pc; ++j) {
            std::cout << std::setw(width) << std::right << format_elem(data[i * ld + j]);
        }
        std::cout << " ]\n";
    }

    // 截断提示
    if (rows > pr || cols > pc) {
        std::cout << name << " (showing "
                  << pr << "x" << pc
                  << " of " << rows << "x" << cols << ")\n";
    }
}

