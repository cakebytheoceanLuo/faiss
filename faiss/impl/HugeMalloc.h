// -*- c++ -*-

#pragma once

#include <sys/mman.h>

namespace faiss {

// $ man mmap
// > [...] If addr is NULL, then the kernel chooses the (page-aligned) address
//       at which to create the mapping; this is the most portable method of
//       creating a new mapping. [...]
//   "page-aligned" implies 64-Bytes-aligned
void* malloc_huge(size_t size) {
    void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    assert(p != MAP_FAILED);
    madvise(p, size, MADV_HUGEPAGE);
    memset(p, 0, size);
    return p;
}

}  // namespace faiss
