// -*- c++ -*-

#include <faiss/impl/HugeMalloc.h>
#include <sys/mman.h>
#include <cassert>
#include <cstdlib>
#include <cstring>

namespace faiss {

void* malloc_huge(size_t size) {
  void* p = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  assert(p != MAP_FAILED);
  madvise(p, size, MADV_HUGEPAGE);
  memset(p, 0, size);
  return p;
}

}  // namespace faiss
