// -*- c++ -*-

#pragma once

#include <sys/mman.h>

namespace faiss {

// $ man mmap
// > [...] If addr is NULL, then the kernel chooses the (page-aligned) address
//       at which to create the mapping; this is the most portable method of
//       creating a new mapping. [...]
//   "page-aligned" implies 64-Bytes-aligned
void* malloc_huge(size_t size);

}  // namespace faiss
