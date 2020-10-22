/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/HugeMalloc.h>

#include "PerfEvent.hpp"

using idx_t = faiss::Index::idx_t;

int main() {
    static constexpr int d = 512;                           // dimension
    static constexpr int nb = 100000;                       // database size
    static constexpr int nq = 10000;                        // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float *xb = reinterpret_cast<float*>(faiss::malloc_huge(d * nb * sizeof(float)));
    float *xq = reinterpret_cast<float*>(faiss::malloc_huge(d * nq * sizeof(float)));

    int32_t *xb_int = reinterpret_cast<int32_t*>(faiss::malloc_huge(d * nb * sizeof(int32_t)));
    int32_t *xq_int = reinterpret_cast<int32_t*>(faiss::malloc_huge(d * nq * sizeof(int32_t)));

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++) {
            xb[d * i + j] = distrib(rng);
            xb_int[d * i + j] = static_cast<int32_t>(xb[d * i + j]);
        }
        xb[d * i] += i / 1000.;
        xb_int[d * i] += i / 1000;
    }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++) {
            xq[d * i + j] = distrib(rng);
            xq_int[d * i + j] = static_cast<int32_t>(xq[d * i + j]);
        }
        xq[d * i] += i / 1000.;
        xq_int[d * i] += i / 1000;
    }


    static constexpr int nlist = 100;
    static constexpr int k = 4;

    faiss::IndexFlatL2 quantizer(d);       // the other index
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    assert(!index.is_trained);
    index.train(nb, xb);
    assert(index.is_trained);
    index.add(nb, xb);
    index.nprobe = 10;

    {       // search xq
        idx_t *I = reinterpret_cast<idx_t*>(faiss::malloc_huge(k * nq * sizeof(idx_t)));
        float *D = reinterpret_cast<float*>(faiss::malloc_huge(k * nq * sizeof(float)));

        {
            PerfEvent e;
            e.startCounters();

            index.search_new(nq, xq, xq_int, k, D, I);

            e.stopCounters();
            e.printReport(std::cout, nq);
        }

        printf("I=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        munmap(I, k * nq * sizeof(idx_t));
        munmap(D, k * nq * sizeof(float));
    }



    munmap(xb, d * nb * sizeof(float));
    munmap(xq, d * nq * sizeof(float));

    return 0;
}
