// CS 179 GPU Programming
// Problem Set 2
// Jalen Green

// Q1.1
latency of arithmetic instruction: ~10ns
# instructions issuable in 1 clock cycle: 8
a GK110 can start instructions in up to 4 warps each clock, and up to 2 
instructions in each warp

GPU clock cycle = 1ns

Thus we have 10ns / (1ns/cycle) = 10 cycles of latency. Since the instructions
issuable per clock cycle is 8, we need 8 * 10 = 80 arithmetic instructions to
hide latency.


// Q1.2.a
int idx = threadIdx.y + blockSize.y * threadIdx.x;
Since threadIdx.y is the same for all threads in a warp, and blockSize.y = 32,
all of the indexes will be equivalent (mod 32), therefore there is no warp
divergence; all threads take the same branch of the if statement.


// Q1.2.b
This code diverges because threads in a warp take difference branches.
For example, at threadIdx.x == 0, the statement inside the for loop is never
executed, whereas at threadIdx.x == 1, the statement is executed. So some
threads will be idle when other threads in the same warp are doing work.

The threads that are idle are finished, however, so no thread is idly waiting
in order to execute more instructions. 


// Q1.3.a
data[threadIdx.x + blockSize.x * threadIdx.y] is accessed.
blockSize.x = 32, threadIdx.y is the same for all threads, call it Y.
data[threadIdx.x + 32 * Y]

The first thread will be 128-byte-aligned since 
(0 + 32 * Y) * sizeof(float) = 128 * Y = 0 (mod 128).
Increasing threadIdx.x will add 4 to the address, thus all memory accesses of
the 32 threads in the warp will be in the same 128 byte cache line.
This is write coalesced.


// Q1.3.b
Incrementing threadIdx.x will change the index accessed by 32 (128 bytes).
Thus each thread in the warp will hit a new cache line.
This is non-coalesced, it will hit 32 cache lines.


// Q1.3.c
Each index is one after the index of the previous thread, but the first access
is not 128-byte-aligned, so 2 cache lines are written to.
This is not write coalesced.


// Q1.4.a
accessing 
    output[i + 32 * j]
    lhs   [i + 32 * k]
    rhs   [k + 128 * j]
    lhs   [i + 32 * (k + 1)]
    rhs   [(k + 1) + 128 * j]

j is the threadIdx.y, stays same within warp, call Y
k is also the same among threads in a warp, call K

accessing 
    output[[0..31] + 32 * Y]       // all different (mod 32)
    lhs   [[0..31] + 32 * K]       // all different (mod 32)
    rhs   [K + 128 * Y]            // all accessing the same memory
    lhs   [[0..31] + 32 * (K + 1)] // all different (mod 32)
    rhs   [(K + 1) + 128 * Y]      // all accessing the same memory

This code does not have bank conflicts.


// Q1.4.b

Original:

output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];

Expanded:

 1: l1 = lhs[i + 32 * k];
 2: r1 = rhs[k + 128 * j];
 3: o1 = output[i + 32 * j];
 4: o1 += l1 * r1;
 5: output[i + 32 * j] = o1;

 6: l2 = lhs[i + 32 * (k + 1)];
 7: r2 = rhs[(k + 1) + 128 * j];
 8: o2 = output[i + 32 * j];
 9: o2 += l2 * r2;
10: output[i + 32 * j] = o2;


//Q1.4.c
line | dependencies (lines)
4    | 1,2,3
5    | 4
8    | 5
9    | 6,7,8
10   | 9

//Q1.4.d
int i = threadIdx.x;
int j = threadIdx.y;
ac1 = 0;
ac2 = 0;
for (int k = 0; k < 128; k += 2) {
    int v1 = k + 128 * j;
    int v2 = i + 32 * k;
    r1 = rhs[v1];
    r2 = rhs[v1 + 1];
    l1 = lhs[v2];
    l2 = lhs[v2 + 32];
    ac1 += l1 * r1;
    ac2 += l2 * r2;
}
output[i + 32 * j] += ac1 + ac2;


//Q1.4.e
We could change the loop such that we process more values of k at a time and
have more accumulators to have better instruction-level parallelism (do some
better reduction).


//Q2
Output:
[jpgreen@mx:~/cs179/ps2/cs179_set2]> ./transpose 4096 all
Size 4096 naive CPU: 253.455139 ms
Size 4096 GPU memcpy: 0.642752 ms
Size 4096 naive GPU: 1.866816 ms
Size 4096 shmem GPU: 0.688672 ms
Size 4096 optimal GPU: 0.578848 ms

Strategy:
Read from global memory coalesced.
Transpose the matrix as we write it to shared memory.
Add a row in the shmem array (so it is 65X64) in order to avoid bank conflicts
when writing to shmem.
Sync threads.
Read from shmem (not the same elements the thread wrote) 
and write to global memory coalesced.

Optimizing:
Unroll loop, switch order of instructions to increase ILP, do index math before
__syncthreads().


//BONUS
1) The for loop requires less writes to a and less reads from a.
   The vec_add way has more IO, and thus is slower.

   The for loop does not read from a, and it writes to each element of a only
   once; this totals 1 write, 0 reads for each element in a.
   The vec_add way writes to each element in a once, then reads from each
   element in a, then writes to each element in a; this totals 2 writes, 1 read
   for each element in a.

2) The for loop has better ILP. Each
   a[i] = x[i] + y[i] + z[i]
   line can be computed in parallel for all the different i's.

   In the vec_add way, the second line vec_add(a, z, a, n) depends on
   the first line vec_add(x, y, a, n). Thus no progress can be made on the
   second line while the first one is in progress.