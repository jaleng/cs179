// Q1.1
latency of arithmetic instruction: ~10ns
# instructions issuable in 1 clock cycle: 8
a GK110 can start instructions in up to 4 warps each clock, and up to 2 
instructions in each warp

GPU clock cycle = 1ns

Need (#arith. instr) * (1 ns / 8 instr) * 10ns < (#arith. instr) * 1ns
(#arith. instr)


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
threadIdx.x and blockSize.y are the same within a warp. Incrementing 
threadIdx.y will increase the index into data by 1 * (sizeof(float)) = 4.
All accessed memory is thus in the same 128 byte cache line.
This is write coalesced.


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
k is also the same among threads in a warp

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
    r1 = rhs[k + 128 * j];
    r2 = rhs[(k + 1) + 128 * j];
    l1 = lhs[i + 32 * k];
    l2 = lhs[i + 32 * (k + 1)];
    ac1 += l1 * r1;
    ac2 += l2 * r2;
}
output[i + 32 * j] += ac1 + ac2;

//Q1.4.e
We could change the loop such that we process more values of k at a time and
have more accumulators to have better instruction-level parallelism