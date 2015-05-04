CS 179 Set 4
Jalen Green
README
----------------------

1.1

We could use shared memory to speed up the computation.
Read the frontier, visited, cost, and compact adjacency list arrays 
from global memory and write it to shared memory. Since these arrays can be 
accessed multiple times within the duration of the kernel, putting them in shmem
prevent many accesses of the same global data, which is slower than shmem
accesses. When exploring nodes' neighbors, there is a high chance of
uncoalesced memory accesses if global memory is used, so using shmem guards
against that too (at the risk of bank conflicts, which are still faster).
After computing, we can then sync threads and write to global memory coalesced.

1.2

We can find the sum of all elements in the frontier array quickly using a
parallelized reduction. We can then check to see if the result is zero.
If it is, then F is all false, otherwise F is not all false.

1.3

We can have a flag that indicates whether a node has become a frontier node
in the last round. We can clear the flag before calling the kernel, and then
have the kernel set the flag any time it adds a node to F. This performs better
with dense and sparse graphs.

2

List-mode format data does not group data by spatially close measurements, thus
linear interpolation won't work and spatial caching will not be as useful. So
texture memory cannot be used efficiently.

We cannot iterate over spatially close data and accumulate computed
contributions into a register before writing to a pixel, so PET must have more
accesses to global memory.

Thus PET will be slower if we use a similar approach as with X-ray CT