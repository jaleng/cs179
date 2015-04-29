CS 179 Set 3
Jalen Green
README.txt
-----------
Can only run on GPU's of compute capability 3.0 or higher
due to the use of warp shuffle operations.

threadsPerBlock must be a power of 2 as the reduction algorithm
relies on that.

