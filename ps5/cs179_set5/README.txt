CS 179 Set 5
Jalen Green
README
----------------
Note: when viewing the nvvp profiler, I did not see any
overlapping memcpy's or kernels.

batch size 2048:
latency for 1 batch: ~9.5ms
sample HtoD->compute->DtoH
  Memcpy HtoD Duration: 77.794us
    end 8,492,701,932ns
  sloppyClusterKernel Duration: 9.362ms
    start 8,492,712,252ns
    end   8,502,076,522ns
  Memcpy DtoH Duration: 4.704us
    start 8,502,079,498ns
    end   8,502,084,202ns

total duration: 27.624s
throughput 1,568,767 reviews / 27.624s = 56,790 reviews/s


batch size 256:
latency for 1 batch: ~1.4ms
total duration: 28.825s
throughput = 54,424 reviews/s

sample HtoD->compute->DtoH
  Memcpy HtoD
    Duration: 12.209us
    end:      8,370,811,068ns
  sloppyClusterKernel
    Duration: 1.362ms
    Start:    8,370,850,781ns
    End:      8,372,213,175ns
  Memcpy DtoH
    Duration: 2.72us
    Start:    8,372,223,639ns

The throughput did not change much with batch size.

The latency of a batch increased with the batch size.

loader.py throughput = 1,568,767 reviews/ 210.173s = 7,424 reviews/s

loader.py has a much lower throughput than cluster.cc, so running the
cluster program on pre-computed LSA's will be much faster than piping
the output of loader.py.

It would be difficult to get better performance on multiple gpu's 
because when computing we must always have up-to-date values of
the cluster locations and be able to write to them. With 1 GPU
we can keep this data in the GPUs memory, but if we have multiple,
we would have to move this data back and forth from GPUs.

  
