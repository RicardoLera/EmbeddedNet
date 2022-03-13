==430300== Cachegrind, a cache and branch-prediction profiler
==430300== Copyright (C) 2002-2017, and GNU GPL'd, by Nicholas Nethercote et al.
==430300== Using Valgrind-3.15.0 and LibVEX; rerun with -h for copyright info
==430300== Command: ./EmbeddedNet test 1
==430300== Parent PID: 170008
==430300== 
--430300-- warning: L3 cache found, using its data for the LL simulation.
==430300== 
==430300== I   refs:      6,369,594,192
==430300== I1  misses:            4,271
==430300== LLi misses:            2,442
==430300== I1  miss rate:          0.00%
==430300== LLi miss rate:          0.00%
==430300== 
==430300== D   refs:      2,064,233,110  (1,359,003,131 rd   + 705,229,979 wr)
==430300== D1  misses:       34,476,969  (   19,010,839 rd   +  15,466,130 wr)
==430300== LLd misses:        5,311,277  (    1,603,955 rd   +   3,707,322 wr)
==430300== D1  miss rate:           1.7% (          1.4%     +         2.2%  )
==430300== LLd miss rate:           0.3% (          0.1%     +         0.5%  )
==430300== 
==430300== LL refs:          34,481,240  (   19,015,110 rd   +  15,466,130 wr)
==430300== LL misses:         5,313,719  (    1,606,397 rd   +   3,707,322 wr)
==430300== LL miss rate:            0.1% (          0.0%     +         0.5%  )
