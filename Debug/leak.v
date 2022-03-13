==427722== Memcheck, a memory error detector
==427722== Copyright (C) 2002-2017, and GNU GPL'd, by Julian Seward et al.
==427722== Using Valgrind-3.15.0 and LibVEX; rerun with -h for copyright info
==427722== Command: ./EmbeddedNet test 1
==427722== Parent PID: 170008
==427722== 
==427722== Invalid read of size 4
==427722==    at 0x10BBD5: test (actions.c:813)
==427722==    by 0x113248: main (main.c:371)
==427722==  Address 0x51b2ac0 is 0 bytes after a block of size 800 alloc'd
==427722==    at 0x4E1DD99: calloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==427722==    by 0x10BA5C: test (actions.c:785)
==427722==    by 0x113248: main (main.c:371)
==427722== 
==427722== Invalid write of size 4
==427722==    at 0x10BBF4: test (actions.c:813)
==427722==    by 0x113248: main (main.c:371)
==427722==  Address 0x51b2ac0 is 0 bytes after a block of size 800 alloc'd
==427722==    at 0x4E1DD99: calloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==427722==    by 0x10BA5C: test (actions.c:785)
==427722==    by 0x113248: main (main.c:371)
==427722== 
==427722== Invalid read of size 4
==427722==    at 0x10BEE3: test (actions.c:847)
==427722==    by 0x113248: main (main.c:371)
==427722==  Address 0x51b2ac0 is 0 bytes after a block of size 800 alloc'd
==427722==    at 0x4E1DD99: calloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==427722==    by 0x10BA5C: test (actions.c:785)
==427722==    by 0x113248: main (main.c:371)
==427722== 
==427722== Invalid read of size 4
==427722==    at 0x10BF2C: test (actions.c:847)
==427722==    by 0x113248: main (main.c:371)
==427722==  Address 0x51b2ac0 is 0 bytes after a block of size 800 alloc'd
==427722==    at 0x4E1DD99: calloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==427722==    by 0x10BA5C: test (actions.c:785)
==427722==    by 0x113248: main (main.c:371)
==427722== 
==427722== 
==427722== Process terminating with default action of signal 27 (SIGPROF)
==427722==    at 0x50AF52A: __open_nocancel (open64_nocancel.c:45)
==427722==    by 0x50BD30F: write_gmon (gmon.c:370)
==427722==    by 0x50BDB6E: _mcleanup (gmon.c:444)
==427722==    by 0x4FE300D: __cxa_finalize (cxa_finalize.c:83)
==427722==    by 0x109566: ??? (in /home/ricardo/EmbeddedNet/Debug/EmbeddedNet)
==427722==    by 0x45F1F5A: _dl_fini (dl-fini.c:138)
==427722==    by 0x4FE28D6: __run_exit_handlers (exit.c:108)
==427722==    by 0x4FE2A8F: exit (exit.c:139)
==427722==    by 0x4FC00B9: (below main) (libc-start.c:342)
==427722== 
==427722== HEAP SUMMARY:
==427722==     in use at exit: 132,074 bytes in 3 blocks
==427722==   total heap usage: 387 allocs, 384 frees, 27,595,922 bytes allocated
==427722== 
==427722== 16 bytes in 1 blocks are definitely lost in loss record 1 of 3
==427722==    at 0x4E1DD99: calloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==427722==    by 0x113216: main (main.c:369)
==427722== 
==427722== 20,480 bytes in 1 blocks are definitely lost in loss record 2 of 3
==427722==    at 0x4E1DD99: calloc (in /usr/lib/x86_64-linux-gnu/valgrind/vgpreload_memcheck-amd64-linux.so)
==427722==    by 0x1131FA: main (main.c:368)
==427722== 
==427722== LEAK SUMMARY:
==427722==    definitely lost: 20,496 bytes in 2 blocks
==427722==    indirectly lost: 0 bytes in 0 blocks
==427722==      possibly lost: 0 bytes in 0 blocks
==427722==    still reachable: 111,578 bytes in 1 blocks
==427722==         suppressed: 0 bytes in 0 blocks
==427722== Reachable blocks (those to which a pointer was found) are not shown.
==427722== To see them, rerun with: --leak-check=full --show-leak-kinds=all
==427722== 
==427722== For lists of detected and suppressed errors, rerun with: -s
==427722== ERROR SUMMARY: 6 errors from 6 contexts (suppressed: 0 from 0)
