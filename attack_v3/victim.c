#include <stdint.h>
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#define ITERS 200000UL


static inline void do_elim() {
    register uint64_t a = 1;
    register uint64_t b = 2;

    // Serialize on entry
    asm volatile("lfence" ::: "memory");

    for (uint64_t i = 0; i < ITERS; i++) {

        asm volatile("lfence" ::: "memory");

        __asm__ volatile(
            "mov %1, %0\n\t"      // eliminatable
            : "+r"(a), "+r"(b)
            :
            : "memory"
        );

        asm volatile("lfence" ::: "memory");
    }

    // Serialize on exit
    asm volatile("lfence" ::: "memory");

    asm volatile("" :: "r"(a), "r"(b));
}


static inline void do_noelim() {
     register uint64_t a = 1;

    asm volatile("lfence" ::: "memory");

    for (uint64_t i = 0; i < ITERS; i++) {

        asm volatile("lfence" ::: "memory");

        __asm__ volatile(
            "mov %0, %0\n\t"      // must execute → never eliminated
            : "+r"(a)
            :
            : "memory"
        );

        asm volatile("lfence" ::: "memory");
    }

    asm volatile("lfence" ::: "memory");

    asm volatile("" :: "r"(a));
}

int main(int argc,char**argv){
    if(argc!=2){printf("usage: victim <0|1>\n");return 1;}
    int bit=atoi(argv[1]);

    for(int i=0;i<200000;i++){
        if(bit==0) do_elim();
        else       do_noelim();
    }
    return 0;
}
