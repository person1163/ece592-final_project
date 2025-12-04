#include <stdint.h>
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#define ITERS 200000UL


__attribute__((noinline,optimize("O0")))
void do_elim() {
    register uint64_t a = 1, b = 2;

    for (uint64_t i = 0; i < ITERS; i++) {
        asm volatile("lfence" ::: "memory");

        asm volatile(
            "mov %1, %0\n\t"
            : "+r"(a), "+r"(b)
            :
            : "memory"
        );

        asm volatile("lfence" ::: "memory");
    }
}

__attribute__((noinline,optimize("O0")))
void do_noelim() {
    register uint64_t a = 1;

    for (uint64_t i = 0; i < ITERS; i++) {
        asm volatile("lfence" ::: "memory");

        asm volatile(
            "mov %0, %0\n\t"
            : "+r"(a)
            :
            : "memory"
        );

        asm volatile("lfence" ::: "memory");
    }
}


int main(int argc,char**argv){
    if(argc!=2){printf("usage: victim <0|1>\n");return 1;}
    int bit=atoi(argv[1]);

    for(int i=0;i<200;i++){
        if(bit==0) do_elim();
        else       do_noelim();
    }
    return 0;
}
