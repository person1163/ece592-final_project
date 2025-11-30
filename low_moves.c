#include <stdint.h>
int main() {
    register int a=1,b=2,c=3,d=4;
    for (uint64_t i=0;i<100000000;i++) {
        __asm__ volatile(
            "add %1,%0\n\t"   // real data dependency
            "add %2,%1\n\t"
            "add %3,%2\n\t"
            "add %0,%3\n\t"
            : "+r"(a), "+r"(b), "+r"(c), "+r"(d)
        );
    }
    volatile int sink=a+b+c+d;
    return sink;
}
