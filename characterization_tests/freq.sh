#!/bin/bash

echo "1" | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
echo "performance" | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# sudo cpupower frequency-set -g userspace
# sudo cpupower frequency-set -d 2500000
# sudo cpupower frequency-set -u 2500000
# sudo cpupower frequency-set -f 2500000