# Black + mypyc

Hello! You've found the home base for the ongoing work getting [psf/black][black]
compiled with [mypyc][mypyc]. Mypyc is a compiler for typed Python code. The
end goal is to increase runtime performance. Since Black is already a well
typed project with type checking provided by mypy, mypyc won out over other
options like Cython. Although using mypyc isn't all smooth sailing, there's a
lot of infrastructural and tuning work to be done (not to mention compatibility
work since mypyc is in alpha). All of that is tracked here.

The issue tracker details specific TO-DOs and goals, while the
["Getting Black compiled with mypyc"][project-board] project is intended as
a general overview of what has been done and what's still pending.

## Performance status

While there's still a lot performance tuning and testing work to be done, some initial
numbers have been collected:

<details>

<summary>System metadata</summary>

```text
- aslr: Full randomization
- boot_time: 2021-06-29 17:11:34
- cpu_affinity: 1
- cpu_config: 1=driver:acpi-cpufreq, governor:userspace, isolated; idle:acpi_idle
- cpu_count: 2
- cpu_model_name: AMD A6-9220 RADEON R4, 5 COMPUTE CORES 2C+3G
- hostname: acer-ubuntu
- perf_version: 2.2.0
- platform: Linux-5.8.0-59-generic-x86_64-with-glibc2.29
- python_cflags: -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall
- python_compiler: GCC 9.3.0
- python_executable: /home/ichard26/.local/share/virtualenvs/black-Q9x7i-w-/bin/python
- python_implementation: cpython
- python_version: 3.8.5 (64-bit)
- timer: clock_gettime(CLOCK_MONOTONIC), resolution: 1.00 ns
- unit: second
```

</details>

- Formatting with safety checks: 1.74x faster
- Formatting without safety checks: 1.90x faster
- Importing black: 1.14x faster (this one is a very rough figure)
- blib2to3 parsing: 1.72x faster

System was tuned with `isolcpus=1`, `rcu_nocbs=1`, `nohz_full=1`, and a fixed CPU
frequency.

Benchmark results are in part from [blackbench][blackbench], an in-development
benchmarking suite for Black.

## Acknowledgements

The project is led by [@ichard26](https://github.com/ichard26).

Thank you to [@JelleZijlstra](https://github.com/JelleZijlstra) for their support and
insights while I debugged or tried to understand mypy[c]'s quirks.

Many thanks goes to [@msullivan](https://github.com/msullivan) for laying the
[groundwork][initial-mypyc-pr] that this project stands on, and to the mypyc team
for building the tool.

Finally, thank you to everyone else who chipped in whose name I have forgetten (sorry!).

[black]: https://github.com/psf/black/
[mypyc]: https://mypyc.readthedocs.io/
[blackbench]: https://github.com/ichard26/blackbench
[project-board]: https://github.com/ichard26/black-mypyc-wheels/projects/1
[initial-mypyc-pr]: https://github.com/psf/black/pull/1009
