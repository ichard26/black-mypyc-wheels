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

See the latest performance report here: https://gist.github.com/ichard26/b996ccf410422b44fcd80fb158e05b0d

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
