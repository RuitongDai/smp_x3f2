"""Train wrapper: registers SMP tasks then delegates to mjlab.scripts.train.main."""

from __future__ import annotations

from mjlab.scripts.train import main

import smp.rl.tasks  # noqa: F401  # registers Smp-* tasks in the mjlab registry

if __name__ == "__main__":
  main()
