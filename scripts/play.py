"""Play wrapper: registers SMP tasks then delegates to mjlab.scripts.play.main."""

from __future__ import annotations

from mjlab.scripts.play import main

import smp.rl.tasks  # noqa: F401  # registers Smp-* tasks in the mjlab registry

if __name__ == "__main__":
  main()
