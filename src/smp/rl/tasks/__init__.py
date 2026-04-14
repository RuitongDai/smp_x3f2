"""SMP downstream RL tasks.

Importing this package registers all SMP tasks in ``mjlab.tasks.registry``
via side-effect imports of each task sub-package.
"""

# from smp.rl.tasks import base_height  # noqa: F401  # registers Smp-BaseHeight-G1
from smp.rl.tasks import velocity  # noqa: F401  # registers Smp-Velocity-G1
