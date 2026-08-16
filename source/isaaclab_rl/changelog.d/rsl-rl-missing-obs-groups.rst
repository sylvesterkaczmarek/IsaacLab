Fixed
^^^^^

* Fixed RSL-RL 4+ compatibility handling leaving ``obs_groups`` as ``dataclasses.MISSING`` for legacy runner configurations. Missing observation groups are now passed to RSL-RL as an empty mapping so its native resolver can infer ``actor`` and ``critic`` observation sets from the environment instead of blocking runner initialisation.
