# EE Body Configs

This directory stores swappable end-effector hardware configs and their control
implementations.

The arm hardware config references one end-effector by name with `ee_body`. To
change the physical end effector, add a new JSON file here and change only that
name in `configs/hardware/openarm_v1.json`.

```text
ee_body/
  config/       JSON configs
  drivers/      Python control implementations
```

End-effectors are intentionally independent from GMR IK and policy models.
Arm joints still come from robot qpos; grippers or other end-effectors should
be controlled from buttons, triggers, or a dedicated command source.

Drivers can optionally expose a hardware-tuner UI by implementing
`create_tuner_panel(parent=..., bridge=..., status_callback=..., hz=...)`.
The main OpenArm tuner only mounts this hook; all end-effector-specific widgets
and commands stay inside the driver.

`openarm_v1_dm_gripper` uses PICO trigger values as continuous close commands:
released maps to `open_position`, fully pressed maps to `safe_close_position`.
Closing uses `close_speed_rad_s` and `close_torque_pu`; opening uses
`open_speed_rad_s` and `open_torque_pu`. Contact protection can use a torque
threshold if `torque_stop_threshold` is set, and otherwise uses stall detection
from velocity feedback.
