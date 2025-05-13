# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0-Unreleased]

### Added

- `NEURACORE_REMOTE_RECORDING_TRIGGER_ENABLED` New environment variable to disable other machines from starting a recording
- `nc.connect_endpoint()` and `nc.connect_local_endpoint()` have new arguments `robot_name` and `instance`

### Changed

- `nc.connect_endpoint()` endpoint name argument has been renamed `name` -> `endpoint_name` to avoid confusion with the new `robot_name` argument

### Removed

- None

### Fixed

- Connect Endpoint (`nc.connect_endpoint()` and `nc.connect_local_endpoint()`) now supports multiple simultaneous robot instances


## [1.4.0]

### Added

- Added additional logging functions
- Example dataset upload scripts

### Changed

- Improved data streaming
- Open-sourced more example algorithms
- `nc.log_joints` -> `nc.log_joint_positions`
- `nc.log_actions` -> `nc.log_joint_target_positions`



## [1.3.1]

### Fixed

- API call to get algorithms.


## [1.3.0] 

### Added

- Ability to launch and monitor training jobs using python.
- Ability to deploy and terminate model endpoints using python.

### Changed

- Allow python versions >= 3.9


## [1.2.0] - 2025-03-12

### Added

- Open source initial neuracore algorithms


## [1.1.1] - 2025-03-11

### Fixed

- Fixed streaming data example


## [1.1.0] - 2025-01-31

### Added

- Ability to upload MJCF in addition to URDF
- Value/type checks on logging funtions
- Ability to stream datasets to python
- Ability to download model endpoints via 'nc.connect_local_endpoint(train_run_name="MyTrainRun")'

### Changed

- Depth encoding always assumes depth in m.
- Rate limiting default is not DROP rather than BUFFER. Messages will be dropped if logging frequency is too high. 
- Each camera now has its own websocket


## [1.0.0] - 2025-01-07

### Added

- Initial release
