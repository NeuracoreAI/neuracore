# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

### Added

- None

### Changed

- None

### Removed

- None

### Fixed

- weights_only False for loading from cache


## [3.0.0]

### Added

- `nc.get_latest_sync_point()` a new method that creates a sync point from gathering data logged by a robot. 
- `nc.check_remote_nodes_connected()` a new method to be used in tandem with `get_latest_sync_point` to ensure the expected remote nodes have been fully connected.
- `nc.policy()`, `nc.policy_local_server()` and `nc.policy_remote_server()` now also gather data from remote nodes by default now. to avoid this behavior use the new `NEURACORE_CONSUME_LIVE_DATA` environment variable or provide your own sync point.
- `nc-select-org` and `nc-login` now have non-interactive options.

### Changed

- Policy predictions now return a list of SyncPoint's, rather than a ModelPrediction object.
- `NEURACORE_LIVE_DATA_ENABLED` has been split into two new environment variables `NEURACORE_PROVIDE_LIVE_DATA` and `NEURACORE_CONSUME_LIVE_DATA`

### Removed

- None

### Fixed

- None


## [2.0.0]

### Added

- `nc.set_organization(id_or_name)` method to simplify changing organizations
- `nc-select-org` to interactively change organization
- `nc.list_my_orgs()` method to find the details of your organizations
- Support for Python 3.9
- Added Pi0
- Ability to launch a policy outside of a server using `nc.policy()`

### Changed

- The current organization is now stored locally at `~/.neuracore/config.json` rather than being set globally
- Training now supports all data types. See CNNMLP for an example.
- Endpoints now use our own custom server, rather than torchserve
- `nc.connect_local_endpoint()` -> `nc.policy_local_server()`
- `nc.connect_endpoint()` -> `nc.policy_remote_server()`
- Moved from .mar format to .nc.zip format for model archives

### Removed

- None

### Fixed

- Error message when the wong credentials are provided to `nc-login` 
- Streaming will resume correctly after network disruptions


## [1.6.0]

### Added

- Open sourced our training code!
- Additional lossy upload stream for faster viewing in the dataset viewer
- CLI validation tool for validating algorithms locally. Can be used by running `neuracore-validate /PATH/TO/MY/ALG`

### Changed

- Improved validation error messages on validation of uploaded algorithms

### Removed

- None

### Fixed

- None


## [1.5.0]

### Added

- `NEURACORE_REMOTE_RECORDING_TRIGGER_ENABLED` New environment variable to disable other machines from starting a recording
- `nc.connect_endpoint()` and `nc.connect_local_endpoint()` have new arguments `robot_name` and `instance`

### Changed

- `nc.connect_endpoint()` endpoint name argument has been renamed `name` -> `endpoint_name` to avoid confusion with the new `robot_name` argument

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
