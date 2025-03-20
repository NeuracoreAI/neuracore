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

- None

## [1.3.1]

### Added

- None

### Changed

- None

### Removed

- None

### Fixed

- API call to get algorithms.

## [1.3.0] 

### Added

- Ability to launch and monitor training jobs using python.
- Ability to deploy and terminate model endpoints using python.

### Changed

- Allow python versions >= 3.9

### Removed

- None

### Fixed

- None


## [1.2.0] - 2025-03-12

### Added

- Open source intitial neuracore algorithms


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

### Removed

- None

### Fixed

- None


## [1.0.0] - 2025-01-07

### Added

- Initial release

### Changed

- None

### Removed

- None

### Fixed

- None