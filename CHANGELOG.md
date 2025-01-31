# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Ability to upload MJCF in addition to URDF
- Value/type checks on logging funtions
- Ability to stream datasets to python

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