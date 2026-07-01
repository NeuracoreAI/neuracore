//! Daemon profile storage: load, create, update, delete, and list the YAML
//! profile files under `~/.neuracore/data_daemon/profiles/`.
//!
//! Mirrors `config_manager/profiles.py::ProfileManager`. The on-disk format is
//! YAML to match the existing layout the integration tests write directly
//! (e.g. `shared/profiles.py::scoped_offline_profile`).

use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use thiserror::Error;

use crate::config::{build_default_daemon_config, DaemonConfig};

/// Errors raised while managing profiles. The `Display` strings are surfaced
/// verbatim in CLI output, so their wording is part of the CLI contract.
#[derive(Debug, Error)]
pub enum ProfileError {
    /// The requested profile file does not exist.
    #[error("Profile '{0}' not found.")]
    NotFound(String),
    /// A profile with the same name already exists.
    #[error("Profile '{0}' already exists.")]
    AlreadyExists(String),
    /// An I/O error occurred while reading or writing a profile file.
    #[error(transparent)]
    Io(#[from] io::Error),
    /// A profile file could not be parsed as a valid `DaemonConfig`.
    #[error(transparent)]
    Yaml(#[from] serde_yaml::Error),
}

/// Manages daemon profiles stored on disk.
pub struct ProfileManager {
    home: PathBuf,
}

impl Default for ProfileManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ProfileManager {
    /// Create a `ProfileManager` rooted at the current user's home directory.
    pub fn new() -> Self {
        ProfileManager {
            home: super::env::home_dir(),
        }
    }

    /// Create a `ProfileManager` rooted at an explicit home directory. Used by
    /// the config watcher to resolve against an injected root (production passes
    /// the real home) and by tests to avoid touching the developer's profiles.
    pub fn with_home(home: PathBuf) -> Self {
        ProfileManager { home }
    }

    /// Directory where daemon profiles are stored.
    fn profiles_dir(&self) -> PathBuf {
        self.home
            .join(".neuracore")
            .join("data_daemon")
            .join("profiles")
    }

    /// Ensure the profiles directory exists and return its path.
    fn ensure_profiles_dir(&self) -> io::Result<PathBuf> {
        let profiles_dir = self.profiles_dir();
        fs::create_dir_all(&profiles_dir)?;
        Ok(profiles_dir)
    }

    /// Filesystem path for a named profile (creating the directory if needed).
    fn profile_path(&self, profile: &str) -> io::Result<PathBuf> {
        Ok(self.ensure_profiles_dir()?.join(format!("{profile}.yaml")))
    }

    /// List available profile names, sorted, without the `.yaml` suffix.
    pub fn list_profiles(&self) -> Vec<String> {
        let profiles_dir = self.profiles_dir();
        let entries = match fs::read_dir(&profiles_dir) {
            Ok(entries) => entries,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Vec::new(),
            Err(error) => {
                tracing::warn!(%error, dir = %profiles_dir.display(), "failed to read profiles directory");
                return Vec::new();
            }
        };

        let mut names: Vec<String> = entries
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                if path.is_file() && path.extension().is_some_and(|ext| ext == "yaml") {
                    path.file_stem()
                        .map(|stem| stem.to_string_lossy().into_owned())
                } else {
                    None
                }
            })
            .collect();
        names.sort();
        names
    }

    /// Load a profile configuration from disk.
    ///
    /// When `profile` is `None`, returns the computed default configuration —
    /// matching `ProfileManager.get_profile(None)`.
    pub fn get_profile(&self, profile: Option<&str>) -> Result<DaemonConfig, ProfileError> {
        let Some(name) = profile else {
            return Ok(build_default_daemon_config()?);
        };

        let profile_path = self.profile_path(name)?;
        let contents = match fs::read_to_string(&profile_path) {
            Ok(contents) => contents,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(ProfileError::NotFound(name.to_string()));
            }
            Err(error) => return Err(error.into()),
        };

        // An empty file parses to an all-default config.
        if contents.trim().is_empty() {
            return Ok(DaemonConfig::default());
        }
        Ok(serde_yaml::from_str(&contents)?)
    }

    /// Create a new profile populated with default configuration values.
    pub fn create_profile(&self, profile: &str) -> Result<(), ProfileError> {
        let profile_path = self.profile_path(profile)?;
        let config = build_default_daemon_config()?;
        let serialized = serde_yaml::to_string(&config)?;

        match fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&profile_path)
        {
            Ok(mut file) => {
                file.write_all(serialized.as_bytes())?;
                Ok(())
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                Err(ProfileError::AlreadyExists(profile.to_string()))
            }
            Err(error) => Err(error.into()),
        }
    }

    /// Update an existing profile by overlaying the provided field values.
    ///
    /// Returns the updated configuration. Fields left unset in `updates` keep
    /// their existing values, matching pydantic's `model_copy(update=...)`.
    pub fn update_profile(
        &self,
        profile: &str,
        updates: &DaemonConfig,
    ) -> Result<DaemonConfig, ProfileError> {
        let profile_path = self.profile_path(profile)?;
        let mut config = self.get_profile(Some(profile))?;
        config.overlay(updates);

        let serialized = serde_yaml::to_string(&config)?;
        // Write to a sibling temp file then rename so a crash mid-write can't
        // leave a truncated, unparseable profile behind (rename is atomic on the
        // same filesystem).
        let temp_path = profile_path.with_extension("yaml.tmp");
        fs::write(&temp_path, serialized)?;
        fs::rename(&temp_path, &profile_path)?;
        Ok(config)
    }

    /// Delete an existing profile.
    pub fn delete_profile(&self, profile: &str) -> Result<(), ProfileError> {
        let profile_path = self.profile_path(profile)?;
        match fs::remove_file(&profile_path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                Err(ProfileError::NotFound(profile.to_string()))
            }
            Err(error) => Err(error.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `ProfileManager` rooted at a throwaway temp directory.
    fn temp_manager() -> (tempfile::TempDir, ProfileManager) {
        let home = tempfile::tempdir().expect("create temp home");
        let manager = ProfileManager::with_home(home.path().to_path_buf());
        (home, manager)
    }

    #[test]
    fn list_is_empty_before_any_profile_exists() {
        let (_home, manager) = temp_manager();
        assert!(manager.list_profiles().is_empty());
    }

    #[test]
    fn create_then_get_and_list_round_trips() {
        let (_home, manager) = temp_manager();

        manager.create_profile("alpha").expect("create alpha");
        let config = manager.get_profile(Some("alpha")).expect("get alpha");

        assert_eq!(config.offline, Some(false));
        assert_eq!(config.num_threads, Some(1));
        assert_eq!(manager.list_profiles(), vec!["alpha".to_string()]);
    }

    #[test]
    fn create_rejects_duplicate_profile() {
        let (_home, manager) = temp_manager();
        manager.create_profile("alpha").expect("create alpha");

        let error = manager.create_profile("alpha").expect_err("duplicate");
        assert!(matches!(error, ProfileError::AlreadyExists(name) if name == "alpha"));
    }

    #[test]
    fn update_overlays_fields_and_persists() {
        let (_home, manager) = temp_manager();
        manager.create_profile("alpha").expect("create alpha");

        let updates = DaemonConfig {
            storage_limit: Some(4096),
            offline: Some(true),
            ..DaemonConfig::default()
        };
        manager.update_profile("alpha", &updates).expect("update");

        let reloaded = manager.get_profile(Some("alpha")).expect("reload");
        assert_eq!(reloaded.storage_limit, Some(4096));
        assert_eq!(reloaded.offline, Some(true));
        // Untouched fields survive the update.
        assert_eq!(reloaded.num_threads, Some(1));
    }

    #[test]
    fn update_missing_profile_reports_not_found() {
        let (_home, manager) = temp_manager();
        let error = manager
            .update_profile("ghost", &DaemonConfig::default())
            .expect_err("missing");
        assert!(matches!(error, ProfileError::NotFound(name) if name == "ghost"));
    }

    #[test]
    fn get_missing_profile_reports_not_found() {
        let (_home, manager) = temp_manager();
        let error = manager.get_profile(Some("ghost")).expect_err("missing");
        assert_eq!(error.to_string(), "Profile 'ghost' not found.");
    }

    #[test]
    fn delete_removes_profile_and_then_reports_not_found() {
        let (_home, manager) = temp_manager();
        manager.create_profile("alpha").expect("create alpha");

        manager.delete_profile("alpha").expect("delete alpha");
        assert!(manager.list_profiles().is_empty());

        let error = manager.delete_profile("alpha").expect_err("second delete");
        assert!(matches!(error, ProfileError::NotFound(name) if name == "alpha"));
    }

    #[test]
    fn partial_yaml_profile_loads_with_defaults() {
        let (_home, manager) = temp_manager();
        let profiles_dir = manager.ensure_profiles_dir().expect("profiles dir");
        // Matches what the integration tests write directly.
        fs::write(profiles_dir.join("partial.yaml"), "offline: true\n").expect("write");

        let config = manager.get_profile(Some("partial")).expect("load partial");
        assert_eq!(config.offline, Some(true));
        assert_eq!(config.storage_limit, None);
    }

    #[test]
    fn yaml_profile_accepts_unit_suffixed_byte_values() {
        let (_home, manager) = temp_manager();
        let profiles_dir = manager.ensure_profiles_dir().expect("profiles dir");
        fs::write(profiles_dir.join("units.yaml"), "storage_limit: 1G\n").expect("write");

        let config = manager.get_profile(Some("units")).expect("load units");
        assert_eq!(config.storage_limit, Some(1024 * 1024 * 1024));
    }
}
