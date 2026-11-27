//! The URDF package the SDK saves for a robot it connects.
//!
//! `nc.connect_robot` builds the package it would upload online and saves it
//! under [`crate::storage::paths::robot_packages_root`], then names it in its
//! connect request; the daemon keeps the path on the robot row. Once the
//! server has registered the robot, the robot resolver sends the package if
//! the server has none or the robot was connected with `overwrite_urdf`, as
//! `nc.connect_robot` decides online, and deletes it.
//!
//! A package the server rejects is logged and deleted: the robot is
//! registered without it, so recordings made with it still upload, and an
//! online `nc.connect_robot` can upload a fixed one later.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::api::models::RobotRegistrationResponse;
use crate::api::{ApiClient, ApiClientError};

/// How old a saved package no robot row names must be before the robot
/// resolver deletes it at startup. A package is saved just before the robot
/// is connected, so a young one may be about to be named.
pub const UNREFERENCED_PACKAGE_RETENTION: Duration = Duration::from_secs(24 * 60 * 60);

/// Why a saved package could not be sent this time.
#[derive(Debug, thiserror::Error)]
pub enum SendError {
    /// The package could not be read.
    #[error("failed to read the robot package: {0}")]
    Read(#[source] std::io::Error),
    /// The server did not take the package.
    #[error(transparent)]
    Upload(#[from] ApiClientError),
}

/// Send the package at `path` for the robot the server registered as
/// `registered`, if the server has no URDF or `overwrite_urdf` was asked.
///
/// `Ok` means the package is dealt with: sent, not needed, gone, or rejected
/// by the server, which sending again cannot help. `Err` means it could not
/// be sent this time, so the robot is tried again.
pub async fn send_package(
    client: &ApiClient,
    org_id: &str,
    path: &Path,
    registered: &RobotRegistrationResponse,
    shared: bool,
    overwrite_urdf: bool,
) -> Result<(), SendError> {
    if registered.has_urdf && !overwrite_urdf {
        return Ok(());
    }
    let package = match tokio::fs::read(path).await {
        Ok(package) => package,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(SendError::Read(error)),
    };
    match client
        .upload_robot_package(org_id, &registered.robot_id, shared, &package)
        .await
    {
        Ok(()) => tracing::info!(
            cloud_robot_id = registered.robot_id,
            "uploaded the robot's description"
        ),
        Err(error) if rejected(&error) => tracing::warn!(
            %error,
            cloud_robot_id = registered.robot_id,
            "the server rejected the robot's description; the robot is registered without it"
        ),
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

/// Delete a package the robot resolver dealt with.
pub async fn remove_package(path: &Path) {
    match tokio::fs::remove_file(path).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            tracing::warn!(%error, path = %path.display(), "failed to delete a sent robot package")
        }
    }
}

/// Delete the packages in `root` that no robot row names and that are older
/// than [`UNREFERENCED_PACKAGE_RETENTION`]: one replaced before it was sent,
/// or saved by a process that stopped before connecting the robot.
pub async fn remove_unreferenced(root: &Path, referenced: &HashSet<PathBuf>) {
    let mut entries = match tokio::fs::read_dir(root).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
        Err(error) => {
            tracing::warn!(%error, root = %root.display(), "failed to list saved robot packages");
            return;
        }
    };
    let now = SystemTime::now();
    while let Ok(Some(entry)) = entries.next_entry().await {
        let path = entry.path();
        if referenced.contains(&path) {
            continue;
        }
        let old_enough = entry
            .metadata()
            .await
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age >= UNREFERENCED_PACKAGE_RETENTION);
        if old_enough {
            tracing::info!(path = %path.display(), "deleting a robot package no robot uses");
            remove_package(&path).await;
        }
    }
}

/// Whether the server rejected the package itself (an invalid URDF or file
/// path, a robot id it does not know, a package too large), so sending it
/// again cannot help. A rejected key, a missing role and throttling are not
/// about the package: they are retried.
fn rejected(error: &ApiClientError) -> bool {
    matches!(
        error,
        ApiClientError::Status { status, .. }
            if status.is_client_error() && !matches!(status.as_u16(), 401 | 403 | 408 | 429)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use tempfile::TempDir;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::api::auth::StaticAuthProvider;
    use crate::api::ApiClientOptions;

    fn client(server: &MockServer) -> ApiClient {
        let options = ApiClientOptions {
            base_url: server.uri(),
            timeout: Duration::from_secs(5),
            max_retries: 1,
            max_backoff: Duration::from_secs(1),
        };
        ApiClient::new(options, Arc::new(StaticAuthProvider::new("token-1"))).unwrap()
    }

    fn registered(has_urdf: bool) -> RobotRegistrationResponse {
        RobotRegistrationResponse {
            robot_id: "cloud-robot-9".into(),
            has_urdf,
        }
    }

    fn upload(status: u16) -> Mock {
        Mock::given(method("PUT"))
            .and(path("/org/org-1/robots/cloud-robot-9/package"))
            .respond_with(ResponseTemplate::new(status))
    }

    fn saved(dir: &TempDir) -> PathBuf {
        let path = dir.path().join("package.zip");
        std::fs::write(&path, b"zip bytes").unwrap();
        path
    }

    #[tokio::test]
    async fn a_package_is_sent_when_the_server_has_none() {
        let dir = TempDir::new().unwrap();
        let server = MockServer::start().await;
        upload(200).expect(1).mount(&server).await;

        send_package(
            &client(&server),
            "org-1",
            &saved(&dir),
            &registered(false),
            false,
            false,
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn a_package_is_only_sent_over_one_on_the_server_when_asked() {
        let dir = TempDir::new().unwrap();
        let server = MockServer::start().await;
        upload(200).expect(1).mount(&server).await;
        let path = saved(&dir);

        send_package(
            &client(&server),
            "org-1",
            &path,
            &registered(true),
            false,
            false,
        )
        .await
        .unwrap();
        send_package(
            &client(&server),
            "org-1",
            &path,
            &registered(true),
            false,
            true,
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn a_rejected_package_is_dealt_with_and_a_failed_one_is_not() {
        let dir = TempDir::new().unwrap();
        let path = saved(&dir);
        let rejecting = MockServer::start().await;
        upload(422).mount(&rejecting).await;
        let failing = MockServer::start().await;
        upload(503).mount(&failing).await;

        assert!(send_package(
            &client(&rejecting),
            "org-1",
            &path,
            &registered(false),
            false,
            false
        )
        .await
        .is_ok());
        assert!(send_package(
            &client(&failing),
            "org-1",
            &path,
            &registered(false),
            false,
            false
        )
        .await
        .is_err());
    }

    #[tokio::test]
    async fn a_missing_package_needs_nothing() {
        let dir = TempDir::new().unwrap();
        let server = MockServer::start().await;
        upload(200).expect(0).mount(&server).await;

        send_package(
            &client(&server),
            "org-1",
            &dir.path().join("gone.zip"),
            &registered(false),
            false,
            false,
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn only_old_packages_no_robot_names_are_deleted() {
        let dir = TempDir::new().unwrap();
        let old_unused = dir.path().join("old-unused.zip");
        let old_named = dir.path().join("old-named.zip");
        let young = dir.path().join("young.zip");
        for path in [&old_unused, &old_named, &young] {
            std::fs::write(path, b"zip").unwrap();
        }
        let two_days_ago = SystemTime::now() - Duration::from_secs(2 * 24 * 60 * 60);
        for path in [&old_unused, &old_named] {
            std::fs::File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_modified(two_days_ago)
                .unwrap();
        }

        remove_unreferenced(dir.path(), &HashSet::from([old_named.clone()])).await;

        assert!(!old_unused.exists());
        assert!(old_named.exists());
        assert!(young.exists());
    }
}
