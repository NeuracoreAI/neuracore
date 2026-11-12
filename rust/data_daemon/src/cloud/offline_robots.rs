//! The URDF package an offline `nc.connect_robot` saves, uploaded when a
//! recording needs the robot.
//!
//! Offline, `connect_robot` builds the package it would have uploaded and
//! saves it as `{recordings_root}/.offline_robots/{local robot id}/package.zip`
//! (`neuracore/data_daemon/offline_robots.py`). The offline resolver registers
//! a robot instance only when a recording that uses it waits, and uploads the
//! package then, before the robot counts as registered, as `connect_robot`
//! uploads it online. A package nothing recorded with is never uploaded: the
//! startup sweep deletes it once it is a day old.

use std::path::Path;
use std::time::SystemTime;

use crate::api::models::RobotRegistrationResponse;
use crate::api::{ApiClient, ApiClientError};

/// The package file an offline `connect_robot` saves.
pub const PACKAGE_FILENAME: &str = "package.zip";

/// Why a saved package could not be sent this time.
#[derive(Debug, thiserror::Error)]
pub enum SendError {
    /// The package could not be read.
    #[error("failed to read the robot package: {0}")]
    Read(#[source] std::io::Error),
    /// The backend did not take the package.
    #[error(transparent)]
    Upload(#[from] ApiClientError),
}

/// The package [`send_package`] dealt with: its modified time when it was
/// read, or `None` when there was none.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sent(Option<SystemTime>);

/// Upload the package saved for a robot connected offline, now that the
/// backend has registered it, if the backend has none or the robot was
/// connected with `overwrite_urdf`, as `connect_robot` decides online.
///
/// A robot saved with no package needs nothing. The backend refusing the
/// package itself is logged and the package counts as dealt with: sending it
/// again cannot help, and the robot is registered without it. Returns why
/// when the package could not be sent this time, so the robot is tried
/// again.
pub async fn send_package(
    client: &ApiClient,
    org_id: &str,
    robot_dir: &Path,
    registered: &RobotRegistrationResponse,
    shared: bool,
    overwrite_urdf: bool,
) -> Result<Sent, SendError> {
    let package_path = robot_dir.join(PACKAGE_FILENAME);
    let package = read_package(&package_path).await.map_err(SendError::Read)?;
    let Some((bytes, modified)) = package else {
        return Ok(Sent(None));
    };
    if registered.has_urdf && !overwrite_urdf {
        return Ok(Sent(Some(modified)));
    }
    match client
        .upload_robot_package(org_id, &registered.robot_id, shared, &bytes)
        .await
    {
        Ok(()) => tracing::info!(
            cloud_robot_id = registered.robot_id,
            "uploaded the description of a robot connected offline"
        ),
        Err(error) if rejected(&error) => tracing::warn!(
            %error,
            cloud_robot_id = registered.robot_id,
            "backend rejected the description of a robot connected offline; it is not uploaded"
        ),
        Err(error) => return Err(error.into()),
    }
    Ok(Sent(Some(modified)))
}

/// Delete the package [`send_package`] dealt with, unless it has been saved
/// again since, then the robot's directory if nothing else is in it.
pub async fn remove_package(robot_dir: &Path, sent: Sent) {
    let package_path = robot_dir.join(PACKAGE_FILENAME);
    let unchanged = match sent.0 {
        Some(read) => tokio::fs::metadata(&package_path)
            .await
            .and_then(|metadata| metadata.modified())
            .is_ok_and(|modified| modified == read),
        None => false,
    };
    if unchanged {
        let _ = tokio::fs::remove_file(&package_path).await;
    }
    // Fails, as it should, when a new save has written into the directory.
    let _ = tokio::fs::remove_dir(robot_dir).await;
}

/// Delete everything saved for a robot the backend refused: it can never be
/// registered, so its package can never be sent.
pub async fn discard(robot_dir: &Path) {
    match tokio::fs::remove_dir_all(robot_dir).await {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => tracing::warn!(
            %error,
            path = %robot_dir.display(),
            "failed to delete a refused robot's package"
        ),
    }
}

/// The package and its modified time, or `None` when the robot has none.
/// The time is read first, so a package replaced while it is read looks
/// newer than the one sent and is kept.
async fn read_package(path: &Path) -> std::io::Result<Option<(Vec<u8>, SystemTime)>> {
    let modified = match tokio::fs::metadata(path).await {
        Ok(metadata) => metadata.modified()?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    Ok(Some((tokio::fs::read(path).await?, modified)))
}

/// Whether the backend rejected the package itself (an invalid URDF or file
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

    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    use tempfile::TempDir;
    use wiremock::matchers::{body_string_contains, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use crate::api::auth::StaticAuthProvider;
    use crate::api::ApiClientOptions;

    struct Saved {
        dir: PathBuf,
        _root: TempDir,
    }

    impl Saved {
        fn new(package: Option<&[u8]>) -> Self {
            let root = TempDir::new().unwrap();
            let dir = root.path().join("local-robot-1");
            if let Some(package) = package {
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(dir.join(PACKAGE_FILENAME), package).unwrap();
            }
            Saved { dir, _root: root }
        }

        fn package(&self) -> PathBuf {
            self.dir.join(PACKAGE_FILENAME)
        }

        async fn send(
            &self,
            server: &MockServer,
            has_urdf: bool,
            overwrite_urdf: bool,
        ) -> Result<Sent, SendError> {
            let registered = RobotRegistrationResponse {
                robot_id: "cloud-robot-9".into(),
                has_urdf,
            };
            send_package(
                &client(server),
                "org-1",
                &self.dir,
                &registered,
                false,
                overwrite_urdf,
            )
            .await
        }
    }

    fn client(server: &MockServer) -> ApiClient {
        let options = ApiClientOptions {
            base_url: server.uri(),
            timeout: Duration::from_secs(5),
            max_retries: 1,
            max_backoff: Duration::from_secs(1),
        };
        ApiClient::new(options, Arc::new(StaticAuthProvider::new("token-1"))).unwrap()
    }

    fn upload(status: u16) -> Mock {
        Mock::given(method("PUT"))
            .and(path("/org/org-1/robots/cloud-robot-9/package"))
            .and(query_param("is_shared", "false"))
            .respond_with(ResponseTemplate::new(status))
    }

    #[tokio::test]
    async fn uploads_the_package_the_backend_lacks() {
        let saved = Saved::new(Some(b"zip bytes"));
        let server = MockServer::start().await;
        Mock::given(method("PUT"))
            .and(path("/org/org-1/robots/cloud-robot-9/package"))
            .and(body_string_contains("zip bytes"))
            .respond_with(ResponseTemplate::new(200))
            .expect(1)
            .mount(&server)
            .await;

        let sent = saved.send(&server, false, false).await.unwrap();
        remove_package(&saved.dir, sent).await;

        assert!(!saved.dir.exists());
    }

    #[tokio::test]
    async fn a_robot_with_a_description_already_keeps_it() {
        let saved = Saved::new(Some(b"zip bytes"));
        let server = MockServer::start().await;
        upload(200).expect(0).mount(&server).await;

        let sent = saved.send(&server, true, false).await.unwrap();
        remove_package(&saved.dir, sent).await;

        assert!(!saved.dir.exists());
    }

    #[tokio::test]
    async fn overwrite_replaces_the_description_already_there() {
        let saved = Saved::new(Some(b"zip bytes"));
        let server = MockServer::start().await;
        upload(200).expect(1).mount(&server).await;

        saved.send(&server, true, true).await.unwrap();
    }

    #[tokio::test]
    async fn a_robot_saved_without_a_package_sends_nothing() {
        let saved = Saved::new(None);
        let server = MockServer::start().await;
        upload(200).expect(0).mount(&server).await;

        assert_eq!(saved.send(&server, false, false).await.unwrap(), Sent(None));
    }

    #[tokio::test]
    async fn a_failed_upload_keeps_the_package_for_the_next_attempt() {
        let saved = Saved::new(Some(b"zip bytes"));
        let server = MockServer::start().await;
        upload(503).mount(&server).await;

        let result = saved.send(&server, false, false).await;

        assert!(matches!(result, Err(SendError::Upload(_))), "{result:?}");
        assert!(saved.package().exists());
    }

    #[tokio::test]
    async fn a_rejected_key_is_retried_rather_than_dropping_the_package() {
        let saved = Saved::new(Some(b"zip bytes"));
        let server = MockServer::start().await;
        upload(401).mount(&server).await;

        let result = saved.send(&server, false, false).await;

        assert!(matches!(result, Err(SendError::Upload(_))), "{result:?}");
    }

    #[tokio::test]
    async fn a_rejected_description_counts_as_dealt_with() {
        let saved = Saved::new(Some(b"not a urdf"));
        let server = MockServer::start().await;
        upload(422).expect(1).mount(&server).await;

        let sent = saved.send(&server, false, false).await.unwrap();
        remove_package(&saved.dir, sent).await;

        assert!(!saved.dir.exists());
    }

    #[tokio::test]
    async fn a_package_saved_again_after_it_was_read_is_kept() {
        let saved = Saved::new(Some(b"new zip"));
        let read_earlier = SystemTime::now() - Duration::from_secs(60);

        remove_package(&saved.dir, Sent(Some(read_earlier))).await;

        assert!(saved.package().exists());
    }

    #[tokio::test]
    async fn discarding_a_refused_robot_removes_its_package() {
        let saved = Saved::new(Some(b"zip bytes"));

        discard(&saved.dir).await;

        assert!(!saved.dir.exists());
    }
}
