"""Fast, peer-free unit tests for the Rust WebRTC stack's Python surface.

These pin the pure logic added in PR0-PR3 (the recording bridge, the
feature-flag selection/loader, and the PR1 test-helper logic) without a native
module, a peer connection, sockets, or sleeps. The native transport, the
negotiation queue, and the manifest model are unit-tested on the Rust side in
``neuracore_webrtc``'s ``#[cfg(test)]`` modules; everything requiring a live
``PeerConnection`` stays in ``tests/integration/webrtc``.
"""
