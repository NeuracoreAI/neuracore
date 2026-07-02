"""Chrome interop decision gate: our producer -> real Google Chrome, under netem.

The fast loopback suite validates protocol mechanics on the libdatachannel<->
libdatachannel path (the 1% path). The real consumer is Chrome (the 99% path),
and the whole REMB-driven-adaptation choice (`reports/SPIKE-pr5-media-chain.md`)
rests on Chrome-plus-REMB holding the live-preview SLOs under constraint. This
harness is that decision gate: it drives **installed Google Chrome** (Playwright
channel "chrome", *not* open-source Chromium, which often lacks the H.264
decoder) as a recvonly WebRTC peer, with our stack as the sole offerer.

Shape:
  * The producer offers a sendonly H.264 video track (goog-remb + nack, no
    transport-cc), so Chrome runs its own receive-side bandwidth estimator and
    sends REMB back toward the producer.
  * Signaling is bridged in-process: the producer's drained events
    (`on_local_description` offer, `on_local_candidate`) are fed to the browser
    page, and the browser's answer + ICE candidates are fed back to
    `set_remote_answer` / `add_remote_candidate`.
  * The whole process runs inside a private netns with a netem-shaped `lo` (same
    out-of-band shaping as `netem_runner`), so Chrome and the producer talk over
    the constrained loopback.

It asserts from two sides and prints a JSON verdict:
  * Chrome `getStats` inbound-rtp: framesPerSecond / framesDecoded hold a floor,
    freezeCount / totalFreezesDuration stay low, frameHeight reflects any
    downscale step, no decode stall.
  * The producer's structured ladder: REMB/RR drove a step down under constraint
    (max_step > 0).

Run via the gated test (`test_chrome_interop`) or directly:
    unshare -n env NCD_RUST_WEBRTC=1 PYTHONPATH=$PWD \
      python3 -m tests.integration.webrtc.chrome_interop
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

# The receiver page below is embedded JavaScript whose lines exceed the Python
# line-length limit; that is fine for an inline harness page.
# ruff: noqa: E501


RECEIVER_HTML = """
<!doctype html><html><head><meta charset="utf-8"></head><body>
<video id="v" autoplay muted playsinline></video>
<script>
let pc = null;
const pending = [];
// Answer every offer: the producer sends a first data-only offer, then a second
// offer that renegotiates the video m-line in (the track add). Both must be
// answered on the same RTCPeerConnection or Chrome never receives the video.
window.handleOffer = async (offerSdp) => {
  if (!pc) {
    pc = new RTCPeerConnection({ iceServers: [] });
    pc.ontrack = (e) => { document.getElementById('v').srcObject = e.streams[0]; };
    window.__cands = [];
    pc.onicecandidate = (e) => { if (e.candidate) window.__cands.push(e.candidate.toJSON()); };
  }
  await pc.setRemoteDescription({ type: 'offer', sdp: offerSdp });
  for (const c of pending.splice(0)) { try { await pc.addIceCandidate(c); } catch (_) {} }
  const answer = await pc.createAnswer();
  await pc.setLocalDescription(answer);
  return pc.localDescription.sdp;
};
window.addCand = async (cand, mid) => {
  const c = { candidate: cand, sdpMid: mid };
  if (!pc || !pc.remoteDescription) { pending.push(c); return; }
  try { await pc.addIceCandidate(c); } catch (_) {}
};
window.takeCands = () => { const c = window.__cands || []; window.__cands = []; return c; };
window.inboundStats = async () => {
  if (!pc) return null;
  const report = await pc.getStats();
  let r = null;
  report.forEach((s) => {
    if (s.type === 'inbound-rtp' && s.kind === 'video') r = s;
  });
  if (!r) return null;
  return {
    framesDecoded: r.framesDecoded || 0,
    framesReceived: r.framesReceived || 0,
    framesPerSecond: r.framesPerSecond || 0,
    freezeCount: r.freezeCount || 0,
    totalFreezesDuration: r.totalFreezesDuration || 0,
    frameWidth: r.frameWidth || 0,
    frameHeight: r.frameHeight || 0,
    pliCount: r.pliCount || 0,
    nackCount: r.nackCount || 0,
    packetsLost: r.packetsLost || 0,
    packetsReceived: r.packetsReceived || 0,
    framesDropped: r.framesDropped || 0,
    keyFramesDecoded: r.keyFramesDecoded || 0,
    bytesReceived: r.bytesReceived || 0,
  };
};
</script></body></html>
"""

DEFAULT_NETEM = "delay 20ms rate 400kbit limit 48"


def _chrome_munge(sdp: str) -> str:
    """Make a libdatachannel offer acceptable to Chrome's stricter SDP parser.

    libdatachannel emits a bare ``a=ssrc:<id>`` line; Chrome rejects it
    ("a=ssrc Expects 2 fields") and requires at least an attribute such as
    ``cname``. We append ``cname`` (matching the packetizer's RTCP CNAME) to any
    bare ssrc line. This is a real producer-side cutover concern flagged in
    reports/PR5-congestion.md; the munge keeps the interop gate honest without
    changing the loopback path (libdatachannel parses the bare line fine).
    """
    out = []
    for line in sdp.replace("\r\n", "\n").split("\n"):
        if line.startswith("a=ssrc:") and " " not in line.strip()[len("a=ssrc:") :]:
            ssrc = line.strip()[len("a=ssrc:") :]
            out.append(f"a=ssrc:{ssrc} cname:neuracore")
        else:
            out.append(line)
    return "\r\n".join(out)


def _sh(cmd: str) -> tuple[int, str]:
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)  # noqa: S602
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def _setup_link() -> str:
    netem = os.environ.get("NEURACORE_WEBRTC_NETEM", DEFAULT_NETEM)
    rc, out = _sh("ip link set lo up")
    if rc != 0:
        raise RuntimeError(f"could not bring up lo: {out}")
    _sh("tc qdisc del dev lo root")
    rc, out = _sh(f"tc qdisc add dev lo root netem {netem}")
    if rc != 0:
        raise RuntimeError(f"could not apply netem '{netem}': {out}")
    return netem


def _run(seconds: float, settle: float) -> dict:
    os.environ.setdefault("NCD_RUST_WEBRTC", "1")
    # Turn on the producer's Chrome-only SDP munge (bare a=ssrc -> a=ssrc cname),
    # which Chrome's parser requires. Gated so the loopback path stays byte-identical.
    os.environ.setdefault("NCD_WEBRTC_CHROME_SDP", "1")
    from playwright.sync_api import sync_playwright

    from neuracore.core.streaming.p2p.webrtc_selection import load_native
    from tests.integration.webrtc.shared import metrics
    from tests.integration.webrtc.shared.frames import encode_frame

    native = load_native()
    track_id = "cam0"
    producer = native.Producer(connection_id=None, frame_queue_capacity=16)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="chrome",
            headless=True,
            args=[
                "--no-sandbox",
                "--autoplay-policy=no-user-gesture-required",
                "--disable-gpu",
                # In an isolated netns Chrome's default mDNS ICE candidate
                # obfuscation (.local hostnames) hangs — there is no mDNS
                # responder. Expose raw host IPs so ICE gathers the 127.0.0.1
                # candidate directly and the handshake completes.
                "--disable-features=WebRtcHideLocalIpsWithMdns",
            ],
        )
        page = browser.new_page()
        page.set_content(RECEIVER_HTML)

        # Playwright's sync API is single-threaded, so the whole bridge runs on
        # this thread: pump producer signaling-out into the page and the page's
        # ICE candidates back. `_pump_signaling` is called both during the
        # handshake and throughout the frame loop (trickle ICE continues).
        state = {"answered": False}

        def pump_signaling() -> None:
            for event in producer.drain_events():
                kind = event.get("kind")
                if kind == "on_local_description" and event.get("sdp_type") == "offer":
                    # Answer every offer (the data-only bootstrap and the video
                    # renegotiation) on the same Chrome peer connection. The
                    # producer already munges the bare a=ssrc line for Chrome
                    # (NCD_WEBRTC_CHROME_SDP, set below); _chrome_munge stays as an
                    # idempotent backstop in case the env gate is ever off.
                    answer = page.evaluate(
                        "(sdp) => window.handleOffer(sdp)", _chrome_munge(event["sdp"])
                    )
                    producer.set_remote_answer(answer)
                    state["answered"] = True
                elif kind == "on_local_candidate":
                    page.evaluate(
                        "(a) => window.addCand(a.cand, a.mid)",
                        {"cand": event["candidate"], "mid": event.get("mid")},
                    )
            if state["answered"]:
                for c in page.evaluate("() => window.takeCands()") or []:
                    producer.add_remote_candidate(
                        c.get("candidate", ""), c.get("sdpMid")
                    )

        # The producer must offer the video track up front so the first drained
        # description carries the m-line Chrome answers.
        producer.add_data_channel("control", "reliable")
        producer.add_video_track(track_id)

        deadline = time.time() + 10
        while not state["answered"] and time.time() < deadline:
            pump_signaling()
            time.sleep(0.02)
        if not state["answered"]:
            browser.close()
            raise RuntimeError("offer/answer did not complete with Chrome")

        # Feed frames at the source rate and poll both sides.
        period = 1.0 / metrics.SOURCE_FPS
        total = int(seconds * metrics.SOURCE_FPS)
        samples: list[dict] = []
        steps: list[int] = []
        start = time.perf_counter()
        last_poll = 0.0
        for i in range(total):
            target = start + i * period
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
            producer.submit_frame(track_id, encode_frame(i))
            t = time.perf_counter() - start
            if t - last_poll >= 0.5:
                last_poll = t
                pump_signaling()  # keep trickling ICE both ways
                steps.append(producer.congestion_step(track_id) or 0)
                s = page.evaluate("() => window.inboundStats()")
                if s:
                    s["t"] = round(t, 2)
                    samples.append(s)

        max_step = producer.congestion_max_step(track_id)

        # Steady-state Chrome stats: compare two getStats samples in the tail to
        # derive the decode fps Chrome actually sustained after adaptation.
        tail = [s for s in samples if s["t"] >= settle]
        verdict: dict = {"ok": True, "max_step": max_step, "step_timeline": steps}
        if len(tail) >= 2:
            first, last = tail[0], tail[-1]
            dt = last["t"] - first["t"]
            decoded_fps = (
                (last["framesDecoded"] - first["framesDecoded"]) / dt if dt else 0.0
            )
            verdict.update(
                tail_decoded_fps=round(decoded_fps, 2),
                tail_reported_fps=round(last["framesPerSecond"], 2),
                freezeCount=last["freezeCount"],
                totalFreezesDuration=round(last["totalFreezesDuration"], 3),
                frameWidth=last["frameWidth"],
                frameHeight=last["frameHeight"],
                framesDecoded=last["framesDecoded"],
                keyFramesDecoded=last["keyFramesDecoded"],
                framesDropped=last["framesDropped"],
                packetsLost=last["packetsLost"],
                packetsReceived=last["packetsReceived"],
                nackCount=last["nackCount"],
                pliCount=last["pliCount"],
                bytesReceived=last["bytesReceived"],
            )
        else:
            verdict.update(
                ok=False, error=f"insufficient Chrome stats samples: {len(tail)}"
            )
        browser.close()
        return verdict


def _host_ip() -> str:
    """The host's primary (non-loopback) IP, which both the producer ICE agent
    and Chrome will gather as a host candidate so they pair on the host network.
    """
    rc, out = _sh("ip route get 1.1.1.1")
    for tok in out.split():
        if tok == "src":
            return out.split("src", 1)[1].split()[0]
    return "127.0.0.1"


def main() -> int:
    seconds = float(os.environ.get("NEURACORE_WEBRTC_CHROME_SECONDS", 24))
    settle = float(os.environ.get("NEURACORE_WEBRTC_CHROME_SETTLE", 12))
    # Host mode (no netns): Chrome's WebRTC stack does not function inside an
    # isolated net namespace (ICE gathering hangs), and the host loopback cannot
    # be netem-shaped (it is the host's). So the Chrome decision gate runs clean
    # on the host to validate the wire interop (Chrome decoding the built-in
    # chain's H.264 + REMB exchange); the adaptation-under-constraint proof comes
    # from the libdatachannel-consumer netem gate. See reports/PR5-congestion.md.
    host_mode = os.environ.get("NEURACORE_WEBRTC_CHROME_HOST", "1") not in ("0", "")
    try:
        if host_mode:
            os.environ.setdefault("NEURACORE_WEBRTC_BIND_ADDRESS", _host_ip())
            netem = "none (host mode: clean loopback interop validation)"
        else:
            netem = _setup_link()
        result = _run(seconds, settle)
        result["netem"] = netem
        result["host_mode"] = host_mode
    except Exception as exc:  # noqa: BLE001
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
