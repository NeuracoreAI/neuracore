//! Minimal NUT-container muxer for a single video stream of lossless PNG
//! frames.
//!
//! The video trace actor spools captured frames into a `.nut` file with this
//! writer; the file is then handed off to an `ffmpeg` transcoder. The caller
//! always supplies packed RGB24; the writer PNG-encodes each frame (see
//! [`PngScratch::encode`]) so the on-disk spool — and the daemon's transcode
//! read-back of it — is a small fraction of the raw video bandwidth.
//!
//! The output is intentionally the bare minimum NUT spec elements needed for
//! `ffprobe`/`ffmpeg` to demux the PNG stream: file id string, main header,
//! stream header, one syncpoint, and one frame packet per captured frame. We
//! deliberately skip the optional index packet so the file stays crash-safe — a
//! truncated tail still demuxes up to the last complete frame.
//!
//! See `https://ffmpeg.org/~michael/nut.txt` for the authoritative spec. The
//! bit-level layout is non-obvious in several places (frame-code table
//! run-length encoding, `coded_pts` lsb/msb form, CRC-32/MPEG-2 with
//! MSB-first polynomial 0x04C11DB7) so the helpers below carry inline
//! commentary explaining *why* each magic value is what it is.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
// Only the Linux `sync_file_range` writeback hint needs the raw fd; gated to
// avoid an unused-import warning on platforms without that syscall (e.g. macOS).
use std::cell::RefCell;
#[cfg(target_os = "linux")]
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use miniz_oxide::deflate::core::{
    compress, create_comp_flags_from_zip_params, CompressorOxide, TDEFLFlush, TDEFLStatus,
};

/// Fixed 25-byte file identifier mandated by the NUT spec. The trailing NUL
/// is part of the signature.
const FILE_ID_STRING: &[u8] = b"nut/multimedia container\0";

/// 64-bit big-endian startcodes. All begin with the ASCII byte `'N'` (0x4E)
/// so demuxers can resync by scanning for that byte and then matching the
/// remaining 56 bits.
const MAIN_STARTCODE: u64 = 0x4E4D_7A56_1F5F_04AD;
const STREAM_STARTCODE: u64 = 0x4E53_1140_5BF2_F9DB;
const SYNCPOINT_STARTCODE: u64 = 0x4E4B_E4AD_EECA_4569;

/// On-disk width of a startcode — the full 64 bits, written big-endian.
const STARTCODE_BYTES: usize = size_of::<u64>();

/// NUT bitstream version we emit. Version 3 is the long-stable spec; version
/// 4 introduced extra stream-header fields we do not need.
const NUT_VERSION: u64 = 3;

/// `msb_pts_shift` advertised in the stream header. Conventional value of 7
/// means the short-form `coded_pts` representation occupies the low 7 bits.
/// We always write the full-pts form (encoded as `pts + (1 << 7)`), but the
/// stream header still has to advertise a value so demuxers can compute the
/// short form when they encounter it.
const MSB_PTS_SHIFT: u64 = 7;

/// Maximum distance (bytes) the spec allows between consecutive startcodes.
/// 65536 disables the policy in practice for our use case while still being a
/// legal value (the spec clamps values above 65536 back down to 65536).
const MAX_DISTANCE: u64 = 65536;

/// `data_size_mul` used in the single populated frame-code table entry. With
/// `mul = 1` and `lsb = 0`, the frame's `data_size_msb` carries the entire
/// frame byte count, which is exactly what we want for variable-size raw RGB
/// frames.
const TABLE_MUL: u64 = 1;

// Frame-flag bit positions defined by the NUT spec (see `nut.txt` §"flags").
// These are *bit positions*, not packed values, so the actual flag is `1 <<
// bit`. We keep them as constants for readability in the frame-emit path.
const FLAG_KEY: u64 = 1 << 0; // bit 0
const FLAG_CODED_PTS: u64 = 1 << 3; // bit 3
const FLAG_STREAM_ID: u64 = 1 << 4; // bit 4
const FLAG_SIZE_MSB: u64 = 1 << 5; // bit 5
const FLAG_CHECKSUM: u64 = 1 << 6; // bit 6
const FLAG_CODED: u64 = 1 << 12; // bit 12 — "read coded_flags from the stream"
const FLAG_INVALID: u64 = 1 << 13; // bit 13 — entry is unusable

/// Frame code byte used for every frame we emit. The spec earmarks 0xFF as
/// an "all explicit" entry by convention; combined with `FLAG_CODED` in the
/// table, this means every frame carries its own flags inline.
const FRAME_CODE_ALL_EXPLICIT: u8 = 0xFF;

/// zlib level for the PNG IDAT stream. Level 1 (fastest) is deliberate: the
/// encode runs on the writer thread that drains the frame queue, so a slow
/// encode backs the queue up into producer backpressure — speed matters more
/// than ratio. And with the Up filter the flat frames compress to almost
/// nothing at level 1 anyway. Raise only with a measured CPU/ratio trade.
const PNG_COMPRESSION_LEVEL: u8 = 1;

/// `window_bits` for [`create_comp_flags_from_zip_params`]: a positive value
/// asks miniz_oxide to wrap the deflate stream in a zlib header — what PNG IDAT
/// requires (and what `compress_to_vec_zlib` passed).
const ZLIB_WINDOW_BITS: i32 = 1;

/// `strategy` for [`create_comp_flags_from_zip_params`]: 0 is the default
/// (no forced Huffman-only/RLE strategy).
const DEFLATE_STRATEGY_DEFAULT: i32 = 0;

/// 8-byte PNG file signature.
const PNG_SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// PNG row-filter tag for the "Up" predictor (residual vs the pixel directly
/// above). Prefixes every scanline; see [`PngScratch::encode`] for why Up.
const PNG_FILTER_UP: u8 = 2;

/// Configuration captured at writer-creation time. The writer is single
/// stream; the caller always supplies packed RGB24 (3 bytes per pixel, no
/// padding), which the writer stores as per-frame lossless PNG.
#[derive(Debug, Clone, Copy)]
pub struct NutVideoConfig {
    /// Frame width in pixels. Must be non-zero.
    pub width: u32,
    /// Frame height in pixels. Must be non-zero.
    pub height: u32,
    /// Time-base numerator. For 30 fps capture use `1`.
    pub time_base_num: u32,
    /// Time-base denominator. For 30 fps capture use `30`.
    pub time_base_den: u32,
}

/// Errors raised by [`NutWriter`].
#[derive(Debug, thiserror::Error)]
pub enum NutError {
    /// Configuration values violated NUT spec invariants (e.g. zero width).
    #[error("invalid NUT configuration: {0}")]
    InvalidConfig(&'static str),
    /// Frame buffer size did not match `width * height * 3`.
    #[error("frame buffer size mismatch: expected {expected} bytes, got {actual}")]
    FrameSize {
        /// Expected number of bytes for one packed RGB24 frame.
        expected: usize,
        /// Actual buffer length supplied by the caller.
        actual: usize,
    },
    /// Failed to create the parent directory or open the output file.
    #[error("failed to open NUT file {path}: {source}")]
    Open {
        /// Path that failed to open.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// Failed to write buffered bytes to disk.
    #[error("failed to write NUT file {path}: {source}")]
    Write {
        /// Path being written to.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: io::Error,
    },
}

/// Append-only NUT muxer.
///
/// `create` writes the header packets (file id, main, stream, syncpoint) and
/// flushes them so the file is parseable from the very first frame onwards.
/// Each [`write_frame`](Self::write_frame) call appends one frame packet
/// **without** an explicit per-frame `flush` — the [`BufWriter`] holds up to
/// [`BUF_WRITER_CAPACITY_BYTES`] of pending bytes and drains either when the
/// next write would overflow it or when [`finish`](Self::finish) is called
/// at chunk close. Per-frame flushing was previously used as a crash-safety
/// belt-and-braces, but the chunk-based design now relies on the daemon's
/// startup recovery sweep to delete any partial chunk anyway, so the
/// in-buffer tail is going to be discarded if the producer dies mid-chunk.
/// In exchange we amortise small writes (syncpoint + frame header, ~30
/// bytes) into the next big frame's syscall, cutting tail latency for
/// every frame size.
pub struct NutWriter {
    path: PathBuf,
    writer: BufWriter<File>,
    config: NutVideoConfig,
    /// Bytes physically written to disk so far (header + flushed frames).
    /// Tracked by us rather than queried from the file because `BufWriter`
    /// doesn't expose a cheap byte counter.
    bytes_written: u64,
    /// Number of bytes a well-formed RGB24 frame must occupy. Cached so the
    /// per-frame size check stays in a single `usize`.
    expected_frame_bytes: usize,
    /// Decoded-equivalent bytes seen so far: `expected_frame_bytes` summed over
    /// every appended frame. The chunk-roll threshold keys off this rather than
    /// the compressed [`bytes_written`], so a chunk holds the same frame count
    /// — and thus the same daemon per-chunk transcode granularity — regardless
    /// of how well the PNG frames happened to compress.
    ///
    /// [`bytes_written`]: Self::bytes_written
    logical_bytes: u64,
    /// File offset of the most recently written syncpoint packet. Drives the
    /// bytes-since-last-syncpoint check that triggers the next periodic
    /// syncpoint.
    last_syncpoint_offset: u64,
    /// File offset up to which an async writeback hint has been issued. The
    /// next hint covers `[last_writeback_hint, bytes_written)`.
    last_writeback_hint: u64,
}

/// Emit a new syncpoint when the bytes-since-last-syncpoint would exceed
/// this threshold once the next frame is appended. We pick well below
/// [`MAX_DISTANCE`] (65536) so even a worst-case oversized header keeps the
/// distance within spec; ffmpeg's NUT demuxer rejects the file with
/// `Last frame must have been damaged X > 100 + max_distance` once that
/// budget is blown.
const SYNCPOINT_INTERVAL_BYTES: u64 = 32_768;

/// [`BufWriter`] capacity. Picked from a matrix benchmark as the size
/// that's never meaningfully slower than the alternatives across the
/// 64×64 (12 KiB) → 1920×1920 (10 MiB) frame range we expect, and
/// strictly better than the 8 KiB default for very small frames (where
/// many frames coalesce per syscall) and for very large frames (where the
/// buffer absorbs short writes the kernel returns under writeback
/// pressure). The caveat: in the 0.5-3 MiB frame zone this buffer can cause
/// occasional ~7 ms flush spikes when a frame tips it past capacity and the
/// whole 8 MiB drains in one `write` — a known tail-latency cost, absorbed by
/// the writer thread rather than a `log_*` caller.
const BUF_WRITER_CAPACITY_BYTES: usize = 8 * 1024 * 1024;

/// Issue an async writeback hint at least this often, measured by bytes
/// appended since the last hint.
///
/// A chunk is 256 MiB ([`CHUNK_FLUSH_BYTES`](crate)) and the kernel's
/// `balance_dirty_pages` throttle fires on the *system-wide* dirty-page count —
/// so at 1080p@60 across multiple cameras hundreds of MiB of dirty pages can
/// pile up before a chunk closes, then a single `write()` hard-stalls for
/// hundreds of ms. Hinting writeback every 16 MiB starts the kernel draining
/// those pages continuously, keeping the dirty footprint bounded (≈ interval ×
/// active streams) so the throttle never reaches a hard stall. It's only a
/// handful of cheap syscalls per chunk — far coarser than per-frame, far finer
/// than per-chunk (which would arrive after the stall already happened).
///
/// We only hint writeback (`SYNC_FILE_RANGE_WRITE`); we do **not** drop the
/// pages (`fadvise(DONTNEED)`), because the daemon re-reads each NUT to
/// transcode it — cleaning the pages keeps them cache-warm for that read while
/// still relieving write pressure.
const WRITEBACK_HINT_INTERVAL_BYTES: u64 = 16 * 1024 * 1024;

impl NutWriter {
    /// Create a NUT file at `path` and emit the four mandatory header
    /// elements. The parent directory is created if missing.
    pub fn create(path: &Path, config: NutVideoConfig) -> Result<Self, NutError> {
        if config.width == 0 || config.height == 0 {
            return Err(NutError::InvalidConfig("width and height must be non-zero"));
        }
        if config.time_base_num == 0 || config.time_base_den == 0 {
            return Err(NutError::InvalidConfig(
                "time_base_num and time_base_den must be non-zero",
            ));
        }

        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|source| NutError::Open {
                    path: parent.to_path_buf(),
                    source,
                })?;
            }
        }

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|source| NutError::Open {
                path: path.to_path_buf(),
                source,
            })?;

        let expected_frame_bytes = (config.width as usize)
            .checked_mul(config.height as usize)
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or(NutError::InvalidConfig(
                "width * height * 3 overflows usize",
            ))?;

        let mut writer = NutWriter {
            path: path.to_path_buf(),
            writer: BufWriter::with_capacity(BUF_WRITER_CAPACITY_BYTES, file),
            config,
            bytes_written: 0,
            expected_frame_bytes,
            logical_bytes: 0,
            last_syncpoint_offset: 0,
            last_writeback_hint: 0,
        };
        writer.write_headers()?;
        Ok(writer)
    }

    /// Path being written to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Total bytes appended to the chunk so far. Counts bytes handed to the
    /// `BufWriter` (which may still be buffered in memory, not yet `fsync`'d),
    /// so it is the logical chunk size — what the flush threshold keys off — not
    /// a durability guarantee.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Decoded-equivalent bytes appended so far — `width * height * 3` per
    /// frame, not the compressed on-disk size. See the
    /// [`logical_bytes`](Self::logical_bytes) field for why chunk rolls key off
    /// this rather than [`bytes_written`](Self::bytes_written).
    pub fn logical_bytes(&self) -> u64 {
        self.logical_bytes
    }

    /// Append one frame at the supplied PTS (frame index in time-base ticks).
    /// The frame is supplied as packed RGB24 — exactly `width * height * 3`
    /// bytes — and stored as a per-frame lossless PNG.
    ///
    /// Compresses on the calling thread. The hot producer path instead
    /// compresses on a worker pool and calls
    /// [`write_frame_precompressed`](Self::write_frame_precompressed); this
    /// wrapper is retained for the round-trip tests, which drive raw RGB
    /// straight through.
    pub fn write_frame(&mut self, pts: u64, rgb_bytes: &[u8]) -> Result<(), NutError> {
        if rgb_bytes.len() != self.expected_frame_bytes {
            return Err(NutError::FrameSize {
                expected: self.expected_frame_bytes,
                actual: rgb_bytes.len(),
            });
        }
        let png = encode_png_frame(self.config.width, self.config.height, rgb_bytes);
        self.write_frame_precompressed(pts, &png)
    }

    /// Append one already-PNG-compressed frame (as produced by
    /// [`encode_png_frame`]) at the supplied PTS. The writer thread's half of
    /// the split: it only muxes the payload into the NUT chunk — the expensive
    /// compression has already run on a pool worker — so the single writer keeps
    /// up with multi-camera capture.
    pub fn write_frame_precompressed(&mut self, pts: u64, png: &[u8]) -> Result<(), NutError> {
        let payload_len = png.len();

        // Emit a periodic syncpoint before appending the frame so the gap
        // between consecutive syncpoints stays within the demuxer's reach.
        // Without this, files containing more than a few frames of >16 KiB
        // each are rejected as "Last frame must have been damaged".
        let bytes_since_last = self
            .bytes_written
            .saturating_sub(self.last_syncpoint_offset);
        let projected_frame_bytes = payload_len as u64 + 32;
        if bytes_since_last.saturating_add(projected_frame_bytes) > SYNCPOINT_INTERVAL_BYTES {
            self.write_syncpoint(pts)?;
        }

        // Frame packets are the *only* NUT packets without a startcode. The
        // demuxer differentiates them by inspecting the next byte: any value
        // other than 'N' (0x4E) is treated as a `frame_code`. 0xFF is our
        // "all explicit" entry, so we follow it with the inline coded_flags,
        // stream_id, coded_pts, data_size_msb, and a checksum over that
        // header.
        let mut header = Vec::with_capacity(16);
        header.push(FRAME_CODE_ALL_EXPLICIT);

        // coded_flags XORed against the table entry's FLAG_CODED produces the
        // actual frame flags. Table entry starts as FLAG_CODED only, and we
        // want KEY|STREAM_ID|CODED_PTS|SIZE_MSB|CHECKSUM, so:
        //     coded_flags = FLAG_CODED ^ (KEY|STREAM_ID|CODED_PTS|SIZE_MSB|CHECKSUM)
        let target_flags =
            FLAG_KEY | FLAG_STREAM_ID | FLAG_CODED_PTS | FLAG_SIZE_MSB | FLAG_CHECKSUM;
        let coded_flags = FLAG_CODED ^ target_flags;
        vencode(&mut header, coded_flags);

        // stream_id — always 0, single-stream file.
        vencode(&mut header, 0);

        // coded_pts: the "full pts" form is `pts + (1 << msb_pts_shift)`.
        // Decoder side: any value >= (1 << msb_pts_shift) is treated as full
        // pts and the offset is subtracted back off. This keeps us correct
        // for arbitrarily large pts values without worrying about the
        // lsb/last_pts reconstruction path.
        let coded_pts = pts
            .checked_add(1u64 << MSB_PTS_SHIFT)
            .ok_or(NutError::InvalidConfig("pts overflow when encoding"))?;
        vencode(&mut header, coded_pts);

        // data_size = data_size_lsb (0) + data_size_msb * data_size_mul (1)
        //           = data_size_msb, so we encode the on-disk frame length here.
        vencode(&mut header, payload_len as u64);

        // Frame header checksum: CRC32/MPEG-2 over framecode + all header
        // bytes up to (but not including) the checksum itself.
        let checksum = crc32_nut(&header);
        header.extend_from_slice(&checksum.to_be_bytes());

        self.write_all(&header)?;
        self.write_all(png)?;

        // Track decoded-equivalent volume so chunk rolls track frame count, not
        // the (much smaller, content-dependent) compressed PNG size.
        self.logical_bytes = self
            .logical_bytes
            .saturating_add(self.expected_frame_bytes as u64);

        // Once enough has accumulated, kick off async writeback for it so dirty
        // pages drain continuously instead of piling up to the throttle's hard
        // stall. Best-effort: never fails a frame.
        if self.bytes_written.saturating_sub(self.last_writeback_hint)
            >= WRITEBACK_HINT_INTERVAL_BYTES
        {
            self.hint_writeback();
        }
        Ok(())
    }

    /// Ask the kernel to start writing back the bytes appended since the last
    /// hint, *without waiting* (`SYNC_FILE_RANGE_WRITE`). This bounds the
    /// producer's dirty-page footprint so a later `write()` doesn't hard-stall
    /// under `balance_dirty_pages` throttling. The pages are cleaned but not
    /// evicted, so the daemon's subsequent transcode read still hits cache.
    ///
    /// Best-effort: the buffered tail is flushed into the page cache first so
    /// the whole range is eligible, then the marker advances regardless of the
    /// syscall result — a writeback hint is a pure optimisation and must never
    /// drop a frame or surface an error.
    fn hint_writeback(&mut self) {
        // Push the BufWriter's contents into the page cache so the full range
        // is writeback-eligible. A real write error here is surfaced by the
        // next `write_all`; here we just skip the hint.
        if self.flush().is_err() {
            return;
        }
        let offset = self.last_writeback_hint;
        let nbytes = self.bytes_written.saturating_sub(offset);
        self.last_writeback_hint = self.bytes_written;
        if nbytes == 0 {
            return;
        }
        // `sync_file_range` is Linux-only. On other platforms (macOS) the
        // `flush()` above has already pushed the tail into the page cache, which
        // is all the hint guarantees; there is no portable async-writeback
        // equivalent worth the complexity, so we simply skip it.
        #[cfg(target_os = "linux")]
        {
            let fd = self.writer.get_ref().as_raw_fd();
            // SAFETY: `fd` is the open NUT file's descriptor (valid for the
            // lifetime of `self.writer`), offset/nbytes are non-negative, and
            // `SYNC_FILE_RANGE_WRITE` only *queues* writeback — it does not block,
            // mutate user memory, or take ownership of the fd.
            let result = unsafe {
                libc::sync_file_range(
                    fd,
                    offset as libc::off64_t,
                    nbytes as libc::off64_t,
                    libc::SYNC_FILE_RANGE_WRITE,
                )
            };
            if result != 0 {
                tracing::debug!(
                    errno = %io::Error::last_os_error(),
                    path = %self.path.display(),
                    "sync_file_range writeback hint failed (ignored)",
                );
            }
        }
        #[cfg(not(target_os = "linux"))]
        let _ = offset;
    }

    /// Flush any remaining buffered bytes and return the total bytes
    /// written.
    pub fn finish(mut self) -> Result<u64, NutError> {
        self.flush()?;
        Ok(self.bytes_written)
    }

    fn write_headers(&mut self) -> Result<(), NutError> {
        // file_id_string is the only part of the file that is *not* wrapped
        // in a packet; everything after it is a sequence of packet_header /
        // payload / packet_footer triples (or, for frames, a bare
        // frame_code-based packet).
        self.write_all(FILE_ID_STRING)?;

        let main_payload = build_main_header_payload(self.config);
        let main_packet = wrap_packet(MAIN_STARTCODE, &main_payload);
        self.write_all(&main_packet)?;

        let stream_payload = build_stream_header_payload(self.config);
        let stream_packet = wrap_packet(STREAM_STARTCODE, &stream_payload);
        self.write_all(&stream_packet)?;

        // First syncpoint: global_key_pts = 0, back_ptr_div16 = 0 (no prior
        // syncpoint to chain back to). Record its on-disk offset so the
        // periodic re-emit in `write_frame` can chain `back_ptr_div16`.
        let syncpoint_offset = self.bytes_written;
        let syncpoint_payload = build_syncpoint_payload(0, 0);
        let syncpoint_packet = wrap_packet(SYNCPOINT_STARTCODE, &syncpoint_payload);
        self.write_all(&syncpoint_packet)?;
        self.last_syncpoint_offset = syncpoint_offset;

        Ok(())
    }

    /// Emit a fresh syncpoint with `global_key_pts = pts`. `back_ptr_div16`
    /// is always 0 — the field is a *seek* hint and the spec requires the
    /// real distance to be 16-byte-aligned, which we cannot guarantee
    /// without padding every packet. Setting 0 advertises "no usable back
    /// chain"; ffmpeg's NUT demuxer falls back to linear scanning, which
    /// is exactly what the on-demand transcode pass needs.
    fn write_syncpoint(&mut self, pts: u64) -> Result<(), NutError> {
        let new_offset = self.bytes_written;
        let payload = build_syncpoint_payload(pts, 0);
        let packet = wrap_packet(SYNCPOINT_STARTCODE, &payload);
        self.write_all(&packet)?;
        self.last_syncpoint_offset = new_offset;
        Ok(())
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<(), NutError> {
        self.writer
            .write_all(bytes)
            .map_err(|source| NutError::Write {
                path: self.path.clone(),
                source,
            })?;
        self.bytes_written = self.bytes_written.saturating_add(bytes.len() as u64);
        Ok(())
    }

    fn flush(&mut self) -> Result<(), NutError> {
        self.writer.flush().map_err(|source| NutError::Write {
            path: self.path.clone(),
            source,
        })
    }
}

/// Build the main-header payload (everything between `forward_ptr` and the
/// trailing packet checksum). Kept separate so the packet-framing helper
/// can compute lengths without re-deriving the payload.
fn build_main_header_payload(config: NutVideoConfig) -> Vec<u8> {
    let mut payload = Vec::with_capacity(64);

    vencode(&mut payload, NUT_VERSION);
    vencode(&mut payload, 1); // stream_count
    vencode(&mut payload, MAX_DISTANCE);
    vencode(&mut payload, 1); // time_base_count
    vencode(&mut payload, config.time_base_num as u64);
    vencode(&mut payload, config.time_base_den as u64);

    // Frame-code table. The decode loop walks i from 0..256, consuming
    // count entries per "row". The slot i == 'N' (78) is auto-marked
    // INVALID without consuming a count, so a run of 254 entries starting
    // at i=0 lands the cursor on i=255 even though 254 + 1 (the 'N' freebie)
    // = 255 increments. The second row of count=1 then fills entry 0xFF
    // with the all-explicit FLAG_CODED behaviour we use for every frame.
    //
    // tmp_fields = 6 means the row carries tmp_pts, tmp_mul, tmp_stream,
    // tmp_size, tmp_res, and count (in that order). Everything past that
    // (tmp_match, tmp_head_idx, etc.) keeps its prior value.
    // Row 1: 254 INVALID entries covering i = 0..='N'-1, 'N'+1..=0xFE.
    vencode(&mut payload, FLAG_INVALID); // tmp_flag
    vencode(&mut payload, 6); // tmp_fields
    sencode(&mut payload, 0); // tmp_pts
    vencode(&mut payload, TABLE_MUL); // tmp_mul
    vencode(&mut payload, 0); // tmp_stream
    vencode(&mut payload, 0); // tmp_size
    vencode(&mut payload, 0); // tmp_res
    vencode(&mut payload, 254); // count

    // Row 2: single FLAG_CODED entry that lands on i = 0xFF.
    vencode(&mut payload, FLAG_CODED); // tmp_flag
    vencode(&mut payload, 6); // tmp_fields
    sencode(&mut payload, 0); // tmp_pts (unused for FLAG_CODED frames)
    vencode(&mut payload, TABLE_MUL); // tmp_mul (= 1 so data_size_msb = data_size)
    vencode(&mut payload, 0); // tmp_stream
    vencode(&mut payload, 0); // tmp_size (lsb = 0)
    vencode(&mut payload, 0); // tmp_res
    vencode(&mut payload, 1); // count

    // Version >= 3 main header tail.
    vencode(&mut payload, 0); // header_count_minus1 — no elision headers
    vencode(&mut payload, 0); // main_flags — no BROADCAST_MODE

    payload
}

/// Build the video stream-header payload.
fn build_stream_header_payload(config: NutVideoConfig) -> Vec<u8> {
    let mut payload = Vec::with_capacity(48);

    vencode(&mut payload, 0); // stream_id
    vencode(&mut payload, 0); // stream_class — 0 = video

    // fourcc as a `vb`: length-prefixed bytes. "MPNG" tells ffmpeg's NUT
    // demuxer the frame packets are PNG (`AV_CODEC_ID_PNG`) — the tag ffmpeg
    // itself writes for png-in-NUT, verified via `ffprobe` codec_tag_string.
    let fourcc: &[u8] = b"MPNG";
    vencode(&mut payload, fourcc.len() as u64);
    payload.extend_from_slice(fourcc);

    vencode(&mut payload, 0); // time_base_id
    vencode(&mut payload, MSB_PTS_SHIFT);
    vencode(&mut payload, 1); // max_pts_distance — we always include FLAG_CHECKSUM anyway
    vencode(&mut payload, 0); // decode_delay — PNG is all intra, no B-frames
                              // stream_flags = 0. We deliberately do *not* set FLAG_FIXED_FPS: our
                              // time_base is microsecond ticks (1/1_000_000), and FLAG_FIXED_FPS would
                              // tell downstream demuxers the stream runs at exactly 1/time_base fps
                              // i.e. one million fps. ffmpeg honours that on transcode by inflating the
                              // output to ~10 million frames per 10 s clip (duplicating every real
                              // input frame across all 1-µs slots), which makes the encode effectively
                              // never complete. Real camera capture is variable-rate; an honest VFR
                              // stream is what we want.
    vencode(&mut payload, 0); // stream_flags
    vencode(&mut payload, 0); // codec_specific_data length

    // Video-class tail.
    vencode(&mut payload, config.width as u64);
    vencode(&mut payload, config.height as u64);
    vencode(&mut payload, 1); // sample_width — square pixels
    vencode(&mut payload, 1); // sample_height
    vencode(&mut payload, 0); // colorspace_type — unknown

    payload
}

/// Build the syncpoint payload. The writer emits these periodically (see
/// `write_frame`) to keep the inter-syncpoint distance within the demuxer's
/// reach.
fn build_syncpoint_payload(global_key_pts: u64, back_ptr_div16: u64) -> Vec<u8> {
    let mut payload = Vec::with_capacity(8);

    // global_key_pts (t): tmp = pts * time_base_count + time_base_id. With
    // time_base_count = 1 and time_base_id = 0 this is just the supplied
    // pts value, which is the PTS of the first frame after this syncpoint.
    vencode(&mut payload, global_key_pts);
    // back_ptr_div16 — distance back to the previous syncpoint, in 16-byte
    // units. This writer always passes 0 (no back-chain); demuxers fall back to
    // a linear scan when seeking.
    vencode(&mut payload, back_ptr_div16);
    payload
}

/// Wrap a payload into a complete NUT packet: startcode, forward_ptr,
/// optional header_checksum, payload, trailing CRC.
///
/// `forward_ptr` is defined by the spec as the distance from the first byte
/// after the `packet_header` (i.e. the start of the payload) to the first
/// byte of the *next* packet. That distance equals `payload.len() + 4` (the
/// trailing checksum). If `forward_ptr > 4096` the spec requires an extra
/// `header_checksum u32` between `forward_ptr` and the payload — we'd never
/// hit that for our header packets in practice, but we honour it so the
/// helper is reusable.
fn wrap_packet(startcode: u64, payload: &[u8]) -> Vec<u8> {
    let forward_ptr = payload.len() as u64 + 4; // +4 for the trailing checksum
    let needs_header_checksum = forward_ptr > 4096;

    let mut packet = Vec::with_capacity(STARTCODE_BYTES + 9 + 4 + payload.len() + 4);
    packet.extend_from_slice(&startcode.to_be_bytes());
    vencode(&mut packet, forward_ptr);
    if needs_header_checksum {
        // header_checksum covers startcode + forward_ptr bytes only.
        let header_checksum = crc32_nut(&packet);
        packet.extend_from_slice(&header_checksum.to_be_bytes());
    }

    // The packet checksum covers everything between the packet_header and
    // the checksum itself — i.e. the payload bytes only. Snapshot the
    // position so the slice we hash is unambiguous.
    let payload_start = packet.len();
    packet.extend_from_slice(payload);
    let checksum = crc32_nut(&packet[payload_start..]);
    packet.extend_from_slice(&checksum.to_be_bytes());
    packet
}

/// Append `value` as a NUT variable-length unsigned integer.
///
/// Encoding is big-endian: every byte except the last has its high bit set
/// to mean "more bytes follow"; the last byte has its high bit clear. The
/// low 7 bits of each byte carry value bits, with the most-significant 7
/// bits of `value` emitted first.
fn vencode(out: &mut Vec<u8>, value: u64) {
    // Count how many 7-bit groups are needed. At least one (so 0 encodes as
    // a single 0x00 byte).
    let mut bits_needed = 7;
    while bits_needed < 64 && (value >> bits_needed) != 0 {
        bits_needed += 7;
    }

    // Emit groups MSB first, setting the continuation bit on all but the
    // last byte.
    let mut shift = bits_needed - 7;
    loop {
        let chunk = ((value >> shift) & 0x7F) as u8;
        if shift == 0 {
            out.push(chunk);
            return;
        }
        out.push(chunk | 0x80);
        shift -= 7;
    }
}

/// Append `value` as a NUT signed variable-length integer.
///
/// Zig-zag encoding: positive `x` maps to `2x`, negative `x` to `2|x| - 1`.
/// This keeps small-magnitude values short regardless of sign.
fn sencode(out: &mut Vec<u8>, value: i64) {
    let encoded = if value >= 0 {
        (value as u64).wrapping_mul(2)
    } else {
        // Widen through i128 so unsigned_abs() is correct even for i64::MIN.
        let magnitude = (value as i128).unsigned_abs() as u64;
        magnitude.wrapping_mul(2).wrapping_sub(1)
    };
    vencode(out, encoded);
}

/// Decode every frame's PTS from a finished NUT chunk, in write order — the
/// test-side inverse of [`NutWriter::write_frame_precompressed`]'s framing, so
/// writer tests can assert on the exact timestamps written into a chunk.
#[cfg(test)]
pub(crate) fn read_frame_pts(bytes: &[u8]) -> Vec<u64> {
    let mut pts_values = Vec::new();
    let mut offset = FILE_ID_STRING.len();
    while offset < bytes.len() {
        if bytes[offset] == b'N' {
            // Startcode-wrapped packet (headers / syncpoints): skip startcode,
            // forward_ptr, the header checksum forward_ptr > 4096 implies, and
            // the payload-plus-checksum span forward_ptr covers.
            offset += STARTCODE_BYTES;
            let (forward_ptr, consumed) = vdecode(bytes, offset);
            offset += consumed;
            if forward_ptr > 4096 {
                offset += 4;
            }
            offset += forward_ptr as usize;
        } else {
            assert_eq!(
                bytes[offset], FRAME_CODE_ALL_EXPLICIT,
                "unexpected frame_code at offset {offset}"
            );
            offset += 1;
            let (_coded_flags, consumed) = vdecode(bytes, offset);
            offset += consumed;
            let (_stream_id, consumed) = vdecode(bytes, offset);
            offset += consumed;
            let (coded_pts, consumed) = vdecode(bytes, offset);
            offset += consumed;
            let (data_size, consumed) = vdecode(bytes, offset);
            offset += consumed;
            let pts = coded_pts
                .checked_sub(1u64 << MSB_PTS_SHIFT)
                .expect("frame not coded in full-pts form");
            pts_values.push(pts);
            // Skip the frame header checksum and the payload.
            offset += 4 + data_size as usize;
        }
    }
    pts_values
}

/// Decode a NUT variable-length unsigned integer starting at `offset`.
/// Returns `(value, bytes_consumed)`. Used only by the unit tests, but kept
/// in the main module to keep encoding/decoding side by side.
#[cfg(test)]
fn vdecode(bytes: &[u8], offset: usize) -> (u64, usize) {
    let mut value: u64 = 0;
    let mut consumed = 0;
    loop {
        let byte = bytes[offset + consumed];
        consumed += 1;
        value = (value << 7) | u64::from(byte & 0x7F);
        if byte & 0x80 == 0 {
            return (value, consumed);
        }
    }
}

/// Decode a NUT signed variable-length integer (inverse of [`sencode`]).
#[cfg(test)]
fn sdecode(bytes: &[u8], offset: usize) -> (i64, usize) {
    let (raw, consumed) = vdecode(bytes, offset);
    let value = if raw & 1 == 1 {
        -(raw.div_ceil(2) as i64)
    } else {
        (raw / 2) as i64
    };
    (value, consumed)
}

/// CRC32/MPEG-2: polynomial 0x04C11DB7, init 0, MSB-first, no final XOR.
///
/// NUT spec §"crc32 checksum" specifies this exact variant; using the more
/// common IEEE 802.3 reversed CRC silently produces unparseable files.
fn crc32_nut(bytes: &[u8]) -> u32 {
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (index, slot) in table.iter_mut().enumerate() {
            // Build the lookup entry for one input byte by shifting the
            // byte into the high end of the register and reducing eight
            // times by the polynomial whenever the top bit is set.
            let mut value = (index as u32) << 24;
            for _ in 0..8 {
                if value & 0x8000_0000 != 0 {
                    value = (value << 1) ^ 0x04C1_1DB7;
                } else {
                    value <<= 1;
                }
            }
            *slot = value;
        }
        table
    });

    let mut crc: u32 = 0;
    for &byte in bytes {
        let index = ((crc >> 24) as u8 ^ byte) as usize;
        crc = (crc << 8) ^ table[index];
    }
    crc
}

/// Per-frame PNG-encode scratch, owned by the [`NutWriter`] and reused across
/// frames so a steady-state frame allocates nothing. The three buffers and the
/// compressor are bundled (rather than loose fields) because [`encode`] always
/// drives them as a unit.
///
/// [`encode`]: Self::encode
struct PngScratch {
    /// One frame's filtered PNG scanlines (each row: a filter tag + the row's
    /// filtered bytes) — the input to the zlib stream that becomes IDAT.
    filter: Vec<u8>,
    /// The zlib-compressed IDAT payload. Grown once to fit and then held; see
    /// [`compress_zlib_into`].
    idat: Vec<u8>,
    /// The assembled PNG bitstream (signature + IHDR + IDAT + IEND) that becomes
    /// one NUT frame packet. After [`encode`](Self::encode) the frame is
    /// `output[..len]`.
    output: Vec<u8>,
    /// Persistent zlib compressor, [`reset`](CompressorOxide::reset) between
    /// frames. Holding one across frames keeps its internal Huffman/dictionary
    /// allocations off the per-frame path — `compress_to_vec_zlib` would build a
    /// fresh `CompressorOxide` (two heap allocations) every call.
    compressor: CompressorOxide,
}

impl PngScratch {
    fn new() -> Self {
        Self {
            filter: Vec::new(),
            idat: Vec::new(),
            output: Vec::new(),
            compressor: CompressorOxide::new(create_comp_flags_from_zip_params(
                PNG_COMPRESSION_LEVEL as i32,
                ZLIB_WINDOW_BITS,
                DEFLATE_STRATEGY_DEFAULT,
            )),
        }
    }

    /// Encode one packed RGB24 frame as a standalone PNG (truecolour, 8-bit,
    /// "Up" filter) into `self.output`, returning the byte length written (the
    /// PNG is `self.output[..len]`).
    ///
    /// We hand-roll the container — signature + IHDR + IDAT + IEND — rather than
    /// pull in an image crate; the IDAT payload is a miniz_oxide zlib stream and
    /// the chunk checksums use a standard CRC-32 ([`png_crc32`]).
    fn encode(&mut self, width: u32, height: u32, rgb: &[u8]) -> usize {
        let row_bytes = width as usize * 3;

        // Filtered scanlines: each row is a one-byte filter tag followed by the
        // row's filtered bytes — the input to the zlib stream that becomes IDAT.
        //
        // We use the "Up" filter (residual vs the pixel directly above) on every
        // row. It is the cheapest predictor (a flat byte subtraction, no
        // per-pixel bookkeeping) yet collapses the flat/slowly-varying regions
        // typical of camera frames into runs of zeros, which deflate then
        // crushes far faster and smaller than the raw bytes — measured ~3x
        // faster and ~4x smaller than filter "None" on the near-uniform
        // integration frames.
        self.filter.clear();
        self.filter.reserve((row_bytes + 1) * height as usize);
        for row in 0..height as usize {
            let start = row * row_bytes;
            let current = &rgb[start..start + row_bytes];
            self.filter.push(PNG_FILTER_UP);
            if row == 0 {
                // PNG defines the scanline above row 0 as all-zero, so Up leaves
                // the first row's bytes unchanged.
                self.filter.extend_from_slice(current);
            } else {
                let above = &rgb[start - row_bytes..start];
                self.filter.extend(
                    current
                        .iter()
                        .zip(above)
                        .map(|(value, prior)| value.wrapping_sub(*prior)),
                );
            }
        }

        let idat_len = compress_zlib_into(&mut self.compressor, &self.filter, &mut self.idat);
        let idat = &self.idat[..idat_len];

        let mut ihdr = [0u8; 13];
        ihdr[0..4].copy_from_slice(&width.to_be_bytes());
        ihdr[4..8].copy_from_slice(&height.to_be_bytes());
        ihdr[8] = 8; // bit depth
        ihdr[9] = 2; // colour type: truecolour (RGB)
        ihdr[10] = 0; // compression method: zlib/deflate
        ihdr[11] = 0; // filter method: adaptive (per-row tag; all Up here)
        ihdr[12] = 0; // interlace: none

        self.output.clear();
        self.output
            .reserve(PNG_SIGNATURE.len() + 12 + 13 + 12 + idat.len() + 12);
        self.output.extend_from_slice(&PNG_SIGNATURE);
        write_png_chunk(&mut self.output, b"IHDR", &ihdr);
        write_png_chunk(&mut self.output, b"IDAT", idat);
        write_png_chunk(&mut self.output, b"IEND", &[]);
        self.output.len()
    }
}

thread_local! {
    /// Per-thread PNG-encode scratch. Reused across calls on the same thread so
    /// the compression workers don't reallocate their filter/compressor buffers
    /// each frame. Thread-local (not a shared pool) because [`encode_png_frame`]
    /// runs on multiple compression worker threads concurrently.
    static ENCODE_SCRATCH: RefCell<PngScratch> = RefCell::new(PngScratch::new());
}

/// Compress one packed RGB24 frame to a standalone per-frame PNG, returning the
/// owned bitstream. This is the CPU-heavy half of the old inline `write_frame`;
/// running it on a pool worker rather than the single writer thread is what lets
/// the producer keep up with multi-camera high-detail capture. The result is
/// handed to [`NutWriter::write_frame_precompressed`], which only muxes it.
pub fn encode_png_frame(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
    ENCODE_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let len = scratch.encode(width, height, rgb);
        scratch.output[..len].to_vec()
    })
}

/// Compress `input` as a zlib stream into `output`, reusing both `compressor`
/// and `output`'s allocation across calls; returns the compressed length (the
/// IDAT payload is `output[..len]`).
///
/// This is the buffer-reusing equivalent of miniz_oxide's `compress_to_vec_zlib`
/// (which allocates a fresh output `Vec` *and* a fresh `CompressorOxide` every
/// call). Every frame in a trace is the same size, so `output` grows to fit
/// within the first frame or two and then never resizes again.
fn compress_zlib_into(
    compressor: &mut CompressorOxide,
    input: &[u8],
    output: &mut Vec<u8>,
) -> usize {
    compressor.reset();

    // `compress` writes into the slice's *length*, not its spare capacity, so
    // the buffer must be sized (not merely reserved). Seed a small floor on the
    // first frame; later frames keep whatever length earlier ones grew it to.
    const MIN_OUTPUT: usize = 1024;
    if output.len() < MIN_OUTPUT {
        output.resize(MIN_OUTPUT, 0);
    }

    let mut remaining = input;
    let mut written = 0;
    loop {
        let (status, consumed, produced) = compress(
            compressor,
            remaining,
            &mut output[written..],
            TDEFLFlush::Finish,
        );
        written += produced;
        match status {
            TDEFLStatus::Done => return written,
            // Ran out of output room before finishing: advance past the consumed
            // input (the compressor itself holds no input cursor), grow, repeat.
            TDEFLStatus::Okay => {
                remaining = &remaining[consumed..];
                let headroom = output.len() - written;
                if headroom < 64 {
                    output.resize((output.len() * 2).max(written + 64), 0);
                }
            }
            // `compress` only ever returns Done/Okay for a valid compressor and
            // a real output buffer (BadParam/PutBufFailed need a null compressor
            // or the callback-func variant, neither of which we use).
            other => unreachable!("zlib compress returned {other:?}"),
        }
    }
}

/// Append one PNG chunk — length (u32 BE) + type + data + CRC-32 (u32 BE) over
/// type+data — per the PNG spec.
fn write_png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let checksum_start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    out.extend_from_slice(&png_crc32(&out[checksum_start..]).to_be_bytes());
}

/// Standard CRC-32 (ISO-HDLC: reflected, polynomial 0xEDB88320, init/final XOR
/// 0xFFFFFFFF) — the variant PNG chunk checksums use. This is a *different*
/// variant from [`crc32_nut`] (NUT uses CRC-32/MPEG-2), so the two are kept
/// separate rather than parameterised.
fn png_crc32(bytes: &[u8]) -> u32 {
    static TABLE: OnceLock<[u32; 256]> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut table = [0u32; 256];
        for (index, slot) in table.iter_mut().enumerate() {
            let mut crc = index as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    0xEDB8_8320 ^ (crc >> 1)
                } else {
                    crc >> 1
                };
            }
            *slot = crc;
        }
        table
    });

    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc = table[((crc ^ byte as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command;
    use tempfile::TempDir;

    #[test]
    fn vencode_round_trip() {
        // Boundary values from the spec: single-byte cap (127), 14-bit cap
        // (16383), and a representative wide value.
        for value in [
            0u64,
            1,
            126,
            127,
            128,
            255,
            16_383,
            16_384,
            1_000_000,
            u64::MAX,
        ] {
            let mut buffer = Vec::new();
            vencode(&mut buffer, value);
            let (decoded, consumed) = vdecode(&buffer, 0);
            assert_eq!(decoded, value, "vencode/vdecode disagreed for {value}");
            assert_eq!(consumed, buffer.len(), "extra bytes left over");
        }
    }

    #[test]
    fn vencode_known_short_forms() {
        // Spot-check the literal byte sequences called out in the spec so a
        // regression in the encoder is caught even if the decoder is also
        // broken in the same way.
        let mut buffer = Vec::new();
        vencode(&mut buffer, 0);
        assert_eq!(buffer, vec![0x00]);
        buffer.clear();
        vencode(&mut buffer, 127);
        assert_eq!(buffer, vec![0x7F]);
        buffer.clear();
        vencode(&mut buffer, 128);
        assert_eq!(buffer, vec![0x81, 0x00]);
        buffer.clear();
        vencode(&mut buffer, 16_384);
        assert_eq!(buffer, vec![0x81, 0x80, 0x00]);
    }

    #[test]
    fn sencode_round_trip() {
        for value in [
            0i64,
            1,
            -1,
            63,
            -63,
            64,
            -64,
            8_192,
            -8_192,
            i64::from(i32::MAX),
            i64::from(i32::MIN),
        ] {
            let mut buffer = Vec::new();
            sencode(&mut buffer, value);
            let (decoded, consumed) = sdecode(&buffer, 0);
            assert_eq!(decoded, value, "sencode/sdecode disagreed for {value}");
            assert_eq!(consumed, buffer.len(), "extra bytes left over");
        }
    }

    #[test]
    fn crc32_known_vector() {
        // NUT uses polynomial 0x04C11DB7, init = 0, MSB-first, no final
        // XOR. Cross-checked against an independent reference implementation
        // of the same parameters for the standard "123456789" input. (The
        // more famous CRC-32/MPEG-2 check constant 0x0376E6E7 corresponds to
        // init = 0xFFFFFFFF, which NUT does *not* use.)
        assert_eq!(crc32_nut(b"123456789"), 0x89A1_897F);
    }

    #[test]
    fn file_starts_with_nut_id() {
        let tempdir = TempDir::new().unwrap();
        let path = tempdir.path().join("raw.nut");
        let writer = NutWriter::create(
            &path,
            NutVideoConfig {
                width: 4,
                height: 4,
                time_base_num: 1,
                time_base_den: 30,
            },
        )
        .unwrap();
        writer.finish().unwrap();

        let bytes = std::fs::read(&path).unwrap();
        assert!(bytes.len() >= FILE_ID_STRING.len());
        assert_eq!(&bytes[..FILE_ID_STRING.len()], FILE_ID_STRING);
    }

    #[test]
    fn rejects_wrong_frame_size() {
        let tempdir = TempDir::new().unwrap();
        let path = tempdir.path().join("raw.nut");
        let mut writer = NutWriter::create(
            &path,
            NutVideoConfig {
                width: 4,
                height: 4,
                time_base_num: 1,
                time_base_den: 30,
            },
        )
        .unwrap();
        let too_small = vec![0u8; 10];
        let err = writer.write_frame(0, &too_small).unwrap_err();
        assert!(matches!(
            err,
            NutError::FrameSize {
                expected: 48,
                actual: 10
            }
        ));
    }

    /// Locate `binary` on PATH. Returns `None` (so the caller can skip cleanly)
    /// when it is absent — sandboxes without the FFmpeg suite skip these tests
    /// rather than failing.
    fn locate_on_path(binary: &str) -> Option<PathBuf> {
        // `which` is the most portable check across the Linux distributions we
        // run CI on.
        let output = Command::new("which").arg(binary).output().ok()?;
        if !output.status.success() {
            return None;
        }
        let path = String::from_utf8(output.stdout).ok()?;
        let trimmed = path.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(PathBuf::from(trimmed))
        }
    }

    #[test]
    fn ffprobe_recognises_stream_metadata() {
        let ffprobe = match locate_on_path("ffprobe") {
            Some(path) => path,
            None => {
                eprintln!(
                    "ffprobe not on PATH — skipping NUT metadata validation. \
                     Install `ffmpeg` to enable this test."
                );
                return;
            }
        };

        let tempdir = TempDir::new().unwrap();
        let path = tempdir.path().join("raw.nut");
        let config = NutVideoConfig {
            width: 16,
            height: 16,
            time_base_num: 1,
            time_base_den: 30,
        };
        let mut writer = NutWriter::create(&path, config).unwrap();

        // Distinct pixel values per frame so a future regression that
        // duplicates frame data is also caught by inspection.
        let frame_count = 4u64;
        for index in 0..frame_count {
            let mut buffer = vec![0u8; 16 * 16 * 3];
            for (pixel_index, chunk) in buffer.chunks_mut(3).enumerate() {
                chunk[0] = ((pixel_index + index as usize) & 0xFF) as u8;
                chunk[1] = ((pixel_index * 3 + index as usize) & 0xFF) as u8;
                chunk[2] = ((pixel_index * 5 + index as usize) & 0xFF) as u8;
            }
            writer.write_frame(index, &buffer).unwrap();
        }
        writer.finish().unwrap();

        let output = Command::new(&ffprobe)
            .args([
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-count_frames",
            ])
            .arg(&path)
            .output()
            .expect("spawn ffprobe");

        assert!(
            output.status.success(),
            "ffprobe exited with {:?}: stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );

        let parsed: serde_json::Value =
            serde_json::from_slice(&output.stdout).expect("ffprobe should emit valid JSON");
        let streams = parsed["streams"].as_array().expect("streams array present");
        assert_eq!(
            streams.len(),
            1,
            "expected exactly one stream, got {streams:?}"
        );

        let stream = &streams[0];
        assert_eq!(stream["codec_type"], "video");
        assert_eq!(stream["width"], 16);
        assert_eq!(stream["height"], 16);

        // `-count_frames` populates `nb_read_frames`; fall back to
        // `nb_frames` if the build of ffprobe in question prefers it.
        let frame_field = stream
            .get("nb_read_frames")
            .or_else(|| stream.get("nb_frames"))
            .and_then(|value| value.as_str())
            .and_then(|s| s.parse::<u64>().ok());
        if let Some(reported) = frame_field {
            assert_eq!(
                reported, frame_count,
                "ffprobe reported {reported} frames, expected {frame_count}"
            );
        }
    }

    #[test]
    fn png_chunk_round_trips_losslessly() {
        let ffprobe = match locate_on_path("ffprobe") {
            Some(path) => path,
            None => {
                eprintln!("ffprobe not on PATH — skipping PNG round-trip test.");
                return;
            }
        };
        let ffmpeg = match locate_on_path("ffmpeg") {
            Some(path) => path,
            None => {
                eprintln!("ffmpeg not on PATH — skipping PNG round-trip test.");
                return;
            }
        };

        let (width, height) = (32u32, 24u32);
        let tempdir = TempDir::new().unwrap();
        let path = tempdir.path().join("frames.nut");
        let config = NutVideoConfig {
            width,
            height,
            time_base_num: 1,
            time_base_den: 1_000_000,
        };
        let mut writer = NutWriter::create(&path, config).unwrap();

        // Distinct, high-entropy-ish pixels per frame so a regression that
        // dropped/duplicated pixels (or silently lost the alpha/channel order)
        // is caught by the bit-exact comparison below.
        let frame_count = 5usize;
        let frame_bytes = (width * height * 3) as usize;
        let mut expected = Vec::with_capacity(frame_bytes * frame_count);
        for index in 0..frame_count {
            let mut buffer = vec![0u8; frame_bytes];
            for (pixel_index, channels) in buffer.chunks_mut(3).enumerate() {
                channels[0] = ((pixel_index * 7 + index * 13) & 0xFF) as u8;
                channels[1] = ((pixel_index * 5 + index * 29) & 0xFF) as u8;
                channels[2] = ((pixel_index * 3 + index * 31) & 0xFF) as u8;
            }
            writer.write_frame(index as u64, &buffer).unwrap();
            expected.extend_from_slice(&buffer);
        }
        writer.finish().unwrap();

        // The NUT must advertise the PNG codec so the daemon's codec-agnostic
        // transcode picks the right decoder off the stream header.
        let probe = Command::new(&ffprobe)
            .args([
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
            ])
            .arg(&path)
            .output()
            .expect("spawn ffprobe");
        assert!(probe.status.success(), "ffprobe failed: {probe:?}");
        assert_eq!(String::from_utf8_lossy(&probe.stdout).trim(), "png");

        // Decode the whole NUT back to packed RGB24 and assert it is bit-exact
        // to what we wrote — PNG is lossless, so the round-trip must be exact.
        let decoded = Command::new(&ffmpeg)
            .args(["-v", "error", "-vsync", "passthrough"])
            .arg("-i")
            .arg(&path)
            .args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
            .output()
            .expect("spawn ffmpeg");
        assert!(
            decoded.status.success(),
            "ffmpeg decode failed: stderr={}",
            String::from_utf8_lossy(&decoded.stderr)
        );
        assert_eq!(
            decoded.stdout,
            expected,
            "PNG round-trip was not bit-exact (len got {}, want {})",
            decoded.stdout.len(),
            expected.len()
        );
    }
}
