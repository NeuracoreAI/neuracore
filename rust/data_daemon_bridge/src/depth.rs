//! Depth-to-RGB24 storage conversion for the producer's video path.
//!
//! The NUT/PNG pipeline ([`crate::nut_writer`]) only ever accepts packed
//! RGB24. A depth frame is therefore converted into that representation
//! *before* compression, mirroring the Python reference algorithm in
//! `neuracore.core.utils.depth_utils.depth_to_rgb_storage` bit-for-bit:
//!
//! 1. Clip depth values (metres) to `[0, MAX_DEPTH]`.
//! 2. Normalize to `[0, 1]`.
//! 3. Scale to 24-bit range (`0..=2^24 - 1`).
//! 4. Floor and split the 24-bit value into R, G, B bytes (most-significant
//!    byte first).
//!
//! Matching this exactly is what lets `rgb_to_depth_storage` (Python) decode a
//! Rust-converted depth frame correctly.

use data_daemon_shared::FrameDtype;

/// Maximum depth value (metres) the canonical storage encoding represents.
/// Must match `neuracore.core.utils.depth_utils.MAX_DEPTH` — this Rust
/// conversion and the Python decoder are two implementations of the same wire
/// contract, so a drift here would silently corrupt every depth recording's
/// decoded values without either side raising an error.
const MAX_DEPTH: f32 = 10.0;

/// 24-bit maximum value depth is scaled into, as `f32`: `2^24 - 1`.
const MAX_24_BIT: f32 = 16_777_215.0;

/// Clip one decoded depth sample (metres) into `[0, MAX_DEPTH]`.
///
/// `f32::clamp` does not touch `NaN` (comparisons against `NaN` are always
/// false, so the "self" branch is taken unchanged) — left alone, a `NaN`
/// sample would propagate through the scale/floor/cast chain into a
/// platform-dependent byte pattern, which is exactly the "undefined
/// conversion" this module must not produce. The Python reference has the
/// same gap (`np.clip` propagates `NaN`, and the final `astype(np.uint8)`
/// cast of a `NaN` is itself platform/version-dependent), so there is no
/// existing canonical behaviour to match; we deliberately and
/// deterministically treat a `NaN` sample as invalid depth (zero) rather than
/// inherit that ambiguity. `+inf`/`-inf` need no special case: `clamp` already
/// pins them to `MAX_DEPTH`/`0` respectively, matching `np.clip`.
fn clip_depth_meters(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, MAX_DEPTH)
    }
}

/// Convert one clipped depth sample (metres) to its packed 24-bit storage
/// value, matching `depth_to_rgb_storage`'s `normalize -> scale -> floor`
/// chain exactly (see the module docs for why floor-once-then-split is
/// equivalent to Python's per-channel floor/mod on the un-floored float).
fn depth_meters_to_24bit(clipped_meters: f32) -> u32 {
    let normalized = clipped_meters / MAX_DEPTH;
    let scaled = normalized * MAX_24_BIT;
    scaled.floor() as u32
}

/// Decode one little-endian IEEE-754 depth sample from `chunk`, which must be
/// exactly `dtype.bytes_per_pixel()` bytes. Callers get that guarantee from
/// [`depth_to_rgb24`], which asserts the whole buffer's length once and then
/// hands out chunks via [`slice::chunks_exact`] — so this function never has
/// to re-validate a length itself. Numpy arrays are native-endian, and every
/// platform this wheel ships for (Linux x86_64, Apple Silicon macOS) is
/// little-endian.
///
/// # Panics
///
/// Panics unconditionally (in every build profile, not just debug) if
/// `dtype` is [`FrameDtype::Rgb8`]. This function only ever decodes depth
/// samples; every production call site is gated by a `match` on `dtype` that
/// structurally excludes `Rgb8` (see [`depth_to_rgb24`]), so reaching this
/// arm means an internal caller broke that contract. Returning a placeholder
/// value here would silently corrupt the converted frame instead of
/// surfacing the bug.
fn decode_meters(dtype: FrameDtype, chunk: &[u8]) -> f32 {
    match dtype {
        FrameDtype::DepthF16 => {
            let bytes: [u8; 2] = chunk
                .try_into()
                .expect("chunks_exact(2) guarantees a 2-byte chunk");
            half::f16::from_le_bytes(bytes).to_f32()
        }
        FrameDtype::DepthF32 => {
            let bytes: [u8; 4] = chunk
                .try_into()
                .expect("chunks_exact(4) guarantees a 4-byte chunk");
            f32::from_le_bytes(bytes)
        }
        FrameDtype::Rgb8 => unreachable!("decode_meters called with a non-depth dtype"),
    }
}

/// Convert one raw depth frame (`dtype` is [`FrameDtype::DepthF16`] or
/// [`FrameDtype::DepthF32`]) into packed RGB24 storage bytes, matching
/// `depth_to_rgb_storage` exactly.
///
/// Always returns exactly `width * height * 3` bytes.
///
/// # Panics
///
/// Panics if `raw.len()` is not exactly `width * height * dtype.bytes_per_pixel()`,
/// or if `width * height * bytes_per_pixel` overflows `usize`. The buffer
/// length is a **caller-enforced invariant**, not user input to validate
/// defensively here: the native boundary ([`crate::log_frame`](crate))
/// already rejects a mismatched buffer before it is ever copied, and the
/// writer re-checks the same exact length again in `submit_frame` before
/// this function ever runs — so by the time a buffer reaches here, its
/// length has already been validated twice. A mismatch reaching this point
/// means one of those upstream checks was bypassed by a bug, and that bug
/// must surface immediately rather than silently degrade to zero-depth
/// pixels for the missing samples.
pub fn depth_to_rgb24(dtype: FrameDtype, width: u32, height: u32, raw: &[u8]) -> Vec<u8> {
    debug_assert!(
        matches!(dtype, FrameDtype::DepthF16 | FrameDtype::DepthF32),
        "depth_to_rgb24 called with {dtype:?}, expected a depth dtype"
    );
    let bytes_per_pixel = dtype.bytes_per_pixel();
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .expect("width * height overflows usize");
    let expected_len = pixel_count
        .checked_mul(bytes_per_pixel)
        .expect("width * height * bytes_per_pixel overflows usize");
    assert!(
        raw.len() == expected_len,
        "depth_to_rgb24: buffer is {} bytes; expected exactly {expected_len} bytes for a \
         {width}x{height} {dtype:?} frame. The caller must validate frame size before \
         calling — this is an internal invariant violation, not user input.",
        raw.len(),
    );

    let mut rgb = Vec::with_capacity(pixel_count * 3);
    // Branch on `dtype` once, here, rather than re-deciding it for every
    // pixel: each arm below calls `decode_meters` with a compile-time
    // literal `FrameDtype` (not the runtime `dtype` binding), so the
    // function's internal `match` collapses to straight-line code once
    // inlined — there is no per-pixel dtype decision left at runtime.
    match dtype {
        FrameDtype::DepthF16 => {
            for chunk in raw.chunks_exact(bytes_per_pixel) {
                push_pixel(&mut rgb, decode_meters(FrameDtype::DepthF16, chunk));
            }
        }
        FrameDtype::DepthF32 => {
            for chunk in raw.chunks_exact(bytes_per_pixel) {
                push_pixel(&mut rgb, decode_meters(FrameDtype::DepthF32, chunk));
            }
        }
        FrameDtype::Rgb8 => unreachable!("depth_to_rgb24 called with Rgb8"),
    }
    rgb
}

/// Clip, quantize, and append one decoded depth sample's 24-bit storage
/// value to `rgb` as three bytes (R, G, B — most-significant byte first).
fn push_pixel(rgb: &mut Vec<u8>, meters: f32) {
    let value = depth_meters_to_24bit(clip_depth_meters(meters));
    rgb.push((value >> 16) as u8);
    rgb.push(((value >> 8) & 0xFF) as u8);
    rgb.push((value & 0xFF) as u8);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode one `f32` metres value as little-endian bytes for the given
    /// dtype, mirroring how a numpy array's raw buffer looks on the wire.
    fn encode_meters(dtype: FrameDtype, meters: f32) -> Vec<u8> {
        match dtype {
            FrameDtype::DepthF16 => half::f16::from_f32(meters).to_le_bytes().to_vec(),
            FrameDtype::DepthF32 => meters.to_le_bytes().to_vec(),
            FrameDtype::Rgb8 => panic!("not a depth dtype"),
        }
    }

    /// Convert a single 1x1 "frame" at `meters` and return its RGB triple.
    fn single_pixel(dtype: FrameDtype, meters: f32) -> [u8; 3] {
        let raw = encode_meters(dtype, meters);
        let rgb = depth_to_rgb24(dtype, 1, 1, &raw);
        [rgb[0], rgb[1], rgb[2]]
    }

    #[test]
    fn zero_depth_is_black() {
        for dtype in [FrameDtype::DepthF16, FrameDtype::DepthF32] {
            assert_eq!(single_pixel(dtype, 0.0), [0, 0, 0], "dtype {dtype:?}");
        }
    }

    #[test]
    fn max_depth_saturates_all_channels() {
        for dtype in [FrameDtype::DepthF16, FrameDtype::DepthF32] {
            assert_eq!(
                single_pixel(dtype, MAX_DEPTH),
                [255, 255, 255],
                "dtype {dtype:?}"
            );
        }
    }

    #[test]
    fn half_depth_matches_hand_computed_bytes() {
        // 5.0m is exactly half of MAX_DEPTH (10.0): normalized = 0.5, scaled =
        // 0.5 * (2^24 - 1) = 8_388_607.5, floor = 8_388_607 = 0x7FFFFF.
        assert_eq!(single_pixel(FrameDtype::DepthF32, 5.0), [0x7F, 0xFF, 0xFF]);
        // f16 can represent 5.0 exactly too, so both dtypes agree here.
        assert_eq!(single_pixel(FrameDtype::DepthF16, 5.0), [0x7F, 0xFF, 0xFF]);
    }

    #[test]
    fn negative_depth_clips_to_zero() {
        for dtype in [FrameDtype::DepthF16, FrameDtype::DepthF32] {
            assert_eq!(single_pixel(dtype, -3.0), [0, 0, 0], "dtype {dtype:?}");
        }
    }

    #[test]
    fn above_max_depth_clips_to_saturation() {
        for dtype in [FrameDtype::DepthF16, FrameDtype::DepthF32] {
            assert_eq!(
                single_pixel(dtype, 4_000.0),
                [255, 255, 255],
                "dtype {dtype:?}"
            );
        }
    }

    #[test]
    fn nan_is_treated_as_zero_depth_deterministically() {
        // Python's own NaN -> uint8 cast is platform/version-dependent, so
        // there is nothing canonical to match; this pins our deliberate,
        // deterministic choice (never a panic, never UB).
        assert_eq!(single_pixel(FrameDtype::DepthF32, f32::NAN), [0, 0, 0]);
        assert_eq!(
            single_pixel(FrameDtype::DepthF16, half::f16::NAN.to_f32()),
            [0, 0, 0]
        );
    }

    #[test]
    fn positive_infinity_saturates_negative_infinity_clips_to_zero() {
        assert_eq!(
            single_pixel(FrameDtype::DepthF32, f32::INFINITY),
            [255, 255, 255]
        );
        assert_eq!(
            single_pixel(FrameDtype::DepthF32, f32::NEG_INFINITY),
            [0, 0, 0]
        );
    }

    #[test]
    fn output_is_exactly_width_times_height_times_three_bytes() {
        let raw = vec![0u8; 4 * 4 * 4]; // f32 buffer, all-zero depth
        let rgb = depth_to_rgb24(FrameDtype::DepthF32, 4, 4, &raw);
        assert_eq!(rgb.len(), 4 * 4 * 3);
    }

    #[test]
    #[should_panic(expected = "expected exactly 64 bytes")]
    fn truncated_buffer_panics_loudly() {
        // Half the bytes a 4x4 f32 frame needs. The buffer-length invariant
        // is caller-enforced (see `depth_to_rgb24`'s doc comment), so a
        // mismatch here means an internal caller broke that contract — it
        // must panic immediately, not silently decode the missing samples
        // as zero depth.
        let full_len = 4 * 4 * 4;
        let raw = vec![0xFFu8; full_len / 2];
        let _ = depth_to_rgb24(FrameDtype::DepthF32, 4, 4, &raw);
    }

    #[test]
    #[should_panic(expected = "expected exactly 8 bytes")]
    fn empty_buffer_panics_loudly() {
        let _ = depth_to_rgb24(FrameDtype::DepthF16, 2, 2, &[]);
    }

    #[test]
    #[should_panic(expected = "width * height * bytes_per_pixel overflows usize")]
    fn overflowing_dimensions_panic_instead_of_wrapping() {
        let _ = depth_to_rgb24(FrameDtype::DepthF32, u32::MAX, u32::MAX, &[]);
    }

    #[test]
    #[should_panic(expected = "decode_meters called with a non-depth dtype")]
    fn decode_meters_panics_on_rgb8() {
        // Calls `decode_meters` directly — not through `depth_to_rgb24` — so
        // this exercises `decode_meters`'s own unconditional panic rather
        // than `depth_to_rgb24`'s separate (debug-only) outer guard. Rgb8's
        // `bytes_per_pixel()` is 3, so a 3-byte chunk is the "right shape"
        // input for the arm under test.
        let _ = decode_meters(FrameDtype::Rgb8, &[0u8; 3]);
    }

    #[test]
    fn f16_and_f32_agree_on_a_representative_mid_range_value() {
        // 3.25m is exactly representable in both f16 and f32, so both dtypes
        // must produce identical bytes for it.
        assert_eq!(
            single_pixel(FrameDtype::DepthF16, 3.25),
            single_pixel(FrameDtype::DepthF32, 3.25)
        );
    }

    #[test]
    fn multi_pixel_frame_decodes_each_pixel_independently() {
        // Two f32 pixels: 0.0m then MAX_DEPTH — proves pixel offsets advance
        // correctly (not just the single-pixel path).
        let mut raw = Vec::new();
        raw.extend_from_slice(&0.0f32.to_le_bytes());
        raw.extend_from_slice(&MAX_DEPTH.to_le_bytes());
        let rgb = depth_to_rgb24(FrameDtype::DepthF32, 2, 1, &raw);
        assert_eq!(&rgb[0..3], &[0, 0, 0]);
        assert_eq!(&rgb[3..6], &[255, 255, 255]);
    }
}
