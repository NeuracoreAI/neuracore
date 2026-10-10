//! Depth-to-gray16 storage conversion for the producer's video path.
//!
//! Every depth frame is stored as uint16 grey samples in sensor units. The
//! producer logs little-endian samples and the output is big-endian, the byte
//! order 16-bit PNG samples use.

use data_daemon_shared::FrameDtype;

/// Convert one raw depth frame into big-endian gray16 samples.
///
/// `raw` holds little-endian uint16 sensor units. Always returns exactly
/// `width * height * 2` bytes.
///
/// # Panics
///
/// Panics if `dtype` is [`FrameDtype::Rgb8`], or if `raw.len()` is not exactly
/// `width * height * dtype.bytes_per_pixel()`. Both are invariants the native
/// boundary and the writer's `submit_frame` check before a frame gets here.
pub fn depth_to_gray16_be(dtype: FrameDtype, width: u32, height: u32, raw: &[u8]) -> Vec<u8> {
    assert!(dtype.is_depth(), "depth_to_gray16_be called with {dtype:?}");
    let bytes_per_pixel = dtype.bytes_per_pixel();
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .expect("width * height overflows usize");
    let expected_len = pixel_count
        .checked_mul(bytes_per_pixel)
        .expect("width * height * bytes_per_pixel overflows usize");
    assert!(
        raw.len() == expected_len,
        "depth_to_gray16_be: buffer is {} bytes; expected exactly {expected_len} bytes for a \
         {width}x{height} {dtype:?} frame",
        raw.len(),
    );

    let mut gray = Vec::with_capacity(pixel_count * 2);
    match dtype {
        FrameDtype::DepthU16 { .. } => {
            for chunk in raw.as_chunks::<2>().0 {
                gray.extend_from_slice(&[chunk[1], chunk[0]]);
            }
        }
        FrameDtype::Rgb8 => unreachable!("depth_to_gray16_be called with Rgb8"),
    }
    gray
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uint16_passes_through_as_big_endian() {
        let samples: [u16; 4] = [0, 1, 0x1234, 65535];
        let raw: Vec<u8> = samples.iter().flat_map(|v| v.to_le_bytes()).collect();
        let gray = depth_to_gray16_be(FrameDtype::depth_u16(1e-4), 2, 2, &raw);
        let back: Vec<u16> = gray
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
            .collect();
        assert_eq!(back, samples);
    }

    #[test]
    #[should_panic(expected = "expected exactly")]
    fn wrong_buffer_length_panics() {
        let _ = depth_to_gray16_be(FrameDtype::depth_u16(1e-4), 2, 2, &[0u8; 4]);
    }

    #[test]
    #[should_panic(expected = "depth_to_gray16_be called with Rgb8")]
    fn rgb_dtype_panics() {
        let _ = depth_to_gray16_be(FrameDtype::Rgb8, 1, 1, &[0u8; 3]);
    }
}
