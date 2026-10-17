//! Reading the pixel data of a TIFF IFD (strips or tiles; uncompressed, lossless/lossy JPEG, Deflate, JPEG XL), in parallel.

use crate::unpack::*;
use crate::{MAX_SAMPLES, RawData, RawError, Result, ljpeg};
use lightcraft_tiff::image::{Chunk, ImageInfo, chunk_bytes};
use lightcraft_tiff::{ByteOrder, tags::compression as comp};
use rayon::prelude::*;

/// Bit packing for uncompressed integer data with bit depths other than 8/16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // vendor decoders use the other variants
pub enum Packing {
    /// TIFF standard: MSB-first, rows start on byte boundaries.
    Msb,
    /// LSB-first little-endian bit stream (some vendor raws), rows byte aligned.
    Lsb,
    /// Each sample stored in 16 bits in file byte order, regardless of `bits`.
    Word16,
    /// MSB-first bit stream inside 32-bit words stored in file byte order ([`unpack_words32_msb`]); rows start on a
    /// word boundary.
    Words32Msb,
}

pub(crate) enum ChunkPx {
    U16(Vec<u16>),
    F32(Vec<f32>),
}

/// The checks [`read_image`] makes before decoding anything (sample format, size limits, data
/// present); returns the number of samples.
pub fn check_image(data: &[u8], info: &ImageInfo) -> Result<usize> {
    let (w, h) = (info.width as usize, info.height as usize);
    let cpp = info.samples_per_pixel as usize;
    let total = w.checked_mul(h).and_then(|v| v.checked_mul(cpp)).ok_or(RawError::Limit("image too large"))?;
    if total > MAX_SAMPLES || total == 0 {
        return Err(RawError::Limit("image too large"));
    }
    let bits = info.bits() as u32;
    let float = info.sample_format == 3;
    if float && !matches!(bits, 16 | 24 | 32) {
        return Err(RawError::Unsupported(format!("{bits}-bit floating point samples")));
    }
    if !float && !(1..=16).contains(&bits) {
        return Err(RawError::Unsupported(format!("{bits}-bit integer samples")));
    }
    let chunks = info.chunks(data.len() as u64);
    if chunks.is_empty() {
        return Err(RawError::Corrupt("no image data chunks".into()));
    }
    // every visible sample needs a chunk: a short offsets table would leave zero-filled (black) regions
    let (across, down) = info.grid();
    let planes = if info.planar == 2 { cpp } else { 1 };
    let expected = (across as usize).saturating_mul(down as usize).saturating_mul(planes);
    if chunks.len() < expected {
        return Err(RawError::Corrupt(format!("{} image data chunks, expected {expected}", chunks.len())));
    }
    // Plausibility bound before allocating: no supported coding stores more than ~2000 samples per byte
    // (Deflate of constant data is the extreme), so tiny files cannot trigger huge allocations.
    let available: u64 = chunks.iter().map(|c| chunk_bytes(data, c).map_or(0, |s| s.len() as u64)).sum();
    if available.saturating_mul(2048) < total as u64 {
        return Err(RawError::Corrupt(format!("{available} bytes of image data cannot hold {total} samples")));
    }
    // Each chunk is decoded at its nominal size, which for tiles may reach far past the image: a one-row
    // image of 32768-row tiles asks 2 GiB a tile from a few bytes of data (issue #2117).
    let nominal = chunks.iter().map(|c| chunk_samples(info, c)).fold(0usize, usize::saturating_add);
    if nominal > nominal_budget(total) {
        return Err(RawError::Limit("tiles far larger than the image"));
    }
    Ok(total)
}

/// Most samples the nominal chunks of an image of `total` samples may hold together. A tile grid
/// overhangs the image by less than one tile a side, so at most twice the image each way once the
/// image is at least a tile wide and tall; the slack covers images smaller than one tile (a
/// 4096 × 4096 tile of one sample).
fn nominal_budget(total: usize) -> usize {
    total.saturating_mul(4).saturating_add(1 << 24)
}

/// Samples in chunk `c` at its nominal size.
fn chunk_samples(info: &ImageInfo, c: &Chunk) -> usize {
    let ccpp = if info.planar == 2 { 1 } else { info.samples_per_pixel as usize };
    (c.width as usize).saturating_mul(c.height as usize).saturating_mul(ccpp)
}

/// Most nominal samples decoded before they are placed (128 MiB of 16-bit samples): collected all
/// at once, a file's whole nominal grid was held next to the output (issue #2117).
const BATCH_SAMPLES: usize = 1 << 26;

/// Decode `chunks` in parallel, in batches of at most [`BATCH_SAMPLES`] nominal samples (but at
/// least one chunk a thread), handing each result to `place` in order; an error from `place` stops
/// the run.
pub(crate) fn decode_in_batches<T: Send>(
    info: &ImageInfo,
    chunks: &[Chunk],
    decode: impl Fn(&Chunk) -> T + Sync,
    place: impl FnMut(&Chunk, T) -> Result<()>,
) -> Result<()> {
    let per_chunk = chunks.iter().map(|c| chunk_samples(info, c)).max().unwrap_or(1).max(1);
    batched(chunks, (BATCH_SAMPLES / per_chunk).max(rayon::current_num_threads()), decode, place)
}

/// [`decode_in_batches`] with `n` chunks a batch.
fn batched<T: Send>(chunks: &[Chunk], n: usize, decode: impl Fn(&Chunk) -> T + Sync, mut place: impl FnMut(&Chunk, T) -> Result<()>) -> Result<()> {
    for batch in chunks.chunks(n.max(1)) {
        let decoded: Vec<T> = batch.par_iter().map(&decode).collect();
        for (c, d) in batch.iter().zip(decoded) {
            place(c, d)?;
        }
    }
    Ok(())
}

/// Read the samples of `info` in `mode`: [`Mode::Header`] only checks them ([`check_image`]) and
/// returns no samples (floating-point data as an empty `F32`).
pub(crate) fn read_image_in(mode: crate::Mode, data: &[u8], info: &ImageInfo, order: ByteOrder, packing: Packing) -> Result<RawData> {
    match mode {
        crate::Mode::Full => read_image(data, info, order, packing),
        crate::Mode::Header => {
            check_image(data, info)?;
            // a lossless-JPEG layout the decoder rejects (subsampled components, e.g. Sony's
            // lossless M/S sizes) must fail here too, as the full decode will; likewise a JPEG XL
            // chunk whose headers don't match the IFD
            let first = info.chunks(data.len() as u64).first().copied();
            if let Some(c) = first
                && let Some(src) = chunk_bytes(data, &c)
            {
                match info.compression {
                    7 => {
                        ljpeg::frame_info(src)?;
                    }
                    comp::JPEG_XL => {
                        let planar = info.planar == 2 && info.samples_per_pixel > 1;
                        let cpp = if planar { 1 } else { info.samples_per_pixel as usize };
                        let (cw, ch, vw, vh) = chunk_dims(info, &c);
                        jxl_chunk_check(src, cw, ch, vw, vh, cpp, info.bits() as u32, info.sample_format == 3)?;
                    }
                    _ => {}
                }
            }
            Ok(if info.sample_format == 3 { RawData::F32(Vec::new()) } else { RawData::U16(Vec::new()) })
        }
    }
}

/// Decode all chunks of `info` into one buffer of `width × height × cpp` samples.
pub fn read_image(data: &[u8], info: &ImageInfo, order: ByteOrder, packing: Packing) -> Result<RawData> {
    let (w, h) = (info.width as usize, info.height as usize);
    let total = check_image(data, info)?;
    let cpp = info.samples_per_pixel as usize;
    let bits = info.bits() as u32;
    let float = info.sample_format == 3;
    let chunks = info.chunks(data.len() as u64);
    let planar = info.planar == 2 && cpp > 1;
    let ccpp = if planar { 1 } else { cpp };
    let mut out_u16 = if float { Vec::new() } else { vec![0u16; total] };
    let mut out_f32 = if float { vec![0f32; total] } else { Vec::new() };
    let decode = |c: &Chunk| decode_chunk(data, info, c, order, packing, ccpp, bits, float);
    decode_in_batches(info, &chunks, decode, |c, r| {
        // a chunk that cannot be decoded is not a partial image: the file stays preview-only
        let px = r?;
        let (cw, ch) = (c.width as usize, c.height as usize);
        let (x0, y0) = (c.x as usize, c.y as usize);
        if x0 >= w || y0 >= h {
            return Ok(());
        }
        let copy_w = cw.min(w - x0);
        let copy_h = ch.min(h - y0);
        for y in 0..copy_h {
            for x in 0..copy_w {
                for s in 0..ccpp {
                    let src = (y * cw + x) * ccpp + s;
                    let dst_s = if planar { c.plane as usize } else { s };
                    if dst_s >= cpp {
                        continue;
                    }
                    let dst = ((y0 + y) * w + x0 + x) * cpp + dst_s;
                    match &px {
                        ChunkPx::U16(v) => {
                            if let (Some(o), Some(&s)) = (out_u16.get_mut(dst), v.get(src)) {
                                *o = s;
                            }
                        }
                        ChunkPx::F32(v) => {
                            if let (Some(o), Some(&s)) = (out_f32.get_mut(dst), v.get(src)) {
                                *o = s;
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    })?;
    Ok(if float { RawData::F32(out_f32) } else { RawData::U16(out_u16) })
}

#[allow(clippy::too_many_arguments)]
fn decode_chunk(data: &[u8], info: &ImageInfo, c: &Chunk, order: ByteOrder, packing: Packing, cpp: usize, bits: u32, float: bool) -> Result<ChunkPx> {
    let src = chunk_bytes(data, c).ok_or_else(|| RawError::Corrupt("chunk offset past end of file".into()))?;
    let (cw, ch, vw, vh) = chunk_dims(info, c);
    let n = cw.checked_mul(ch).and_then(|v| v.checked_mul(cpp)).ok_or(RawError::Limit("chunk too large"))?;
    if n > MAX_SAMPLES {
        return Err(RawError::Limit("chunk too large"));
    }
    match info.compression {
        comp::NONE => {
            // uncompressed: missing rows are filled with zeros, but a chunk holding a small fraction of what its
            // samples need is no picture: refused before the nominal chunk is allocated (issue #1512)
            let need = (n as u64).saturating_mul(u64::from(bits)) / 8;
            if (src.len() as u64).saturating_mul(64) < need {
                return Err(RawError::Corrupt(format!("an uncompressed chunk of {n} samples has only {} bytes", src.len())));
            }
            unpack_chunk(src, order, packing, cw, ch, cpp, bits, float, 1)
        }
        comp::JPEG => {
            let f = ljpeg::decode(src, n.saturating_mul(2).max(1 << 16))?;
            if f.data.len() < n {
                // Some writers encode edge tiles smaller than nominal: accept a frame that covers the rows inside the image.
                if f.data.is_empty() || f.data.len() % (cw * cpp) != 0 || f.data.len() / (cw * cpp) < vh {
                    return Err(RawError::Corrupt(format!("lossless JPEG tile has {} samples, expected {n}", f.data.len())));
                }
            }
            let mut v = f.data;
            v.resize(n, 0);
            Ok(ChunkPx::U16(v))
        }
        comp::LOSSY_JPEG => {
            // lossy DNG: each chunk is a baseline (DCT) JPEG of 8-bit samples
            // at most the nominal tile (DNG tiles are multiples of 16, as JPEG blocks are)
            let max = (cw.next_multiple_of(16), ch.next_multiple_of(16));
            let (px, jw, jh, jc) = lossy_jpeg(src, cpp, max, max.0.saturating_mul(max.1))?;
            let mut v = vec![0u16; n];
            for y in 0..ch.min(jh) {
                for x in 0..cw.min(jw) {
                    for k in 0..cpp {
                        v[(y * cw + x) * cpp + k] = px[(y * jw + x) * jc + k.min(jc - 1)] as u16;
                    }
                }
            }
            Ok(ChunkPx::U16(v))
        }
        comp::ADOBE_DEFLATE | comp::DEFLATE => {
            let bytes_per = if float { bits.div_ceil(8) as usize } else { 0 };
            let row_bytes = if float { cw * cpp * bytes_per } else { (cw * cpp * bits as usize).div_ceil(8) };
            let raw = inflate(src, row_bytes * ch)?;
            unpack_chunk(&raw, order, packing, cw, ch, cpp, bits, float, info.predictor)
        }
        comp::JPEG_XL => jxl_chunk(src, cw, ch, vw, vh, cpp, bits, float),
        other => Err(RawError::Unsupported(format!("TIFF compression {other}"))),
    }
}

/// A chunk's nominal size and the part of it inside the image (smaller for edge tiles).
fn chunk_dims(info: &ImageInfo, c: &Chunk) -> (usize, usize, usize, usize) {
    let (cw, ch) = (c.width as usize, c.height as usize);
    let vw = cw.min(info.width.saturating_sub(c.x) as usize);
    let vh = ch.min(info.height.saturating_sub(c.y) as usize);
    (cw, ch, vw, vh)
}

#[cfg(feature = "jxl")]
use crate::jxl::{check as jxl_chunk_check, decode as jxl_chunk};

#[cfg(not(feature = "jxl"))]
#[allow(clippy::too_many_arguments)]
fn jxl_chunk(_: &[u8], _: usize, _: usize, _: usize, _: usize, _: usize, _: u32, _: bool) -> Result<ChunkPx> {
    Err(RawError::Unsupported("JPEG XL DNG (built without the `jxl` feature)".into()))
}

#[cfg(not(feature = "jxl"))]
#[allow(clippy::too_many_arguments)]
fn jxl_chunk_check(src: &[u8], cw: usize, ch: usize, vw: usize, vh: usize, cpp: usize, bits: u32, float: bool) -> Result<()> {
    jxl_chunk(src, cw, ch, vw, vh, cpp, bits, float).map(|_| ())
}

/// Decode one baseline JPEG chunk to 8-bit samples: (samples, width, height, channels). A frame
/// wider or taller than `max`, or of more than `max_px` pixels, is refused from its header before
/// zune-jpeg allocates the output it declares: up to 65535² × 3 bytes (12.9 GB) from a few
/// hundred, or on wasm32 a size that overflows `usize` and panics inside zune-jpeg (issue #2118).
pub(crate) fn lossy_jpeg(src: &[u8], cpp: usize, max: (usize, usize), max_px: usize) -> Result<(Vec<u8>, usize, usize, usize)> {
    use zune_core::bytestream::ZCursor;
    use zune_core::colorspace::ColorSpace;
    use zune_core::options::DecoderOptions;
    let cs = if cpp == 1 { ColorSpace::Luma } else { ColorSpace::RGB };
    let (max_w, max_h) = (max.0.clamp(1, u16::MAX as usize), max.1.clamp(1, u16::MAX as usize));
    let opts = DecoderOptions::default().set_max_width(max_w).set_max_height(max_h).set_strict_mode(false).jpeg_set_out_colorspace(cs);
    let mut d = zune_jpeg::JpegDecoder::new_with_options(ZCursor::new(src), opts);
    let corrupt = |e: zune_jpeg::errors::DecodeErrors| RawError::Corrupt(format!("lossy JPEG tile: {e}"));
    d.decode_headers().map_err(corrupt)?;
    let info = d.info().ok_or_else(|| RawError::Corrupt("lossy JPEG tile without a header".into()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    if w > max_w || h > max_h || w.saturating_mul(h) > max_px || d.output_buffer_size().is_none() {
        return Err(RawError::Limit("lossy JPEG frame larger than expected"));
    }
    let px = d.decode().map_err(corrupt)?;
    let c = if cpp == 1 { 1 } else { 3 };
    if w == 0 || h == 0 || px.len() < w * h * c {
        return Err(RawError::Corrupt("lossy JPEG tile: short output".into()));
    }
    Ok((px, w, h, c))
}

#[allow(clippy::too_many_arguments)]
fn unpack_chunk(
    src: &[u8],
    order: ByteOrder,
    packing: Packing,
    cw: usize,
    ch: usize,
    cpp: usize,
    bits: u32,
    float: bool,
    predictor: u16,
) -> Result<ChunkPx> {
    let row_n = cw * cpp;
    let (factor, fp) = match predictor {
        1 => (0, false),
        2 => (1, false),
        3 => (1, true),
        34892 => (2, false),
        34893 => (4, false),
        34894 => (2, true),
        34895 => (4, true),
        p => return Err(RawError::Unsupported(format!("predictor {p}"))),
    };
    if float {
        let bp = bits.div_ceil(8) as usize;
        let row_bytes = row_n * bp;
        let mut out = vec![0f32; row_n * ch];
        let mut rowbuf = vec![0u8; row_bytes];
        for y in 0..ch {
            let s = src.get(y * row_bytes..).unwrap_or(&[]);
            let len = s.len().min(row_bytes);
            rowbuf[..len].copy_from_slice(&s[..len]);
            rowbuf[len..].fill(0);
            let big = if fp {
                undo_float_predictor(&mut rowbuf, row_n, bp, cpp * factor);
                true
            } else {
                if factor > 0 {
                    return Err(RawError::Unsupported("integer predictor on float data".into()));
                }
                order == ByteOrder::Big
            };
            for (i, o) in out[y * row_n..(y + 1) * row_n].iter_mut().enumerate() {
                let b = &rowbuf[i * bp..(i + 1) * bp];
                *o = match (bp, big) {
                    (2, true) => f16_to_f32(u16::from_be_bytes([b[0], b[1]])),
                    (2, false) => f16_to_f32(u16::from_le_bytes([b[0], b[1]])),
                    (3, true) => f24_to_f32(u32::from_be_bytes([0, b[0], b[1], b[2]])),
                    (3, false) => f24_to_f32(u32::from_le_bytes([b[0], b[1], b[2], 0])),
                    (4, true) => f32::from_be_bytes([b[0], b[1], b[2], b[3]]),
                    _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                };
            }
        }
        return Ok(ChunkPx::F32(out));
    }
    if fp {
        return Err(RawError::Unsupported("floating-point predictor on integer data".into()));
    }
    let mut out = vec![0u16; row_n * ch];
    let (row_bytes, kind) = match (bits, packing) {
        (8, _) => (row_n, 0),
        (16, _) | (_, Packing::Word16) => (row_n * 2, 1),
        (_, Packing::Msb) => ((row_n * bits as usize).div_ceil(8), 2),
        (_, Packing::Lsb) => ((row_n * bits as usize).div_ceil(8), 3),
        (_, Packing::Words32Msb) => ((row_n * bits as usize).div_ceil(32) * 4, 4),
    };
    for y in 0..ch {
        let s = src.get(y * row_bytes..).unwrap_or(&[]);
        let s = &s[..s.len().min(row_bytes)];
        let row = &mut out[y * row_n..(y + 1) * row_n];
        match kind {
            0 => row.iter_mut().zip(s).for_each(|(o, &b)| *o = b as u16),
            1 => read_u16s(s, order, row),
            2 => unpack_msb(s, bits, row),
            4 => unpack_words32_msb(s, order, bits, row),
            _ => unpack_lsb(s, bits, row),
        }
        if factor > 0 {
            if bits == 8 {
                for i in cpp * factor..row.len() {
                    row[i] = (row[i] as u8).wrapping_add(row[i - cpp * factor] as u8) as u16;
                }
            } else {
                undo_diff_u16(row, cpp * factor);
            }
        }
    }
    Ok(ChunkPx::U16(out))
}

#[cfg(test)]
mod lossy_tests {
    use super::*;
    use lightcraft_tiff::image::Layout;

    /// A lossy-DNG-style image: two 16×8 tiles, each a baseline JPEG, the right one cut short
    /// by the image edge.
    #[test]
    fn lossy_jpeg_tiles_decode() {
        let (tw, th) = (16usize, 8usize);
        let tile = |shade: u8| {
            let px: Vec<u8> = (0..tw * th).flat_map(|i| [shade, (i % tw * 15) as u8, 200]).collect();
            let mut out = Vec::new();
            jpeg_encoder::Encoder::new(&mut out, 100).encode(&px, tw as u16, th as u16, jpeg_encoder::ColorType::Rgb).unwrap();
            out
        };
        let (a, b) = (tile(40), tile(220));
        let mut file = vec![0u8; 8];
        let oa = file.len() as u64;
        file.extend_from_slice(&a);
        let ob = file.len() as u64;
        file.extend_from_slice(&b);
        let info = ImageInfo {
            width: 24,
            height: 8,
            bits_per_sample: vec![8, 8, 8],
            samples_per_pixel: 3,
            compression: comp::LOSSY_JPEG,
            photometric: 34892,
            planar: 1,
            predictor: 1,
            sample_format: 1,
            new_subfile_type: 0,
            layout: Layout::Tiles { tile_width: tw as u32, tile_height: th as u32 },
            offsets: vec![oa, ob],
            byte_counts: vec![a.len() as u64, b.len() as u64],
        };
        let RawData::U16(v) = read_image(&file, &info, ByteOrder::Little, Packing::Msb).unwrap() else { panic!("integer samples") };
        assert_eq!(v.len(), 24 * 8 * 3);
        let px = |x: usize, y: usize| &v[(y * 24 + x) * 3..(y * 24 + x) * 3 + 3];
        assert!((px(2, 3)[0] as i32 - 40).abs() <= 3 && (px(2, 3)[2] as i32 - 200).abs() <= 3, "{:?}", px(2, 3));
        assert!((px(20, 3)[0] as i32 - 220).abs() <= 3, "second tile: {:?}", px(20, 3));
        assert!((px(10, 5)[1] as i32 - 150).abs() <= 6, "green ramp: {:?}", px(10, 5));
    }

    fn strip_info(compression: u16, offsets: Vec<u64>, byte_counts: Vec<u64>) -> ImageInfo {
        ImageInfo {
            width: 4,
            height: 4,
            bits_per_sample: vec![8],
            samples_per_pixel: 1,
            compression,
            photometric: 1,
            planar: 1,
            predictor: 1,
            sample_format: 1,
            new_subfile_type: 0,
            layout: Layout::Strips { rows_per_strip: 2 },
            offsets,
            byte_counts,
        }
    }

    /// A missing strip or a strip that fails to decode is an error, not a black band in a "successful" image.
    #[test]
    fn missing_or_failed_chunks_are_errors() {
        let file = vec![9u8; 8 + 8];
        // both strips present
        let ok = strip_info(comp::NONE, vec![0, 8], vec![8, 8]);
        assert!(read_image(&file, &ok, ByteOrder::Little, Packing::Msb).is_ok());
        // the second strip has no offset
        let missing = strip_info(comp::NONE, vec![0], vec![8]);
        assert!(read_image(&file, &missing, ByteOrder::Little, Packing::Msb).is_err());
        // the header-only read rejects it too, as the full decode will
        assert!(read_image_in(crate::Mode::Header, &file, &missing, ByteOrder::Little, Packing::Msb).is_err());
        // the second strip starts past the end of the file
        let past_end = strip_info(comp::NONE, vec![0, 100], vec![8, 8]);
        assert!(read_image(&file, &past_end, ByteOrder::Little, Packing::Msb).is_err());
    }

    /// A small image whose uncompressed tile claims to be enormous (16384 x 16384 nominal, a few bytes of data) is
    /// refused before that tile is allocated (issue #1512); an edge tile stored at its full nominal size decodes.
    #[test]
    fn a_huge_nominal_uncompressed_tile_with_little_data_is_refused() {
        let info = |tile: u32, bytes: u64| ImageInfo {
            width: 8,
            height: 8,
            bits_per_sample: vec![16],
            samples_per_pixel: 1,
            compression: comp::NONE,
            photometric: 32803,
            planar: 1,
            predictor: 1,
            sample_format: 1,
            new_subfile_type: 0,
            layout: Layout::Tiles { tile_width: tile, tile_height: tile },
            offsets: vec![0],
            byte_counts: vec![bytes],
        };
        let small = vec![1u8; 2 * 16 * 16];
        assert!(read_image(&small, &info(16, small.len() as u64), ByteOrder::Little, Packing::Msb).is_ok());
        let data = vec![1u8; 4096];
        // refused by the aggregate tile budget now (issue #2117), before the data check of #1512
        let r = read_image(&data, &info(16384, data.len() as u64), ByteOrder::Little, Packing::Msb);
        assert!(matches!(r, Err(RawError::Corrupt(_) | RawError::Limit(_))), "{r:?}");
    }

    fn tiled(width: u32, height: u32, tile: (u32, u32), compression: u16, chunks: usize, bytes: u64) -> ImageInfo {
        ImageInfo {
            width,
            height,
            bits_per_sample: vec![8],
            samples_per_pixel: 1,
            compression,
            photometric: 32803,
            planar: 1,
            predictor: 1,
            sample_format: 1,
            new_subfile_type: 0,
            layout: Layout::Tiles { tile_width: tile.0, tile_height: tile.1 },
            offsets: vec![0; chunks],
            byte_counts: vec![bytes; chunks],
        }
    }

    /// A one-row image of 64 nominal 32768 × 32768 lossy-JPEG tiles, all pointing at the same 16 KB
    /// of data, passes the per-chunk checks; together they asked 64 × 2 GiB before a byte was decoded.
    /// Refused by the aggregate budget, in the header read as in the decode (issue #2117).
    #[test]
    fn nominal_tiles_far_past_the_image_are_refused() {
        let data = vec![0u8; 16384];
        let bomb = tiled(64 * 32768, 1, (32768, 32768), comp::LOSSY_JPEG, 64, 16384);
        assert_eq!(read_image(&data, &bomb, ByteOrder::Little, Packing::Msb), Err(RawError::Limit("tiles far larger than the image")));
        assert!(matches!(read_image_in(crate::Mode::Header, &data, &bomb, ByteOrder::Little, Packing::Msb), Err(RawError::Limit(_))));
        // ordinary overhangs pass: one tile larger than a small image, and a 2 × 2 grid barely past 256
        assert!(check_image(&data, &tiled(300, 300, (4096, 4096), comp::NONE, 1, 16384)).is_ok());
        assert!(check_image(&data, &tiled(300, 300, (256, 256), comp::NONE, 4, 16384)).is_ok());
        assert!(check_image(&data, &tiled(4097, 4097, (4096, 4096), comp::NONE, 4, 16384)).is_ok());
    }

    /// Batches are placed in chunk order, each chunk once, and an error from `place` stops the run.
    #[test]
    fn batches_place_every_chunk_in_order() {
        let info = tiled(64, 64, (16, 16), comp::NONE, 16, 1);
        let chunks = info.chunks(1);
        for n in [1, 3, 16, 100] {
            let mut seen = Vec::new();
            batched(
                &chunks,
                n,
                |c| c.index * 10,
                |c, v| {
                    assert_eq!(v, c.index * 10);
                    seen.push(c.index);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(seen, (0..16).collect::<Vec<_>>(), "batches of {n}");
        }
        let mut placed = 0;
        let r = batched(
            &chunks,
            4,
            |c| c.index,
            |_, v| {
                placed += 1;
                if v == 5 { Err(RawError::Corrupt("stop".into())) } else { Ok(()) }
            },
        );
        assert!(r.is_err());
        assert_eq!(placed, 6);
    }

    /// A lossy-JPEG frame whose SOF declares 65535 × 65535 is refused from its header, before
    /// zune-jpeg allocates the 12.9 GB output it declares (issue #2118): as a DNG tile and as the
    /// embedded preview the white-balance estimate decodes.
    #[test]
    fn lossy_jpeg_larger_than_expected_is_refused_from_its_header() {
        let px: Vec<u8> = (0..16 * 8 * 3).map(|i| (i * 7) as u8).collect();
        let mut jpeg = Vec::new();
        jpeg_encoder::Encoder::new(&mut jpeg, 90).encode(&px, 16, 8, jpeg_encoder::ColorType::Rgb).unwrap();
        assert!(lossy_jpeg(&jpeg, 3, (16, 16), 256).is_ok());
        assert!(matches!(lossy_jpeg(&jpeg, 3, (8, 16), 256), Err(_)), "wider than the tile");
        let sof = jpeg.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        jpeg[sof + 5..sof + 9].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
        assert!(lossy_jpeg(&jpeg, 3, (16, 16), 256).is_err());
        assert_eq!(lossy_jpeg(&jpeg, 3, (1 << 16, 1 << 16), 1 << 24), Err(RawError::Limit("lossy JPEG frame larger than expected")));
    }

    /// A lossless-JPEG tile may be coded only as tall as the part inside the image, never shorter.
    #[test]
    fn short_lossless_jpeg_tiles_are_errors() {
        let tile = |rows: usize| ljpeg::encode(&vec![300u16; 8 * rows], 8, rows, 1, 12, 1, 0);
        let decode = |height: u32, rows: usize| {
            let t = tile(rows);
            let info = ImageInfo {
                width: 8,
                height,
                bits_per_sample: vec![12],
                samples_per_pixel: 1,
                compression: comp::JPEG,
                photometric: 32803,
                planar: 1,
                predictor: 1,
                sample_format: 1,
                new_subfile_type: 0,
                layout: Layout::Tiles { tile_width: 8, tile_height: 8 },
                offsets: vec![0],
                byte_counts: vec![t.len() as u64],
            };
            read_image(&t, &info, ByteOrder::Little, Packing::Msb)
        };
        // an edge tile coded at its in-image height, and a full tile
        assert_eq!(decode(6, 6), Ok(RawData::U16(vec![300; 8 * 6])));
        assert_eq!(decode(8, 8), Ok(RawData::U16(vec![300; 8 * 8])));
        // half of the visible rows are missing
        assert!(decode(8, 4).is_err());
    }
}
