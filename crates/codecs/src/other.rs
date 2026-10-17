//! GIF (first frame) and BMP via the `image` crate's pure-Rust decoders.

use crate::convert::{Buf, Meta, Model, Raw, check_budget, finish, try_filled};
use crate::{DecodeOptions, Decoded, Error, Format, HEADER_CHECK_PIXELS, Result};
use image::{DynamicImage, ImageDecoder};

/// The `image` crate's decoder for `format`, its sample buffer checked against `opts` before it is
/// allocated: a GIF's buffer is its whole logical screen (RGBA), whatever the size of its frames.
fn decoder<'a>(bytes: &'a [u8], format: Format, opts: &DecodeOptions) -> Result<Box<dyn ImageDecoder + 'a>> {
    let err = |e: image::ImageError| Error::Malformed(format, e.to_string());
    let cursor = std::io::Cursor::new(bytes);
    let d: Box<dyn ImageDecoder> = match format {
        Format::Gif => Box::new(image::codecs::gif::GifDecoder::new(cursor).map_err(err)?),
        Format::Bmp => Box::new(image::codecs::bmp::BmpDecoder::new(cursor).map_err(err)?),
        _ => return Err(Error::Unsupported(format, "not handled by this decoder")),
    };
    let (w, h) = d.dimensions();
    check_budget(format, w as u64, h as u64, u64::from(d.color_type().bytes_per_pixel()), opts)?;
    Ok(d)
}

pub(crate) fn decode(bytes: &[u8], format: Format, opts: &DecodeOptions) -> Result<Decoded> {
    let err = |e: image::ImageError| Error::Malformed(format, e.to_string());
    let mut d = decoder(bytes, format, opts)?;
    let icc = d.icc_profile().ok().flatten();
    let img = DynamicImage::from_decoder(d).map_err(err)?;
    let (w, h) = (img.width(), img.height());
    let alpha = img.color().has_alpha();
    let gray = !img.color().has_color();
    let (model, alpha, buf) = match (gray, alpha) {
        (true, false) => (Model::Gray, false, img.into_luma8().into_raw()),
        (true, true) => (Model::Gray, true, img.into_luma_alpha8().into_raw()),
        (false, false) => (Model::Rgb, false, img.into_rgb8().into_raw()),
        (false, true) => (Model::Rgb, true, img.into_rgba8().into_raw()),
    };
    let raw = Raw { width: w as usize, height: h as usize, model, alpha, premultiplied: false, buf: Buf::U8(buf), bit_depth: 8 };
    finish(format, raw, Meta { icc, ..Default::default() }, (w, h), opts)
}

/// Stored dimensions from the GIF logical screen or the BMP info header (orientation 1: neither
/// carries one), refused as a decode at the default limits would refuse them. Up to
/// [`HEADER_CHECK_PIXELS`] the pixel data is read too, into a scratch buffer, so a damaged or
/// truncated small file is refused as its decode would be; the colour conversion a decode adds
/// can't fail.
pub(crate) fn header(bytes: &[u8], format: Format) -> Result<(u32, u32, u16)> {
    let d = decoder(bytes, format, &DecodeOptions::default())?;
    let (w, h) = d.dimensions();
    if u64::from(w) * u64::from(h) <= HEADER_CHECK_PIXELS {
        let len = usize::try_from(d.total_bytes()).map_err(|_| Error::TooLarge(w.into(), h.into()))?;
        let mut scratch = try_filled(len, 0u8, (w as usize, h as usize))?;
        d.read_image(&mut scratch).map_err(|e| Error::Malformed(format, e.to_string()))?;
    }
    Ok((w, h, 1))
}
