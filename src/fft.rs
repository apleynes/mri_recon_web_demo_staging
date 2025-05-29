use image::{GrayImage, ImageReader, RgbImage, RgbaImage};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use num_complex::Complex;
use ndrustfft::{self, R2cFftHandler, FftHandler};
use nshare::{self, IntoNdarray2, IntoNdarray3, IntoImageLuma};

// CPU implementations (existing)
pub fn fft2(gray_img: &ArrayView2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (ny, nx) = gray_img.dim();
    let mut fft_img = Array2::<Complex<f64>>::zeros((ny, nx));
    let mut fft_handler = FftHandler::new(nx);
    ndrustfft::ndfft_par(&gray_img, &mut fft_img, &mut fft_handler, 1);
    let mut fft_handler = FftHandler::new(ny);
    ndrustfft::ndfft_par(&fft_img.clone(), &mut fft_img, &mut fft_handler, 0);
    fft_img
}

pub fn ifft2(fft_img: &ArrayView2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (ny, nx) = fft_img.dim();
    let mut ifft_img = Array2::<Complex<f64>>::zeros((ny, nx));
    let mut ifft_handler = FftHandler::new(ny);
    ndrustfft::ndifft_par(&fft_img, &mut ifft_img, &mut ifft_handler, 0);
    let mut ifft_handler = FftHandler::new(nx);
    ndrustfft::ndifft_par(&ifft_img.clone(), &mut ifft_img, &mut ifft_handler, 1);
    ifft_img
}

// High-level API with async interface (CPU-only for now, placeholder for future GPU implementation)
pub async fn fft2_auto(gray_img: &ArrayView2<'_, Complex<f64>>) -> Array2<Complex<f64>> {
    leptos::logging::log!("Using CPU FFT (GPU FFT disabled due to WASM compatibility issues)");
    fft2(gray_img)
}

pub async fn ifft2_auto(fft_img: &ArrayView2<'_, Complex<f64>>) -> Array2<Complex<f64>> {
    leptos::logging::log!("Using CPU IFFT (GPU IFFT disabled due to WASM compatibility issues)");
    ifft2(fft_img)
}

pub fn fft2shift(img: &ArrayView2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (ny, nx) = img.dim();
    let mut shifted = Array2::zeros((ny, nx));
    let half_ny = ny / 2;
    let half_nx = nx / 2;

    for i in 0..ny {
        for j in 0..nx {
            let new_i = (i + half_ny) % ny;
            let new_j = (j + half_nx) % nx;
            shifted[[new_i, new_j]] = img[[i, j]];
        }
    }
    shifted
}

pub fn ifft2shift(img: &ArrayView2<Complex<f64>>) -> Array2<Complex<f64>> {
    let (ny, nx) = img.dim();
    let mut shifted = Array2::zeros((ny, nx));
    let half_ny = ny / 2;
    let half_nx = nx / 2;

    for i in 0..ny {
        for j in 0..nx {
            let new_i = (i + half_ny) % ny;
            let new_j = (j + half_nx) % nx;
            shifted[[new_i, new_j]] = img[[i, j]];
        }
    }
    shifted
}

fn convert_to_image_and_save(img: &Array2<u8>, filename: &str) {
    let (ny, nx) = img.dim();
    let img: GrayImage = GrayImage::from_raw(nx as u32, ny as u32, img.clone().into_iter().collect()).unwrap();
    img.save(filename).unwrap();
}
