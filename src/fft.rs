use image::{GrayImage, ImageReader, RgbImage, RgbaImage};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use num_complex::Complex;
use ndrustfft::{self, R2cFftHandler, FftHandler};
use nshare::{self, IntoNdarray2, IntoNdarray3, IntoImageLuma};


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
    // println!("fft_gray_img shape: {:?}", fft_gray_img.dim());

//     // FFTshift
//     let shifted_gray_img = fft2shift(&fft_gray_img.view());
    
//     // Display log magnitude spectrum
//     let eps = 1e-6;
//     let log_magnitude = shifted_gray_img
//         .mapv(|x| x.norm())
//         .mapv(|x| (x + eps).ln());
//     let max_val = log_magnitude.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
//     let min_val = log_magnitude.fold(f64::INFINITY, |a, &b| a.min(b));
//     println!("max_val: {}, min_val: {}", max_val, min_val);
//     let log_magnitude = log_magnitude.mapv(|x| ((x - min_val) / (max_val - min_val) * 255.0) as u8);
//     let log_magnitude_img: GrayImage = GrayImage::from_raw(log_magnitude.dim().1 as u32, 
//         log_magnitude.dim().0 as u32,
//         log_magnitude.into_iter().collect()).unwrap();
//     log_magnitude_img.save("log_magnitude.png").unwrap();

//     // Inverse FFT
    
//     ndrustfft::ndifft(&fft_gray_img.clone(), &mut fft_gray_img, &mut fft_handler, 0);
//     let mut recovered_gray_img = Array2::<f64>::zeros((ny, nx));
//     ndrustfft::ndifft_r2c(&fft_gray_img, &mut recovered_gray_img, &mut r2c_handler, 1);
//     // let recovered_gray_img = recovered_gray_img.mapv(|x| x.re);
//     // Normalize to 0-255
//     let max_val = recovered_gray_img.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
//     let min_val = recovered_gray_img.fold(f64::INFINITY, |a, &b| a.min(b));
//     println!("max_val: {}, min_val: {}", max_val, min_val);


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

// fn main() {
//     let gray_img = ImageReader::open("test_output.png")
//         .expect("Failed to open image")
//         .decode()
//         .expect("Failed to decode image")
//         .into_luma8();
//     // let gray_img = image::DynamicImage::from(img.clone()).into_luma8();
//     // let img = img.into_ndarray3();
//     gray_img.save("gray_img.png").unwrap();
//     let gray_img_array = gray_img.into_ndarray2().mapv(|x| x as f64);

//     // let mut gray_img_array = Array2::<f64>::zeros((256, 256));
//     // for i in 0..256 {
//     //     for j in 0..256 {
//     //         gray_img_array[[i, j]] = (i as f64 + j as f64) / 2.0;
//     //     }
//     // }


//     println!("gray_img_array shape: {:?}", gray_img_array.dim());

//     convert_to_image_and_save(&gray_img_array.mapv(|x| x as u8), "gray_img_array.png");

//     let (ny, nx) = gray_img_array.dim();
//     // Pad if not even
//     // if (nx % 2) == 1 {
//     //     let gray_img_array = ndarray_ndimage::pad(
//     //         &gray_img_array, 
//     //         &[[0, 0], [0, 1]], 
//     //         PadMode::Constant(0.0));
//     // }
//     // if (ny % 2) == 1 {
//     // let gray_img_array = ndarray_ndimage::pad(
//     //     &gray_img_array, 
//     //     &[[0, 1], [0, 0]], 
//     //     PadMode::Constant(0.0));
//     // }
//     let (ny, nx) = gray_img_array.dim();
//     println!("gray_img_array shape: {:?}", gray_img_array.dim());
//     // let mut fft_gray_img = Array2::<Complex<f64>>::zeros((ny, (nx / 2) + 1));
//     let mut fft_gray_img = Array2::<Complex<f64>>::zeros((ny, nx / 2 + 1));
//     println!("fft_gray_img shape: {:?}", fft_gray_img.dim());
//     let mut r2c_handler = R2cFftHandler::new(nx);
//     // let gray_img_array = gray_img_array.mapv(|x| Complex::new(x, 0.0));
//     ndrustfft::ndfft_r2c(&gray_img_array, &mut fft_gray_img, &mut r2c_handler, 1);

//     println!("fft_gray_img shape: {:?}", fft_gray_img.dim());

//     let mut fft_handler = FftHandler::new(ny);
//     ndrustfft::ndfft(&fft_gray_img.clone(), &mut fft_gray_img, &mut fft_handler, 0);

//     // println!("fft_gray_img shape: {:?}", fft_gray_img.dim());

//     // FFTshift
//     let shifted_gray_img = fft2shift(&fft_gray_img.view());
    
//     // Display log magnitude spectrum
//     let eps = 1e-6;
//     let log_magnitude = shifted_gray_img
//         .mapv(|x| x.norm())
//         .mapv(|x| (x + eps).ln());
//     let max_val = log_magnitude.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
//     let min_val = log_magnitude.fold(f64::INFINITY, |a, &b| a.min(b));
//     println!("max_val: {}, min_val: {}", max_val, min_val);
//     let log_magnitude = log_magnitude.mapv(|x| ((x - min_val) / (max_val - min_val) * 255.0) as u8);
//     let log_magnitude_img: GrayImage = GrayImage::from_raw(log_magnitude.dim().1 as u32, 
//         log_magnitude.dim().0 as u32,
//         log_magnitude.into_iter().collect()).unwrap();
//     log_magnitude_img.save("log_magnitude.png").unwrap();

//     // Inverse FFT
    
//     ndrustfft::ndifft(&fft_gray_img.clone(), &mut fft_gray_img, &mut fft_handler, 0);
//     let mut recovered_gray_img = Array2::<f64>::zeros((ny, nx));
//     ndrustfft::ndifft_r2c(&fft_gray_img, &mut recovered_gray_img, &mut r2c_handler, 1);
//     // let recovered_gray_img = recovered_gray_img.mapv(|x| x.re);
//     // Normalize to 0-255
//     let max_val = recovered_gray_img.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
//     let min_val = recovered_gray_img.fold(f64::INFINITY, |a, &b| a.min(b));
//     println!("max_val: {}, min_val: {}", max_val, min_val);
//     let recovered_gray_img = recovered_gray_img.mapv(|x| ((x - min_val) / (max_val - min_val) * 255.0) as u8);
//     let recovered_gray_img: GrayImage = GrayImage::from_raw(nx as u32, 
//         ny as u32, 
//         recovered_gray_img.into_iter().collect()).unwrap();
//     recovered_gray_img.save("recovered_gray_img.png").unwrap();

// }
