pub use wasm_bindgen_rayon::init_thread_pool;
use std::{fs::File, io::Cursor};

use gloo_events::EventListener;
use leptos::{html::{Canvas, Input}, logging::log, prelude::*, task::spawn_local};
use ndrustfft::Complex;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{window, Blob, CanvasRenderingContext2d, Element, HtmlCanvasElement, HtmlInputElement, MouseEvent, Url};
use image::{imageops::FilterType, GrayImage, ImageFormat, ImageReader, RgbImage};
use base64::{engine::general_purpose, Engine as _};
use nshare::{AsNdarray2, AsNdarray3, IntoNdarray3};
use ndarray::{Array2, Array3, s, azip};
mod fft;
mod tgv;


async fn convert_image_input_to_base_64(input: Option<HtmlInputElement>) -> Result<(String, u32, u32), String> {
    let input = input.ok_or("No input element found")?;
    let files = input.files().ok_or("No files selected")?;
    let file = files.get(0).ok_or("No file found")?;

    // Read file as ArrayBuffer
    let array_buffer_promise = file.array_buffer();
    let array_buffer = wasm_bindgen_futures::JsFuture::from(array_buffer_promise)
        .await
        .map_err(|e| format!("Failed to read file: {:?}", e))?;

    // Convert to Uint8Array and then to Vec<u8>
    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    let buffer_vec = uint8_array.to_vec();

    // Decode the image
    let mut img = image::load_from_memory(&buffer_vec)
        .map_err(|e| format!("Failed to decode image: {:?}", e))?;
    // Get the width and height of the image
    let mut width = img.width();
    let mut height = img.height();

    // If image is bigger than 512x512, resize it to 512x512
    if width > 512 || height > 512 {
        img = img.resize_exact(512, 512, FilterType::CatmullRom);
        width = 512;
        height = 512;
    }
    let img: RgbImage = img.into_rgb8();


    // Convert original image to base64 for display
    let mut original_buffer = Vec::new();
    img.write_to(&mut Cursor::new(&mut original_buffer), ImageFormat::Png)
        .map_err(|e| format!("Failed to encode original image: {:?}", e))?;
    let original_base64 = general_purpose::STANDARD.encode(&original_buffer);
    Ok((format!("data:image/png;base64,{}", original_base64), width, height))
}


fn convert_data_url_to_image(data_url: &str) -> Result<image::DynamicImage, String> {
    // Remove the data URL prefix (e.g., "data:image/png;base64,")
    let parts: Vec<&str> = data_url.split(",").collect();
    if parts.len() != 2 {
        return Err("Invalid data URL format".to_string());
    }
    let base64_data = parts[1];

    // Decode the Base64 data
    let image_data = match base64::engine::general_purpose::STANDARD.decode(base64_data) {
        Ok(data) => data,
        Err(_) => return Err("Failed to decode base64 data".to_string()),
    };

    // Load the image from memory
    let img = match image::load_from_memory(&image_data) {
        Ok(image) => image,
        Err(_) => return Err("Failed to load image from memory".to_string()),
    };

    Ok(img)
}

fn draw_point(ctx: &CanvasRenderingContext2d, x: f64, y: f64, erase: bool, point_size: f64) {
    if erase {
        ctx.set_fill_style_str("black");
    } else {
        ctx.set_fill_style_str("white");
    }
    // Center the point
    ctx.fill_rect(x - point_size / 2.0, y - point_size / 2.0, point_size, point_size);
}

#[component]
fn PointSizeSlider(point_size: ReadSignal<f64>, set_point_size: WriteSignal<f64>) -> impl IntoView {
    view! {
        <input type="range" min="1" max="100" value=point_size on:input=move |evt| set_point_size.set(event_target_value(&evt).parse().unwrap()) />
        {point_size}
    }
}


#[component]
fn AdaptiveCanvas(img_width: ReadSignal<u32>, img_height: ReadSignal<u32>, canvas_ref: NodeRef<Canvas>) -> impl IntoView {
    let overlay_ref = NodeRef::<Canvas>::new();

    // Effect to draw the crosshair on the overlay
    Effect::new(move |_| {
        let width = img_width.get() as f64;
        let height = img_height.get() as f64;
        let overlay = overlay_ref
            .get()
            .expect("overlay canvas should be in the DOM");
        overlay.set_width(width as u32);
        overlay.set_height(height as u32);
        let ctx = overlay
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()
            .unwrap();

        // Clear the overlay
        ctx.clear_rect(0.0, 0.0, width, height);

        // Draw the crosshair
        ctx.set_stroke_style_str("rgba(255, 0, 0, 1.0)"); // Semi-transparent red
        ctx.set_line_width(1.0);
        ctx.begin_path();
        let center_x = width / 2.0;
        let center_y = height / 2.0;
        // Horizontal line
        ctx.move_to(0.0, center_y);
        ctx.line_to(width, center_y);
        // Vertical line
        ctx.move_to(center_x, 0.0);
        ctx.line_to(center_x, height);
        ctx.stroke();
    });

    view! {
        <div style="position: relative;">
            <canvas
                node_ref=canvas_ref
                width=move || img_width.get()
                height=move || img_height.get()
                style="border:1px solid black; background:black;"
            />
            <canvas
                node_ref=overlay_ref
                // width=move || img_width.get()  // Need to set width and height in the effect for the crosshair update to work
                // height=move || img_height.get()
                style="position: absolute; top: 0; left: 0; opacity: 0.5; pointer-events: none;"
            />
        </div>
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ReconInteractivityMode {
    OnMouseUp,
    OnDraw,
}

#[derive(Clone, Copy, PartialEq)]
enum ReconMode {
    ZeroFilled,
    TGV2,
}

struct ReconParams {
    tgv2_lam: f32,
    tgv2_iter: usize,
}

fn App() -> impl IntoView {

    // signal: true = erase, false = draw
    let (is_erase, set_erase) = signal(false);
    let (point_size, set_point_size) = signal(4.0);
    // ref to the canvas element
    let canvas_ref = NodeRef::<Canvas>::new();

    let (img_width, set_img_width) = signal(32);
    let (img_height, set_img_height) = signal(32);

    // Initialize with empty vector instead of dummy 32x32 data
    let (img_fft_vec, set_img_fft_vec) = signal(Vec::<Complex<f64>>::new());
    let (reconstructed_img_zero_filled, set_reconstructed_img_zero_filled) = signal(String::new());
    let (reconstructed_img_compressed_sensing, set_reconstructed_img_compressed_sensing) = signal(String::new());

    // Add reconstruction in progress flag to prevent overlapping calls
    let (reconstruction_in_progress, set_reconstruction_in_progress) = signal(false);
    
    let file_input: NodeRef<Input> = NodeRef::new();
    let (original_img_src, set_original_img_src) = signal(String::new());
    let (processed_img_src, set_processed_img_src) = signal(String::new());

    let (recon_interactivity_mode, set_recon_interactivity_mode) = signal(ReconInteractivityMode::OnDraw);
    let (zero_filled_recon_enabled, set_zero_filled_recon_enabled) = signal(true);
    let (compressed_sensing_recon_enabled, set_compressed_sensing_recon_enabled) = signal(false);

    // TGV2 parameters
    let (tgv2_lam, set_tgv2_lam) = signal(1.0);
    let (tgv2_iter, set_tgv2_iter) = signal(5 as usize);
    
    // Debounced reconstruction function to prevent excessive calls
    let debounced_reconstruct = move |mode: ReconMode| {
        // Skip if reconstruction is already in progress
        if reconstruction_in_progress.get_untracked() {
            leptos::logging::log!("DEBUG: Reconstruction already in progress, skipping");
            return;
        }
        
        set_reconstruction_in_progress.set(true);
        
        let canvas_ref = canvas_ref;
        let img_fft_vec = img_fft_vec;
        let img_width = img_width;
        let img_height = img_height;
        let set_reconstruction_in_progress = set_reconstruction_in_progress;
        let tgv2_lam = tgv2_lam.get_untracked();
        let tgv2_iter = tgv2_iter.get_untracked();
        
        match mode {
            ReconMode::ZeroFilled => {
                read_canvas_and_reconstruct(
                    canvas_ref, 
                    img_fft_vec, 
                    img_width, 
                    img_height, 
                    set_reconstructed_img_zero_filled, 
                    ReconMode::ZeroFilled, 
                    ReconParams { tgv2_lam, tgv2_iter },
                    set_reconstruction_in_progress
                );
            },
            ReconMode::TGV2 => {
                read_canvas_and_reconstruct(
                    canvas_ref, 
                    img_fft_vec, 
                    img_width, 
                    img_height, 
                    set_reconstructed_img_compressed_sensing, 
                    ReconMode::TGV2, 
                    ReconParams { tgv2_lam, tgv2_iter },
                    set_reconstruction_in_progress
                );
            },
        }
    };
    
    let reconstruct_img_and_set_reconstructed_img = move |_| {
        if zero_filled_recon_enabled.get() {
            debounced_reconstruct(ReconMode::ZeroFilled);
        }
        if compressed_sensing_recon_enabled.get() {
            debounced_reconstruct(ReconMode::TGV2);
        }
    };


    // set up pointer listeners once the canvas is in the DOM
    Effect::new(move |evt| {
        let canvas = canvas_ref
            .get()
            .expect("canvas should be in the DOM");
        // get 2D context
        let ctx = canvas
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()
            .unwrap();

        // track whether pointer is down
        let is_drawing = std::rc::Rc::new(std::cell::Cell::new(false));
        let drawing_flag = is_drawing.clone();
        let erase_flag = is_erase;

        // mousedown → start drawing
        // let canvas_clone = canvas.clone();
        EventListener::new(&canvas, "pointerdown", move |evt| {
            drawing_flag.set(true);
            let pe = evt.dyn_ref::<web_sys::PointerEvent>().unwrap();
            let canvas = canvas_ref.get_untracked().unwrap();
            let rect = canvas.get_bounding_client_rect();
            // adjust for canvas position
            let x = pe.client_x() as f64 - rect.left();
            let y = pe.client_y() as f64 - rect.top();
            // let x = pe.client_x() as f64;
            // let y = pe.client_y() as f64;
            draw_point(&ctx, x, y, erase_flag.get_untracked(), point_size.get_untracked());

            let recon_interactivity_mode: ReconInteractivityMode = recon_interactivity_mode.get_untracked();
            if recon_interactivity_mode == ReconInteractivityMode::OnDraw {
                if zero_filled_recon_enabled.get() {
                    debounced_reconstruct(ReconMode::ZeroFilled);
                }
                if compressed_sensing_recon_enabled.get() {
                    debounced_reconstruct(ReconMode::TGV2);
                }
            }
        })
        .forget();

        // pointerup anywhere → stop drawing
        let drawing_flag_up = is_drawing.clone();
        EventListener::new(&window().unwrap(), "pointerup", move |_| {
            drawing_flag_up.set(false);

            if recon_interactivity_mode.get() == ReconInteractivityMode::OnMouseUp {
                if zero_filled_recon_enabled.get() {
                    debounced_reconstruct(ReconMode::ZeroFilled);
                }
                if compressed_sensing_recon_enabled.get() {
                    debounced_reconstruct(ReconMode::TGV2);
                }
            }

        })
        .forget();

        // // pointerup → reconstruct image
        // let reconstruct_img_and_set_reconstructed_img_clone = reconstruct_img_and_set_reconstructed_img.clone();
        // EventListener::new(&window().unwrap(), "pointerup", move |evt| {
        //     reconstruct_img_and_set_reconstructed_img_clone(evt);
        // })
        // .forget();

        // pointermove → draw if pointerdown
        let ctx = canvas
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()
            .unwrap();  // Refresh context
        EventListener::new(&canvas, "pointermove", move |evt| {
            if !is_drawing.get() {
                return;
            }
            let pe = evt.dyn_ref::<web_sys::PointerEvent>().unwrap();
            let canvas = canvas_ref.get_untracked().unwrap();
            let rect = canvas.get_bounding_client_rect();
            let x = pe.client_x() as f64 - rect.left();
            let y = pe.client_y() as f64 - rect.top();
            // let x = pe.client_x() as f64;
            // let y = pe.client_y() as f64;
            draw_point(&ctx, x, y, erase_flag.get_untracked(), point_size.get_untracked());

            if recon_interactivity_mode.get() == ReconInteractivityMode::OnDraw {
                if zero_filled_recon_enabled.get() {
                    debounced_reconstruct(ReconMode::ZeroFilled);
                }
                if compressed_sensing_recon_enabled.get() {
                    debounced_reconstruct(ReconMode::TGV2);
                }
            }
        })
        .forget();
    });

    let update_image = move |_| {
        spawn_local(async move {
            let input_element = file_input.get();
            let (base64, width, height) = convert_image_input_to_base_64(input_element.clone()).await.expect("Failed to process image");
            set_original_img_src.set(base64);
            set_img_width.set(width);
            set_img_height.set(height);
            // Convert image to array
            let input = input_element.clone().ok_or("No input element found").expect("No input element found");
            let files = input.files().ok_or("No files selected").expect("No files selected");
            let file = files.get(0).ok_or("No file found").expect("No file found");
        
            // Read file as ArrayBuffer
            let array_buffer_promise = file.array_buffer();
            let array_buffer = wasm_bindgen_futures::JsFuture::from(array_buffer_promise)
                .await
                .map_err(|e| format!("Failed to read file: {:?}", e)).expect("Failed to read file");
        
            // Convert to Uint8Array and then to Vec<u8>
            let uint8_array = js_sys::Uint8Array::new(&array_buffer);
            let buffer_vec = uint8_array.to_vec();
            let mut img = image::load_from_memory(&buffer_vec)
                .map_err(|e| format!("Failed to decode image: {:?}", e)).expect("Failed to decode image");

            // Resize image to 512x512 if it's bigger (redo this. TODO: Refactor)
            let mut width = img.width();
            let mut height = img.height();
            if width > 512 || height > 512 {
                img = img.resize_exact(512, 512, FilterType::CatmullRom);
                width = 512;
                height = 512;
            }
            let img: GrayImage = img.into_luma8();
            let img = img.as_ndarray2();
            let complex_img: Array2<Complex<f64>> = img.map(|x| Complex::new(*x as f64, 0.0));
            
            // Add logging before FFT processing
            leptos::logging::log!("DEBUG: Input image dimensions before FFT: {}x{} (total elements: {})", complex_img.nrows(), complex_img.ncols(), complex_img.len());
            leptos::logging::log!("DEBUG: img_width signal: {}, img_height signal: {}", width, height);
            
            // Use GPU-accelerated FFT with fallback
            let fft_vec = fft::fft2_auto(&complex_img.view()).await;
            
            // Add logging after FFT processing
            leptos::logging::log!("DEBUG: FFT result dimensions: {}x{} (total elements: {})", fft_vec.nrows(), fft_vec.ncols(), fft_vec.len());
            
            let fft_vec = fft::fft2shift(&fft_vec.view());
            
            // Add logging after shift
            leptos::logging::log!("DEBUG: After fft2shift dimensions: {}x{} (total elements: {})", fft_vec.nrows(), fft_vec.ncols(), fft_vec.len());
            
            let (v, offset) = fft_vec.into_raw_vec_and_offset();
            
            // Add logging for stored vector
            leptos::logging::log!("DEBUG: Storing FFT vector with {} elements", v.len());
            
            set_img_fft_vec.set(v);

            // Clear the canvas
            let canvas = canvas_ref.get().unwrap();
            let ctx = canvas.get_context("2d").unwrap().unwrap().dyn_into::<CanvasRenderingContext2d>().unwrap();
            ctx.set_fill_style_str("black");
            ctx.fill_rect(0.0, 0.0, img_width.get() as f64, img_height.get() as f64);

            // Clear the reconstructed image
            let reconstructed_img = Array2::zeros((img_height.get() as usize, img_width.get() as usize));
    
            let reconstructed_img = GrayImage::from_raw(width as u32, height as u32, reconstructed_img.into_iter().collect()).unwrap();
            let mut reconstructed_buffer = Vec::new();
            reconstructed_img.write_to(&mut Cursor::new(&mut reconstructed_buffer), ImageFormat::Png)
                .map_err(|e| format!("Failed to encode reconstructed image: {:?}", e)).expect("Failed to encode reconstructed image");
            let reconstructed_base64 = general_purpose::STANDARD.encode(&reconstructed_buffer);
            set_reconstructed_img_zero_filled.set(format!("data:image/png;base64,{}", reconstructed_base64));
            set_reconstructed_img_compressed_sensing.set(format!("data:image/png;base64,{}", reconstructed_base64));
        })
    };

    view! {
        <div>
            <h1>"MR Image Sampling and Reconstruction Demo"</h1>
            <h2>"Instructions"</h2>
            <p>"Upload an image. Then, draw a sampling mask on the canvas by clicking and dragging inside the canvas. The red crosshair indicates the center of the canvas (center of k-space). The image will be reconstructed from the mask as soon as you draw or release the mouse button. "</p>
            <p>"If it's too slow, try a smaller image or select 'On mouse up' in the reconstruction interactivity mode."</p>
            <p>"The demo runs entirely in the browser using your machine's CPU. No data is sent to any servers."</p>
            <h2>"Upload image"</h2>
            <input 
                type="file" 
                accept="image/*" 
                node_ref=file_input
                // on:
                on:change=update_image
            />

            <Show when=move || !original_img_src.get().is_empty()>
                <div class="image-box">
                    <h2>"Original Image"</h2>
                    <img src=original_img_src alt="Original Image" />
                </div>
            </Show>

            <div style="display: flex; flex-direction: row; gap: 10px;">
                <div class="image-box">
                    <h2>"Sampling mask"</h2>
                    <AdaptiveCanvas img_width=img_width img_height=img_height canvas_ref=canvas_ref />
                </div>

                <Show when=move || !reconstructed_img_zero_filled.get().is_empty() && !original_img_src.get().is_empty() && zero_filled_recon_enabled.get()>
                    <div class="image-box">
                        <h2>"Basic (Zero-filled) Reconstructed Image"</h2>
                        <img src=reconstructed_img_zero_filled alt="Reconstructed Image" />
                    </div>
                </Show>

                <Show when=move || !reconstructed_img_compressed_sensing.get().is_empty() && !original_img_src.get().is_empty() && compressed_sensing_recon_enabled.get()>
                    <div class="image-box">
                        <h2>"Compressed sensing (TGV2) Reconstructed Image"</h2>
                        <img src=reconstructed_img_compressed_sensing alt="Reconstructed Image" />
                        // Parameter control for the compressed sensing reconstruction
                        <p>Regularization Strength (lambda)</p>
                        // Slider: -1e-12 to 1e3 but in log scale
                        <input
                          type="range"
                          min="-6"
                          max="6"
                          step="0.01"
                          // Bind slider thumb to param
                          prop:value=move || tgv2_lam.get().to_string()  // Let the slider cursor go from min to max
                          // On input, parse and update `param`
                          on:input=move |ev| {
                            let v = event_target_value(&ev)
                                      .parse::<f32>()
                                      .unwrap_or(tgv2_lam.get());
                            set_tgv2_lam.set(v);
                            debounced_reconstruct(ReconMode::TGV2);
                          }
                        />
                        // Number input: shows same param but in linear scale
                        <input
                          type="number"
                          step="1e-3"
                          min="1e-6"
                          max="1e6"
                          prop:value=move || 10.0_f32.powf(tgv2_lam.get()).to_string()  // Show hte displayed value in log scale
                        //   On input, parse and update `param`
                          on:input=move |ev| {
                            let v = event_target_value(&ev)
                                      .parse::<f32>()
                                      .map(|x| x.log10())
                                      .unwrap_or(tgv2_lam.get());  // Int inputted, store it as log scale
                            set_tgv2_lam.set(v);
                            debounced_reconstruct(ReconMode::TGV2);
                          }
                          style="width: 4em;"
                        />
                        <p>Number of iterations</p>
                        // Slider: -1e-12 to 1e3 but in log scale
                        <input
                          type="range"
                          min="1"
                          max="30"
                          step="1"
                          // Bind slider thumb to param
                          prop:value=move || tgv2_iter.get().to_string()  // Let the slider cursor go from min to max
                          // On input, parse and update `param`
                          on:input=move |ev| {
                            let v = event_target_value(&ev)
                                      .parse::<usize>()
                                      .unwrap_or(tgv2_iter.get());
                                    // tgv_lam.set(v);
                            set_tgv2_iter.set(v);
                          }
                        />
                        // Number input: shows same param but in linear scale
                        <input
                          type="number"
                          step="1"
                          min="1"
                          max="30"
                          prop:value=move || tgv2_iter.get().to_string()  // Show hte displayed value in log scale
                        //   On input, parse and update `param`
                          on:input=move |ev| {
                            let v = event_target_value(&ev)
                                      .parse::<usize>()
                                      .unwrap_or(tgv2_iter.get());  // Int inputted, store it as log scale
                            // param_set.set(v);
                            set_tgv2_iter.set(v);
                          }
                          style="width: 4em;"
                        />
                    </div>
                </Show>
            </div>

            <br />
            <p>Reconstruction interactivity:</p>
            <input type="radio" name="recon_interactivity_mode" value="on_mouse_up" on:change=move |_| set_recon_interactivity_mode.set(ReconInteractivityMode::OnMouseUp) checked=move || recon_interactivity_mode.get() == ReconInteractivityMode::OnMouseUp />
            <label for="on_mouse_up">"On mouse up"</label>
            <input type="radio" name="recon_interactivity_mode" value="on_draw" on:change=move |_| set_recon_interactivity_mode.set(ReconInteractivityMode::OnDraw) checked=move || recon_interactivity_mode.get() == ReconInteractivityMode::OnDraw />
            <label for="on_draw">"On draw"</label>
            <br />
            <p>Reconstruction modes:</p>
            <input type="checkbox" name="zero_filled_recon_enabled" 
                on:change=move |evt| {
                    set_zero_filled_recon_enabled.set(event_target_checked(&evt));
                    if zero_filled_recon_enabled.get() {
                        debounced_reconstruct(ReconMode::ZeroFilled);
                    }
                }
                checked=move || zero_filled_recon_enabled.get() />
            <label for="zero_filled_recon_enabled">"Basic (Zero-filled)"</label>
            <input type="checkbox" name="compressed_sensing_recon_enabled" 
                on:change=move |evt| {
                    set_compressed_sensing_recon_enabled.set(event_target_checked(&evt));
                    if compressed_sensing_recon_enabled.get() {
                        debounced_reconstruct(ReconMode::TGV2);
                    }
                }
                checked=move || compressed_sensing_recon_enabled.get() />
            <label for="compressed_sensing_recon_enabled">"Compressed sensing (TGV2)"</label>
            <br />
            <button on:click=move |_| set_erase.set(false)>
                "Draw"
            </button>
            <button on:click=move |_| set_erase.set(true)>
                "Erase"
            </button>
            <br />
            <p>Point size:</p>
            <PointSizeSlider point_size=point_size set_point_size=set_point_size />
            <p>Mode: {move || if is_erase.get() { "Erase" } else { "Draw" }}</p>
            <br />
            <button on:click=move |evt| {
                let canvas = canvas_ref.get().unwrap();
                let ctx = canvas.get_context("2d").unwrap().unwrap().dyn_into::<CanvasRenderingContext2d>().unwrap();
                ctx.set_fill_style_str("black");
                ctx.fill_rect(0.0, 0.0, img_width.get() as f64, img_height.get() as f64);

                reconstruct_img_and_set_reconstructed_img(evt);
            }>
                "Clear"
            </button>
            <br />
            // <button on:click=move |_| {
            //     let canvas = canvas_ref
            //         .get()
            //         .expect("canvas should be in the DOM");
            //     let image_string = canvas.to_data_url_with_type("image/png").expect("Failed to convert canvas to image");
            //     // let image = image::load_from_memory(&image_string.as_bytes()).expect("Failed to load image");
            //     // let image_array = image.as_luma8().unwrap();

            //     let window = web_sys::window().unwrap();
            //     let document = window.document().unwrap();
            //     let a = document
            //         .create_element("a")
            //         .unwrap()
            //         .dyn_into::<web_sys::HtmlAnchorElement>()
            //         .unwrap();
            //     a.set_href(&image_string);
            //     a.set_download("canvas.png");
            //     // // hide the link
            //     // a.set_property("display", "none").unwrap();
            //     // insert into DOM, trigger download, then remove
            //     let body = document.body().unwrap();
            //     body.append_child(&a).unwrap();
            //     a.click();
            //     body.remove_child(&a).unwrap();

            // }>
            //     "Save as image"
            // </button>
            <br />
            <button on:click=reconstruct_img_and_set_reconstructed_img>
                "Reconstruct image"
            </button>
        </div>
    }
}

fn read_canvas_and_reconstruct(
    canvas_ref: NodeRef<Canvas>, 
    img_fft_vec: ReadSignal<Vec<Complex<f64>>>,
    img_width: ReadSignal<u32>,
    img_height: ReadSignal<u32>,
    set_reconstructed_img: WriteSignal<String>,
    recon_mode: ReconMode,
    recon_params: ReconParams,
    set_reconstruction_in_progress: WriteSignal<bool>,
) {
    spawn_local(async move {
        let width = img_width.get() as usize;
        let height = img_height.get() as usize;
        let fft_vec = img_fft_vec.get();
        
        // Early guard: Don't proceed if no valid image data is available
        if fft_vec.is_empty() {
            leptos::logging::log!("DEBUG: No image data available, skipping reconstruction");
            set_reconstruction_in_progress.set(false);
            return;
        }
        
        if fft_vec.len() != height * width {
            leptos::logging::log!("ERROR: Size mismatch! Expected {} elements, got {}. Skipping reconstruction.", height * width, fft_vec.len());
            set_reconstruction_in_progress.set(false);
            return;
        }
        
        // Check available memory before proceeding with large allocations
        leptos::logging::log!("DEBUG: Starting reconstruction for {}x{} image ({} MB estimated)", 
            height, width, (height * width * 32) / (1024 * 1024)); // rough estimate
        
        // Get sampling mask from canvas using ImageData instead of PNG conversion
        let canvas = canvas_ref
            .get()
            .expect("canvas should be in the DOM");
        
        let ctx = canvas
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()
            .unwrap();
        
        // Get ImageData directly from canvas instead of PNG conversion
        let image_data = ctx
            .get_image_data(0.0, 0.0, width as f64, height as f64)
            .expect("Failed to get image data from canvas");
        
        let data = image_data.data();
        
        // Create FFT image array first (reuse the existing vector to avoid extra allocation)
        let fft_img = match Array2::from_shape_vec((height, width), fft_vec) {
            Ok(arr) => arr,
            Err(e) => {
                leptos::logging::log!("ERROR: Failed to create FFT array: {:?}", e);
                set_reconstruction_in_progress.set(false);
                return;
            }
        };
        
        // Create masked FFT image and apply mask directly without intermediate mask array
        let mut masked_fft_img = Array2::<Complex<f64>>::zeros((height, width));
        
        for i in 0..height {
            for j in 0..width {
                let pixel_index = (i * width + j) * 4; // RGBA format
                let r = data[pixel_index] as f64;
                let g = data[pixel_index + 1] as f64; 
                let b = data[pixel_index + 2] as f64;
                // Average RGB values and threshold
                let brightness = (r + g + b) / 3.0;
                
                if brightness > 128.0 {
                    masked_fft_img[[i, j]] = fft_img[[i, j]];
                }
                // else leave as zero (already initialized)
            }
        }
        
        leptos::logging::log!("DEBUG: Created masked FFT array, starting reconstruction");

        let reconstructed_img = match recon_mode {
            ReconMode::ZeroFilled => {
                // Use GPU-accelerated IFFT with fallback
                let shifted = fft::ifft2shift(&masked_fft_img.view());
                let ifft_result = fft::ifft2_auto(&shifted.view()).await;
                ifft_result.map(|x| x.re as f32)
            },
            ReconMode::TGV2 => {
                // For TGV, create a minimal mask array only when needed
                let mut mask = Array2::<f32>::zeros((height, width));
                for i in 0..height {
                    for j in 0..width {
                        let pixel_index = (i * width + j) * 4;
                        let r = data[pixel_index] as f64;
                        let g = data[pixel_index + 1] as f64; 
                        let b = data[pixel_index + 2] as f64;
                        let brightness = (r + g + b) / 3.0;
                        mask[[i, j]] = if brightness > 128.0 { 1.0 } else { 0.0 };
                    }
                }
                
                // Use GPU-accelerated TGV with fallback
                tgv::tgv_mri_reconstruction_auto(
                    &masked_fft_img.view(), 
                    &mask.view(), 
                    recon_params.tgv2_lam, 
                    1.0, 
                    2.0, 
                    1.0/(12.0_f32).sqrt(), 
                    1.0/(12.0_f32).sqrt(), 
                    recon_params.tgv2_iter as usize
                ).await
            },
        };
        
        leptos::logging::log!("DEBUG: Reconstruction complete, normalizing image");
        
        let reconstructed_img = normalize_image_by_min_max(reconstructed_img);
        
        let reconstructed_img = GrayImage::from_raw(width as u32, height as u32, reconstructed_img.into_iter().collect()).unwrap();
        let mut reconstructed_buffer = Vec::new();
        reconstructed_img.write_to(&mut Cursor::new(&mut reconstructed_buffer), ImageFormat::Png)
            .map_err(|e| format!("Failed to encode reconstructed image: {:?}", e)).expect("Failed to encode reconstructed image");
        let reconstructed_base64 = general_purpose::STANDARD.encode(&reconstructed_buffer);
        set_reconstructed_img.set(format!("data:image/png;base64,{}", reconstructed_base64));
        
        leptos::logging::log!("DEBUG: Reconstruction and encoding complete");
        
        // Clear the in-progress flag
        set_reconstruction_in_progress.set(false);
    })
}

fn normalize_image_by_min_max(img: Array2<f32>) -> Array2<u8> {
    let min_val = img.fold(f32::INFINITY, |a, &b| a.min(b));
    img.map(|x| x - min_val);

    let max_val = img.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    img.map(|x| (x * 255.0 / max_val) as u8)
}


fn main() {
    console_error_panic_hook::set_once();
    leptos::attr::csp("worker-src 'self' blob:; script-src 'unsafe-inline' 'self' blob:;");
    // mount the app to <body>
    mount_to_body(|| view! { <App/> });
}
