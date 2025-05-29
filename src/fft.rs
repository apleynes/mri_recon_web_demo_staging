use image::{GrayImage, ImageReader, RgbImage, RgbaImage};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use num_complex::Complex;
use ndrustfft::{self, R2cFftHandler, FftHandler};
use nshare::{self, IntoNdarray2, IntoNdarray3, IntoImageLuma};
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ComplexF32 {
    real: f32,
    imag: f32,
}

impl From<Complex<f64>> for ComplexF32 {
    fn from(c: Complex<f64>) -> Self {
        ComplexF32 {
            real: c.re as f32,
            imag: c.im as f32,
        }
    }
}

impl From<ComplexF32> for Complex<f64> {
    fn from(c: ComplexF32) -> Self {
        Complex::new(c.real as f64, c.imag as f64)
    }
}

// GPU FFT context
pub struct GpuFftContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    fft_pipeline: Option<wgpu::ComputePipeline>,
    bit_reverse_pipeline: Option<wgpu::ComputePipeline>,
}

impl GpuFftContext {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find an appropriate adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await?;

        Ok(Self {
            device,
            queue,
            fft_pipeline: None,
            bit_reverse_pipeline: None,
        })
    }

    fn create_fft_pipeline(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FFT Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/fft.wgsl").into()),
        });

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FFT Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FFT Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        self.fft_pipeline = Some(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FFT Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        }));

        Ok(())
    }

    fn create_bit_reverse_pipeline(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bit Reverse Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/bit_reverse.wgsl").into()),
        });

        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bit Reverse Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Bit Reverse Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        self.bit_reverse_pipeline = Some(self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bit Reverse Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        }));

        Ok(())
    }

    pub async fn fft_1d(&mut self, data: &mut [Complex<f64>], inverse: bool) -> Result<(), Box<dyn std::error::Error>> {
        let len = data.len();
        
        // Only support power-of-two sizes for now
        if !len.is_power_of_two() {
            return Err("GPU FFT only supports power-of-two sizes".into());
        }

        // Initialize pipelines if needed
        if self.fft_pipeline.is_none() {
            self.create_fft_pipeline()?;
        }
        if self.bit_reverse_pipeline.is_none() {
            self.create_bit_reverse_pipeline()?;
        }

        // Convert data to GPU format
        let gpu_data: Vec<ComplexF32> = data.iter().map(|&c| c.into()).collect();

        // Create GPU buffer
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFT Data Buffer"),
            contents: bytemuck::cast_slice(&gpu_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        });

        // Create parameters buffer
        #[repr(C)]
        #[derive(Copy, Clone, Pod, Zeroable)]
        struct FftParams {
            size: u32,
            inverse: u32,
            stage: u32,
            _padding: u32,
        }

        let log2_len = len.trailing_zeros();

        // Step 1: Bit-reverse reordering
        let params = FftParams {
            size: len as u32,
            inverse: if inverse { 1 } else { 0 },
            stage: 0,
            _padding: 0,
        };

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFT Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = &self.bit_reverse_pipeline.as_ref().unwrap().get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bit Reverse Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("FFT Compute Encoder"),
        });

        // Bit-reverse pass
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bit Reverse Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(self.bit_reverse_pipeline.as_ref().unwrap());
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups((len as u32 + 63) / 64, 1, 1);
        }

        // FFT stages
        let fft_bind_group_layout = &self.fft_pipeline.as_ref().unwrap().get_bind_group_layout(0);
        
        for stage in 0..log2_len {
            let stage_params = FftParams {
                size: len as u32,
                inverse: if inverse { 1 } else { 0 },
                stage: stage,
                _padding: 0,
            };

            self.queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&[stage_params]));

            let fft_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("FFT Bind Group"),
                layout: fft_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FFT Stage Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(self.fft_pipeline.as_ref().unwrap());
            compute_pass.set_bind_group(0, &fft_bind_group, &[]);
            compute_pass.dispatch_workgroups((len as u32 + 63) / 64, 1, 1);
        }

        // Read back results
        let result_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT Result Buffer"),
            size: (gpu_data.len() * std::mem::size_of::<ComplexF32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&buffer, 0, &result_buffer, 0, result_buffer.size());

        self.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = result_buffer.slice(..);
        let (sender, receiver) = futures_channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.await.unwrap()?;

        let data_slice = buffer_slice.get_mapped_range();
        let result_data: &[ComplexF32] = bytemuck::cast_slice(&data_slice);

        // Copy results back
        for (i, &gpu_val) in result_data.iter().enumerate() {
            data[i] = gpu_val.into();
        }

        Ok(())
    }
}

// Static GPU context (will be initialized lazily)
static mut GPU_FFT_CONTEXT: Option<GpuFftContext> = None;

async fn get_gpu_context() -> Result<&'static mut GpuFftContext, Box<dyn std::error::Error>> {
    unsafe {
        if GPU_FFT_CONTEXT.is_none() {
            GPU_FFT_CONTEXT = Some(GpuFftContext::new().await?);
        }
        Ok(GPU_FFT_CONTEXT.as_mut().unwrap())
    }
}

// GPU implementations
async fn fft2_gpu(gray_img: &ArrayView2<'_, Complex<f64>>) -> Result<Array2<Complex<f64>>, Box<dyn std::error::Error>> {
    let (ny, nx) = gray_img.dim();
    
    // Find next power-of-two sizes
    let padded_ny = ny.next_power_of_two();
    let padded_nx = nx.next_power_of_two();
    
    // Check if padding would create too large arrays (potential OOM)
    let padded_elements = padded_ny * padded_nx;
    let original_elements = ny * nx;
    let memory_ratio = padded_elements as f64 / original_elements as f64;
    
    leptos::logging::log!("DEBUG: GPU FFT - original: {}x{} ({}), padded: {}x{} ({}), ratio: {:.2}x", 
        ny, nx, original_elements, padded_ny, padded_nx, padded_elements, memory_ratio);
    
    // If padding increases memory usage by more than 3x, fall back to CPU
    if memory_ratio > 3.0 {
        leptos::logging::log!("DEBUG: GPU FFT padding ratio too high ({:.2}x), falling back to CPU", memory_ratio);
        return Err("Padding ratio too high for GPU".into());
    }
    
    // Try to create padded array - if this fails, we'll get an error instead of OOM
    let mut padded = match Array2::<Complex<f64>>::zeros((padded_ny, padded_nx)) {
        arr => arr,
    };
    
    // Copy original data to top-left corner
    for i in 0..ny {
        for j in 0..nx {
            padded[[i, j]] = gray_img[[i, j]];
        }
    }
    
    let ctx = match get_gpu_context().await {
        Ok(ctx) => ctx,
        Err(e) => {
            leptos::logging::log!("DEBUG: Failed to get GPU context: {}", e);
            return Err(e);
        }
    };
    
    // Row-wise FFTs on padded data
    for mut row in padded.axis_iter_mut(Axis(0)) {
        let mut row_data: Vec<Complex<f64>> = row.to_vec();
        if let Err(e) = ctx.fft_1d(&mut row_data, false).await {
            leptos::logging::log!("DEBUG: GPU FFT row failed: {}", e);
            return Err(e);
        }
        for (i, &val) in row_data.iter().enumerate() {
            row[i] = val;
        }
    }
    
    // Column-wise FFTs on padded data
    for j in 0..padded_nx {
        let mut col_data: Vec<Complex<f64>> = padded.column(j).to_vec();
        if let Err(e) = ctx.fft_1d(&mut col_data, false).await {
            leptos::logging::log!("DEBUG: GPU FFT column failed: {}", e);
            return Err(e);
        }
        for (i, &val) in col_data.iter().enumerate() {
            padded[[i, j]] = val;
        }
    }
    
    // Extract result back to original size
    let mut result = Array2::<Complex<f64>>::zeros((ny, nx));
    for i in 0..ny {
        for j in 0..nx {
            result[[i, j]] = padded[[i, j]];
        }
    }
    
    leptos::logging::log!("DEBUG: GPU FFT completed successfully");
    Ok(result)
}

async fn ifft2_gpu(fft_img: &ArrayView2<'_, Complex<f64>>) -> Result<Array2<Complex<f64>>, Box<dyn std::error::Error>> {
    let (ny, nx) = fft_img.dim();
    
    // Find next power-of-two sizes
    let padded_ny = ny.next_power_of_two();
    let padded_nx = nx.next_power_of_two();
    
    // Check if padding would create too large arrays (potential OOM)
    let padded_elements = padded_ny * padded_nx;
    let original_elements = ny * nx;
    let memory_ratio = padded_elements as f64 / original_elements as f64;
    
    leptos::logging::log!("DEBUG: GPU IFFT - original: {}x{} ({}), padded: {}x{} ({}), ratio: {:.2}x", 
        ny, nx, original_elements, padded_ny, padded_nx, padded_elements, memory_ratio);
    
    // If padding increases memory usage by more than 3x, fall back to CPU
    if memory_ratio > 3.0 {
        leptos::logging::log!("DEBUG: GPU IFFT padding ratio too high ({:.2}x), falling back to CPU", memory_ratio);
        return Err("Padding ratio too high for GPU".into());
    }
    
    // Try to create padded array - if this fails, we'll get an error instead of OOM
    let mut padded = match Array2::<Complex<f64>>::zeros((padded_ny, padded_nx)) {
        arr => arr,
    };
    
    // Copy original data to top-left corner
    for i in 0..ny {
        for j in 0..nx {
            padded[[i, j]] = fft_img[[i, j]];
        }
    }
    
    let ctx = match get_gpu_context().await {
        Ok(ctx) => ctx,
        Err(e) => {
            leptos::logging::log!("DEBUG: Failed to get GPU context: {}", e);
            return Err(e);
        }
    };
    
    // Row-wise IFFTs on padded data
    for mut row in padded.axis_iter_mut(Axis(0)) {
        let mut row_data: Vec<Complex<f64>> = row.to_vec();
        if let Err(e) = ctx.fft_1d(&mut row_data, true).await {
            leptos::logging::log!("DEBUG: GPU IFFT row failed: {}", e);
            return Err(e);
        }
        for (i, &val) in row_data.iter().enumerate() {
            row[i] = val;
        }
    }
    
    // Column-wise IFFTs on padded data
    for j in 0..padded_nx {
        let mut col_data: Vec<Complex<f64>> = padded.column(j).to_vec();
        if let Err(e) = ctx.fft_1d(&mut col_data, true).await {
            leptos::logging::log!("DEBUG: GPU IFFT column failed: {}", e);
            return Err(e);
        }
        for (i, &val) in col_data.iter().enumerate() {
            padded[[i, j]] = val;
        }
    }
    
    // Extract result back to original size with proper normalization
    let mut result = Array2::<Complex<f64>>::zeros((ny, nx));
    // The normalization factor should account for the FFT size, not the crop
    let normalization = (padded_ny * padded_nx) as f64;
    
    for i in 0..ny {
        for j in 0..nx {
            result[[i, j]] = padded[[i, j]] / normalization;
        }
    }
    
    leptos::logging::log!("DEBUG: GPU IFFT completed successfully");
    Ok(result)
}

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

// High-level API with GPU/CPU fallback
pub async fn fft2_auto(gray_img: &ArrayView2<'_, Complex<f64>>) -> Array2<Complex<f64>> {
    let (ny, nx) = gray_img.dim();
    
    leptos::logging::log!("DEBUG: fft2_auto called with input dimensions: {}x{}", ny, nx);
    
    // Temporarily force CPU FFT to debug issues
    leptos::logging::log!("DEBUG: Forcing CPU FFT for debugging");
    let result = fft2(gray_img);
    leptos::logging::log!("DEBUG: CPU FFT result dimensions: {}x{}", result.nrows(), result.ncols());
    return result;
    
    // GPU FFT code (temporarily disabled)
    /*
    // Try GPU FFT first (now supports arbitrary sizes via padding)
    match fft2_gpu(gray_img).await {
        Ok(result) => {
            leptos::logging::log!("Using GPU FFT (size: {}x{}, padded to {}x{}) - result size: {}x{}", ny, nx, ny.next_power_of_two(), nx.next_power_of_two(), result.nrows(), result.ncols());
            return result;
        },
        Err(e) => {
            leptos::logging::log!("GPU FFT failed ({}), falling back to CPU", e);
        }
    }
    
    leptos::logging::log!("Using CPU FFT (size: {}x{})", ny, nx);
    let result = fft2(gray_img);
    leptos::logging::log!("DEBUG: CPU FFT result dimensions: {}x{}", result.nrows(), result.ncols());
    result
    */
}

pub async fn ifft2_auto(fft_img: &ArrayView2<'_, Complex<f64>>) -> Array2<Complex<f64>> {
    let (ny, nx) = fft_img.dim();
    
    leptos::logging::log!("DEBUG: ifft2_auto called with input dimensions: {}x{}", ny, nx);
    
    // Temporarily force CPU IFFT to debug the black image issue
    leptos::logging::log!("DEBUG: Forcing CPU IFFT for debugging");
    let result = ifft2(fft_img);
    leptos::logging::log!("DEBUG: CPU IFFT result dimensions: {}x{}", result.nrows(), result.ncols());
    return result;
    
    // GPU IFFT code (temporarily disabled)
    /*
    // Try GPU IFFT first (now supports arbitrary sizes via padding)
    match ifft2_gpu(fft_img).await {
        Ok(result) => {
            leptos::logging::log!("Using GPU IFFT (size: {}x{}, padded to {}x{}) - result size: {}x{}", ny, nx, ny.next_power_of_two(), nx.next_power_of_two(), result.nrows(), result.ncols());
            return result;
        },
        Err(e) => {
            leptos::logging::log!("GPU IFFT failed ({}), falling back to CPU", e);
        }
    }
    
    leptos::logging::log!("Using CPU IFFT (size: {}x{})", ny, nx);
    let result = ifft2(fft_img);
    leptos::logging::log!("DEBUG: CPU IFFT result dimensions: {}x{}", result.nrows(), result.ncols());
    result
    */
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
