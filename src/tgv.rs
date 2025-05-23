use web_time::Instant;

use leptos::logging::log;
use ndarray::{azip, par_azip, s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, Zip};
use num_complex::{Complex, ComplexFloat};
use rayon::{array, prelude::*};
use crate::fft::*;
use num_cpus;


fn roll1d(a: &ArrayView1<f32>, roll_amount: i32) -> Array1<f32> {
    
    ndarray::concatenate![Axis(0), a.slice(s![-roll_amount..]), a.slice(s![..-roll_amount])]
}

fn roll2d(a: &ArrayView2<f32>, axis: usize, roll_amount: i32) -> Array2<f32> {
    assert!(roll_amount.abs() > 0);
    if axis == 0 {
        return ndarray::concatenate![Axis(0), a.slice(s![-roll_amount.., ..]), a.slice(s![..-roll_amount, ..])]
    } else if axis == 1 {
        ndarray::concatenate![Axis(1), a.slice(s![.., -roll_amount..]), a.slice(s![.., ..-roll_amount,])]
    } else {
        return a.to_owned()
    }
}

fn gradient(u: &ArrayView2<f32>) -> Array3<f32> {
    // let grad_x = roll2d(&u.view(), 1, -1) - u;
    let num_cpus = num_cpus::get();
    let (y_size, x_size) = u.dim();

    let chunk_size = y_size / num_cpus;
    let mut grad_x = u.clone().to_owned();
    grad_x.axis_chunks_iter_mut(Axis(0), chunk_size).par_bridge().for_each( |mut chunk| {
        chunk.axis_iter_mut(Axis(0)).for_each(|mut row| {
            let mut diff = row.to_owned();
            for i in 0..row.len() {
                if i == (row.len() - 1) {
                    diff[i] = row[0] - row[i];
                } else {
                    diff[i] = row[i + 1] - row[i];
                }
            };
            row.assign(&diff);
        });
    });
    
    // let grad_y = roll2d(&u.view(), 0, -1) - u;

    let chunk_size = x_size / num_cpus;
    let mut grad_y = u.clone().to_owned();
    grad_y.axis_chunks_iter_mut(Axis(1), chunk_size).par_bridge().for_each( |mut chunk| {
        chunk.axis_iter_mut(Axis(1)).for_each(|mut col| {
            // let owned_col_view = col.view();
            // let shifted_col = roll1d(&owned_col_view, -1);
            // let diff = shifted_col - col.to_owned();
            // // let mut diff = diff.clone();
            // // col = diff.view_mut();
            // col.assign(&diff);

            let mut diff = col.to_owned();
            for i in 0..col.len() {
                if i == (col.len() - 1) {
                    diff[i] = col[0] - col[i];
                } else {
                    diff[i] = col[i + 1] - col[i];
                }
            };
            col.assign(&diff);
        });
    });

    ndarray::stack![Axis(2), grad_x, grad_y]
}

fn divergence(p: &ArrayView3<f32>) -> Array2<f32> {
    // let first_term = p.slice(s![.., .., 0]).to_owned() 
    //     - roll2d(&p.slice(s![.., .., 0]), 1, 1);

    // Calculate first term in parallel
    let num_cpus = num_cpus::get();
    let (y_size, x_size, _) = p.dim();
    let chunk_size = y_size / num_cpus;
    let mut first_term = p.slice(s![.., .., 0]).clone().to_owned();
    first_term.axis_chunks_iter_mut(Axis(0), chunk_size).par_bridge().for_each( |mut chunk| {
        chunk.axis_iter_mut(Axis(0)).for_each(|mut row| {
            // let owned_row_view = row.view();
            // let shifted_row = roll1d(&owned_row_view, 1);
            // let diff = row.to_owned() - shifted_row;
            let mut diff = row.to_owned();
            for i in 0..row.len() {
                if i == 0 {
                    diff[i] = row[i] - row[row.len() - 1];
                } else {
                    diff[i] = row[i] - row[i - 1];
                }
            };
            row.assign(&diff);
        });
    });

    // let second_term = p.slice(s![.., .., 1]).to_owned() 
    // - roll2d(&p.slice(s![.., .., 1]), 0, 1);

    // Calculate second term in parallel
    let chunk_size = x_size / num_cpus;
    let mut second_term = p.slice(s![.., .., 1]).clone().to_owned();
    second_term.axis_chunks_iter_mut(Axis(1), chunk_size).par_bridge().for_each( |mut chunk| {
        chunk.axis_iter_mut(Axis(1)).for_each(|mut col| {
            // let owned_col_view = col.view();
            // let shifted_col = roll1d(&owned_col_view, 1);
            // let diff = col.to_owned() - shifted_col;
            let mut diff = col.to_owned();
            for i in 0..col.len() {
                if i == 0 {
                    diff[i] = col[i] - col[col.len() - 1];
                } else {
                    diff[i] = col[i] - col[i - 1];
                }
            };
            col.assign(&diff);
        });
    });
    // -(first_term + second_term)
    // Calculate output
    // par_azip!((x in &mut first_term, &y in &second_term) {
    //     *x = -(*x + y);
    // });
    let chunk_size = y_size / num_cpus;
    let mut first_term_chunks = first_term.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let second_term_chunks = second_term.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    first_term_chunks.zip(second_term_chunks).par_bridge().for_each(|(mut x, y)| {
        // x = -(x + y);
        x.assign(&(-(x.to_owned() + y)));
    });
    first_term

}

fn sym_gradient(w: &ArrayView3<f32>) -> Array3<f32> {
    let num_cpus = num_cpus::get();
    let (y_size, x_size, _) = w.dim();

    // Calculate first diagonal: ∂x w_0 (shift right, forward difference)
    let chunk_size = y_size / num_cpus;
    let mut first_diagonal = w.slice(s![.., .., 0]).clone().to_owned();
    first_diagonal.axis_chunks_iter_mut(Axis(0), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(0)).for_each(|mut row| {
            let mut diff = row.to_owned();
            for i in 0..row.len() {
                if i == (row.len() - 1) {
                    diff[i] = row[0 as usize] - row[i];
                } else {
                    diff[i] = row[i + 1] - row[i];
                }
            };
            row.assign(&diff);
        });
    });

    // Calculate second diagonal: ∂y w_1 (shift down, forward difference)
    let chunk_size = x_size / num_cpus;
    let mut second_diagonal = w.slice(s![.., .., 1]).clone().to_owned();
    second_diagonal.axis_chunks_iter_mut(Axis(1), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(1)).for_each(|mut col| {
            let mut diff = col.to_owned();
            for i in 0..col.len() {
                if i == (col.len() - 1) {
                    diff[i] = col[0 as usize] - col[i];
                } else {
                    diff[i] = col[i + 1] - col[i];
                }
            };
            col.assign(&diff);
        });
    });

    // Calculate off-diagonals: 0.5*(∂y w_0 + ∂x w_1)
    // tmp1: ∂y w_0 (shift down, forward difference)
    let chunk_size = x_size / num_cpus;
    let mut tmp1 = w.slice(s![.., .., 0]).clone().to_owned();
    tmp1.axis_chunks_iter_mut(Axis(1), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(1)).for_each(|mut col| {
            let mut diff = col.to_owned();
            for i in 0..col.len() {
                if i == (col.len() - 1) {
                    diff[i] = col[0 as usize] - col[i];
                } else {
                    diff[i] = col[i + 1] - col[i];
                }
            };
            col.assign(&diff);
        });
    });

    // tmp2: ∂x w_1 (shift right, forward difference)
    let chunk_size = y_size / num_cpus;
    let mut tmp2 = w.slice(s![.., .., 1]).clone().to_owned();
    tmp2.axis_chunks_iter_mut(Axis(0), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(0)).for_each(|mut row| {
            let mut diff = row.to_owned();
            for i in 0..row.len() {
                if i == (row.len() - 1) {
                    diff[i] = row[0 as usize] - row[i];
                } else {
                    diff[i] = row[i + 1] - row[i];
                }
            };
            row.assign(&diff);
        });
    });

    let mut off_diagonals = tmp1;
    let mut off_diagonals_chunks = off_diagonals.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let tmp2_chunks = tmp2.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    off_diagonals_chunks.zip(tmp2_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(0.5 * (x.to_owned() + y)));
    });

    ndarray::stack![Axis(2), first_diagonal, second_diagonal, off_diagonals]
}

fn sym_divergence(q: &ArrayView3<f32>) -> Array3<f32> {
    let num_cpus = num_cpus::get();
    let (y_size, x_size, _) = q.dim();

    // First component: ∂x q_0 - ∂y q_2
    // ∂x q_0: backward difference (shift left)
    let chunk_size = y_size / num_cpus;
    let mut first_component = q.slice(s![.., .., 0]).clone().to_owned();
    first_component.axis_chunks_iter_mut(Axis(0), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(0)).for_each(|mut row| {
            let mut diff = row.to_owned();
            for i in 0..row.len() {
                if i == 0 {
                    diff[i] = row[i] - row[row.len() - 1];
                } else {
                    diff[i] = row[i] - row[i - 1];
                }
            };
            row.assign(&diff.map(|x| -x));
        });
    });

    // ∂y q_2: backward difference (shift up)
    let chunk_size = x_size / num_cpus;
    let mut second_term = q.slice(s![.., .., 2]).clone().to_owned();
    second_term.axis_chunks_iter_mut(Axis(1), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(1)).for_each(|mut col| {
            let mut diff = col.to_owned();
            for i in 0..col.len() {
                if i == 0 {
                    diff[i] = col[i] - col[col.len() - 1];
                } else {
                    diff[i] = col[i] - col[i - 1];
                }
            };
            col.assign(&diff.map(|x| -0.5 * x));
        });
    });
    let mut first_component_chunks = first_component.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let second_term_chunks = second_term.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    first_component_chunks.zip(second_term_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() + y));
    });

    // Second component: ∂y q_1 - ∂x q_2
    // ∂y q_1: backward difference (shift up)
    let chunk_size = x_size / num_cpus;
    let mut second_component = q.slice(s![.., .., 1]).clone().to_owned();
    second_component.axis_chunks_iter_mut(Axis(1), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(1)).for_each(|mut col| {
            let mut diff = col.to_owned();
            for i in 0..col.len() {
                if i == 0 {
                    diff[i] = col[i] - col[col.len() - 1];
                } else {
                    diff[i] = col[i] - col[i - 1];
                }
            };
            col.assign(&diff.map(|x| -x));
        });
    });

    // ∂x q_2: backward difference (shift left)
    let chunk_size = y_size / num_cpus;
    let mut second_term = q.slice(s![.., .., 2]).clone().to_owned();
    second_term.axis_chunks_iter_mut(Axis(0), chunk_size).par_bridge().for_each(|mut chunk| {
        chunk.axis_iter_mut(Axis(0)).for_each(|mut row| {
            let mut diff = row.to_owned();
            for i in 0..row.len() {
                if i == 0 {
                    diff[i] = row[i] - row[row.len() - 1];
                } else {
                    diff[i] = row[i] - row[i - 1];
                }
            };
            row.assign(&diff.map(|x| -0.5 * x));
        });
    });
    let mut second_component_chunks = second_component.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let second_term_chunks = second_term.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    second_component_chunks.zip(second_term_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() + y));
    });

    ndarray::stack![Axis(2), first_component, second_component]
}

fn proj_p(p: &ArrayView3<f32>, alpha1: &f32) -> Array3<f32> {
    let norm = p.clone().to_owned().map_axis_mut(Axis(2), |slice| {
        slice.powi(2).sum().sqrt()
    });

    let mut factor = norm;
    factor.par_iter_mut().for_each(|x| {
        if (*x / alpha1) > 1. {
            *x = *x / alpha1;
        } else {
            *x = 1.0;
        }
    });
    let mut p_proj = p.to_owned();
    
    // Apply chunked parallelization for slice1
    let num_cpus = num_cpus::get();
    let (y_size, _, _) = p_proj.dim();
    let chunk_size = y_size / num_cpus;
    let mut sliced_p_proj = p_proj.slice_mut(s![.., .., 0]);
    let mut slice1_chunks = sliced_p_proj.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let factor_chunks = factor.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    slice1_chunks.zip(factor_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() / &y));
    });
    
    // Apply chunked parallelization for slice2
    let mut sliced_p_proj = p_proj.slice_mut(s![.., .., 1]);
    let mut slice2_chunks = sliced_p_proj.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let factor_chunks = factor.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    slice2_chunks.zip(factor_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() / &y));
    });
    
    p_proj
}

fn proj_q(q: &ArrayView3<f32>, alpha0: &f32) -> Array3<f32> {
    let norm = q.clone().to_owned().map_axis_mut(Axis(2), |slice| {
        slice.powi(2).sum().sqrt()
    });

    let mut factor = norm;
    factor.par_iter_mut().for_each(|x| {
        if (*x / alpha0) > 1. {
            *x = *x / alpha0;
        } else {
            *x = 1.0;
        }
    });
    let mut q_proj = q.to_owned();
    
    // Apply chunked parallelization for all three slices
    let num_cpus = num_cpus::get();
    let (y_size, _, _) = q_proj.dim();
    let chunk_size = y_size / num_cpus;
    
    // Slice 0
    let mut sliced_q_proj = q_proj.slice_mut(s![.., .., 0]);
    let mut slice1_chunks = sliced_q_proj.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let factor_chunks = factor.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    slice1_chunks.zip(factor_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() / &y));
    });
    
    // Slice 1
    let mut sliced_q_proj = q_proj.slice_mut(s![.., .., 1]);
    let mut slice2_chunks = sliced_q_proj.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let factor_chunks = factor.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    slice2_chunks.zip(factor_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() / &y));
    });
    
    // Slice 2
    let mut sliced_q_proj = q_proj.slice_mut(s![.., .., 2]);
    let mut slice3_chunks = sliced_q_proj.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
    let factor_chunks = factor.axis_chunks_iter(Axis(0), chunk_size).into_iter();
    slice3_chunks.zip(factor_chunks).par_bridge().for_each(|(mut x, y)| {
        x.assign(&(x.to_owned() / &y));
    });
    
    q_proj
}


// pub struct TGV {
//     pub lambda: f32,
//     pub mu: f32,
//     pub nu: f32,
// }

// impl TGV {
//     pub fn new(lambda: f32, mu: f32, nu: f32) -> Self {
//         Self { lambda, mu, nu }
//     }
    
// }

// enum ArrayInputType {
//     Array2(Array2<f32>),
//     Array3(Array3<f32>),
// }

// fn report_min_max(array: ArrayInputType) {
//     log!("Min max: {}, {}", array.clone().into_iter().reduce(f32::min).unwrap(), array.clone().into_iter().reduce(f32::max).unwrap());
// }

pub fn tgv_mri_reconstruction(
    centered_kspace: &ArrayView2<Complex<f64>>,
    centered_mask: &ArrayView2<f32>,
    lambda: f32,
    alpha1: f32,
    alpha0: f32,
    tau: f32,
    sigma: f32,
    max_iter: usize,
) -> Array2<f32> {

    // Start timing
    let start_time = Instant::now();

    let (ny, nx) = centered_kspace.dim();
    // Initialize with zero-filled reconstruction
    let mut masked_kspace = Array2::<Complex<f64>>::zeros((ny, nx));
    par_azip!((x in &mut masked_kspace, &y in centered_kspace, &z in centered_mask) {
        if z > 0. {
            *x = y;
        }
    });
    let masked_kspace_view = masked_kspace.view();
    let shifted_masked_kspace = ifft2shift(&masked_kspace_view);
    let u = ifft2(&shifted_masked_kspace.view());
    let mut u = Array2::<f32>::from_shape_vec((ny, nx), u.into_iter().map(|x| x.re as f32).collect()).unwrap();
    
    // Initialize TGV variables
    let mut w = Array3::<f32>::zeros((ny, nx, 2));
    let mut p = Array3::<f32>::zeros((ny, nx, 2));
    let mut q = Array3::<f32>::zeros((ny, nx, 3));

    let mut u_bar = u.clone();
    let mut w_bar = w.clone();

    let t: f32 = 1.0;  // Adaptive momentum

    let num_cpus = num_cpus::get();
    let (y_size, _, _) = p.dim();
    let chunk_size = y_size / num_cpus;
    for i in 0..max_iter {
        let grad_u_bar = gradient(&u_bar.view());
        
        // Replace par_azip with chunked parallelization

        let mut p_chunks = p.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let grad_chunks = grad_u_bar.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        let w_bar_chunks = w_bar.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        p_chunks.zip(grad_chunks.zip(w_bar_chunks)).par_bridge().for_each(|(mut x, (y, z))| {
            x.assign(&(x.to_owned() + sigma * (y.to_owned() - z.to_owned())));
        });
        
        let p = proj_p(&p.view(), &(&alpha0));

        let sym_grad_w_bar = sym_gradient(&w_bar.view());
        
        // Replace par_azip with chunked parallelization
        let mut q_chunks = q.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let sym_grad_chunks = sym_grad_w_bar.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        q_chunks.zip(sym_grad_chunks).par_bridge().for_each(|(mut x, y)| {
            x.assign(&(x.to_owned() + sigma * lambda * y.to_owned()));
        });
        
        let q = proj_q(&q.view(), &(&alpha1));

        // Primal updates
        let u_old = u.clone();
        let w_old = w.clone();

        // Update u: TGV + data fidelity
        let div_p = divergence(&p.view());
        
        // Replace par_azip with chunked parallelization for u update
        let mut u_chunks = u.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let div_p_chunks = div_p.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        u_chunks.zip(div_p_chunks).par_bridge().for_each(|(mut x, y)| {
            x.assign(&(x.to_owned() - tau * lambda * y.to_owned()));
        });

        // Data fidelity term
        let complex_u = Array2::<Complex<f64>>::from_shape_vec((ny, nx), (&u).into_iter().map(|x| Complex::new(*x as f64, 0.0)).collect()).unwrap();
        let fft_u = fft2shift(&fft2(&complex_u.view()).view());
        let mut residual = Array2::<Complex<f64>>::zeros((ny, nx));
        
        // Replace par_azip with chunked parallelization
        let mut residual_chunks = residual.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let fft_u_chunks = fft_u.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        let centered_kspace_chunks = centered_kspace.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        let centered_mask_chunks = centered_mask.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        residual_chunks.zip(fft_u_chunks.zip(centered_kspace_chunks.zip(centered_mask_chunks))).par_bridge().for_each(|(mut x, (y, (z, w)))| {
            for ((x_elem, y_elem), (z_elem, w_elem)) in x.iter_mut().zip(y.iter()).zip(z.iter().zip(w.iter())) {
                if *w_elem > 0. {
                    *x_elem = *y_elem - *z_elem;
                }
            }
        });
        
        let ifft_residual = ifft2(&ifft2shift(&residual.view()).view());
        let ifft_residual = Array2::<f32>::from_shape_vec((ny, nx), ifft_residual.into_iter().map(|x| x.re as f32).collect()).unwrap();
        // Replace par_azip with chunked parallelization for final u update
        let mut u_chunks = u.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let ifft_residual_chunks = ifft_residual.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        u_chunks.zip(ifft_residual_chunks).par_bridge().for_each(|(mut x, y)| {
            x.assign(&(x.to_owned() - tau * y.to_owned()));
        });

        // Update w
        let sym_div_q = sym_divergence(&q.view());
        
        // Replace par_azip with chunked parallelization for w update
        let mut w_chunks = w.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let p_chunks = p.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        let sym_div_q_chunks = sym_div_q.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        w_chunks.zip(p_chunks.zip(sym_div_q_chunks)).par_bridge().for_each(|(mut x, (y, z))| {
            x.assign(&(x.to_owned() - tau * (-y.to_owned() + z.to_owned())));
        });
        
        // Extrapolation
        // Update momentum
        let t_old = t;
        let t = (1. + (1. + 4. * t_old.powi(2)).sqrt()) / 2.0;
        let theta = (1. - t_old) / t;
        
        // Replace par_azip with chunked parallelization for u_bar update
        let mut u_bar_chunks = u_bar.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let u_chunks = u.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        let u_old_chunks = u_old.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        u_bar_chunks.zip(u_chunks.zip(u_old_chunks)).par_bridge().for_each(|(mut x, (y, z))| {
            x.assign(&(y.to_owned() + theta * (y.to_owned() - z.to_owned())));
        });
        
        // Replace par_azip with chunked parallelization for w_bar update
        let mut w_bar_chunks = w_bar.axis_chunks_iter_mut(Axis(0), chunk_size).into_iter();
        let w_chunks = w.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        let w_old_chunks = w_old.axis_chunks_iter(Axis(0), chunk_size).into_iter();
        w_bar_chunks.zip(w_chunks.zip(w_old_chunks)).par_bridge().for_each(|(mut x, (y, z))| {
            x.assign(&(y.to_owned() + theta * (y.to_owned() - z.to_owned())));
        });

        let total_residual: f64 = residual.map(|x| x.re.powi(2)).sum();
        // log!("Iteration: {}, Total residual: {:.3e}", i, total_residual);


        // log!("Min max: {}, {}", u.clone().into_iter().reduce(f32::min).unwrap(), u.clone().into_iter().reduce(f32::max).unwrap());
    }
    // Convert to real
    // let u = Array2::<f32>::from_shape_vec((ny, nx), u.into_iter().map(|x| x.re as f32).collect()).unwrap();
    // log!("Min max: {}, {}", u.clone().into_iter().reduce(f32::min).unwrap(), u.clone().into_iter().reduce(f32::max).unwrap());

    let end_time = Instant::now();
    log!("Time taken: {:?}", end_time.duration_since(start_time));
    u
}

