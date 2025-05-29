use web_time::Instant;

use leptos::logging::log;
use ndarray::{par_azip, s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, Zip};
use num_complex::{Complex, ComplexFloat};
use rayon::{array, prelude::*};
use crate::fft::*;


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

    let mut grad_x = u.clone().to_owned();
    grad_x.axis_iter_mut(Axis(0)) 
        .into_par_iter()
        .for_each(|mut row| {
            let owned_row_view = row.view();
            let shifted_row = roll1d(&owned_row_view, -1);
            let diff = shifted_row - row.to_owned();
            // let mut diff = diff.clone();
            // row = diff.view_mut();
            row.assign(&diff);
        });
    
    // let grad_y = roll2d(&u.view(), 0, -1) - u;

    let mut grad_y = u.clone().to_owned();
    grad_y.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let owned_col_view = col.view();
            let shifted_col = roll1d(&owned_col_view, -1);
            let diff = shifted_col - col.to_owned();
            // let mut diff = diff.clone();
            // col = diff.view_mut();
            col.assign(&diff);
        });

    ndarray::stack![Axis(2), grad_x, grad_y]
}

fn divergence(p: &ArrayView3<f32>) -> Array2<f32> {
    // let first_term = p.slice(s![.., .., 0]).to_owned() 
    //     - roll2d(&p.slice(s![.., .., 0]), 1, 1);

    // Calculate first term in parallel
    let mut first_term = p.slice(s![.., .., 0]).clone().to_owned();
    first_term.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let owned_row_view = row.view();
            let shifted_row = roll1d(&owned_row_view, 1);
            let diff = row.to_owned() - shifted_row;
            row.assign(&diff);
        });


    // let second_term = p.slice(s![.., .., 1]).to_owned() 
    // - roll2d(&p.slice(s![.., .., 1]), 0, 1);

    // Calculate second term in parallel
    let mut second_term = p.slice(s![.., .., 1]).clone().to_owned();
    second_term.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let owned_col_view = col.view();
            let shifted_col = roll1d(&owned_col_view, 1);
            let diff = col.to_owned() - shifted_col;
            col.assign(&diff);
        });

    // -(first_term + second_term)
    // Calculate output
    par_azip!((x in &mut first_term, &y in &second_term) {
        *x = -(*x + y);
    });
    first_term

}

fn sym_gradient(w: &ArrayView3<f32>) -> Array3<f32> {
    // // First diagonal: ∂x w_0
    // let first_diagonal = roll2d(&w.slice(s![.., .., 0]), 1, -1) 
    //     - w.slice(s![.., .., 0]);
    // // Second diagonal: ∂y w_1
    // let second_diagonal = roll2d(&w.slice(s![.., .., 1]), 0, -1) 
    //     - w.slice(s![.., .., 1]);
    // // Off-diagonals: 0.5*(∂y w_0 + ∂x w_1)
    // let tmp1 = roll2d(&w.slice(s![.., .., 0]), 0, -1) 
    //     - w.slice(s![.., .., 0]);
    // let tmp2 = roll2d(&w.slice(s![.., .., 1]), 1, -1) 
    //     - w.slice(s![.., .., 1]);
    // let off_diagonals = 0.5 * (tmp1 + tmp2);

    // Calculate first diagonal in parallel
    let mut first_diagonal = w.slice(s![.., .., 0]).clone().to_owned();
    first_diagonal.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let owned_row_view = row.view();
            let shifted_row = roll1d(&owned_row_view, -1);
            let diff = shifted_row - row.to_owned();
            row.assign(&diff);
        });

    // Calculate second diagonal in parallel
    let mut second_diagonal = w.slice(s![.., .., 1]).clone().to_owned();
    second_diagonal.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let owned_col_view = col.view();
            let shifted_col = roll1d(&owned_col_view, -1);
            let diff = shifted_col - col.to_owned();
            col.assign(&diff);
        });

    // Calculate off-diagonals in parallel
    let mut tmp1 = w.slice(s![.., .., 0]).clone().to_owned();
    tmp1.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut row| {
            let owned_row_view = row.view();
            let shifted_row = roll1d(&owned_row_view, -1);
            let diff = shifted_row - row.to_owned();
            row.assign(&diff);
        });

    let mut tmp2 = w.slice(s![.., .., 1]).clone().to_owned();
    tmp2.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut col| {
            let owned_col_view = col.view();
            let shifted_col = roll1d(&owned_col_view, -1);
            let diff = shifted_col - col.to_owned();
            col.assign(&diff);
        });
    let mut off_diagonals = tmp1;
    par_azip!((x in &mut off_diagonals, &y in &tmp2) {
        *x = 0.5 * (*x + y);
    });

    ndarray::stack![Axis(2), first_diagonal, second_diagonal, off_diagonals]
}

fn sym_divergence(q: &ArrayView3<f32>) -> Array3<f32> {
    // // First component: ∂x q_0 - ∂y q_2
    // let first_term = -(q.slice(s![.., .., 0]).to_owned() 
    //     - roll2d(&q.slice(s![.., .., 0]), 1, 1));
    // let second_term = -0.5 * (q.slice(s![.., .., 2]).to_owned() 
    //     - roll2d(&q.slice(s![.., .., 2]), 0, 1));
    // let first_component = first_term + second_term;
    // // Second component: ∂y q_1 - ∂x q_2
    // let first_term = -(q.slice(s![.., .., 1]).to_owned() 
    //     - roll2d(&q.slice(s![.., .., 1]), 0, 1));
    // let second_term = -0.5 * (q.slice(s![.., .., 2]).to_owned() 
    //     - roll2d(&q.slice(s![.., .., 2]), 1, 1));
    // let second_component = first_term + second_term;


    let mut first_component = q.slice(s![.., .., 0]).clone().to_owned();
    first_component.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let owned_row_view = row.view();
            let shifted_row = roll1d(&owned_row_view, 1);
            let diff = -(row.to_owned() - shifted_row);
            row.assign(&diff);
        });
    let mut second_term = q.slice(s![.., .., 2]).clone().to_owned();
    second_term.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let owned_col_view = col.view();
            let shifted_col = roll1d(&owned_col_view, 1);
            let diff = -0.5 * (col.to_owned() - shifted_col);
            col.assign(&diff);
        });
    par_azip!((x in &mut first_component, &y in &second_term) {
        *x += y;
    });

    let mut second_component = q.slice(s![.., .., 1]).clone().to_owned();
    second_component.axis_iter_mut(Axis(1))
        .into_par_iter()
        .for_each(|mut col| {
            let owned_col_view = col.view();
            let shifted_col = roll1d(&owned_col_view, 1);
            let diff = -(col.to_owned() - shifted_col);
            col.assign(&diff);
        });
    let mut second_term = q.slice(s![.., .., 2]).clone().to_owned();
    second_term.axis_iter_mut(Axis(0))
        .into_par_iter()
        .for_each(|mut row| {
            let owned_row_view = row.view();
            let shifted_row = roll1d(&owned_row_view, 1);
            let diff = -0.5 * (row.to_owned() - shifted_row);
            row.assign(&diff);
        });
    par_azip!((x in &mut second_component, &y in &second_term) {
        *x += y;
    });

    ndarray::stack![Axis(2), first_component, second_component]
}

fn proj_p(p: &ArrayView3<f32>, alpha1: &f32) -> Array3<f32> {
    // let norm = (p.slice(s![.., .., 0]).map(|x| x.powi(2)) 
    //     + p.slice(s![.., .., 1]).map(|x| x.powi(2)))
    //     .sqrt();
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
    let mut slice1 = p_proj.slice_mut(s![.., .., 0]);
    // slice1 /= &factor;
    par_azip!((x in &mut slice1, &y in &factor) {
        *x /= y;
    });
    let mut slice2 = p_proj.slice_mut(s![.., .., 1]);
    // slice2 /= &factor;
    par_azip!((x in &mut slice2, &y in &factor) {
        *x /= y;
    });
    p_proj
}

fn proj_q(q: &ArrayView3<f32>, alpha0: &f32) -> Array3<f32> {
    // let norm = (q.slice(s![.., .., 0]).map(|x| x.powi(2)) 
    //     + q.slice(s![.., .., 1]).map(|x| x.powi(2)) 
    //     + q.slice(s![.., .., 2]).map(|x| x.powi(2)))
    //     .sqrt();
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
    let mut slice1 = q_proj.slice_mut(s![.., .., 0]);
    // slice1 /= &factor;
    par_azip!((x in &mut slice1, &y in &factor) {
        *x /= y;
    });
    let mut slice2 = q_proj.slice_mut(s![.., .., 1]);
    // slice2 /= &factor;
    par_azip!((x in &mut slice2, &y in &factor) {
        *x /= y;
    });
    let mut slice3 = q_proj.slice_mut(s![.., .., 2]);
    // slice3 /= &factor;
    par_azip!((x in &mut slice3, &y in &factor) {
        *x /= y;
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
    for i in 0..max_iter {
        let grad_u_bar = gradient(&u_bar.view());
        par_azip!((x in &mut p, &y in &grad_u_bar, &z in &w_bar) {
            *x += &sigma * (y - z);
        });
        let p = proj_p(&p.view(), &(&alpha0));
        // log!("Min max: {}, {}", p.clone().into_iter().reduce(f32::min).unwrap(), p.clone().into_iter().reduce(f32::max).unwrap());

        let sym_grad_w_bar = sym_gradient(&w_bar.view());
        par_azip!((x in &mut q, &y in &sym_grad_w_bar) {
            *x += &sigma * &lambda * y;
        });
        let q = proj_q(&q.view(), &(&alpha1));

        // Primal updates
        let u_old = u.clone();
        let w_old = w.clone();

        // Update u: TGV + data fidelity
        let div_p = divergence(&p.view());
        par_azip!((x in &mut u, &y in &div_p) {
            *x -= tau * &lambda * y;
        });

        // Data fidelity term
        let complex_u = Array2::<Complex<f64>>::from_shape_vec((ny, nx), (&u).into_iter().map(|x| Complex::new(*x as f64, 0.0)).collect()).unwrap();
        let fft_u = fft2shift(&fft2(&complex_u.view()).view());
        let mut residual = Array2::<Complex<f64>>::zeros((ny, nx));
        par_azip!((x in &mut residual, &y in &fft_u, &z in centered_kspace, &w in centered_mask) {
            if w > 0. {
                *x = y - z;
            }
        });
        let ifft_residual = ifft2(&ifft2shift(&residual.view()).view());
        let ifft_residual = Array2::<f32>::from_shape_vec((ny, nx), ifft_residual.into_iter().map(|x| x.re as f32).collect()).unwrap();
        par_azip!((x in &mut u, &y in &ifft_residual) {
            *x -= tau * y;
        });

        // Update w
        let sym_div_q = sym_divergence(&q.view());
        par_azip!((x in &mut w, &y in &p, &z in &sym_div_q) {
            *x -= tau * (-y) * z;
        });
        
        // Extrapolation
        // Update momentum
        let t_old = t;
        let t = (1. + (1. + 4. * t_old.powi(2)).sqrt()) / 2.0;
        let theta = (1. - t_old) / t;
        par_azip!((x in &mut u_bar, &y in &u, &z in &u_old) {
            // *x = 2. * y - z;
            *x = y + theta * (y - z);
        });
        par_azip!((x in &mut w_bar, &y in &w, &z in &w_old) {
            // *x = 2. * y - z;
            *x = y + theta * (y - z);
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

