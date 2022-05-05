use approx::AbsDiffEq;
use nalgebra as na;
use regex::Regex;
use std::io::BufRead;
use std::io::BufReader;

/// conversion factor for force constants written out in fort.9903
const _FAC: f64 = 4.359813653e0;
/// threshold for considering an element of the gradient or Hessian to be zero
const THR: f64 = 1e-10;

type Dmat = na::DMatrix<f64>;
type Dvec = na::DVector<f64>;

#[derive(Debug)]
pub struct Anpass {
    pub disps: Dmat,
    /// empty if loaded from a template without energies, as determined by the
    /// documentation for `load`
    pub energies: Dvec,
    /// i32 for compatibility with `f64::powi`
    pub exponents: na::DMatrix<i32>,
    ///  empty if not running at a stationary point
    pub biases: Vec<f64>,
}

impl PartialEq for Anpass {
    fn eq(&self, other: &Self) -> bool {
        self.disps.abs_diff_eq(&other.disps, 1e-12)
            && self.energies.abs_diff_eq(&other.energies, 1e-11)
            && self.exponents.eq(&other.exponents)
            && self.biases.eq(&other.biases)
    }
}

impl Anpass {
    /// Load an Anpass from `filename`. Everything before a line like
    /// `(3F12.8,f20.12)` is ignored. This line signals the start of the
    /// displacements. If the number of formats given in this line matches the
    /// number of fields in each displacement line, the last field is treated as
    /// an energy. Otherwise, every field is treated as a displacement
    pub fn load(filename: &str) -> Self {
        let f = match std::fs::File::open(filename) {
            Ok(f) => f,
            Err(e) => panic!("failed to open {filename} with {e}"),
        };
        let lines = BufReader::new(f).lines().flatten();
        let start =
            Regex::new(r"(?i)^\s*\((\d+)f[0-9.]+,f[0-9.]+\)\s*$").unwrap();
        let mut ndisp_fields = usize::default();
        #[derive(PartialEq)]
        enum State {
            Disp,
            Exps,
            Unks,
            Stat,
            None,
        }
        use State::*;
        let mut state = None;
        let mut disps = Vec::new();
        let mut ndisps = 0;
        let mut energies = Vec::new();
        let mut nunk = usize::default();
        let mut exponents = Vec::new();
        let mut biases = Vec::new();
        for line in lines {
            if start.is_match(&line) {
                ndisp_fields =
                    start.captures(&line).unwrap()[1].parse().unwrap();
                state = Disp;
            } else if line.contains("UNKNOWNS") {
                state = Unks;
            } else if line.contains("STATIONARY POINT") {
                state = Stat;
            } else if state == Disp {
                let f = line
                    .split_whitespace()
                    .flat_map(|s| s.parse::<f64>())
                    .collect::<Vec<_>>();
                let fl = f.len() - 1;
                if fl == ndisp_fields {
                    // disps + energy
                    disps.extend_from_slice(&f[..fl]);
                    energies.push(f[fl]);
                } else {
                    // only disps
                    disps.extend(f);
                }
                ndisps += 1;
            } else if state == Unks {
                nunk = line.trim().parse().unwrap();
                state = Exps;
            } else if state == Exps {
                exponents.extend(
                    line.split_whitespace().flat_map(|s| s.parse::<i32>()),
                );
            } else if state == Stat {
                biases.extend(
                    line.split_whitespace().flat_map(|s| s.parse::<f64>()),
                );
            }
        }
        Self {
            disps: Dmat::from_row_slice(ndisps, ndisp_fields, &disps),
            energies: Dvec::from(energies),
            exponents: na::DMatrix::from_row_slice(
                exponents.len() / nunk,
                nunk,
                &exponents,
            ),
            biases,
        }
    }

    /// determine the [ordinary least
    /// squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) solution
    /// to the [polynomial
    /// regression](https://en.wikipedia.org/wiki/Polynomial_regression) problem
    /// described by `self.disps`, `self.energies`, and `self.exponents`, and
    /// return the solution vector along with the evaluated matrix describing
    /// the function. See the PDF documentation for further details
    pub fn fit(&self) -> (Dvec, Dmat) {
        let (ndisps, ncols) = self.disps.shape();
        let (_, nunks) = self.exponents.shape();
        let mut x = Dmat::repeat(ndisps, nunks, 1.0);
        // TODO this is probably too naive
        for i in 0..ndisps {
            for k in 0..nunks {
                for j in 0..ncols {
                    x[(i, k)] *=
                        self.disps[(i, j)].powi(self.exponents[(j, k)]);
                }
            }
        }
        let xtx = x.transpose() * &x;
        let chol = na::Cholesky::new(xtx)
            .expect("Cholesky decomposition failed in `fit`");
        let inv = chol.inverse();
        let a = inv * x.transpose();
        let f = a * &self.energies;
        (f, x)
    }

    /// compute the gradient of the function described by `coeffs` at `x`
    fn grad(&self, x: &Dvec, coeffs: &Dvec) -> Dvec {
        let (nvbl, nunk) = self.exponents.shape();
        let mut grad = vec![0.0; nvbl];
        for i in 0..nvbl {
            let mut sum = 0.0;
            for j in 0..nunk {
                let fij = self.exponents[(i, j)];
                let mut coj = coeffs[j] * fij as f64;
                if coj.abs() < THR {
                    continue;
                }
                if fij != 1 {
                    coj *= x[i].powi(fij - 1);
                }
                for k in 0..nvbl {
                    let ekj = self.exponents[(k, j)];
                    if k != i && ekj != 0 {
                        coj *= x[k].powi(ekj);
                    }
                }
                sum += coj;
            }
            grad[i] = sum;
        }
        Dvec::from(grad)
    }

    /// compute the hessian of the function described by `coeffs` at `x`
    fn hess(&self, x: &Dvec, coeffs: &Dvec) -> Dmat {
        let (nvbl, nunk) = self.exponents.shape();
        let mut hess = Dmat::zeros(nvbl, nvbl);
        for i in 0..nvbl {
            for l in 0..=i {
                let mut sum = 0.0;
                if i != l {
                    // off-diagonal
                    for j in 0..nunk {
                        let mut coj = coeffs[j];
                        let eij = self.exponents[(i, j)];
                        let elj = self.exponents[(l, j)];
                        let fij = eij as f64;
                        let flj = elj as f64;
                        coj *= fij * flj;
                        if coj.abs() < THR {
                            continue;
                        }
                        if eij != 1 {
                            coj *= x[i].powi(eij - 1);
                        }
                        if elj != 1 {
                            coj *= x[l].powi(elj - 1);
                        }
                        for k in 0..nvbl {
                            if k != i && k != l {
                                let ekj = self.exponents[(k, j)];
                                if ekj != 0 {
                                    coj *= x[k].powi(ekj);
                                }
                            }
                        }
                        sum += coj;
                    }
                    hess[(i, l)] = sum;
                    hess[(l, i)] = sum;
                } else {
                    // diagonal
                    for j in 0..nunk {
                        let mut coj = coeffs[j];
                        let eij = self.exponents[(i, j)];
                        let fij = eij as f64;
                        coj *= fij * (fij - 1.);
                        if coj.abs() < THR {
                            continue;
                        }
                        if eij != 2 {
                            coj *= x[i].powi(eij - 2);
                        }
                        for k in 0..nvbl {
                            if k != i {
                                let ekj = self.exponents[(k, j)];
                                if ekj != 0 {
                                    coj *= x[k].powi(ekj);
                                }
                            }
                        }
                        sum += coj;
                    }
                    hess[(i, l)] = sum;
                }
            }
        }
        hess
    }

    /// use [Newton's optimization
    /// method](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)
    /// to find the roots of the equation described by `coeffs` and
    /// `self.exponents`.
    pub fn newton(&self, coeffs: &Dvec) -> Dvec {
        const MAXIT: usize = 100;
        let (nvbl, _) = self.exponents.shape();
        let mut x = Dvec::repeat(nvbl, 0.0);
        for _ in 0..MAXIT {
            let grad = self.grad(&x, coeffs);
            let hess = self.hess(&x, coeffs);
            let inv = na::Cholesky::new(hess)
                .expect("Cholesky decomposition failed in `newton`")
                .inverse();
            let delta = 0.5 * inv * grad;
            if delta.iter().all(|x| *x <= 1.1e-8) {
                return x;
            }
            x -= delta;
        }
        panic!("too many Newton iterations");
    }
}
