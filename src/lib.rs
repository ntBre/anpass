use approx::AbsDiffEq;
use fc::Fc;
use nalgebra as na;
use regex::Regex;
use std::fmt::Display;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;

pub mod fc;

/// conversion factor for force constants written out in fort.9903
const FAC: f64 = 4.359813653e0;
/// threshold for considering an element of the gradient or Hessian to be zero
const THR: f64 = 1e-10;

pub type Dmat = na::DMatrix<f64>;
pub type Dvec = na::DVector<f64>;

#[derive(Debug, PartialEq, Clone)]
pub struct Bias {
    pub disp: Dvec,
    pub energy: f64,
}

#[derive(Debug, Clone)]
pub struct Anpass {
    pub disps: Dmat,
    /// empty if loaded from a template without energies, as determined by the
    /// documentation for `load`
    pub energies: Dvec,
    /// i32 for compatibility with `f64::powi`
    pub exponents: na::DMatrix<i32>,
    ///  empty if not running at a stationary point
    pub bias: Option<Bias>,
}

impl PartialEq for Anpass {
    fn eq(&self, other: &Self) -> bool {
        self.disps.abs_diff_eq(&other.disps, 1e-12)
            && self.energies.abs_diff_eq(&other.energies, 1e-11)
            && self.exponents.eq(&other.exponents)
            && self.bias.eq(&other.bias)
    }
}

impl Display for Anpass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (rows, cols) = self.disps.shape();
        for row in 0..rows {
            for col in 0..cols {
                write!(f, "{:12.8}", self.disps[(row, col)])?;
            }
            writeln!(f, "{:20.12}", self.energies[row])?;
        }
        writeln!(f, "UNKNOWNS")?;
        let (rows, cols) = self.exponents.shape();
        writeln!(f, "{:4}", cols)?;
        writeln!(f, "FUNCTION")?;
        for row in 0..rows {
            for col in 0..cols {
                if col > 0 && col % 16 == 0 {
                    writeln!(f)?;
                }
                write!(f, "{:5}", self.exponents[(row, col)])?;
            }
            writeln!(f)?;
        }
        writeln!(f, "END OF DATA")?;
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub enum StatKind {
    Max,
    Min,
    Stat,
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
        let mut bias = std::option::Option::None;
        for line in lines {
            if start.is_match(&line) {
                ndisp_fields =
                    start.captures(&line).unwrap()[1].parse().unwrap();
                state = Disp;
            } else if line.contains("UNKNOWNS") {
                state = Unks;
            } else if line.contains("STATIONARY POINT")
                && !line.starts_with("!")
            {
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
                let line = line
                    .split_whitespace()
                    .flat_map(|s| s.parse::<f64>())
                    .collect::<Vec<_>>();
                let l = line.len();
                bias = Some(Bias {
                    disp: Dvec::from(line[..l - 1].to_vec()),
                    energy: line[l - 1],
                });
                state = None;
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
            bias,
        }
    }

    /// determine the [ordinary least
    /// squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) solution
    /// to the [polynomial
    /// regression](https://en.wikipedia.org/wiki/Polynomial_regression) problem
    /// described by `self.disps`, `self.energies`, and `self.exponents`, and
    /// return the solution vector along with the evaluated matrix describing
    /// the function. The latter is for checking the residuals. See the PDF
    /// documentation for further details
    pub fn fit(&self) -> (Dvec, Dmat) {
        let (ndisps, ncols) = self.disps.shape();
        let (_, nunks) = self.exponents.shape();
        let mut x = Dmat::repeat(ndisps, nunks, 1.0);
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

    /// characterize the stationary point described by `hess`
    fn characterize(&self, hess: &Dmat) -> StatKind {
        let evals = hess
            .eigenvalues()
            .expect("eigendcomposition failed in `newton`");
        let prod = evals.fold(0, |acc, v| {
            if v < 0.0 {
                acc - 1
            } else if v > 0.0 {
                acc + 1
            } else {
                acc
            }
        });
        let l = evals.len() as isize;
        if prod == -l {
            StatKind::Max
        } else if prod == l {
            StatKind::Min
        } else {
            StatKind::Stat
        }
    }

    /// use [Newton's optimization
    /// method](https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization)
    /// to find the roots of the equation described by `coeffs` and
    /// `self.exponents`. return the stationary point and the final Hessian
    /// matrix
    pub fn newton(&self, coeffs: &Dvec) -> (Dvec, StatKind) {
        const MAXIT: usize = 100;
        let (nvbl, _) = self.exponents.shape();
        let mut x = Dvec::repeat(nvbl, 0.0);
        for _ in 0..MAXIT {
            let grad = self.grad(&x, coeffs);
            let hess = self.hess(&x, coeffs);
            let chol = match na::Cholesky::new(hess.clone()) {
                Some(mat) => mat,
                None => {
		    let mut f = std::fs::File::create("anpass.bad").unwrap();
		    write!(f, "{}", self).unwrap();
                    panic!("Cholesky decomposition failed in `newton`");
                }
            };
            let inv = chol.inverse();
            let delta = 0.5 * inv * grad;
            if delta.iter().all(|x| *x <= 1.1e-8) {
                return (x, self.characterize(&hess));
            }
            x -= delta;
        }
        panic!("too many Newton iterations");
    }

    /// evaluate the function at the point `x`
    pub fn eval(&self, x: &Dvec, coeffs: &Dvec) -> f64 {
        let mut sum = 0.0;
        for (k, prod) in coeffs.iter().enumerate() {
            let mut prod = prod.clone();
            if prod.abs() < THR {
                continue;
            }
            for (j, xi) in x.iter().enumerate() {
                let ejk = self.exponents[(j, k)];
                if ejk != 0 {
                    prod *= xi.powi(ejk);
                }
            }
            sum += prod;
        }
        sum
    }

    pub fn bias(&self, bias: &Bias) -> Self {
        let (rows, cols) = self.disps.shape();
        let mut disps = Vec::with_capacity(rows * cols);
        let mut energies = Vec::with_capacity(rows);
        for r in 0..rows {
            disps.extend(
                (self.disps.row(r).transpose() - bias.disp.clone()).iter(),
            );
            energies.push(self.energies[r] - bias.energy);
        }
        Self {
            disps: Dmat::from_row_slice(rows, cols, &disps),
            energies: Dvec::from(energies),
            ..self.clone()
        }
    }

    pub fn make9903(&self, coeffs: &Dvec) -> Vec<Fc> {
        let (c, r) = self.exponents.shape();
        let mut ret = Vec::new();
        for i in 0..r {
            let mut ifact = 1.0;
            let mut ictmp = [0; 4];
            let mut iccount: usize = 0;
            for j in (0..c).rev() {
                let iexpo = self.exponents[(j, i)];
                ifact *= [1.0, 1.0, 2.0, 6.0, 24.0][iexpo as usize];
                if iexpo > 0 {
                    for k in 0..iexpo {
                        ictmp[iccount + k as usize] = j + 1;
                    }
                    iccount += iexpo as usize;
                }
            }
            let ffcc = coeffs[i] * ifact * FAC;
            let [a, b, c, d] = ictmp;
            ret.push(Fc(a, b, c, d, ffcc));
        }
        ret
    }

    pub fn write9903<W: Write>(&self, w: &mut W, fcs: &[Fc]) {
        writeln!(w).unwrap();
        for fc in fcs {
            writeln!(w, "{fc}",).unwrap();
        }
    }

    /// perform the initial fitting, find the stationary point, bias to the new
    /// stationary point, and refit. returns the force constants at the
    /// stationary point and the bias (long line)
    pub fn run(&self) -> (Vec<Fc>, Bias) {
        let (coeffs, _) = self.fit();
        // find stationary point
        let (x, _) = self.newton(&coeffs);
        // determine energy at stationary point
        let e = self.eval(&x, &coeffs);
        // bias the displacements and energies to the new stationary point
        let bias = Bias { disp: x, energy: e };
        let anpass = self.bias(&bias);
        // perform the refitting
        let (coeffs, _) = anpass.fit();
        (anpass.make9903(&coeffs), bias)
    }
}
