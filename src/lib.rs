use approx::AbsDiffEq;
use nalgebra as na;
use regex::Regex;
use std::io::BufRead;
use std::io::BufReader;

type Dmat = na::DMatrix<f64>;
type Dvec = na::DVector<f64>;

#[derive(Debug)]
pub struct Anpass {
    pub disps: Dmat,
    /// empty if loaded from a template without energies, as determined by the
    /// documentation for `load`
    pub energies: Dvec,
    /// i32 for compatibility with `f64::powi`
    pub exponents: Vec<Vec<i32>>,
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
        let mut exp_buf = Vec::new();
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
                exp_buf.extend(
                    line.split_whitespace().flat_map(|s| s.parse::<i32>()),
                );
                if exp_buf.len() == nunk {
                    exponents.push(exp_buf);
                    exp_buf = Vec::with_capacity(nunk);
                }
            } else if state == Stat {
                biases.extend(
                    line.split_whitespace().flat_map(|s| s.parse::<f64>()),
                );
            }
        }
        Self {
            disps: Dmat::from_row_slice(ndisps, ndisp_fields, &disps),
            energies: Dvec::from(energies),
            exponents,
            biases,
        }
    }

    /// determine the [ordinary least
    /// squares](https://en.wikipedia.org/wiki/Ordinary_least_squares) solution
    /// to the [polynomial
    /// regression](https://en.wikipedia.org/wiki/Polynomial_regression) problem
    /// described by `self.disps`, `self.energies`, and `self.exponents`. See
    /// the PDF documentation for further details
    pub fn fit(&self) {
    }
}
