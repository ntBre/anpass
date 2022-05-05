#![allow(unused)]
use approx::AbsDiffEq;
use nalgebra as na;
use regex::Regex;
use std::io::BufRead;
use std::io::BufReader;

type Dmat = na::DMatrix<f64>;
type Dvec = na::DVector<f64>;

#[derive(Debug)]
pub struct Anpass {
    disps: Dmat,
    /// empty if loaded from a template without energies, as determined by the
    /// documentation for `load`
    energies: Dvec,
    /// i32 for compatibility with `f64::powi`
    exponents: Vec<Vec<i32>>,
    ///  empty if not running at a stationary point
    biases: Vec<f64>,
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
        let mut in_disps = false;
        let mut disps = Vec::new();
        let mut ndisps = 0;
        let mut energies = Vec::new();
        let mut in_exps = false;
        let mut in_unk = false;
        let mut nunk = usize::default();
        let mut exponents = Vec::new();
        let mut exp_buf = Vec::new();
        let mut in_stat = false;
        let mut biases = Vec::new();
        for line in lines {
            if start.is_match(&line) {
                ndisp_fields =
                    start.captures(&line).unwrap()[1].parse().unwrap();
                in_disps = true;
            } else if line.contains("UNKNOWNS") {
                in_disps = false;
                in_unk = true;
            } else if line.contains("STATIONARY POINT") {
                in_stat = true;
            } else if in_disps {
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
            } else if in_unk {
                nunk = line.trim().parse().unwrap();
                in_unk = false;
                in_exps = true;
            } else if in_exps {
                exp_buf.extend(
                    line.split_whitespace().flat_map(|s| s.parse::<i32>()),
                );
                if exp_buf.len() == nunk {
                    exponents.push(exp_buf);
                    exp_buf = Vec::with_capacity(nunk);
                }
            } else if in_stat {
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
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    #[test]
    fn test_load() {
        let anpass = Anpass::load("testfiles/anpass.in");
        let want = Anpass {
            #[rustfmt::skip]
            disps: Dmat::from_row_slice(
                69,
                3,
                &vec![
                    -0.00500000, -0.00500000, -0.01000000,
                    -0.00500000, -0.00500000, 0.00000000,
                    -0.00500000, -0.00500000, 0.01000000,
                    -0.00500000, -0.01000000, 0.00000000,
                    -0.00500000, -0.01500000, 0.00000000,
                    -0.00500000, 0.00000000, -0.01000000,
                    -0.00500000, 0.00000000, 0.00000000,
                    -0.00500000, 0.00000000, 0.01000000,
                    -0.00500000, 0.00500000, -0.01000000,
                    -0.00500000, 0.00500000, 0.00000000,
                    -0.00500000, 0.00500000, 0.01000000,
                    -0.00500000, 0.01000000, 0.00000000,
                    -0.00500000, 0.01500000, 0.00000000,
                    -0.01000000, -0.00500000, 0.00000000,
                    -0.01000000, -0.01000000, 0.00000000,
                    -0.01000000, 0.00000000, -0.01000000,
                    -0.01000000, 0.00000000, 0.00000000,
                    -0.01000000, 0.00000000, 0.01000000,
                    -0.01000000, 0.00500000, 0.00000000,
                    -0.01000000, 0.01000000, 0.00000000,
                    -0.01500000, -0.00500000, 0.00000000,
                    -0.01500000, 0.00000000, 0.00000000,
                    -0.01500000, 0.00500000, 0.00000000,
                    -0.02000000, 0.00000000, 0.00000000,
                    0.00000000, -0.00500000, -0.01000000,
                    0.00000000, -0.00500000, 0.00000000,
                    0.00000000, -0.00500000, 0.01000000,
                    0.00000000, -0.01000000, -0.01000000,
                    0.00000000, -0.01000000, 0.00000000,
                    0.00000000, -0.01000000, 0.01000000,
                    0.00000000, -0.01500000, 0.00000000,
                    0.00000000, -0.02000000, 0.00000000,
                    0.00000000, 0.00000000, -0.01000000,
                    0.00000000, 0.00000000, -0.02000000,
                    0.00000000, 0.00000000, 0.00000000,
                    0.00000000, 0.00000000, 0.01000000,
                    0.00000000, 0.00000000, 0.02000000,
                    0.00000000, 0.00500000, -0.01000000,
                    0.00000000, 0.00500000, 0.00000000,
                    0.00000000, 0.00500000, 0.01000000,
                    0.00000000, 0.01000000, -0.01000000,
                    0.00000000, 0.01000000, 0.00000000,
                    0.00000000, 0.01000000, 0.01000000,
                    0.00000000, 0.01500000, 0.00000000,
                    0.00000000, 0.02000000, 0.00000000,
                    0.00500000, -0.00500000, -0.01000000,
                    0.00500000, -0.00500000, 0.00000000,
                    0.00500000, -0.00500000, 0.01000000,
                    0.00500000, -0.01000000, 0.00000000,
                    0.00500000, -0.01500000, 0.00000000,
                    0.00500000, 0.00000000, -0.01000000,
                    0.00500000, 0.00000000, 0.00000000,
                    0.00500000, 0.00000000, 0.01000000,
                    0.00500000, 0.00500000, -0.01000000,
                    0.00500000, 0.00500000, 0.00000000,
                    0.00500000, 0.00500000, 0.01000000,
                    0.00500000, 0.01000000, 0.00000000,
                    0.00500000, 0.01500000, 0.00000000,
                    0.01000000, -0.00500000, 0.00000000,
                    0.01000000, -0.01000000, 0.00000000,
                    0.01000000, 0.00000000, -0.01000000,
                    0.01000000, 0.00000000, 0.00000000,
                    0.01000000, 0.00000000, 0.01000000,
                    0.01000000, 0.00500000, 0.00000000,
                    0.01000000, 0.01000000, 0.00000000,
                    0.01500000, -0.00500000, 0.00000000,
                    0.01500000, 0.00000000, 0.00000000,
                    0.01500000, 0.00500000, 0.00000000,
                    0.02000000, 0.00000000, 0.00000000,
                ],
            ),
            #[rustfmt::skip]
            energies: Dvec::from(vec![
                0.000128387078, 0.000027809414, 0.000128387078,
                0.000035977201, 0.000048243883, 0.000124321064,
                0.000023720402, 0.000124321065, 0.000124313373,
                0.000023689948, 0.000124313373, 0.000027697745,
                0.000035723392, 0.000102791171, 0.000113093098,
                0.000199639109, 0.000096581025, 0.000199639109,
                0.000094442297, 0.000096354531, 0.000228163468,
                0.000219814727, 0.000215550318, 0.000394681651,
                0.000100159437, 0.000001985383, 0.000100159437,
                0.000106187756, 0.000008036587, 0.000106187756,
                0.000018173585, 0.000032416257, 0.000098196697,
                0.000392997365, 0.000000000000, 0.000098196697,
                0.000392997364, 0.000100279477, 0.000002060371,
                0.000100279477, 0.000106387616, 0.000008146336,
                0.000106387616, 0.000018237641, 0.000032313930,
                0.000119935606, 0.000024112936, 0.000119935606,
                0.000028065156, 0.000036090120, 0.000120058596,
                0.000024213636, 0.000120058597, 0.000124214356,
                0.000028347337, 0.000124214356, 0.000036494030,
                0.000048633604, 0.000093011998, 0.000094882871,
                0.000188725453, 0.000095181193, 0.000188725453,
                0.000101370691, 0.000111560627, 0.000207527972,
                0.000211748039, 0.000219975758, 0.000372784451,
            ]),
            exponents: vec![
                vec![
                    0, 1, 0, 2, 1, 0, 0, 3, 2, 1, 0, 1, 0, 4, 3, 2, 1, 0, 2, 1,
                    0, 0,
                ],
                vec![
                    0, 0, 1, 0, 1, 2, 0, 0, 1, 2, 3, 0, 1, 0, 1, 2, 3, 4, 0, 1,
                    2, 0,
                ],
                vec![
                    0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2,
                    2, 4,
                ],
            ],
            biases: vec![],
        };
        assert_abs_diff_eq!(anpass.disps, want.disps);
        assert_eq!(anpass.energies.len(), want.energies.len());
        assert_abs_diff_eq!(anpass.energies, want.energies);
        assert_eq!(anpass.exponents, want.exponents);
        assert_eq!(anpass.biases, want.biases);
    }
}
