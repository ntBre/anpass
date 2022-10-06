use rust_anpass::{Anpass, Bias};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let infile = args.get(1);
    let anpass = match infile {
        Some(s) => Anpass::load_file(s),
        None => Anpass::load(std::io::stdin()),
    };
    // perform the initial fitting
    let (coeffs, _) = anpass.fit();
    // find the stationary point
    let (x, _) = anpass.newton(&coeffs);
    // determine energy at the stationary point
    let e = anpass.eval(&x, &coeffs);
    // bias the displacements and energies to the new stationary point
    let anpass = anpass.bias(&Bias { disp: x, energy: e });
    // perform the refitting
    let (coeffs, _) = anpass.fit();
    // make and write the fort.9903 file expected by intder
    let f9903 = anpass.make9903(&coeffs);
    let filename = "fort.9903";
    let mut f = match std::fs::File::create(filename) {
        Ok(f) => f,
        Err(e) => panic!("failed to create {filename} with {e}"),
    };
    anpass.write9903(&mut f, &f9903);
}
