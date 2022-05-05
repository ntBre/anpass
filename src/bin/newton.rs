use rust_anpass::Anpass;

fn main() {
    let anpass = Anpass::load("testfiles/c3h2.in");
    let (coeffs, _) = anpass.fit();
    anpass.newton(&coeffs);
}
