use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fftvsdp::Image;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;
use rustfft::num_complex::Complex64;

const WIDTH: usize = 1000;
const HEIGHT: usize = 1000;
const FILTER_SIZE: usize = 3;

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut rng = Pcg64::seed_from_u64(2);
    let rand_img = Image::<f64>::rand_gen(WIDTH, HEIGHT, &mut rng);
    c.bench_function("acm", |b| b.iter(|| t(black_box(rand_img.clone()))));
}

fn t(img: Image<f64>) {
    let mut i = img.clone();
    i.acm();
    let mut t = img;
    for y in FILTER_SIZE..HEIGHT {
        for x in FILTER_SIZE..WIDTH {
            let v = *unsafe { i.pixel_unchecked(x, y) }
                - *unsafe { i.pixel_unchecked(x - FILTER_SIZE, y) }
                - *unsafe { i.pixel_unchecked(x, y - FILTER_SIZE) }
                + *unsafe { i.pixel_unchecked(x - FILTER_SIZE, y - FILTER_SIZE) };
            *unsafe { t.pixel_unchecked_mut(x, y) } = v;
        }
    }
}
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
