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
    c.bench_function("fft", |b| b.iter(|| t(black_box(rand_img.clone()))));
}

fn t(img: Image<f64>) {
    let mut i = img.map_pixel(|x| Complex64::new(x, 0.0));
    i.par_fft();
    let filter = {
        let mut f = Image::<Complex64> {
            pixels: (0..HEIGHT)
                .flat_map(|y| std::iter::repeat(y).zip((0..WIDTH)))
                .map(|(y, x)| {
                    if y < FILTER_SIZE && x < FILTER_SIZE {
                        Complex64::new(1.0, 0.0)
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                })
                .collect(),
            width: WIDTH,
            height: HEIGHT,
        };
        f.par_fft();
        f
    };
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            *unsafe { i.pixel_unchecked_mut(x, y) } *= unsafe { filter.pixel_unchecked(x, y) };
        }
    }
    i.par_ifft();
    i.map_pixel(|x| x.re);
}
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
