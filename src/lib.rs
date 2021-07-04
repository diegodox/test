use std::ops::{Add, AddAssign, Index, IndexMut, Sub};

use itertools::Itertools;
use rand::{distributions::Standard, prelude::Distribution, Rng};
use rayon::{
    iter::{IntoParallelRefMutIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use rustfft::num_complex::Complex64;

pub struct Point2D<T> {
    x: T,
    y: T,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Image<T> {
    pub pixels: Vec<T>,
    pub width: usize,
    pub height: usize,
}

impl<T> Index<usize> for Image<T> {
    type Output = [T];
    fn index(&self, index: usize) -> &Self::Output {
        &self.pixels[index * self.width..(index + 1) * self.width]
    }
}
impl<T> IndexMut<usize> for Image<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.pixels[index * self.width..(index + 1) * self.width]
    }
}

impl<T> Index<Point2D<usize>> for Image<T> {
    type Output = T;
    fn index(&self, index: Point2D<usize>) -> &Self::Output {
        self.pixel(index.x, index.y).unwrap()
    }
}
impl<T> IndexMut<Point2D<usize>> for Image<T> {
    fn index_mut(&mut self, index: Point2D<usize>) -> &mut Self::Output {
        self.pixel_mut(index.x, index.y).unwrap()
    }
}

impl<T> Image<T> {
    pub fn map_pixel<P>(self, f: impl Fn(T) -> P) -> Image<P> {
        let Image {
            pixels,
            width,
            height,
        } = self;
        Image {
            pixels: pixels.into_iter().map(f).collect(),
            width,
            height,
        }
    }
    pub fn pixel(&self, x: usize, y: usize) -> Option<&T> {
        self.pixels.get(y * self.width + x)
    }
    pub fn pixel_mut(&mut self, x: usize, y: usize) -> Option<&mut T> {
        self.pixels.get_mut(y * self.width + x)
    }
    pub unsafe fn pixel_unchecked(&self, x: usize, y: usize) -> &T {
        self.pixels.get_unchecked(y * self.width + x)
    }
    pub unsafe fn pixel_unchecked_mut(&mut self, x: usize, y: usize) -> &mut T {
        self.pixels.get_unchecked_mut(y * self.width + x)
    }
    pub fn size(&self) -> Point2D<usize> {
        Point2D {
            x: self.width,
            y: self.height,
        }
    }
    pub fn rand_gen(width: usize, height: usize, rng: &mut impl Rng) -> Self
    where
        Standard: Distribution<T>,
    {
        Self {
            pixels: (0..height * width).map(|_| rng.gen::<T>()).collect(),
            width,
            height,
        }
    }

    /// Get a image's width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get a image's height.
    pub fn height(&self) -> usize {
        self.height
    }
}
impl<T: Default> Image<T> {
    pub fn default(width: usize, height: usize) -> Self {
        Self {
            pixels: (0..height * width).map(|_| T::default()).collect(),
            width,
            height,
        }
    }
}
impl<T: AddAssign + Add<Output = T> + Sub<Output = T> + Clone> Image<T> {
    pub fn acm(&mut self) {
        for (y, yp) in (0..self.height()).into_iter().tuple_windows() {
            let x = self[y][0].clone();
            self[yp][0] += x;
        }
        for (x, xp) in (0..self.width()).into_iter().tuple_windows() {
            let x = self[0][x].clone();
            self[0][xp] += x;
        }
        for (y, yp) in (0..self.height()).into_iter().tuple_windows() {
            for (x, xp) in (0..self.width()).tuple_windows() {
                let x = self[yp][x].clone() + self[y][xp].clone() - self[y][x].clone();
                self[yp][xp] += x;
            }
        }
    }
}
impl Image<Complex64> {
    pub fn par_fft(&mut self) -> &mut Self {
        let Point2D {
            x: width,
            y: height,
        } = self.size();
        let mut fft_planner = rustfft::FftPlanner::new();
        // fft x dir
        let fft_x = fft_planner.plan_fft_forward(width);
        let mut output = self.pixels.clone();
        output
            .par_chunks_exact_mut(width)
            .for_each(|input| fft_x.process(input));
        // transpose fft x output
        let mut t = {
            let mut t = output.clone();
            transpose::transpose(&output, &mut t, width, height);
            t
        };
        // fft y dir
        let fft_y = fft_planner.plan_fft_forward(height);
        t.par_chunks_exact_mut(height)
            .for_each(|input| fft_y.process(input));
        // transpose fft y output
        transpose::transpose(&t, &mut output, height, width);
        self.pixels = output;
        self
    }
    pub fn par_ifft(&mut self) -> &mut Self {
        let Point2D {
            x: width,
            y: height,
        } = self.size();
        let mut fft_planner = rustfft::FftPlanner::new();
        // transpose self
        let mut t = {
            let mut t = self.pixels.clone();
            transpose::transpose(&self.pixels, &mut t, width, height);
            t
        };
        // ifft y dir
        let fft_y = fft_planner.plan_fft_inverse(height);
        t.par_chunks_exact_mut(height).for_each(|input| {
            fft_y.process(input);
        });
        // transpose ifft y output
        transpose::transpose(&t, &mut self.pixels, height, width);
        // ifft x dir
        let fft_x = fft_planner.plan_fft_inverse(width);
        self.pixels
            .par_chunks_exact_mut(width)
            .for_each(|input| fft_x.process(input));
        self.pixels
            .par_iter_mut()
            .for_each(|x| *x /= (width * height) as f64);
        self
    }
}
#[cfg(test)]
mod test {
    use approx::{assert_relative_eq, relative_eq};
    use itertools::Itertools;
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;
    use rustfft::num_complex::Complex64;

    use crate::Image;

    #[test]
    fn test_acm() {
        let img = Image {
            pixels: vec![1, 2, 3, 4],
            width: 2,
            height: 2,
        };
        let ans = Image {
            pixels: vec![1, 3, 4, 10],
            width: 2,
            height: 2,
        };
        let acm = {
            let mut acm = img.clone();
            acm.acm();
            acm
        };
        assert_eq!(acm, ans);
    }

    #[test]
    fn test_acm2() {
        let img = Image {
            pixels: vec![1, 1, 1, 1],
            width: 2,
            height: 2,
        };
        let ans = Image {
            pixels: vec![1, 2, 2, 4],
            width: 2,
            height: 2,
        };
        let acm = {
            let mut acm = img.clone();
            acm.acm();
            acm
        };
        assert_eq!(acm, ans);
    }

    #[test]
    fn test_fft() {
        let mut rng = Pcg64::seed_from_u64(2);
        let rand_img =
            Image::<f64>::rand_gen(100, 100, &mut rng).map_pixel(|x| Complex64::new(x, 0.0));
        let fft = {
            let mut img = rand_img.clone();
            img.par_fft().par_ifft();
            img
        };

        for (a, b) in rand_img.pixels.into_iter().zip(fft.pixels.into_iter()) {
            assert_relative_eq!(a.re, b.re, epsilon = 1e-15);
        }
    }

    #[test]
    fn ave() {
        const WIDTH: usize = 1000;
        const HEIGHT: usize = 1000;
        const FILTER_SIZE: usize = 3;
        let mut rng = Pcg64::seed_from_u64(2);
        let rand_img = Image::<f64>::rand_gen(WIDTH, HEIGHT, &mut rng);
        // let rand_img = Image::<f64> {
        //     width: WIDTH,
        //     height: HEIGHT,
        //     pixels: (0..WIDTH * HEIGHT).map(|_| 1.0).collect(),
        // };
        let fft = {
            let mut i = rand_img.clone().map_pixel(|x| Complex64::new(x, 0.0));
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
                    *unsafe { i.pixel_unchecked_mut(x, y) } *=
                        unsafe { filter.pixel_unchecked(x, y) };
                }
            }
            i.par_ifft();
            i.map_pixel(|x| x.re)
        };
        let acm = {
            let mut i = rand_img.clone();
            i.acm();
            let mut t = rand_img.clone();
            for y in FILTER_SIZE..HEIGHT {
                for x in FILTER_SIZE..WIDTH {
                    let v = *unsafe { i.pixel_unchecked(x, y) }
                        - *unsafe { i.pixel_unchecked(x - FILTER_SIZE, y) }
                        - *unsafe { i.pixel_unchecked(x, y - FILTER_SIZE) }
                        + *unsafe { i.pixel_unchecked(x - FILTER_SIZE, y - FILTER_SIZE) };
                    *unsafe { t.pixel_unchecked_mut(x, y) } = v;
                }
            }
            t
        };
        for y in FILTER_SIZE..HEIGHT - FILTER_SIZE {
            for x in FILTER_SIZE..WIDTH - FILTER_SIZE {
                let a = *unsafe { fft.pixel_unchecked(x, y) };
                let b = *unsafe { acm.pixel_unchecked(x, y) };
                assert!(
                    relative_eq!(a, b, epsilon = 1e-8),
                    "a: {}, b: {}, x: {}, y: {}",
                    a,
                    b,
                    x,
                    y
                );
            }
        }
    }
}
