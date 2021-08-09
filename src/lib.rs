use std::{
    fmt::{Display, Write},
    iter::{self, Once},
    marker::PhantomData,
    mem,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    ptr, slice,
};

pub trait Field:
    Sized
    + Clone
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Neg<Output = Self>
{
    /// The zero value of the field. This must correspond to the default value.
    fn zero() -> Self;

    /// The one value of the field.
    fn one() -> Self;
}

impl Field for f32 {
    fn zero() -> Self {
        0.
    }

    fn one() -> Self {
        1.
    }
}

impl Field for f64 {
    fn zero() -> Self {
        0.
    }

    fn one() -> Self {
        1.
    }
}

/// An algebra over a field.
pub trait Algebra<T: Field>:
    Field + From<T> + Mul<T, Output = Self> + MulAssign<T> + Div<T, Output = Self> + DivAssign<T>
{
    /// The dimension of the algebra over the field.
    const DIM: usize;

    /// The conjugate of a value.
    fn conj(&self) -> Self;

    /// Converts a number into its own conjugate.
    fn conj_mut(&mut self);

    /// The norm of a value.
    fn norm(&self) -> T;

    /// The multiplicative inverse of a value.
    fn inv(&self) -> Self {
        self.conj() / self.norm()
    }
}

impl Algebra<f32> for f32 {
    const DIM: usize = 1;

    fn conj(&self) -> Self {
        *self
    }

    fn conj_mut(&mut self) {}

    fn norm(&self) -> Self {
        self * self
    }

    fn inv(&self) -> Self {
        1.0 / self
    }
}

impl Algebra<f64> for f64 {
    const DIM: usize = 1;

    fn conj(&self) -> Self {
        *self
    }

    fn conj_mut(&mut self) {}

    fn norm(&self) -> Self {
        self * self
    }

    fn inv(&self) -> Self {
        1.0 / self
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
/// Applies the Cayley–Dickson construction to a type `U`.
pub struct Cayley<T: Field, U: Algebra<T>> {
    a: U,
    b: U,
    _phantom: PhantomData<T>,
}

impl<T: Field, U: Algebra<T>> Cayley<T, U> {
    /// Initializes a new number.
    pub fn new(a: U, b: U) -> Self {
        Self {
            a,
            b,
            _phantom: PhantomData,
        }
    }
}

impl<T: Field, U: Algebra<T>> From<T> for Cayley<T, U> {
    fn from(t: T) -> Self {
        Self::new(t.into(), U::zero())
    }
}

impl<T: Field, U: Algebra<T>> Add for Cayley<T, U> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.a + rhs.a, self.b + rhs.b)
    }
}

impl<T: Field, U: Algebra<T>> AddAssign for Cayley<T, U> {
    fn add_assign(&mut self, rhs: Self) {
        self.a += rhs.a;
        self.b += rhs.b;
    }
}

impl<T: Field, U: Algebra<T>> Sub for Cayley<T, U> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.a - rhs.a, self.b - rhs.b)
    }
}

impl<T: Field, U: Algebra<T>> SubAssign for Cayley<T, U> {
    fn sub_assign(&mut self, rhs: Self) {
        self.a -= rhs.a;
        self.b -= rhs.b;
    }
}

impl<T: Field, U: Algebra<T>> Mul<T> for Cayley<T, U> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::new(self.a * rhs.clone(), self.b * rhs)
    }
}

impl<T: Field, U: Algebra<T>> MulAssign<T> for Cayley<T, U> {
    fn mul_assign(&mut self, rhs: T) {
        self.a *= rhs.clone();
        self.b *= rhs;
    }
}

impl<T: Field, U: Algebra<T>> Mul for Cayley<T, U> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let a = self.a;
        let b = self.b;
        let c = rhs.a;
        let d = rhs.b;
        let c_conj = c.conj();

        Self::new(a.clone() * c - d.conj() * b.clone(), d * a + b * c_conj)
    }
}

impl<T: Field, U: Algebra<T>> MulAssign for Cayley<T, U> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<T: Field, U: Algebra<T>> Div<T> for Cayley<T, U> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self::new(self.a / rhs.clone(), self.b / rhs)
    }
}

impl<T: Field, U: Algebra<T>> DivAssign<T> for Cayley<T, U> {
    fn div_assign(&mut self, rhs: T) {
        self.a /= rhs.clone();
        self.b /= rhs;
    }
}

impl<T: Field, U: Algebra<T>> Div for Cayley<T, U> {
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inv()
    }
}

impl<T: Field, U: Algebra<T>> DivAssign for Cayley<T, U> {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone() / rhs;
    }
}

impl<T: Field, U: Algebra<T>> Neg for Cayley<T, U> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.a, -self.b)
    }
}

impl<T: Field, U: Algebra<T>> Field for Cayley<T, U> {
    fn zero() -> Self {
        Self::new(U::zero(), U::zero())
    }

    fn one() -> Self {
        Self::new(U::one(), U::zero())
    }
}

impl<T: Field, U: Algebra<T>> Algebra<T> for Cayley<T, U> {
    const DIM: usize = 2 * U::DIM;

    fn conj(&self) -> Self {
        Self::new(self.a.conj(), -self.b.clone())
    }

    fn conj_mut(&mut self) {
        self.a.conj_mut();
        self.b *= -U::one();
    }

    fn norm(&self) -> T {
        self.a.norm() + self.b.norm()
    }
}

impl<T: Field, U: Algebra<T>> Cayley<T, U> {
    /// Returns a slice to the inner vector of values.
    pub fn as_slice(&self) -> &[T] {
        unsafe { &*ptr::slice_from_raw_parts(self as *const _ as *const T, Self::DIM) }
    }

    /// Returns a mutable slice to the inner vector of values.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        unsafe { &mut *ptr::slice_from_raw_parts_mut(self as *mut _ as *mut T, Self::DIM) }
    }

    /// Returns an iterator over the inner vector.
    pub fn iter(&self) -> slice::Iter<T> {
        self.as_slice().iter()
    }

    /// Returns a mutable iterator over the inner vector.
    pub fn iter_mut(&mut self) -> slice::IterMut<T> {
        self.as_slice_mut().iter_mut()
    }

    /// Initializes a new number from an iterator.
    pub fn from_iterator(iter: &mut impl Iterator<Item = T>) -> Self {
        let mut uninit = mem::MaybeUninit::uninit();
        let ptr = uninit.as_mut_ptr() as *mut T;

        for i in 0..Self::DIM {
            let val = iter.next().expect("iterator not long enough!");
            unsafe {
                ptr.add(i).write(val);
            }
        }

        unsafe { uninit.assume_init() }
    }
}

impl<T: Field, U: Algebra<T>, const N: usize> From<[T; N]> for Cayley<T, U> {
    fn from(arr: [T; N]) -> Self {
        assert_eq!(N, Self::DIM, "dimension mismatch");
        unsafe { mem::transmute_copy(&arr) }
    }
}

impl<T: Field, U: Algebra<T>, const N: usize> From<Cayley<T, U>> for [T; N] {
    fn from(cayley: Cayley<T, U>) -> Self {
        assert_eq!(N, Cayley::<T, U>::DIM, "dimension mismatch");
        unsafe { mem::transmute_copy(&cayley) }
    }
}

impl<T: Field + Display, U: Algebra<T>> Display for Cayley<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_char('[')?;
        let mut iter = self.iter();
        write!(f, "{}", iter.next().unwrap())?;

        for x in iter {
            write!(f, ", {}", x)?;
        }

        f.write_char(']')
    }
}

pub trait AlgebraIntoIterator<T: Field> {
    type IntoIter: Iterator<Item = T>;

    fn into_iter(self) -> Self::IntoIter;
}

impl<T: Field> AlgebraIntoIterator<T> for T {
    type IntoIter = Once<T>;

    fn into_iter(self) -> Self::IntoIter {
        iter::once(self)
    }
}

impl<T: Field, U: Algebra<T> + AlgebraIntoIterator<T>> AlgebraIntoIterator<T> for Cayley<T, U> {
    type IntoIter = iter::Chain<U::IntoIter, U::IntoIter>;

    fn into_iter(self) -> Self::IntoIter {
        self.a.into_iter().chain(self.b.into_iter())
    }
}

impl<T: Field, U: Algebra<T> + AlgebraIntoIterator<T>> IntoIterator for Cayley<T, U>
where
    U: IntoIterator<Item = T>,
{
    type Item = T;
    type IntoIter = <Self as AlgebraIntoIterator<T>>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        AlgebraIntoIterator::<T>::into_iter(self)
    }
}

impl<T: Field, U: Algebra<T>> Index<usize> for Cayley<T, U> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<T: Field, U: Algebra<T>> IndexMut<usize> for Cayley<T, U> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_slice_mut()[index]
    }
}

impl<T: Field, U: Algebra<T>> Cayley<T, U> {
    pub fn unit(idx: usize) -> Self {
        let mut res = Self::zero();
        res[idx] = T::one();
        res
    }
}

pub type Complex<T> = Cayley<T, T>;
pub type Quaternion<T> = Cayley<T, Complex<T>>;
pub type Octonion<T> = Cayley<T, Quaternion<T>>;
pub type Sedenion<T> = Cayley<T, Octonion<T>>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sedenion_zero_divisors() {
        let a: Sedenion<f32> = Sedenion::unit(3) + Sedenion::unit(10);
        let b: Sedenion<f32> = Sedenion::unit(6) - Sedenion::unit(15);

        println!("{} × {} = {}", &a, &b, a * b)
    }
}
