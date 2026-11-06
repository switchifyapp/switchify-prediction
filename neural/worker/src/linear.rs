//! Portable x86-64 Q8 dot products using the baseline SSE2 instruction set.
//! Other targets, quantizations, precision overrides and accelerated builds
//! continue through Candle's QMatMul. The SSE2 path also accepts rank-one and
//! noncontiguous logical inputs; Candle's fallback retains its own restrictions.
use candle_core::{
    Module, Result, Tensor,
    quantized::{QMatMul, QTensor},
};
use std::sync::Arc;

#[derive(Clone)]
pub struct Linear(QMatMul);

impl Linear {
    pub fn from_qtensor(tensor: QTensor) -> Result<Self> {
        Ok(Self(QMatMul::from_qtensor(tensor)?))
    }
    pub fn from_arc(tensor: Arc<QTensor>) -> Result<Self> {
        Ok(Self(QMatMul::from_arc(tensor)?))
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        #[cfg(all(
            target_arch = "x86_64",
            not(feature = "accelerated"),
            not(target_feature = "avx2")
        ))]
        if let QMatMul::QTensor(weight) = &self.0
            && weight.dtype() == candle_core::quantized::GgmlDType::Q8_0
            && matches!(x.device(), candle_core::Device::Cpu)
            && matches!(weight.device(), candle_core::Device::Cpu)
            && x.dtype() == candle_core::DType::F32
        {
            return portable::forward(weight, x);
        }
        self.0.forward(x)
    }
}

#[cfg(all(
    target_arch = "x86_64",
    not(feature = "accelerated"),
    not(target_feature = "avx2")
))]
mod portable {
    use super::*;
    use candle_core::{Device, quantized::GgmlDType};
    use rayon::prelude::*;
    use std::arch::x86_64::*;

    const _: () = assert!(std::mem::size_of::<candle_core::quantized::k_quants::BlockQ8_0>() == 34);

    /// All inputs are checked 32-byte blocks. Signed products and their sum
    /// fit i32, including -128 * -128; no saturating instructions are used.
    pub(super) fn dot_bytes(a: &[u8; 32], b: &[u8; 32]) -> i32 {
        // SAFETY: the caller passes 32-byte slices from checked 34-byte blocks.
        // Both unaligned loads cover 16 bytes within each slice. The store
        // covers exactly the four i32s in the local array. SSE2 is mandatory
        // on x86-64; this module is never compiled for other architectures.
        unsafe {
            let zero = _mm_setzero_si128();
            let mut sum = _mm_setzero_si128();
            for offset in [0, 16] {
                let x = _mm_loadu_si128(a.as_ptr().add(offset).cast());
                let y = _mm_loadu_si128(b.as_ptr().add(offset).cast());
                let sx = _mm_cmpgt_epi8(zero, x);
                let sy = _mm_cmpgt_epi8(zero, y);
                let low = _mm_madd_epi16(_mm_unpacklo_epi8(x, sx), _mm_unpacklo_epi8(y, sy));
                let high = _mm_madd_epi16(_mm_unpackhi_epi8(x, sx), _mm_unpackhi_epi8(y, sy));
                sum = _mm_add_epi32(sum, _mm_add_epi32(low, high));
            }
            let mut lanes = [0i32; 4];
            _mm_storeu_si128(lanes.as_mut_ptr().cast(), sum);
            lanes.into_iter().sum()
        }
    }

    pub(super) fn dot(a: &[u8], b: &[u8]) -> f32 {
        let mut result = 0.;
        for (x, y) in a.chunks_exact(34).zip(b.chunks_exact(34)) {
            let sx = half::f16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32();
            let sy = half::f16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32();
            // Match Candle's scalar float multiplication and accumulation order.
            result +=
                dot_bytes(x[2..].try_into().unwrap(), y[2..].try_into().unwrap()) as f32 * sx * sy;
        }
        result
    }

    pub(super) fn forward(weight: &QTensor, x: &Tensor) -> Result<Tensor> {
        let (n, k) = weight.shape().dims2()?;
        if k == 0 || n == 0 || !k.is_multiple_of(32) || x.dims().last() != Some(&k) {
            candle_core::bail!("invalid Q8 linear dimensions");
        }
        let rows = x.elem_count() / k;
        let stride = k / 32 * 34;
        // QTensor::quantize reads the underlying storage, not its view offset.
        // A contiguous last-position view can still start partway through that
        // storage. Materialize the logical values into an offset-zero tensor.
        let input = Tensor::from_vec(x.flatten_all()?.to_vec1::<f32>()?, x.dims(), &Device::Cpu)?;
        let lhs = QTensor::quantize(&input, GgmlDType::Q8_0)?;
        let lhs_bytes = lhs.data()?;
        let weights = weight.data()?;
        if weights.len() != n * stride || lhs_bytes.len() != rows * stride {
            candle_core::bail!("invalid Q8 linear storage");
        }
        let mut output = vec![0.; rows * n];
        let fill = |(chunk_index, chunk): (usize, &mut [f32])| {
            let start = chunk_index * 64;
            let (mut row, mut column) = (start / n, start % n);
            for value in chunk {
                *value = dot(
                    &weights[column * stride..(column + 1) * stride],
                    &lhs_bytes[row * stride..(row + 1) * stride],
                );
                column += 1;
                if column == n {
                    column = 0;
                    row += 1;
                }
            }
        };
        if candle_core::utils::get_num_threads() > 1 {
            candle_core::utils::with_threadpool(|| {
                output.par_chunks_mut(64).enumerate().for_each(fill);
            });
        } else {
            output.chunks_mut(64).enumerate().for_each(fill);
        }
        let mut shape = x.dims().to_vec();
        *shape.last_mut().unwrap() = n;
        Tensor::from_vec(output, shape, &Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, quantized::GgmlDType};

    #[cfg(all(
        target_arch = "x86_64",
        not(feature = "accelerated"),
        not(target_feature = "avx2")
    ))]
    #[test]
    fn all_signed_byte_extremes_match_scalar_products() {
        for x in -128i16..=127 {
            for y in -128i16..=127 {
                assert_eq!(
                    super::portable::dot_bytes(&[x as u8; 32], &[y as u8; 32]),
                    32 * x as i32 * y as i32
                );
            }
        }
        let a = std::array::from_fn(|i| (i * 37) as u8);
        let b = std::array::from_fn(|i| (i * 19 + 128) as u8);
        let expected: i32 = a
            .iter()
            .zip(b)
            .map(|(&x, y)| x as i8 as i32 * y as i8 as i32)
            .sum();
        assert_eq!(super::portable::dot_bytes(&a, &b), expected);
    }

    #[test]
    fn q8_shapes_match_candle_bit_for_bit() -> Result<()> {
        for n in [1, 3, 63, 64, 65, 192, 576, 1536] {
            for k in [32, 64, 576, 1536] {
                let values: Vec<f32> = (0..n * k)
                    .map(|i| ((i * 37 % 257) as f32 - 128.) / 17.)
                    .collect();
                let weights = Tensor::from_vec(values, (n, k), &Device::Cpu)?;
                let q = Arc::new(QTensor::quantize(&weights, GgmlDType::Q8_0)?);
                let actual = Linear(QMatMul::QTensor(q.clone()));
                let expected = QMatMul::QTensor(q);
                for (batch, seq) in [(1, 1), (2, 3), (8, 1)] {
                    let values: Vec<f32> = (0..batch * seq * k)
                        .map(|i| ((i * 13 % 97) as f32 - 48.) / 9.)
                        .collect();
                    let x = Tensor::from_vec(values, (batch, seq, k), &Device::Cpu)?;
                    let got = actual.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
                    let want = expected.forward(&x)?.flatten_all()?.to_vec1::<f32>()?;
                    assert_eq!(
                        got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "n={n} k={k} batch={batch} seq={seq}"
                    );
                    let extended = Tensor::from_vec(
                        (0..batch * (seq + 1) * k)
                            .map(|i| (i as f32 * 0.031).sin())
                            .collect(),
                        (batch, seq + 1, k),
                        &Device::Cpu,
                    )?;
                    let view = extended.narrow(1, 1, seq)?;
                    let view = if cfg!(all(
                        target_arch = "x86_64",
                        not(feature = "accelerated"),
                        not(target_feature = "avx2")
                    )) {
                        view
                    } else {
                        // Candle's unchanged fallback requires contiguous input.
                        view.contiguous()?
                    };
                    let got = actual.forward(&view)?.flatten_all()?.to_vec1::<f32>()?;
                    let want = expected
                        .forward(&view.contiguous()?)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    assert_eq!(
                        got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        "view n={n} k={k} batch={batch} seq={seq}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn full_vocabulary_projection_matches_candle() -> Result<()> {
        let (n, k) = (49152, 576);
        let weights = Tensor::from_vec(
            (0..n * k)
                .map(|i| ((i * 37 % 257) as f32 - 128.) / 17.)
                .collect(),
            (n, k),
            &Device::Cpu,
        )?;
        let q = Arc::new(QTensor::quantize(&weights, GgmlDType::Q8_0)?);
        drop(weights);
        let actual = Linear(QMatMul::QTensor(q.clone()));
        let expected = QMatMul::QTensor(q);
        let input = Tensor::from_vec(
            (0..2 * k)
                .map(|i| ((i * 13 % 97) as f32 - 48.) / 9.)
                .collect(),
            (2, k),
            &Device::Cpu,
        )?;
        let got = actual.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
        let want = expected.forward(&input)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn precision_overrides_and_other_types_use_candle() -> Result<()> {
        let weights = Tensor::from_vec(
            (0..96).map(|i| i as f32 / 97.).collect(),
            (3, 32),
            &Device::Cpu,
        )?;
        let input = Tensor::from_vec(
            (0..64).map(|i| i as f32 / 67.).collect(),
            (2, 32),
            &Device::Cpu,
        )?;
        for matmul in [
            QMatMul::Tensor(weights.clone()),
            QMatMul::TensorF16(weights.to_dtype(DType::F16)?),
            QMatMul::from_qtensor(QTensor::quantize(&weights, GgmlDType::Q4_0)?)?,
        ] {
            let actual = Linear(matmul.clone()).forward(&input)?.to_vec2::<f32>()?;
            let expected = matmul.forward(&input)?.to_vec2::<f32>()?;
            assert_eq!(actual, expected);
        }
        Ok(())
    }
}
