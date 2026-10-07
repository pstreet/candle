//! Typed device architecture identity for GPU backends.
//!
//! Policy code matches on `GpuArch`, never on raw compute-capability ints:
//! the HIP-synthetic `cc` encoding (gfx1151 -> 1150, gfx942 -> 942) survives
//! only at the kernel bridge as an implementation detail of the llama.cpp
//! derived kernels.

#[cfg(any(feature = "cuda", feature = "rocm"))]
use crate::{Error, Result};

/// NVIDIA: `sm` is the compute-capability pair, e.g. `(9, 0)` for Hopper.
/// AMD: `gfx` is the `gcnArchName` suffix split one char per field after the
/// family digits, letters hex-coded: gfx942 -> `(9, 4, 2)`, gfx1151 ->
/// `(11, 5, 1)`, gfx90a -> `(9, 0, 10)`. Family alone carries the policy
/// granularity (9 = CDNA, 11 = RDNA3/3.5, 12 = RDNA4); the step field
/// distinguishes sibling parts (gfx1150 vs gfx1151) that the synthetic
/// compute caps collide.
///
/// Vulkan is reserved: a future backend resolves identity from
/// `VkPhysicalDeviceProperties` (device name/vendor-ID strings, the
/// ggml-vulkan precedent for vendor quirk workarounds).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GpuArch {
    Cuda { sm: (u8, u8) },
    Rocm { gfx: (u8, u8, u8) },
    Vulkan,
}

// AMD-role arch resolution for dual builds, sharing the rocm branch shape
// below but bound to the unambiguous `cudarc_hip` paths (the crate-root
// `cudarc` alias is the real NVIDIA crate there).
#[cfg(feature = "rocm")]
macro_rules! rocm_resolve_on {
    ($krate:ident, $stream:expr) => {{
        let cu_device = $stream.context().cu_device();
        let cap = |attr| {
            unsafe { $krate::driver::result::device::get_attribute(cu_device, attr) }
                .map_err(crate::Error::wrap)
        };
        let major = cap(
            $krate::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )?;
        match $stream
            .context()
            .gcn_arch_name()
            .ok()
            .as_deref()
            .and_then(GpuArch::parse_gfx_name)
        {
            Some(arch) => Ok(arch),
            // Unparseable name: fall back to the synthetic compute-cap
            // major as the family so family-level policy still applies;
            // the step fields stay zeroed (conservative defaults).
            None if (0..=u8::MAX as i32).contains(&major) => Ok(GpuArch::Rocm {
                gfx: (major as u8, 0, 0),
            }),
            None => crate::bail!("could not resolve AMD gcnArchName"),
        }
    }};
}

#[cfg(any(feature = "cuda", feature = "rocm"))]
impl GpuArch {
    #[cfg(feature = "cuda")]
    pub fn resolve(dev: &crate::CudaDevice) -> Result<Self> {
        use crate::cudarc::driver::{result, sys};
        let stream = dev.cuda_stream();
        let cu_device = stream.context().cu_device();
        let cap =
            |attr| unsafe { result::device::get_attribute(cu_device, attr) }.map_err(Error::wrap);
        let major = cap(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;

        #[cfg(all(feature = "rocm", not(feature = "cuda")))]
        {
            match stream
                .context()
                .gcn_arch_name()
                .ok()
                .as_deref()
                .and_then(Self::parse_gfx_name)
            {
                Some(arch) => Ok(arch),
                // Unparseable name: fall back to the synthetic compute-cap
                // major as the family so family-level policy still applies;
                // the step fields stay zeroed (conservative defaults).
                None if (0..=u8::MAX as i32).contains(&major) => Ok(GpuArch::Rocm {
                    gfx: (major as u8, 0, 0),
                }),
                None => crate::bail!("could not resolve AMD gcnArchName"),
            }
        }
        #[cfg(feature = "cuda")]
        {
            let minor = cap(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
            if major < 0 || minor < 0 || major > u8::MAX as i32 || minor > u8::MAX as i32 {
                crate::bail!("unexpected compute capability {major}.{minor}");
            }
            Ok(GpuArch::Cuda {
                sm: (major as u8, minor as u8),
            })
        }
    }

    /// Arch resolution for the AMD-role device of a dual build.
    #[cfg(feature = "rocm")]
    pub fn resolve_hip(dev: &crate::HipDevice) -> Result<Self> {
        let stream = dev.cuda_stream();
        rocm_resolve_on!(cudarc_hip, stream)
    }
}

impl GpuArch {
    #[cfg_attr(not(feature = "rocm"), allow(dead_code))]
    fn parse_gfx_name(name: &str) -> Option<Self> {
        let name = name.split(':').next()?;
        let digits = name.strip_prefix("gfx")?;
        if digits.is_empty() {
            return None;
        }
        let (family, rest) = digits.split_at(digits.len().saturating_sub(2));
        if family.is_empty() || !family.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let family: u8 = family.parse().ok()?;
        if family == 0 {
            return None;
        }
        let mut fields = [0u8; 2];
        for (slot, ch) in rest.bytes().enumerate() {
            let value = match ch {
                b'0'..=b'9' => ch - b'0',
                b'a'..=b'f' => ch - b'a' + 10,
                _ => return None,
            };
            fields[slot] = value;
        }
        Some(GpuArch::Rocm {
            gfx: (family, fields[0], fields[1]),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::GpuArch;

    #[test]
    fn gfx_name_parsing() {
        let parse = |s: &str| GpuArch::parse_gfx_name(s);
        assert_eq!(parse("gfx1151"), Some(GpuArch::Rocm { gfx: (11, 5, 1) }));
        assert_eq!(parse("gfx1150"), Some(GpuArch::Rocm { gfx: (11, 5, 0) }));
        assert_eq!(parse("gfx942"), Some(GpuArch::Rocm { gfx: (9, 4, 2) }));
        assert_eq!(parse("gfx1100"), Some(GpuArch::Rocm { gfx: (11, 0, 0) }));
        assert_eq!(parse("gfx90a"), Some(GpuArch::Rocm { gfx: (9, 0, 10) }));
        assert_eq!(
            parse("gfx942:sramecc+:xnack-"),
            Some(GpuArch::Rocm { gfx: (9, 4, 2) })
        );
        assert_eq!(parse("gfx1030"), Some(GpuArch::Rocm { gfx: (10, 3, 0) }));
        assert_eq!(parse("gfx1200"), Some(GpuArch::Rocm { gfx: (12, 0, 0) }));
        assert_eq!(parse(""), None);
        assert_eq!(parse("graphics"), None);
        assert_eq!(parse("gfx"), None);
        assert_eq!(parse("gfxzz"), None);
    }
}

#[cfg(all(test, feature = "rocm", not(feature = "cuda")))]
mod probe_tests {
    use super::GpuArch;
    use crate::backend::BackendDevice;
    use crate::CudaDevice;

    #[test]
    #[ignore]
    fn gcn_arch_name_resolves() {
        let dev = CudaDevice::new(0).unwrap();
        let stream = dev.cuda_stream();
        let name = stream.context().gcn_arch_name().unwrap();
        assert!(name.starts_with("gfx"), "{name}");
        assert_eq!(
            GpuArch::resolve(&dev).unwrap(),
            GpuArch::parse_gfx_name(&name).unwrap()
        );
    }
}
