//! Reading and writing tensors in the [safetensors](https://github.com/huggingface/safetensors) format.
//!
//! A file holds a JSON header describing each tensor followed by the raw little-endian data, so weights move freely between Engram, PyTorch and other frameworks.
//!
//! ```no_run
//! # use engram::{nn::{Linear, Module}, safetensors};
//! let model = Linear::new(4, 2);
//! safetensors::save("model.safetensors", &model.state_dict())?;
//! model.load_state_dict(&safetensors::load("model.safetensors")?)?;
//! # Ok::<(), engram::Error>(())
//! ```

mod json;

use std::{collections::BTreeMap, fs, path::Path};

use crate::{Error, Result, Tensor};
use json::Value;

/// Writes named tensors to a file.
pub fn save(path: impl AsRef<Path>, tensors: &BTreeMap<String, Tensor>) -> Result<()> {
    fs::write(path, to_bytes(tensors))?;
    Ok(())
}

/// Reads every tensor in a file.
pub fn load(path: impl AsRef<Path>) -> Result<BTreeMap<String, Tensor>> {
    from_bytes(&fs::read(path)?)
}

/// Serializes named tensors as `F32` data.
pub fn to_bytes(tensors: &BTreeMap<String, Tensor>) -> Vec<u8> {
    let mut entries = Vec::with_capacity(tensors.len());
    let mut offset = 0;
    for (name, tensor) in tensors {
        let end = offset + tensor.numel() * 4;
        let shape: Vec<String> = tensor.shape().iter().map(usize::to_string).collect();
        entries.push(format!(
            r#"{}:{{"dtype":"F32","shape":[{}],"data_offsets":[{offset},{end}]}}"#,
            json::quote(name),
            shape.join(",")
        ));
        offset = end;
    }
    let mut header = format!("{{{}}}", entries.join(","));
    // Padding the header to a multiple of eight bytes keeps the data aligned for readers that map it in place.
    while header.len() % 8 != 0 {
        header.push(' ');
    }

    let mut bytes = Vec::with_capacity(8 + header.len() + offset);
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for tensor in tensors.values() {
        for value in tensor.data().iter() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// Deserializes tensors. `F64` data is converted to `f32`, and other dtypes are rejected.
pub fn from_bytes(bytes: &[u8]) -> Result<BTreeMap<String, Tensor>> {
    let header_len = bytes
        .get(..8)
        .map(|size| u64::from_le_bytes(size.try_into().unwrap()))
        .ok_or_else(|| invalid("data is shorter than the 8 byte header size"))?;
    let data_start = usize::try_from(header_len)
        .ok()
        .and_then(|len| len.checked_add(8))
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| invalid("header extends past the end of the data"))?;
    let header = std::str::from_utf8(&bytes[8..data_start])
        .map_err(|_| invalid("header is not valid UTF-8"))?;
    let Value::Object(entries) = json::parse(header).map_err(Error::Format)? else {
        return Err(invalid("header is not a JSON object"));
    };

    let data = &bytes[data_start..];
    let mut tensors = BTreeMap::new();
    for (name, info) in entries {
        if name != "__metadata__" {
            let tensor = read_tensor(&name, &info, data)?;
            tensors.insert(name, tensor);
        }
    }
    Ok(tensors)
}

fn read_tensor(name: &str, info: &Value, data: &[u8]) -> Result<Tensor> {
    let field = |key: &str| {
        info.get(key)
            .ok_or_else(|| invalid(format!("{name} has no {key}")))
    };
    let shape =
        usizes(field("shape")?).ok_or_else(|| invalid(format!("{name} has an invalid shape")))?;
    let offsets = usizes(field("data_offsets")?)
        .filter(|offsets| offsets.len() == 2)
        .ok_or_else(|| invalid(format!("{name} has invalid data offsets")))?;
    let raw = data
        .get(offsets[0]..offsets[1])
        .ok_or_else(|| invalid(format!("{name} has data offsets outside the file")))?;

    let (width, decode): (usize, fn(&[u8]) -> f32) = match field("dtype")?.as_str() {
        Some("F32") => (4, |b| f32::from_le_bytes(b.try_into().unwrap())),
        Some("F64") => (8, |b| f64::from_le_bytes(b.try_into().unwrap()) as f32),
        _ => return Err(invalid(format!("{name} has an unsupported dtype"))),
    };
    let numel: usize = shape.iter().product();
    if raw.len() != numel * width {
        return Err(invalid(format!(
            "{name} holds {} bytes but its shape needs {}",
            raw.len(),
            numel * width
        )));
    }
    Ok(Tensor::from_vec(
        raw.chunks_exact(width).map(decode).collect(),
        shape,
    ))
}

fn usizes(value: &Value) -> Option<Vec<usize>> {
    value.as_array()?.iter().map(Value::as_usize).collect()
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Format(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor;

    fn sample() -> BTreeMap<String, Tensor> {
        BTreeMap::from([
            ("layer.weight".to_string(), Tensor::randn([3, 4])),
            ("layer.bias".to_string(), tensor![1, 2, 3]),
            ("scale".to_string(), Tensor::scalar(0.5)),
            ("empty".to_string(), Tensor::zeros([0, 2])),
        ])
    }

    #[test]
    fn round_trips_tensors() {
        let tensors = sample();
        let bytes = to_bytes(&tensors);
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!((8 + header_len) % 8, 0);
        assert_eq!(from_bytes(&bytes).unwrap(), tensors);
    }

    #[test]
    fn saves_and_loads_files() {
        let path = std::env::temp_dir().join(format!("engram-{}.safetensors", std::process::id()));
        let tensors = sample();
        save(&path, &tensors).unwrap();
        let loaded = load(&path);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(loaded.unwrap(), tensors);
    }

    #[test]
    fn reads_files_written_by_other_libraries() {
        let header = r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F64","shape":[2],"data_offsets":[0,16]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&1.5f64.to_le_bytes());
        bytes.extend_from_slice(&(-2.0f64).to_le_bytes());
        let tensors = from_bytes(&bytes).unwrap();
        assert_eq!(tensors["w"], tensor![1.5, -2]);
    }

    #[test]
    fn rejects_malformed_data() {
        let header = |json: &str, data_len: usize| {
            let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(json.as_bytes());
            bytes.resize(bytes.len() + data_len, 0);
            from_bytes(&bytes)
        };
        assert!(from_bytes(&[1, 2, 3]).is_err());
        assert!(from_bytes(&u64::MAX.to_le_bytes()).is_err());
        assert!(header("[]", 0).is_err());
        assert!(header(
            r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
            4
        )
        .is_err());
        assert!(header(
            r#"{"x":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#,
            8
        )
        .is_err());
        assert!(header(
            r#"{"x":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#,
            8
        )
        .is_err());
        assert!(header(r#"{"x":{"dtype":"F32","shape":[1]}}"#, 4).is_err());
    }
}
