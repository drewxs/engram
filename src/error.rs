//! Errors from fallible operations such as saving and loading models.

use std::{fmt, io};

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    /// The data is not a valid safetensors file.
    Format(String),
    /// A state dict does not match the module it is loaded into.
    StateDict(String),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(error) => write!(f, "io error: {error}"),
            Error::Format(message) => write!(f, "invalid safetensors data: {message}"),
            Error::StateDict(message) => write!(f, "state dict mismatch: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Error {
        Error::Io(error)
    }
}
