use std::fmt;

use crate::Tensor;

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut res = String::new();
        for row in &self.data {
            res.push_str(&format!("{:?}\n", row));
        }
        write!(f, "{}", res)
    }
}
