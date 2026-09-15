//! A small JSON reader and string quoting for safetensors headers.

/// Nesting beyond this depth is rejected, so hostile headers cannot exhaust the stack.
const MAX_DEPTH: usize = 64;

#[derive(Debug, PartialEq)]
pub(crate) enum Value {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Value>),
    Object(Vec<(String, Value)>),
}

impl Value {
    pub(crate) fn get(&self, key: &str) -> Option<&Value> {
        match self {
            Value::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub(crate) fn as_array(&self) -> Option<&[Value]> {
        match self {
            Value::Array(items) => Some(items),
            _ => None,
        }
    }

    pub(crate) fn as_usize(&self) -> Option<usize> {
        match *self {
            Value::Number(n) if n >= 0.0 && n.fract() == 0.0 && n <= usize::MAX as f64 => {
                Some(n as usize)
            }
            _ => None,
        }
    }
}

pub(crate) fn parse(text: &str) -> Result<Value, String> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        pos: 0,
        depth: 0,
    };
    let value = parser.value()?;
    parser.skip_whitespace();
    if parser.pos != parser.bytes.len() {
        return Err(parser.error("unexpected trailing characters"));
    }
    Ok(value)
}

/// Quotes `text` as a JSON string literal.
pub(crate) fn quote(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
    depth: usize,
}

impl Parser<'_> {
    fn value(&mut self) -> Result<Value, String> {
        self.skip_whitespace();
        match self.peek() {
            Some(b'{') => self.nested(Self::object),
            Some(b'[') => self.nested(Self::array),
            Some(b'"') => self.string().map(Value::String),
            Some(b't') => self.literal("true", Value::Bool(true)),
            Some(b'f') => self.literal("false", Value::Bool(false)),
            Some(b'n') => self.literal("null", Value::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => Err(self.error("expected a value")),
        }
    }

    fn nested(&mut self, parse: fn(&mut Self) -> Result<Value, String>) -> Result<Value, String> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.error("nesting is too deep"));
        }
        let value = parse(self);
        self.depth -= 1;
        value
    }

    fn object(&mut self) -> Result<Value, String> {
        self.pos += 1;
        let mut entries = Vec::new();
        self.skip_whitespace();
        if self.eat(b'}') {
            return Ok(Value::Object(entries));
        }
        loop {
            self.skip_whitespace();
            if self.peek() != Some(b'"') {
                return Err(self.error("expected a string key"));
            }
            let key = self.string()?;
            self.skip_whitespace();
            if !self.eat(b':') {
                return Err(self.error("expected ':'"));
            }
            entries.push((key, self.value()?));
            self.skip_whitespace();
            if self.eat(b'}') {
                return Ok(Value::Object(entries));
            }
            if !self.eat(b',') {
                return Err(self.error("expected ',' or '}'"));
            }
        }
    }

    fn array(&mut self) -> Result<Value, String> {
        self.pos += 1;
        let mut items = Vec::new();
        self.skip_whitespace();
        if self.eat(b']') {
            return Ok(Value::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_whitespace();
            if self.eat(b']') {
                return Ok(Value::Array(items));
            }
            if !self.eat(b',') {
                return Err(self.error("expected ',' or ']'"));
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.pos += 1;
        let mut out = String::new();
        loop {
            let start = self.pos;
            while self
                .peek()
                .is_some_and(|b| b != b'"' && b != b'\\' && b >= 0x20)
            {
                self.pos += 1;
            }
            // The run ends at an ASCII byte, so it never splits a multibyte character of the valid input.
            out.push_str(std::str::from_utf8(&self.bytes[start..self.pos]).unwrap());
            match self.bump() {
                Some(b'"') => return Ok(out),
                Some(b'\\') => {
                    let c = match self.bump() {
                        Some(b'"') => '"',
                        Some(b'\\') => '\\',
                        Some(b'/') => '/',
                        Some(b'b') => '\u{8}',
                        Some(b'f') => '\u{c}',
                        Some(b'n') => '\n',
                        Some(b'r') => '\r',
                        Some(b't') => '\t',
                        Some(b'u') => self.unicode_escape()?,
                        _ => return Err(self.error("invalid escape")),
                    };
                    out.push(c);
                }
                _ => return Err(self.error("unterminated string")),
            }
        }
    }

    fn unicode_escape(&mut self) -> Result<char, String> {
        let high = self.hex4()?;
        let code = if (0xd800..0xdc00).contains(&high) {
            if self.bump() != Some(b'\\') || self.bump() != Some(b'u') {
                return Err(self.error("unpaired surrogate"));
            }
            let low = self.hex4()?;
            if !(0xdc00..0xe000).contains(&low) {
                return Err(self.error("unpaired surrogate"));
            }
            0x10000 + ((high - 0xd800) << 10) + (low - 0xdc00)
        } else {
            high
        };
        char::from_u32(code).ok_or_else(|| self.error("invalid unicode escape"))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let digits = self
            .bytes
            .get(self.pos..self.pos + 4)
            .filter(|digits| digits.iter().all(u8::is_ascii_hexdigit))
            .ok_or_else(|| self.error("invalid unicode escape"))?;
        self.pos += 4;
        Ok(u32::from_str_radix(std::str::from_utf8(digits).unwrap(), 16).unwrap())
    }

    fn number(&mut self) -> Result<Value, String> {
        let start = self.pos;
        while matches!(
            self.peek(),
            Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        ) {
            self.pos += 1;
        }
        std::str::from_utf8(&self.bytes[start..self.pos])
            .unwrap()
            .parse()
            .map(Value::Number)
            .map_err(|_| self.error("invalid number"))
    }

    fn literal(&mut self, word: &str, value: Value) -> Result<Value, String> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(self.error("invalid literal"))
        }
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let byte = self.peek()?;
        self.pos += 1;
        Some(byte)
    }

    fn eat(&mut self, byte: u8) -> bool {
        let matched = self.peek() == Some(byte);
        if matched {
            self.pos += 1;
        }
        matched
    }

    fn error(&self, message: &str) -> String {
        format!("{message} at byte {}", self.pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_nested_values() {
        let value =
            parse(r#" {"a": [1, 2.5, -3e2], "b": {"c": true, "d": null}, "e": "x"} "#).unwrap();
        assert_eq!(
            value.get("a").unwrap().as_array().unwrap()[2],
            Value::Number(-300.0)
        );
        assert_eq!(value.get("b").unwrap().get("c"), Some(&Value::Bool(true)));
        assert_eq!(value.get("e").unwrap().as_str(), Some("x"));
        assert_eq!(parse("[]").unwrap(), Value::Array(Vec::new()));
    }

    #[test]
    fn decodes_escapes() {
        let value = parse(r#""tab\t quote\" slash\/ é 😀""#).unwrap();
        assert_eq!(value.as_str(), Some("tab\t quote\" slash/ é 😀"));
        let name = "odd \"name\"\\\n";
        assert_eq!(parse(&quote(name)).unwrap().as_str(), Some(name));
    }

    #[test]
    fn rejects_invalid_json() {
        for text in [
            "",
            "{",
            r#"{"a" 1}"#,
            "[1,]",
            r#""open"#,
            "tru",
            "1 2",
            r#""\ud800""#,
            r#""\x""#,
        ] {
            assert!(parse(text).is_err(), "{text}");
        }
        assert!(parse(&"[".repeat(1000)).is_err());
    }

    #[test]
    fn reads_only_whole_non_negative_integers_as_sizes() {
        assert_eq!(Value::Number(3.0).as_usize(), Some(3));
        assert_eq!(Value::Number(-1.0).as_usize(), None);
        assert_eq!(Value::Number(1.5).as_usize(), None);
    }
}
