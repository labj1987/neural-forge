//! The bits of Python's `datetime` the report uses: `strptime` with its field patterns (so "26-1-2-3-4-5" parses and
//! "26-11-12-17-58-61" does not), and `timedelta` built from a float. Times are naive microseconds since 1970-01-01.

use std::sync::LazyLock;

use regex::Regex;

const YEAR4: &str = "([0-9]{4})";
const YEAR2: &str = "([0-9]{2})";
const MONTH: &str = "(1[0-2]|0[1-9]|[1-9])";
const DAY: &str = "(3[0-1]|[1-2][0-9]|0[1-9]|[1-9]| [1-9])";
const HOUR: &str = "(2[0-3]|[0-1][0-9]|[0-9])";
const MINUTE: &str = "([0-5][0-9]|[0-9])";
const SECOND: &str = "(6[0-1]|[0-5][0-9]|[0-9])";

fn pattern(year: &str, date_sep: &str, between: &str, time_sep: &str) -> Regex {
    Regex::new(&format!("^{year}{date_sep}{MONTH}{date_sep}{DAY}{between}{HOUR}{time_sep}{MINUTE}{time_sep}{SECOND}")).expect("static regex")
}

/// "%y-%m-%d-%H-%M-%S", GTA's file stamp.
static STAMP_Y: LazyLock<Regex> = LazyLock::new(|| pattern(YEAR2, "-", "-", "-"));
/// "%Y-%m-%d_%H-%M-%S", MangoHud's file stamp.
static STAMP_CSV: LazyLock<Regex> = LazyLock::new(|| pattern(YEAR4, "-", "_", "-"));
/// "%Y/%m/%d %H:%M:%S", nvidia-smi's (a space in the format is `\s+`).
static STAMP_GPU: LazyLock<Regex> = LazyLock::new(|| Regex::new(&format!("^{YEAR4}/{MONTH}/{DAY}\\s+{HOUR}:{MINUTE}:{SECOND}")).expect("static regex"));

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        2 if is_leap(y) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days since 1970-01-01 of a proleptic Gregorian date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn build(re: &Regex, s: &str, two_digit_year: bool) -> Option<i64> {
    let c = re.captures(s)?;
    if c.get(0)?.end() != s.len() {
        return None; // "unconverted data remains"
    }
    let num = |i: usize| c[i].trim().parse::<i64>().ok();
    let mut year = num(1)?;
    if two_digit_year {
        year += if year <= 68 { 2000 } else { 1900 };
    }
    let (month, day, hour, minute, second) = (num(2)?, num(3)?, num(4)?, num(5)?, num(6)?);
    if year < 1 || day > days_in_month(year, month) || second > 59 {
        return None; // datetime() refuses it (strptime lets seconds 60 and 61 through)
    }
    Some((days_from_civil(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second) * 1_000_000)
}

pub(crate) fn parse_gta_stamp(s: &str) -> Option<i64> {
    build(&STAMP_Y, s, true)
}

pub(crate) fn parse_csv_stamp(s: &str) -> Option<i64> {
    build(&STAMP_CSV, s, false)
}

pub(crate) fn parse_gpu_stamp(s: &str) -> Option<i64> {
    build(&STAMP_GPU, s, false)
}

/// `timedelta(seconds=x)` in microseconds: the whole seconds exactly, the fraction times 1e6 rounded half to even.
pub(crate) fn seconds(x: f64) -> i64 {
    let whole = x.trunc();
    whole as i64 * 1_000_000 + ((x - whole) * 1e6).round_ties_even() as i64
}

/// `timedelta(microseconds=x)`.
pub(crate) fn microseconds(x: f64) -> i64 {
    x.round_ties_even() as i64
}

/// `timedelta.total_seconds()`.
pub(crate) fn total_seconds(micros: i64) -> f64 {
    micros as f64 / 1e6
}
