# The German frontend the German checkpoint was trained with: `german_text.py`
# of `sharrnah/index-tts-2.5-german`, ported rule for rule. Python's dictionaries
# keep insertion order and several loops depend on it (abbreviations, currencies
# and units are applied longest first, ties in source order), so the tables are
# ordered lists here. Numbers are `BigInt`, as Python's `int` is unbounded.

const DE_SMALL = ["null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
                  "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn",
                  "achtzehn", "neunzehn"]
const DE_TENS = Dict(20 => "zwanzig", 30 => "dreißig", 40 => "vierzig", 50 => "fünfzig",
                     60 => "sechzig", 70 => "siebzig", 80 => "achtzig", 90 => "neunzig")
const DE_MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August",
                   "September", "Oktober", "November", "Dezember"]
const DE_ABBREVIATIONS = [
    "z. B." => "zum Beispiel", "z.B." => "zum Beispiel", "bzw." => "beziehungsweise",
    "usw." => "und so weiter", "u. a." => "unter anderem", "u.a." => "unter anderem",
    "d. h." => "das heißt", "d.h." => "das heißt", "ggf." => "gegebenenfalls", "ca." => "circa",
    "inkl." => "einschließlich", "Nr." => "Nummer", "Dr." => "Doktor", "Prof." => "Professor",
]
const DE_UNITS = [
    "km/h" => "Kilometer pro Stunde", "km²" => "Quadratkilometer", "km2" => "Quadratkilometer",
    "m²" => "Quadratmeter", "m2" => "Quadratmeter", "m³" => "Kubikmeter", "m3" => "Kubikmeter",
    "km" => "Kilometer", "cm" => "Zentimeter", "mm" => "Millimeter", "kg" => "Kilogramm",
    "mg" => "Milligramm", "g" => "Gramm", "ml" => "Milliliter", "l" => "Liter", "Hz" => "Hertz",
    "kHz" => "Kilohertz", "MHz" => "Megahertz", "GHz" => "Gigahertz", "KB" => "Kilobyte",
    "MB" => "Megabyte", "GB" => "Gigabyte", "TB" => "Terabyte", "°C" => "Grad Celsius",
    "°F" => "Grad Fahrenheit",
]
const DE_CURRENCIES = ["€" => "Euro", "EUR" => "Euro", "Euro" => "Euro", "\$" => "Dollar",
                       "USD" => "Dollar", "£" => "Pfund", "GBP" => "Pfund"]
const DE_FILLERS = ["呃" => "äh", "嗯" => "hm", "噢" => "oh", "哦" => "oh", "诶" => "äh", "欸" => "äh", "啊" => "ah"]

"""Python's `sorted(items, key = -len(key))`: stable, longest first."""
longestfirst(table) = sort(table; by = p -> -length(first(p)), alg = MergeSort)

function underhundred(n::Integer)
    n < 20 && return DE_SMALL[Int(n) + 1]
    tens, ones = divrem(n, 10)
    word = DE_TENS[Int(tens * 10)]
    ones == 0 && return word
    return (ones == 1 ? "ein" : DE_SMALL[Int(ones) + 1]) * "und" * word
end

function underthousand(n::Integer)
    n < 100 && return underhundred(n)
    hundreds, rest = divrem(n, 100)
    prefix = hundreds == 1 ? "einhundert" : DE_SMALL[Int(hundreds) + 1] * "hundert"
    return prefix * (rest == 0 ? "" : underhundred(rest))
end

const DE_SCALES = [(big(10)^15, "Billiarde", "Billiarden"), (big(10)^12, "Billion", "Billionen"),
                   (big(10)^9, "Milliarde", "Milliarden"), (big(10)^6, "Million", "Millionen")]

"""`number_to_words`: an integer in written German."""
function numbertowords(n::Integer)
    n < 0 && return "minus " * numbertowords(-n)
    n < 1000 && return underthousand(n)
    if n < 1_000_000
        thousands, rest = divrem(n, 1000)
        prefix = thousands == 1 ? "eintausend" : numbertowords(thousands) * "tausend"
        return prefix * (rest == 0 ? "" : numbertowords(rest))
    end
    for (value, singular, plural) in DE_SCALES
        if n >= value
            count, rest = divrem(n, value)
            prefix = count == 1 ? "eine " * singular : numbertowords(count) * " " * plural
            return prefix * (rest == 0 ? "" : " " * numbertowords(rest))
        end
    end
    return join((DE_SMALL[d - '0' + 1] for d in string(n)), " ")
end

function ordinaltowords(n::Integer)
    special = Dict(1 => "erste", 3 => "dritte", 7 => "siebte", 8 => "achte")
    haskey(special, Int(n)) && return special[Int(n)]
    return numbertowords(n) * (n < 20 ? "te" : "ste")
end

digitstring(digits::AbstractString) = join((DE_SMALL[d - '0' + 1] for d in digits), " ")
toint(s::AbstractString) = parse(BigInt, s)

function numbertoken(raw::AbstractString)
    raw = replace(raw, "." => "")
    if occursin(',', raw)
        whole, fraction = split(raw, ','; limit = 2)
        return numbertowords(toint(whole)) * " Komma " * digitstring(fraction)
    end
    return numbertowords(toint(raw))
end

function yeartowords(raw::AbstractString)
    year = toint(raw)
    if 1100 <= year <= 1999
        century, rest = divrem(year, 100)
        return numbertowords(century) * "hundert" * (rest == 0 ? "" : numbertowords(rest))
    end
    return numbertowords(year)
end

"""`re.sub(pattern, fn, text)` with `fn` given the match."""
function resub(text::AbstractString, rx::Regex, f)
    out = IOBuffer()
    pos = 1
    for m in eachmatch(rx, text)
        print(out, SubString(text, pos, prevind(text, m.offset)))
        print(out, f(m))
        pos = m.offset + ncodeunits(m.match)
    end
    print(out, SubString(text, pos))
    return String(take!(out))
end

"""`normalize_german_v2`: numbers, dates, times, money, units and abbreviations."""
function normalizegerman(text::AbstractString)
    text = Base.Unicode.normalize(String(text), :NFC)
    for (src, rep) in DE_FILLERS
        text = replace(text, src => " " * rep * " ")
    end
    text = replace(text, r"[\u3400-\u9fff]+" => " ")
    for (abbr, full) in longestfirst(DE_ABBREVIATIONS)
        text = replace(text, Regex("(?<!\\w)" * rxescape(abbr), "i") => full)
    end
    text = resub(text, r"\b([0-3]?\d)\.([01]?\d)\.(\d{4})\b", m -> begin
        day, month = toint(m[1]), toint(m[2])
        (1 <= day <= 31 && 1 <= month <= 12) || return m.match
        ordinaltowords(day) * "n " * DE_MONTHS[Int(month)] * " " * yeartowords(m[3])
    end)
    months = join(rxescape.(DE_MONTHS), "|")
    text = resub(text, Regex("\\b([0-3]?\\d)\\.\\s+(" * months * ")\\s+(\\d{4})\\b", "i"),
                 m -> ordinaltowords(toint(m[1])) * "n " * m[2] * " " * yeartowords(m[3]))
    clock(m) = numbertowords(toint(m[1])) * " Uhr " * numbertowords(toint(m[2]))
    text = resub(text, r"\b([01]?\d|2[0-3]):([0-5]\d)(?:\s*Uhr)?\b"i, clock)
    text = resub(text, r"\b([01]?\d|2[0-3])\.([0-5]\d)\s*Uhr\b"i, clock)
    text = resub(text, r"%\s*(\d[\d.,]*)", m -> numbertoken(m[1]) * " Prozent")
    text = resub(text, r"(\d[\d.,]*)\s*%", m -> numbertoken(m[1]) * " Prozent")
    for (symbol, word) in longestfirst(DE_CURRENCIES)
        e = rxescape(symbol)
        text = resub(text, Regex("(\\d[\\d.]*)[,]([0-9]{2})\\s*" * e * "(?!\\w)", "i"),
                     m -> numbertowords(toint(replace(m[1], "." => ""))) * " " * word * " " *
                          numbertowords(toint(m[2])) * " Cent")
        text = resub(text, Regex(e * "\\s*(\\d[\\d.,]*)", "i"), m -> numbertoken(m[1]) * " " * word)
        text = resub(text, Regex("(\\d[\\d.,]*)\\s*" * e * "(?!\\w)", "i"), m -> numbertoken(m[1]) * " " * word)
    end
    for (unit, word) in longestfirst(DE_UNITS)
        text = resub(text, Regex("\\b(\\d[\\d.,]*)\\s*" * rxescape(unit) * "(?!\\w)", "i"),
                     m -> numbertoken(m[1]) * " " * word)
    end
    text = resub(text, r"\b\d{1,3}(?:\.\d{3})+\b", m -> replace(m.match, "." => ""))
    text = resub(text, r"\b(\d+),(\d+)\b", m -> numbertowords(toint(m[1])) * " Komma " * digitstring(m[2]))
    text = resub(text, r"\b(?:1[1-9]\d{2}|20\d{2})\b", m -> yeartowords(m.match))
    text = resub(text, r"\b\d+\b", m -> numbertowords(toint(m.match)))
    text = replace(text, r"\s+([,.;:!?])" => s"\1")
    return strip(replace(text, r"\s+" => " "))
end

"""
    preparegerman(text) -> String

`prepare_german`: the punctuation table, [`normalizegerman`](@ref), lower case,
pronunciation annotations, upper-cased special-token names. Run before
[`prepare`](@ref) with `lang = "de"`, as the checkpoint's own `inference.py` does
(upstream's normaliser off).
"""
function preparegerman(text::AbstractString)
    t = cleantext(text)
    t = normalizegerman(t)
    t = lowercase(t)
    t = upperspecials(annotate(t))
    return String(strip(t))
end
