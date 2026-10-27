# Interface clarification after the first diagnostic; no completed plot, task
# reference, private assertion or test value is supplied. Preserve v1 failures.
const SCENE_SCHEMA_V2 = """
Return ONE valid JSON object. It has these keys:
"size": [integer width, integer height], each between 200 and 2000.
"axes": array of objects. Each has "id" (a unique STRING, never a number),
"row" and "col" (ONE-BASED integer grid positions 1..4).
Optional "title", "xlabel", "ylabel" are strings.
"plots": array in draw order. Each has "kind", "axis" (an axis id STRING), "x".
The ONLY kinds are "lines", "scatter", "band". Use "lines", NOT "line".
For lines/scatter also include "y"; for band include "lower" and "upper" instead.
Optional color STRING: blue/red/black/green/orange/gray/purple.
Optional alpha NUMBER between 0 and 1, optional label STRING.
Data values x/y/lower/upper are input-name strings, numbers, or actual nested JSON
OBJECTS. An absolute-value expression is {"op":"abs","arg":"a"}.
An elementwise sum is {"op":"+","left":"a","right":"b"}.
Replace a/b with supplied input names or nested expressions. Binary operators
allowed: + - * /. DO NOT encode expression objects inside strings or escape them.
"links": array of {"dimension":"x" or "y","axes":[axis id strings]}.
"legends": array of {"axis":axis id STRING,"position":"lt"/"rt"/"lb"/"rb"}.
"returns": ordered array containing "fig" and requested axis id STRINGS.
Use [] for empty links or legends. No extra fields, code, comments, or prose.
"""
