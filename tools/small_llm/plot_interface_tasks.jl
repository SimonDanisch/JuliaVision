"""Public requests ONLY. References, probes and grades stay in another file."""
function plot_interface_tasks()
    [
        (;id="plot_transfer_01",prompt="Create one720×480 panel titled Calibration. Label x as Time (s), y as Value; both scales linear. Draw a against x as a blue line labeled Sensor A. Put its legend at upper right."),
        (;id="plot_transfer_02",prompt="Create one800×500 panel titled Samples. Label x as Time, y as Score; both scales linear. Draw b against x as green scatter points labeled Measured B. Put the legend at upper left."),
        (;id="plot_transfer_03",prompt="Create one900×500 panel titled Envelope check, with x label Time and y label Signal, both scales linear. FIRST draw a blue band against x from lo to hi, labeled Range. THEN draw a as an orange line against x labeled Center. Put the legend at upper right."),
        (;id="plot_transfer_04",prompt="Create two horizontal panels in a1000×480 figure; link their linear x axes. Both x labels are Time. Left: title Two sensors, y label Value, linear y; FIRST blue line a labeled A, THEN red line b labeled B, both against x; legend upper right. Right: title Replicates, y label Reading, linear y; green scatter b against x labeled Cloud; legend upper left."),
        (;id="plot_transfer_05",prompt="Create one760×500 panel titled Positive signals. Label x as Time and y as Response. Use linear x and log10 y. FIRST draw b as a green line labeled B first, THEN a as a red line labeled A second, both against x. Put the legend at lower left."),
        (;id="plot_transfer_06",prompt="Create two horizontal panels in a1000×500 figure, with LINKED log10 x axes and linear y axes. Both x labels are Input and both y labels are Value. Left title Log-x A: blue scatter a against x labeled Raw A, legend lower left. Right title Log-x B: orange line b against x labeled Raw B, legend upper right."),
        (;id="plot_transfer_07",prompt="Create one720×480 panel titled No legend, x label x and y label Reading, both scales linear. FIRST draw a blue band from lo to hi against x, THEN draw a as a red line against x. Give BOTH series empty labels and do NOT create any legend."),
        (;id="plot_transfer_08",prompt="Create one800×500 panel titled Double log. Label x as Input and y as Positive output. Use log10 on BOTH axes. FIRST draw orange scatter a labeled Samples, THEN a green line b labeled Trend, both against x. Put the legend at lower right.")
    ]
end

const PLOT_INTERFACE_PUBLIC_CONTRACT="""
Supplied input data is a Dict with STRING keys x,a,b,lo,hi. Values are nonempty,
finite, positive host vectors of equal length; lo<=hi elementwise. Use these
arrays by reference, NEVER synthesize/transform/mutate them. One or two axes
occupy figure row1, columns1:2, in left-to-right declaration order. A request
defines plot order as drawing order. Do not save/display files or install code.
Palette roles are blue=#0072B2, orange=#E69F00, green=#009E73, red=#D55E00.
Fixed style: font16; lines linewidth2.5; scatter markersize9; band alpha0.25;
legends frameless. This is a narrow plot-construction interface comparison,
not unrestricted Julia. Both output representations below are supported.

Julia representation: optional using/import CairoMakie or Makie; exactly ONE
long-form, UNTYPED function makeplot(data). Its body permits only simple local
assignments, the listed constructor calls, and a final return of its Figure.
No loops, macros, helper functions, property mutation, type annotations or I/O.
Calls: Figure(size=(width,height),fontsize=16); Axis(fig[1,column];title="...",
xlabel="...",ylabel="...",xscale=identity or log10,yscale=identity or log10);
lines!(ax,data["x"],data["a"];color="#0072B2",linewidth=2.5,label="...");
scatter! analogously with markersize=9; band!(ax,data["x"],data["lo"],data["hi"];
color=("#0072B2",0.25),label="..."); axislegend(ax;position=:rt/:lt/:rb/:lb,
framevisible=false); linkxaxes!(left,right). Use explicit fixed style keywords.
Native Julia uses exact palette HEX values, not CSS :blue etc. Axis scales may
be omitted ONLY for linear. Create legends on ALL labeled axes or none, per
request. Return fig explicitly.

JSON representation: one object with size:[width,height], legend:boolean,
linkx:boolean, axes:[{title,xlabel,ylabel,xscale,yscale,legend_position}],
series:[{kind,axis,x,y,label,color} OR {kind,axis,x,low,high,label,color}].
Axes xscale/yscale are strings linear or log10; legend_position is rt,lt,rb,lb
(ignored when no legend). Series kind is line,scatter,band; axis is a1-based
integer; x/y/low/high are data-key STRINGS; color is a palette ROLE string.
Band requires low/high and forbids y; line/scatter require y, not low/high.
Every axis requires title/xlabel/ylabel/xscale/yscale. Every series requires
kind/axis/x/label/color and its data reference fields. No unknown keys, data
literals, expressions, comments, fences or explanations. Styles are supplied
by the trusted addon; JSON does NOT repeat linewidth/alpha/markersize/fontsize.

Bounds in both interfaces: width400:1600, height300:1200,1:2 axes,1:8 series,
plain text labels, at most100000 input points. The experiment never executes
generated Julia; it accepts only the documented constructor subset.
"""

function plot_interface_messages(task,arm)
    arm in ("julia_source","plot_spec")||throw(ArgumentError("Unknown interface"))
    format=arm=="julia_source" ? "Return ONLY the permitted Julia function program, optional supported import. No fences or explanation." :
        "Return ONLY the permitted JSON plot specification. No Julia, fences or explanation."
    ["system"=>"Translate the user's plot request exactly. "*format,
        "user"=>PLOT_INTERFACE_PUBLIC_CONTRACT*"\n\nUser request:\n"*task.prompt]
end
