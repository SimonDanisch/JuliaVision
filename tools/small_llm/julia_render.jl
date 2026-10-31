using JSON3, SHA

function bounded_png_dimensions(path)
    islink(path) && throw(ArgumentError("artifact symlink refused"))
    isfile(path) && 33<=filesize(path)<=8_000_000 || throw(ArgumentError("artifact size"))
    header=open(io->read(io,24),path)
    header[1:8]==UInt8[0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a] || throw(ArgumentError("not PNG"))
    header[13:16]==collect(codeunits("IHDR")) || throw(ArgumentError("missing PNG header"))
    integer(bytes)=foldl((n,b)->(n<<8)|Int(b),bytes;init=0)
    width=integer(header[17:20]);height=integer(header[21:24])
    1<=width<=2000 && 1<=height<=2000 || throw(ArgumentError("image dimension budget"))
    (;width,height)
end

"""Render a candidate ONLY in the restricted subprocess.

`figure_expression` is trusted harness code using public example inputs, never
assembled from the candidate. This records images, not an aesthetic score.
"""
function render_julia_candidate(code,figure_expression;artifact_path)
    # Never overwrite a previous attempt: first-draw/render timing differences
    # are evidence too, even when the candidate source is identical.
    if ispath(artifact_path)
        stem,extension=splitext(artifact_path)
        artifact_path=stem*"_"*string(time_ns())*extension
    end
    result=mktempdir() do directory
        input=joinpath(directory,"input");output=joinpath(directory,"output");mkdir(input);mkdir(output)
        write(joinpath(input,"candidate.jl"),code)
        layout_probe="""
        canvas=Tuple(Makie.size(figure.scene))
        layouts=Any[]
        for axis in figure.content
            axis isa Makie.Axis || continue
            box=axis.layoutobservables.computedbbox[]
            p=axis.layoutobservables.protrusions[]
            bounds=[box.origin[1]-p.left,box.origin[2]-p.bottom,
                box.origin[1]+box.widths[1]+p.right,box.origin[2]+box.widths[2]+p.top]
            inside=bounds[1]>=-1 && bounds[2]>=-1 && bounds[3]<=canvas[1]+1 && bounds[4]<=canvas[2]+1
            push!(layouts,(;title=String(axis.title[]),decorated_bounds=bounds,within_canvas=inside))
        end
        println("LAYOUT_REPORT=",JSON3.write((;canvas,axes=layouts)))
        """
        write(joinpath(input,"runner.jl"),"using CairoMakie,JSON3\nCairoMakie.activate!()\ninclude(\"/input/candidate.jl\")\nfigure = "*figure_expression*"\nsave(\"/output/plot.png\",figure;px_per_unit=1)\n"*layout_probe*"\nprintln(\"RENDER_COMPLETED\")\n")
        log=joinpath(directory,"render.log");started=time()
        process=open(log,"w") do stream
            command=`bash $(joinpath(@__DIR__,"julia_sandbox.sh")) $input makie $output`
            p=run(pipeline(ignorestatus(command),stdout=stream,stderr=stream);wait=false)
            try
                wait(p)
            finally
                process_running(p) && (kill(p);wait(p))
            end
            p
        end
        diagnostic=open(io->String(read(io,65536)),log)
        if success(process)&&occursin("RENDER_COMPLETED",diagnostic)
            image=joinpath(output,"plot.png");dimensions=bounded_png_dimensions(image)
            lines=filter(l->startswith(l,"LAYOUT_REPORT="),split(diagnostic,'\n'))
            layout=length(lines)==1 ? JSON3.read(only(lines)[length("LAYOUT_REPORT=")+1:end]) : nothing
            mkpath(dirname(artifact_path));cp(image,artifact_path;force=true)
            (;rendered=true,artifact_path=abspath(artifact_path),dimensions,elapsed_seconds=time()-started,
                sha256=bytes2hex(sha256(read(artifact_path))),layout,diagnostic)
        else
            (;rendered=false,exitcode=process.exitcode,elapsed_seconds=time()-started,diagnostic)
        end
    end
    open(io->JSON3.pretty(io,result),artifact_path*".json","w")
    result
end
