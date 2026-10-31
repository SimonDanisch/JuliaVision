module JuliaMethodCards
using SHA

"""Bounded installed-source context for explicitly trusted method objects.

No candidate evaluation or dynamic module lookup. Paths must resolve inside a
caller-supplied source root; symlinks cannot escape it. Context is not claimed
to be a complete method body or a proof that a model-authored call is valid.
"""
function cards(methods_to_read;root,context_lines=24,max_cards=8,max_file_bytes=2_000_000)
    1<=context_lines<=80 && 1<=max_cards<=16 || throw(ArgumentError("card bounds"))
    1<=max_file_bytes<=4_000_000 || throw(ArgumentError("source size bound"))
    source_root=realpath(root);prefix=source_root*string(Base.Filesystem.path_separator)
    selected=collect(methods_to_read);length(selected)<=max_cards || throw(ArgumentError("too many methods"))
    all(method->method isa Method,selected) || throw(ArgumentError("trusted Method objects required"))
    map(selected) do method
        file=String(method.file)
        isfile(file) || throw(ArgumentError("method source unavailable"))
        canonical=realpath(file)
        startswith(canonical,prefix) || throw(ArgumentError("source outside approved root"))
        filesize(canonical)<=max_file_bytes || throw(ArgumentError("source file too large"))
        source=read(canonical,String);lines=split(source,'\n');line=Int(method.line)
        1<=line<=length(lines) || throw(ArgumentError("method line unavailable"))
        stop=min(length(lines),line+context_lines-1)
        (;signature=string(method.sig),owner=string(method.module),file=relpath(canonical,source_root),
            first_line=line,last_line=stop,text=join(lines[line:stop],"\n"),
            bounded_context_not_complete_body=true,source_sha256=bytes2hex(sha256(source)))
    end
end

format_cards(rows)=join(("Installed method $(r.signature), owner $(r.owner), $(r.file):$(r.first_line). Bounded source context, not necessarily a complete body:\n$(r.text)" for r in rows),"\n\n")
end
