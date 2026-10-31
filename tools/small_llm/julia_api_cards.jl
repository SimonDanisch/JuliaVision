module JuliaAPICards
using SHA, JSON3

"""Read documentation from an explicitly trusted, already-loaded module.

Never evaluate candidate code, resolve user-supplied modules, or invent an API
signature. Cards retain missing-doc status and truncation rather than treating a
symbol's mere existence as proof that a candidate call is valid.
"""
function cards(package::Module,symbols;chars=3000)
    128<=chars<=8000 || throw(ArgumentError("card size"))
    length(symbols)<=16 || throw(ArgumentError("too many symbols"))
    map(unique(Symbol.(symbols))) do name
        defined=isdefined(package,name)
        documented=defined && Base.Docs.hasdoc(package,name)
        doc=documented ? Base.Docs.doc(Base.Docs.Binding(package,name)) : nothing
        full=doc===nothing ? "" : sprint(show,MIME("text/plain"),doc)
        (;package=string(package),name=String(name),defined,documented,
            text=first(full,chars),truncated=length(full)>chars,full_document_sha256=bytes2hex(sha256(full)))
    end
end

format_cards(rows)=join(("API documentation for $(r.package).$(r.name); defined=$(r.defined), documented=$(r.documented), truncated=$(r.truncated)\n$(r.text)" for r in rows),"\n\n")
end
