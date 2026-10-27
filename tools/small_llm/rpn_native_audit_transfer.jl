isdefined(@__MODULE__,:RPN_NATIVE_AUDIT_SOURCES)||include("run_rpn_native_audit.jl")
isdefined(@__MODULE__,:deploy_document_study)||include("laptop_document_study_transfer.jl")

"""Sources ONLY; private numerical cases must not be written to the Laptop."""
function rpn_native_audit_source_payload(;source_root=@__DIR__)
    for name in ("rpn_kernel_device_audit_software_validation.json","rpn_native_audit_orchestration_validation.json")
        proof=JSON3.read(read(joinpath(source_root,name),String));proof.complete||error("unqualified audit helper")
        for (file,hash) in pairs(proof.source_sha256)
            basename(String(file))==String(file)&&bytes2hex(open(sha256,joinpath(source_root,String(file))))==hash||error("audit helper qualification drift")
        end
    end
    updates=Dict(file=>read(joinpath(source_root,file)) for file in RPN_NATIVE_AUDIT_SOURCES)
    (;root=DOCUMENT_STUDY_REMOTE_ROOT,updates)
end

deploy_rpn_native_audit_sources(file_session)=deploy_document_study(file_session,rpn_native_audit_source_payload())
