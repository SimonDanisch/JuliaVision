using SHA,JSON3
const BAYES_NATIVE_PUBLIC_FILES=("run_bayes_native_audit.jl","bayes_native_parity.jl","test_bayes_native_parity.jl",
    "bayes_json.jl","portable_evidence.jl","device_finish.jl","runtime_provenance.jl")

"""PUBLIC grader source only. NEVER pass the private packet to file_session."""
function bayes_native_source_updates()
    Dict(file=>read(joinpath(@__DIR__,file)) for file in BAYES_NATIVE_PUBLIC_FILES)
end

function deploy_bayes_native_sources(file_session)
    updates=bayes_native_source_updates()
    expected=Dict(file=>bytes2hex(sha256(bytes)) for (file,bytes) in updates)
    actual=file_session(updates) do updates
        allowed=("run_bayes_native_audit.jl","bayes_native_parity.jl","test_bayes_native_parity.jl",
            "bayes_json.jl","portable_evidence.jl","device_finish.jl","runtime_provenance.jl")
        Set(keys(updates))==Set(allowed)||error("PUBLIC source allowlist only; NO private packet/data")
        S=Base.require(Main,:SHA);root="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm"
        islink(root)&&error("Linked source root");result=Dict{String,String}()
        for (file,bytes) in updates
            basename(file)==file||error("Basenames only")
            path=joinpath(root,file);islink(path)&&error("Linked source")
            if isfile(path);read(path)==bytes||error("Preserve different existing source")
            else;open(io->write(io,bytes),path,"w")
            end
            result[file]=bytes2hex(Base.invokelatest(S.sha256,read(path)))
        end
        output=joinpath(root,"bayes_extraction_native_audit");islink(output)&&error("Linked report directory");mkpath(output)
        result
    end
    actual==expected||error("Grader source transfer drift")
    (;complete=true,public_source_sha256=actual,privacy="NO private packet/expected arrays transferred through this file API")
end

function collect_bayes_native_report(file_session;directory=joinpath(@__DIR__,"bayes_extraction_native_audit"))
    bytes=file_session() do
        path="/tmp/small-llm-bosgame-20261025/dev/JuliaVision/tools/small_llm/bayes_extraction_native_audit/result.json"
        islink(path)&&error("Linked report");read(path)
    end
    path=joinpath(directory,"result.json");mkpath(directory)
    isfile(path)&&read(path)!=bytes&&error("Preserve different native result")
    isfile(path)||open(io->write(io,bytes),path,"w")
    report=JSON3.read(bytes)
    report.complete&&report.normal_final_waitidle&&all(r->r.result.passed,report.rows)||error("Native audit incomplete/failed; do not claim normal release")
    (;complete=true,cases=length(report.rows),harness_checks=report.harness_checks,report_sha256=bytes2hex(sha256(bytes)))
end
