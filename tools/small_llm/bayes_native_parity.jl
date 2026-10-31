isdefined(@__MODULE__,:BayesJSON)||include("bayes_json.jl")
isdefined(@__MODULE__,:PortableEvidence)||include("portable_evidence.jl")
import Mantle

function bayes_native_snapshot(execution)
    out=execution.outputs
    (;status=only(Array(out.status)),prior=Array(out.prior),posterior=Array(out.posterior),
        log_likelihood=Array(out.log_likelihood),defined=Array(out.defined),log_evidence=only(Array(out.log_evidence)))
end

function bayes_parity_float(value)
    value isa Bool&&throw(ArgumentError("Not a numeric parity reference"))
    if value isa AbstractString
        value=="-Inf"&&return -Inf32
        value=="+Inf"&&return Inf32
        throw(ArgumentError("Only explicit signed infinities permitted"))
    end
    value isa Real&&isfinite(value)||throw(ArgumentError("Finite number required"))
    result=Float32(value)
    isfinite(result)||throw(ArgumentError("Reference overflows Float32"))
    result
end

function bayes_parity_encode(value::Real)
    value isa Bool&&throw(ArgumentError("Not a numeric parity reference"))
    isfinite(value) ? value : value==-Inf ? "-Inf" : value==Inf ? "+Inf" : throw(ArgumentError("No NaN references"))
end
function encode_bayes_snapshot(snapshot)
    (;status=Int(snapshot.status),prior=bayes_parity_encode.(snapshot.prior),posterior=bayes_parity_encode.(snapshot.posterior),
        log_likelihood=bayes_parity_encode.(snapshot.log_likelihood),defined=Int.(snapshot.defined),log_evidence=bayes_parity_encode(snapshot.log_evidence))
end

function audit_bayes_native_case(device,network,expected)
    packed=BayesJSON.compile(network;max_states=256)
    execution=PortableEvidence.prepare_bayes(device,packed.factors,packed.query,packed.observed)
    try
        before=Mantle.submissions(device);PortableEvidence.run!(execution);Mantle.waitidle(device)
        first_submissions=Mantle.submissions(device)-before
        first=bayes_native_snapshot(execution)
        before=Mantle.submissions(device);PortableEvidence.run!(execution);Mantle.waitidle(device)
        replay_submissions=Mantle.submissions(device)-before
        second=bayes_native_snapshot(execution)
        close_number(a,b)=isfinite(a)&&isfinite(b) ? isapprox(a,b;atol=3f-5,rtol=3f-5) : a==b
        close_vector(a,b)=length(a)==length(b)&&all(close_number(x,bayes_parity_float(y)) for (x,y) in zip(a,b))
        status_match=first.status==expected.status&&first.defined==Int32.(expected.defined)
        numeric_match=all(close_vector(getproperty(first,k),getproperty(expected,k)) for k in (:prior,:posterior,:log_likelihood))&&close_number(first.log_evidence,bayes_parity_float(expected.log_evidence))
        bitwise_replay=first.status==second.status&&first.defined==second.defined&&
            all(reinterpret(UInt32,getproperty(first,k))==reinterpret(UInt32,getproperty(second,k)) for k in (:prior,:posterior,:log_likelihood))&&
            reinterpret(UInt32,first.log_evidence)==reinterpret(UInt32,second.log_evidence)
        (;passed=status_match&&numeric_match&&bitwise_replay&&first_submissions==replay_submissions==1,
            status_match,numeric_match,bitwise_replay,first_submissions,replay_submissions,nstates=packed.nstates)
    finally
        close(execution)
    end
end
