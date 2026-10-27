isdefined(@__MODULE__,:PublicExampleSynthesis)||include("public_example_synthesis.jl")
isdefined(@__MODULE__,:StencilBoundaryContract)||include("stencil_boundary_contract.jl")
module BoundaryExampleSynthesis
import ..PublicExampleSynthesis
import ..RPNExampleSemantics
import ..RPNPrefixGrammar
import ..StencilBoundaryContract

"""CPU bounded search with boundary feature admission. Every numerical action
uses the existing shared CPU/GPU evaluator, not a second arithmetic backend.
"""
function solve(bank::RPNExampleSemantics.Bank,mode::Symbol;max_cost=8,candidate_budget=2_000_000,state_budget=100_000,seconds=10.0)
    mode in (:periodic,:clamped,:unknown)||throw(ArgumentError("Invalid mode"))
    mode==:unknown&&return PublicExampleSynthesis.solve(bank;max_cost,candidate_budget,state_budget,seconds)
    max_cost isa Integer&&!(max_cost isa Bool)&&1<=max_cost<=32||throw(ArgumentError("Invalid cost"))
    candidate_budget isa Integer&&!(candidate_budget isa Bool)&&1<=candidate_budget<=20_000_000||throw(ArgumentError("Invalid candidates"))
    state_budget isa Integer&&!(state_budget isa Bool)&&1<=state_budget<=1_000_000||throw(ArgumentError("Invalid states"))
    seconds isa Real&&!(seconds isa Bool)&&isfinite(seconds)&&0<seconds<=60||throw(ArgumentError("Invalid deadline"))
    0<length(bank.expected)<=64&&bank.atol==bank.rtol==0f0||throw(ArgumentError("Exact bounded public examples only"))
    _solve(bank,Val(length(bank.expected)),mode,Int(max_cost),Int(candidate_budget),Int(state_budget),Float64(seconds))
end

function _solve(bank,::Val{N},mode,max_cost,candidate_budget,state_budget,seconds) where N
    started=time_ns();deadline=started+round(UInt64,seconds*1e9)
    P=PublicExampleSynthesis
    nodes=P.Node[];values=NTuple{N,Float32}[];seen=Dict{Tuple{NTuple{N,UInt32},Int32},Int32}()
    by_cost=[Int32[] for _ in 1:max_cost];work=zeros(Float32,2N);attempts=0;found=Int32(0);stop=""
    target=Tuple(bank.expected)
    function candidate(action,left,right,cost,peak)
        peak<=bank.max_depth||return
        if attempts>=candidate_budget;stop="candidate_budget";return;end
        if time_ns()>=deadline;stop="deadline";return;end
        attempts+=1
        arity=action<=9 ? Int32(0) : action<=11 ? Int32(1) : Int32(2)
        for example in 1:N
            arity>=1&&(work[2example-1]=values[left][example])
            arity==2&&(work[2example]=values[right][example])
            RPNExampleSemantics.step!(work,bank.features,Int32(example),Int32(action),arity,Int32(2))==0||return
        end
        value=ntuple(i->work[2i-1]==0f0 ? 0f0 : work[2i-1],Val(N))
        key=(ntuple(i->reinterpret(UInt32,value[i]),Val(N)),Int32(peak))
        haskey(seen,key)&&return
        if length(nodes)>=state_budget;stop="state_budget";return;end
        push!(nodes,P.Node(Int32(action),left,right,Int32(cost),Int32(peak)))
        push!(values,value);index=Int32(length(nodes));seen[key]=index;push!(by_cost[cost],index)
        value==target&&(found=index)
    end
    for cost in 1:max_cost
        if cost==1
            for action in 1:9
                StencilBoundaryContract.allowed_push(action,mode)||continue
                candidate(action,Int32(0),Int32(0),cost,1)
                (found!=0||!isempty(stop))&&break
            end
        else
            for left in by_cost[cost-1],action in 10:11
                candidate(action,left,Int32(0),cost,nodes[left].peak)
                (found!=0||!isempty(stop))&&break
            end
            if found==0&&isempty(stop)
                for left_cost in 1:cost-2
                    right_cost=cost-1-left_cost
                    for left in by_cost[left_cost],right in by_cost[right_cost],action in 12:16
                        candidate(action,left,right,cost,max(nodes[left].peak,Int32(1)+nodes[right].peak))
                        (found!=0||!isempty(stop))&&break
                    end
                    (found!=0||!isempty(stop))&&break
                end
            end
        end
        (found!=0||!isempty(stop))&&break
    end
    program=found!=0 ? P.program(nodes,found) : ""
    isempty(program)||StencilBoundaryContract.accepts(program,mode)||error("Search contract violated")
    (;found=found!=0,reason=found!=0 ? "public_example_and_boundary_match" : isempty(stop) ? "cost_exhausted" : stop,
        program,mode=String(mode),operations=found!=0 ? Int(nodes[found].cost) : nothing,
        candidates=attempts,unique_states=length(nodes),elapsed_seconds=Float64(time_ns()-started)/1e9,
        max_cost,candidate_budget,state_budget,seconds,
        scope="First fit to public examples within explicit boundary feature subset; no guarantee beyond these constraints. Shared numeric CPU/GPU evaluator.")
end
end
