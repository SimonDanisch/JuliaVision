isdefined(@__MODULE__,:RPNExampleSemantics)||include("rpn_example_semantics.jl")
isdefined(@__MODULE__,:RPNPrefixGrammar)||include("rpn_prefix_grammar.jl")

"""Bounded enumerative synthesis from supplied examples ONLY.

CPU schedules/deduplicates a finite expression search. All numeric evaluations
call the SAME RPNExampleSemantics.step! used by GPU constraints. No task names,
natural-language specifications, private formulas or benchmark tests are read.
Observational equivalence on examples is NOT equivalence for arbitrary inputs.
"""
module PublicExampleSynthesis
import ..RPNExampleSemantics
import ..RPNPrefixGrammar

struct Node
    action::Int32
    left::Int32
    right::Int32
    cost::Int32
    peak::Int32
end

function program(nodes,index)
    actions=String[]
    function visit(i)
        node=nodes[i]
        node.left!=0&&visit(node.left)
        node.right!=0&&visit(node.right)
        push!(actions,RPNPrefixGrammar.ACTIONS[node.action])
    end
    visit(index);push!(actions,"finish");join(actions,"\n")*"\n"
end

function solve(bank::RPNExampleSemantics.Bank;max_cost=8,candidate_budget=2_000_000,state_budget=100_000,seconds=10.0)
    max_cost isa Integer&&!(max_cost isa Bool)&&1<=max_cost<=32||throw(ArgumentError("invalid cost bound"))
    candidate_budget isa Integer&&!(candidate_budget isa Bool)&&1<=candidate_budget<=20_000_000||throw(ArgumentError("invalid candidate bound"))
    state_budget isa Integer&&!(state_budget isa Bool)&&1<=state_budget<=1_000_000||throw(ArgumentError("invalid state bound"))
    seconds isa Real&&!(seconds isa Bool)&&isfinite(seconds)&&0<seconds<=60||throw(ArgumentError("invalid deadline"))
    0<length(bank.expected)<=64||throw(ArgumentError("bounded public-example cohort required"))
    bank.atol==bank.rtol==0f0||throw(ArgumentError("exact public examples required"))
    _solve(bank,Val(length(bank.expected)),Int(max_cost),Int(candidate_budget),Int(state_budget),Float64(seconds))
end

function _solve(bank,::Val{N},max_cost,candidate_budget,state_budget,seconds) where N
    started=time_ns();deadline=started+round(UInt64,seconds*1e9)
    # Keep separate peak-stack classes. A cheaper but deeper representation
    # must not erase a shallower equivalent needed in a later right subtree.
    nodes=Node[];values=NTuple{N,Float32}[];seen=Dict{Tuple{NTuple{N,UInt32},Int32},Int32}()
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
        push!(nodes,Node(Int32(action),left,right,Int32(cost),Int32(peak)))
        push!(values,value);index=Int32(length(nodes));seen[key]=index;push!(by_cost[cost],index)
        value==target&&(found=index)
        nothing
    end
    for cost in 1:max_cost
        if cost==1
            for action in 1:9
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
                        peak=max(nodes[left].peak,Int32(1)+nodes[right].peak)
                        candidate(action,left,right,cost,peak)
                        (found!=0||!isempty(stop))&&break
                    end
                    (found!=0||!isempty(stop))&&break
                end
            end
        end
        (found!=0||!isempty(stop))&&break
    end
    (;found=found!=0,reason=found!=0 ? "public_example_match" : isempty(stop) ? "cost_exhausted" : stop,
        program=found!=0 ? program(nodes,found) : "",operations=found!=0 ? Int(nodes[found].cost) : nothing,
        peak_stack=found!=0 ? Int(nodes[found].peak) : nothing,candidates=attempts,unique_states=length(nodes),
        elapsed_seconds=Float64(time_ns()-started)/1e9,max_cost,candidate_budget,state_budget,seconds,
        scope="Public-example-only symbolic synthesis, NOT model intelligence or correctness beyond these examples. First match, no hidden-score selection; all numeric operations use shared CPU/GPU evaluator.")
end
end
