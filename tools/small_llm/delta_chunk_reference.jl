# CPU oracle for a future parallel GPU Gated DeltaNet chunk kernel. This is
# separate from the current prefill path, which reuses validated recurrent steps.
module DeltaChunkReference
using LinearAlgebra

function validate(q,k,v,decay,beta,state)
    n=size(q,2)
    n>0 && size(q)==size(k) && size(v,2)==n && length(decay)==length(beta)==n &&
        size(state)==(size(q,1),size(v,1)) || throw(DimensionMismatch("Gated DeltaNet chunk"))
    all(a->all(isfinite,a),(q,k,v,decay,beta,state)) || throw(ArgumentError("nonfinite chunk input"))
    all(x->0<=x<=1,decay) && all(x->0<=x<=1,beta) || throw(ArgumentError("gate outside [0,1]"))
    nothing
end

"""Sequential oracle for normalized Q/K, activated gates, and K-by-V state.

Returns pre-normalization, pre-output-gate values, including 1/sqrt(keydim).
"""
function recurrent(q,k,v,decay,beta,state)
    validate(q,k,v,decay,beta,state)
    s=copy(state);y=similar(v)
    for t in axes(q,2)
        s .*= decay[t]
        delta=beta[t].*(v[:,t]-s'*k[:,t])
        s .+= k[:,t]*delta'
        y[:,t]=s'*q[:,t]/sqrt(size(q,1))
    end
    (;output=y,state=s)
end

"""Equivalent chunk computation using a lower-triangular solve and matmuls.

Write residuals satisfy
    delta[t] + sum(j<t, beta[t]*D[t,j]*dot(k[t],k[j])*delta[j])
      = beta[t]*(v[t]-gamma[t]*state0'*k[t]).
D[t,j] is the product of decay[j+1:t]. Explicit products avoid 0/0 when gates
reset state or cumulative decay underflows. This CPU reference materializes
chunk-by-chunk (not sequence-by-sequence) quadratic scratch.
"""
function chunk(q,k,v,decay,beta,state)
    validate(q,k,v,decay,beta,state)
    n=size(q,2);T=promote_type(eltype(q),eltype(k),eltype(v),eltype(state),eltype(decay),eltype(beta))
    d=zeros(T,n,n)
    for j in 1:n
        d[j,j]=one(T)
        for t in j+1:n
            d[t,j]=decay[t]*d[t-1,j]
        end
    end
    gamma=cumprod(decay)
    lhs=Matrix{T}(I,n,n)
    kk=k'*k
    for t in 1:n, j in 1:t-1
        lhs[t,j]=beta[t]*d[t,j]*kk[t,j]
    end
    rhs=reshape(beta,:,1).*(permutedims(v)-reshape(gamma,:,1).*(k'*state))
    delta=LowerTriangular(lhs)\rhs
    output=permutedims(reshape(gamma,:,1).*(q'*state)+(q'*k.*d)*delta)/sqrt(size(q,1))
    final=gamma[end]*state+k*(reshape(d[end,:],:,1).*delta)
    (;output,state=final)
end

end
