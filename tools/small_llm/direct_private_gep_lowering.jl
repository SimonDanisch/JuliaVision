"""Isolated LLVM-pass prototype, NOT installed in Lava or any model engine.

The reported triangle-tuple compiler issue motivates this narrower experiment.
Only a direct alloca/GEP-leaf PHI consumed immediately after join PHIs qualifies.
Nested pointer chains and conditional predecessor edges are refused, since
moving loads through them can cross writes or speculate an invalid access.
This establishes no cause for earlier LLM failures or driver resets.
"""
module DirectPrivateGEPLowering
using LLVM

function alloca_root(pointer)
    while pointer isa LLVM.GetElementPtrInst
        pointer.value_type.addrspace==0||return nothing
        pointer=pointer.operands[1]
    end
    pointer isa LLVM.AllocaInst&&pointer.value_type.addrspace==0 ? pointer : nothing
end

function eligible(load,block)
    load isa LLVM.LoadInst||return false
    LLVM.API.LLVMGetVolatile(load)==0&&LLVM.API.LLVMGetOrdering(load)==LLVM.API.LLVMAtomicOrderingNotAtomic||return false
    pointer=load.operands[1]
    pointer isa LLVM.PHIInst&&pointer.parent==block&&pointer.value_type.addrspace==0||return false
    # No intervening instruction, including a store or call, may be crossed.
    for before in block.instructions
        before==load&&break
        before isa LLVM.PHIInst||return false
    end
    incoming=collect(pointer.incoming)
    any(pair->first(pair) isa LLVM.GetElementPtrInst,incoming)||return false
    for (leaf,predecessor) in incoming
        leaf isa LLVM.UndefValue&&continue
        leaf isa Union{LLVM.AllocaInst,LLVM.GetElementPtrInst}||return false
        root=alloca_root(leaf);root===nothing&&return false
        terminator=predecessor.terminator
        terminator!==nothing&&LLVM.API.LLVMGetNumSuccessors(terminator)==1||return false
        if leaf isa LLVM.AllocaInst
            layout=root.parent.parent.parent.datalayout
            allocation=LLVM.LLVMType(LLVM.API.LLVMGetAllocatedType(root))
            LLVM.API.LLVMABISizeOfType(layout,allocation)>=LLVM.API.LLVMABISizeOfType(layout,load.value_type)||return false
        end
    end
    true
end

function lower!(mod::LLVM.Module)
    count=0
    for fn in mod.functions
        isempty(fn.blocks)&&continue
        changed=true
        while changed
            changed=false
            for block in fn.blocks
                for load in block.instructions
                    eligible(load,block)||continue
                    pointer=load.operands[1];type=load.value_type
                    value=LLVM.@dispose builder=LLVM.IRBuilder() begin
                        LLVM.position!(builder,LLVM.before(load))
                        LLVM.phi!(builder,type,"direct_gep_value")
                    end
                    for (leaf,predecessor) in pointer.incoming
                        scalar=if leaf isa LLVM.UndefValue
                            LLVM.UndefValue(type)
                        else
                            LLVM.@dispose builder=LLVM.IRBuilder() begin
                                LLVM.position!(builder,LLVM.before(predecessor.terminator))
                                loaded=LLVM.load!(builder,type,leaf,"direct_gep_edge_load")
                                LLVM.API.LLVMSetAlignment(loaded,LLVM.API.LLVMGetAlignment(load))
                                loaded
                            end
                        end
                        push!(value.incoming,(scalar,predecessor))
                    end
                    LLVM.replace_uses!(load,value);LLVM.erase!(load)
                    isempty(LLVM.uses(pointer))&&LLVM.erase!(pointer)
                    count+=1;changed=true;break
                end
                changed&&break
            end
        end
    end
    count
end
end
