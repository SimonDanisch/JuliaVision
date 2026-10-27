import Mantle
import KernelInterface as KI

# One invocation, ordered after the query in the SAME graph. No contending
# writers: cross-token execution follows the session's queue ordering.
function accumulate_memory_status!(history,current)
    KI.get_global_id().x==1 && (@inbounds history[1]=max(history[1],current[1]))
    nothing
end

function declare_memory_status_history!(graph,history,current)
    history!==current && length(history)==length(current)==1 || error("distinct scalar status buffers required")
    eltype(history)===eltype(current)===Int32 || error("Int32 status required")
    Mantle.dispatch!(graph,accumulate_memory_status!,(history,current),1;group=1,name="memory.status_history")
    nothing
end
