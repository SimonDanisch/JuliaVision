# Independent cursor calculation, not the engine's sizing helper. Read-only
# metadata inspection of already compiled compute dispatches; no GPU execution.
function packed_argument_extent(base,sizes)
    cursor=base
    for size in sizes
        size>0 || continue
        while cursor%8!=0;cursor+=1;end
        cursor+=size
    end
    cursor
end

function audit_plan_argument_extents(plan)
    rows=Any[]
    for pass in plan.passes, dispatch in pass.dispatches
        dispatch isa Mantle.CompiledDispatch || continue
        launch=dispatch.launch
        base=launch.arg_buffer_size;sizes=launch.byval_sizes
        required=packed_argument_extent(base,sizes)
        legacy=base+packed_argument_extent(0,sizes)
        push!(rows,(;kernel=string(typeof(dispatch.kernel)),base,required,
            allocated=dispatch.argsize,launch_allocated=launch.total_size,
            legacy_allocated=legacy,legacy_shortfall=max(required-legacy,0),byval_sizes=copy(sizes)))
    end
    isempty(rows) && error("no compiled Vulkan compute dispatches audited")
    (;dispatches=length(rows),all_fit=all(row->row.required<=min(row.allocated,row.launch_allocated),rows),
        legacy_underallocated=count(row->row.legacy_shortfall>0,rows),rows)
end
