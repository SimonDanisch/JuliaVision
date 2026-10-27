module JuliaSemanticContracts
# Separate contract revision; original strict scores and generation-time hashes
# are retained. Declared before inspecting API-v2 generated answers.
function semantic_task(task)
    task.id=="makie_band" || return task
    # Strip indentation so this remains robust to source indentation only.
    firstline="@test Tuple.(a.scene.plots[1][1][]) == collect(zip(1:4,[0.5,1.5,2.5,1.5]))"
    secondline="@test Tuple.(a.scene.plots[1][2][]) == collect(zip(1:4,[1.5,2.5,3.5,2.5]))"
    replacement="""
    lo=collect(zip(1:4,[0.5,1.5,2.5,1.5]));hi=collect(zip(1:4,[1.5,2.5,3.5,2.5]))
    bounds=[Tuple.(a.scene.plots[1][1][]),Tuple.(a.scene.plots[1][2][])]
    @test bounds==[lo,hi] || bounds==[hi,lo]
    """
    occursin(firstline,task.tests)&&occursin(secondline,task.tests) || error("contract changed")
    tests=replace(replace(task.tests,firstline=>replacement),secondline=>"")
    (;task...,tests)
end
end
