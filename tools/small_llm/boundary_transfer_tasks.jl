"""Fresh authored boundary-contract transfer cohort. Public data only.
Same fixed nonmonotonic examples for every task; no hidden tests/reference IR.
"""
function boundary_transfer_tasks()
    x=Float32[1.5,-0.25,2,-1]
    rows=[
        ("previous_clamped_square","Square the previous clamped neighbor.",Float32[2.25,2.25,0.0625,4],0.25f0),
        ("next_periodic_square","Square the next periodic neighbor.",Float32[0.0625,4,1,2.25],0.25f0),
        ("current_above_previous_clamped","Subtract the previous clamped neighbor from the current element, replacing a negative difference with zero.",Float32[0,0,2.25,0],0f0),
        ("periodic_neighbor_minimum","Return the smaller of the previous and next periodic neighbors.",Float32[-1,1.5,-1,1.5],-0.5f0),
        ("clamped_neighbor_product","Multiply the previous clamped neighbor by the next clamped neighbor.",Float32[-0.375,3,0.25,-2],0.25f0),
        ("periodic_threepoint_curvature","Add the previous and next periodic neighbors, then subtract twice the current element.",Float32[-4.25,4,-5.25,5.5],0f0),
        ("clamped_neighbor_separation","Subtract the next clamped neighbor from the previous clamped neighbor, then return the absolute value.",Float32[1.75,0.5,0.75,3],0f0),
        ("periodic_threepoint_upper_envelope","Return the maximum of the current element and the previous and next periodic neighbors.",Float32[1.5,2,2,2],-0.5f0)]
    [(;id=r[1],specification=r[2],public_cases=[(;input=copy(x),output=copy(r[3])),
        (;input=Float32[-0.5],output=Float32[r[4]])]) for r in rows]
end
