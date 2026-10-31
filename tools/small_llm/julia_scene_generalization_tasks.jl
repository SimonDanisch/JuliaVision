# Freeze these before any model run. New combinations for a later paired direct
# Julia / scene-v2 comparison; references and assertions stay out of prompts.
const SCENE_GENERALIZATION_TASKS = [
    (id="scene_ratio",category="makie",inputs=["x","y","scale"],
     prompt="Using Makie, write makeplot(x,y,scale) returning (fig,ax). Figure size (700,420). One Axis titled Scaled, xlabel Sample, ylabel Ratio. Orange scatter of y ./ scale versus x. Inputs are equal-length vectors, scale nonzero. No display, save, backend activation. Code only including imports.",
     reference="""
     using Makie
     function makeplot(x,y,scale)
         f=Figure(size=(700,420));a=Axis(f[1,1],title="Scaled",xlabel="Sample",ylabel="Ratio")
         scatter!(a,x,y./scale,color=:orange);(f,a)
     end
     """,tests="""
     f,a=makeplot([1.,2.,3.],[2.,-6.,8.],[2.,3.,4.])
     @test f isa Makie.Figure;@test a isa Makie.Axis
     @test Tuple(Makie.size(f.scene))==(700,420)
     @test a.title[]=="Scaled";@test a.xlabel[]=="Sample";@test a.ylabel[]=="Ratio"
     @test length(a.scene.plots)==1;@test a.scene.plots[1] isa Makie.Scatter
     @test Tuple.(a.scene.plots[1][1][])==[(1.,1.),(2.,-2.),(3.,2.)]
     @test Makie.to_color(a.scene.plots[1].color[])==Makie.to_color(:orange)
     """),
    (id="scene_shared_y",category="makie",inputs=["x","y"],
     prompt="Using Makie, write makeplot(x,y) returning (fig,left,right). Figure size (800,360). Two side-by-side Axis objects: left title Raw, right title Absolute, both xlabel Time and ylabel Value. Red lines of y versus x on left, green scatter of abs.(y) versus x on right. Link ONLY y axes. No display, save, backend activation. Code only including imports.",
     reference="""
     using Makie
     function makeplot(x,y)
         f=Figure(size=(800,360));a=Axis(f[1,1],title="Raw",xlabel="Time",ylabel="Value")
         b=Axis(f[1,2],title="Absolute",xlabel="Time",ylabel="Value")
         lines!(a,x,y,color=:red);scatter!(b,x,abs.(y),color=:green);linkyaxes!(a,b);(f,a,b)
     end
     """,tests="""
     f,a,b=makeplot([1.,2.,3.],[-2.,0.,3.])
     @test Tuple(Makie.size(f.scene))==(800,360)
     @test a.title[]=="Raw";@test b.title[]=="Absolute"
     @test all(ax->ax.xlabel[]=="Time" && ax.ylabel[]=="Value",(a,b))
     @test length(a.scene.plots)==1;@test length(b.scene.plots)==1
     @test a.scene.plots[1] isa Makie.Lines;@test b.scene.plots[1] isa Makie.Scatter
     @test Tuple.(a.scene.plots[1][1][])==[(1.,-2.),(2.,0.),(3.,3.)]
     @test Tuple.(b.scene.plots[1][1][])==[(1.,2.),(2.,0.),(3.,3.)]
     @test Makie.to_color(a.scene.plots[1].color[])==Makie.to_color(:red)
     @test Makie.to_color(b.scene.plots[1].color[])==Makie.to_color(:green)
     @test a in b.yaxislinks;@test b in a.yaxislinks
     @test isempty(a.xaxislinks);@test isempty(b.xaxislinks)
     """),
]
