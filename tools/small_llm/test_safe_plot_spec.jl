include("safe_plot_spec.jl")
using Test,JSON3
function test_safe_plot_spec()
    base=Dict("size"=>[720,480],"legend"=>true,
        "axes"=>[Dict("title"=>"Example","xlabel"=>"x","ylabel"=>"y","xscale"=>"linear","yscale"=>"linear")],
        "series"=>[Dict("kind"=>"line","axis"=>1,"x"=>"x","y"=>"a","label"=>"A","color"=>"blue")])
    parse(p)=SafePlotSpec.parse_spec(JSON3.write(p))
    @testset "Declarative plotting admits data references, never executable source" begin
        spec=parse(base);data=Dict("x"=>[1.,2.],"a"=>[2.,3.]);before=deepcopy(data)
        @test spec.size==(720,480)
        @test SafePlotSpec.validate_data(spec,data)===nothing
        @test data==before
        for (key,value) in (("size",[true,480]),("size",[10000,480]),("legend",1),("linkx",true),("callback","run(cmd)"))
            invalid=deepcopy(base);invalid[key]=value
            @test_throws ArgumentError parse(invalid)
        end
        for (key,value) in (("kind","eval"),("axis",2),("color","read(file)"),("x","sin(x)"),("label","bad\nlabel"),("callback","eval"))
            invalid=deepcopy(base);invalid["series"][1][key]=value
            @test_throws ArgumentError parse(invalid)
        end
        @test_throws DimensionMismatch SafePlotSpec.validate_data(spec,Dict("x"=>[1.],"a"=>[2.,3.]))
        @test_throws ArgumentError SafePlotSpec.validate_data(spec,Dict("x"=>[1.,2.],"a"=>[NaN,1.]))
        @test_throws ArgumentError SafePlotSpec.validate_data(spec,Dict("x"=>[1.,2.]))
        logarithmic=deepcopy(base);logarithmic["axes"][1]["yscale"]="log10"
        @test_throws ArgumentError SafePlotSpec.validate_data(parse(logarithmic),Dict("x"=>[1.,2.],"a"=>[0.,1.]))
        band=deepcopy(base);band["series"]=[Dict("kind"=>"band","axis"=>1,"x"=>"x","low"=>"lo","high"=>"hi","label"=>"range","color"=>"green")]
        @test SafePlotSpec.validate_data(parse(band),Dict("x"=>[1.,2.],"lo"=>[0.,1.],"hi"=>[1.,2.]))===nothing
        @test_throws ArgumentError SafePlotSpec.validate_data(parse(band),Dict("x"=>[1.,2.],"lo"=>[2.,1.],"hi"=>[1.,2.]))
        @test_throws ArgumentError SafePlotSpec.parse_spec(repeat("x",32769))
        @test_throws ArgumentError SafePlotSpec.parse_spec(repeat("[",17)*repeat("]",17))
        @test_throws ArgumentError SafePlotSpec.parse_spec("]")
        brackets=deepcopy(base);brackets["axes"][1]["title"]="{[quoted]}"
        @test parse(brackets).axes[1].title=="{[quoted]}"
        invalid=deepcopy(base);invalid["axes"][1]["legend_position"]="eval"
        @test_throws ArgumentError parse(invalid)
        linked=deepcopy(base);push!(linked["axes"],deepcopy(linked["axes"][1]));linked["linkx"]=true
        @test parse(linked).linkx
        linked["axes"][2]["xscale"]="log10"
        @test_throws ArgumentError parse(linked)
    end
    base
end
