include("rpn_example_semantics.jl")
using Test
function test_rpn_example_semantics()
    R=RPNExampleSemantics;D=Int32(8)
    @testset "Public-example numeric prefix semantics, not hidden specification" begin
        x=Float32[2,-3,5];f=R.features(x)
        @test f==Float32[2 -3 5;5 2 -3;-3 5 2;2 2 -3;-3 5 5]
        @test R.features(Float32[7])==fill(7f0,5,1)
        @test_throws ArgumentError R.features(Float32[])
        @test_throws ArgumentError R.features(Float32[NaN])
        for action in Int32(1):Int32(9),example in Int32(1):Int32(3)
            s=fill(-999f0,24)
            @test R.step!(s,f,example,action,Int32(0),D)==0
            expected=action<=5 ? f[action,example] : action==6 ? 0f0 : action==7 ? 1f0 : action==8 ? 2f0 : 0.5f0
            @test s[(example-1)*D+1]==expected
            @test count(!=(-999f0),s)==1
        end
        for (action,expected) in ((10,-2f0),(11,4f0),(12,7f0),(13,-3f0),(14,10f0),(15,2f0),(16,5f0))
            s=zeros(Float32,24);s[1]=2;s[2]=5
            depth=Int32(action<=11 ? 1 : 2)
            @test R.step!(s,f,Int32(1),Int32(action),depth,D)==0
            @test s[1]==expected
        end
        s=zeros(Float32,24);s[1]=1f0
        @test R.matches(s,Float32[1,2,3],Int32(1),Int32(1),D,0f0,0f0)
        @test !R.matches(s,Float32[2,2,3],Int32(1),Int32(1),D,0f0,0f0)
        @test R.matches(s,Float32[1.001,2,3],Int32(1),Int32(1),D,0.002f0,0f0)
        for (e,a,d) in ((0,1,0),(typemax(Int32),1,0),(1,0,0),(1,18,1),(1,10,0),(1,12,1),(1,1,8),(1,17,0))
            before=copy(s)
            @test R.step!(s,f,Int32(e),Int32(a),Int32(d),D)==1
            @test isequal(s,before)
        end
        s[1]=floatmax(Float32);before=copy(s)
        @test R.step!(s,f,Int32(1),Int32(11),Int32(1),D)==1
        @test s==before
        for (actual,target,atol,rtol,want) in ((Inf32,1f0,0f0,0f0,false),(NaN32,1f0,0f0,0f0,false),
                (1f0,Inf32,0f0,0f0,false),(1f0,1f0,-1f0,0f0,false),
                (1f0,1f0,0f0,NaN32,false),(floatmax(Float32),-floatmax(Float32),0f0,1f0,false))
            s[1]=actual
            @test R.matches(s,Float32[target],Int32(1),Int32(1),D,atol,rtol)==want
        end
        @test !R.matches(Float32[],Float32[],Int32(1),Int32(1),D,0f0,0f0)
        @test !R.matches(s,Float32[1],typemax(Int32),Int32(1),D,0f0,0f0)
        bank=R.Bank([(;input=Float32[2,3],output=Float32[4,6]),(;input=Float32[9],output=Float32[18])])
        @test bank.expected==Float32[4,6,18]
        @test bank.case_offsets==Int32[1,3,4]
        @test reshape(bank.features,5,3)[:,3]==fill(9f0,5)
        @test reshape(bank.features,5,3)[:,2]==Float32[3,2,2,2,3]
        @test bank.max_depth==8 && bank.atol==bank.rtol==0f0
        bank.expected[1]=0
        @test bank.features[1]==2f0
        @test_throws ArgumentError R.Bank(Any[])
        @test_throws ArgumentError R.Bank([(;input=Float32[],output=Float32[])])
        @test_throws ArgumentError R.Bank([(;input=Float32[1],output=Float32[1,2])])
        @test_throws ArgumentError R.Bank([(;input=Float32[1],output=Float32[NaN])])
        @test_throws ArgumentError R.Bank([(;input=[1.0],output=[1.0])])
        for kwargs in ((;max_depth=true),(;max_depth=65),(;max_points=0),(;max_points=true),(;atol=NaN),(;rtol=-1))
            @test_throws ArgumentError R.Bank([(;input=Float32[1],output=Float32[1])];kwargs...)
        end
        @test_throws ArgumentError R.Bank([(;input=Float32[1,2],output=Float32[1,2])];max_points=1)
    end
end
