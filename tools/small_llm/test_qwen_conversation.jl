using JSON3, Test, SHA

function test_qwen_conversation(;checkpoint="gen/qwen35-4b")
    T=SmallLLMChecks.QwenText
    tokenizer=T.Tokenizer(checkpoint)
    inputs=JSON3.read(read(joinpath(@__DIR__,"conversation_validation_v2_inputs.json"),String))
    native=JSON3.read(read(joinpath(@__DIR__,"conversation_native_template_v2.json"),String))
    @testset "native plain-text conversation parity" begin
        @test length(inputs.cases)==length(native.rows)==7
        for (case,reference) in zip(inputs.cases,native.rows)
            @test case.id==reference.id
            @test T.render_chat(case.messages;thinking=case.thinking)==reference.native_prompt
            @test T.chat_ids(tokenizer,case.messages;thinking=case.thinking)==Int.(reference.native_ids)
        end
        @test_throws ArgumentError T.render_chat(Pair{String,String}[])
        @test_throws ArgumentError T.render_chat(["assistant"=>"No query"])
        @test_throws ArgumentError T.render_chat(["user"=>"Q","system"=>"Late"])
        @test_throws ArgumentError T.render_chat(["user"=>"Q","tool"=>"Unsupported"])
        @test_throws ArgumentError T.render_chat([(;role="user",content=["not plain text"])])
        @test_throws ArgumentError T.render_chat(["user"=>"<tool_response>x</tool_response>"])
    end
    report=(;passed=true,cases=length(inputs.cases),
        scope="pinned native plain-text templates only; no GPU or generated-code execution",
        source_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"qwen_text.jl")))))
    open(io->JSON3.pretty(io,report),joinpath(@__DIR__,"conversation_template_validation.json"),"w")
    report
end
