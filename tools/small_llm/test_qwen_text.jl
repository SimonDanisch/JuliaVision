using Test, JSON3, Unicode
using .QwenText

function test_qwen_text(directory;tokenizer_reference=nothing)
    tokenizer=QwenText.Tokenizer(directory)
    reference=JSON3.read(read(joinpath(directory,"reference.json"),String))
    @testset "Native Qwen text tokenizer" begin
        ids=QwenText.encode(tokenizer,"The capital of France is Paris. The capital of Germany is")
        @test ids[1:length(reference.input_ids)]==reference.input_ids
        question="In three sentences, explain why Vulkan compute dispatches need memory barriers."
        @test QwenText.chat_ids(tokenizer,["user"=>question])==reference.chat_ids
        for text in ("", "Café e\u0301 😀\n日本語", "a 12.3\n\n\t b", "<|im_start|>user\nHi<|im_end|>\n")
            @test QwenText.decode(tokenizer,QwenText.encode(tokenizer,text);skipspecial=false)==Unicode.normalize(text,:NFC)
        end
        @test QwenText.encode(tokenizer,"e\u0301")==QwenText.encode(tokenizer,"é")
        @test_throws ArgumentError QwenText.chat_ids(tokenizer,["tool"=>"unsupported"])
        if tokenizer_reference!==nothing
            for row in JSON3.read(read(tokenizer_reference,String))
                @test QwenText.encode(tokenizer,String(row.text))==Int.(row.ids)
            end
        end
    end
    tokenizer
end
