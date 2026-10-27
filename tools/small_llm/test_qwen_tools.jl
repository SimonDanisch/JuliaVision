# QwenTools.render against the checkpoint's Jinja template, and the call parser
# against the format the template asks for. The expected renders come from
# render.py (jinja2 configured as transformers does) over `tool_cases()` and are
# stored with the template's hash in qwen_tools_template_fixture.json.
isdefined(@__MODULE__,:QwenTools) || include("qwen_tools.jl")
isdefined(@__MODULE__,:SmallLLMChecks) || include("load_attention_checks.jl")
using Test,JSON3,SHA

const TOOL_FIXTURE=joinpath(@__DIR__,"qwen_tools_template_fixture.json")

function tool_cases()
    T=QwenTools
    lookup=T.Tool("lookup","Documentation of an installed Julia name, e.g. `Atomix.@atomic` or `KernelInterface`.",["name"=>"A module, function, macro or type, written as in code."])
    check=T.Tool("check_syntax","Parse Julia code and report the first syntax error.",["code"=>"Complete Julia source.","note"=>"Optional remark with \"quotes\", a\ttab and ünïcode."])
    call=T.Call("lookup",["name"=>"Atomix.@atomic"])
    two=T.Call("check_syntax",["code"=>"function f()\n    1\nend","note"=>"x"])
    m(r,c)=T.Message(r,c)
    [
     (T.Tool[],[m("system","Be brief."),m("user","Hi")],true),
     (T.Tool[],[m("system","Be brief."),m("user","Hi")],false),
     ([lookup,check],[m("system","You are a Julia programmer."),m("user","Write a kernel.")],true),
     ([lookup],[m("user","Write a kernel.")],true),
     ([lookup,check],[m("system","You are a Julia programmer."),m("user","Write a kernel."),
        T.Message("assistant","",  "I should look it up.",[call]),m("tool","@atomic docs here")],true),
     ([lookup,check],[m("user","Write a kernel."),T.Message("assistant","Let me check.","Two checks.",[call,two]),
        m("tool","first result"),m("tool","second result")],true),
     ([lookup],[m("user","First question"),m("assistant","<think>\nold reasoning\n</think>\n\nOld answer."),
        m("user","Second question")],true),
     ([lookup],[m("user","Q"),T.Message("assistant","",nothing,[call]),m("tool","r1"),
        T.Message("assistant","",nothing,[two]),m("tool","r2")],false),
    ]
end

function export_tool_cases(path)
    rows=map(tool_cases()) do (tools,messages,thinking)
        (;tools=[(;name=t.name,description=t.description,parameters=[[k,d] for (k,d) in t.parameters]) for t in tools],
         messages=[(;role=x.role,content=x.content,reasoning=x.reasoning,
            calls=[(;name=c.name,arguments=[[k,v] for (k,v) in c.arguments]) for c in x.calls]) for x in messages],thinking)
    end
    open(io->JSON3.write(io,rows),path,"w")
end

function test_qwen_tools(template_path)
    T=QwenTools
    fixture=JSON3.read(read(TOOL_FIXTURE,String))
    @testset "tool template subset equals the checkpoint template" begin
        @test bytes2hex(sha256(read(template_path,String)))==fixture.template_sha256
        cases=tool_cases()
        @test length(cases)==length(fixture.renders)
        for ((tools,messages,thinking),want) in zip(cases,fixture.renders)
            @test T.render(messages;tools,thinking)==String(want)
        end
        # Without tools it is the plain renderer the frozen studies used.
        for (tools,messages,thinking) in cases[1:2]
            @test T.render(messages;tools,thinking)==SmallLLMChecks.QwenText.render_chat([x.role=>x.content for x in messages];thinking)
        end
        # The in-session suffix is what render writes after a call.
        tools,messages,_=cases[5]
        prefix=T.render(messages[1:3];tools)
        full=T.render(messages;tools)
        turn=prefix[1:findlast("<|im_start|>assistant\n<think>\n",prefix).start-1]
        @test startswith(full,turn)
        @test endswith(full,T.responsesuffix(messages[4].content))
    end
    @testset "tool calls parse in the template's format" begin
        c=T.parsecall("thinking...</think>\n\nI'll look.\n\n<tool_call>\n<function=lookup>\n<parameter=name>\nAtomix.@atomic\n</parameter>\n</function>\n</tool_call>")
        @test c isa T.Call && c.name=="lookup" && c.arguments==["name"=>"Atomix.@atomic"]
        c=T.parsecall("<tool_call>\n<function=check_syntax>\n<parameter=code>\nfunction f()\n    1\nend\n</parameter>\n<parameter=note>\nx\n</parameter>\n</function>\n</tool_call>")
        @test c isa T.Call && c.arguments==["code"=>"function f()\n    1\nend","note"=>"x"]
        @test T.parsecall("no call here</tool_call>") isa String
        @test T.parsecall("<tool_call>\n{\"name\": \"lookup\"}\n</tool_call>") isa String
        @test T.parsecall("<tool_call>\n<function=lookup>\nstray\n<parameter=name>\nx\n</parameter>\n</function>\n</tool_call>") isa String
    end
end
