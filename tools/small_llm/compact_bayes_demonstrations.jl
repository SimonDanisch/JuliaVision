using JSON3

"""Public, unrelated schema examples. These are NOT outputs for study tasks."""
function compact_bayes_demonstrations()
    [
        (;request="Wet has prior P(wet=yes)=0.2. P(chime=yes|wet=no)=0.15 and P(chime=yes|wet=yes)=0.75. Observe chime=no. Request P(wet=yes|chime=no).",
            network=(;nodes=[(;name="wet",parents=String[],table=[0.2]),
                (;name="chime",parents=["wet"],table=[0.15,0.75])],query="wet",observed=Dict("chime"=>"no"))),
        (;request="Color states amber,blue,green have respective priors0.2,0.3,0.5. P(signal=yes|color) is respectively0.1,0.4,0.8. Observe signal=yes. Request P(color=blue|signal=yes).",
            network=(;nodes=[(;name="color",states=["amber","blue","green"],parents=String[],table=[[0.2,0.3,0.5]]),
                (;name="signal",parents=["color"],table=[0.1,0.4,0.8])],query="color",observed=Dict("signal"=>"yes")))
    ]
end

function compact_bayes_demonstration_messages(messages)
    length(messages)==2||throw(ArgumentError("Two original public messages required"))
    normalized=map(messages) do message
        if message isa Pair
            String(first(message))=>String(last(message))
        else
            length(message)==1||throw(ArgumentError("One role per message"))
            String(first(keys(message)))=>String(first(values(message)))
        end
    end
    first(normalized[1])=="system"&&first(normalized[2])=="user"||throw(ArgumentError("Original role order required"))
    user=last(normalized[2]);marker="\nProblem:\n"
    length(findall(marker,user))==1||throw(ArgumentError("Exactly one original problem boundary"))
    contract,problem=split(user,marker;limit=2)
    examples=join(("Example problem:\n"*e.request*"\nExample network:\n"*JSON3.write(e.network) for e in compact_bayes_demonstrations()),"\n\n")
    guide="\n\nPublic format examples for UNRELATED problems (not posterior answers):\n"*examples*
        "\n\nFor the actual problem use ITS OWN names, probabilities, edges and observations. Do not copy example values."
    [normalized[1],"user"=>contract*guide*marker*problem]
end
