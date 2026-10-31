function bayes_extraction_tasks()
    descriptions=[
        ("fault","yes","A manufactured component has fault=yes with probability0.01. The alarm is yes with probability0.9 if fault=yes and0.05 if fault=no. Observe alarm=yes. Find P(fault=yes | alarm=yes)."),
        ("fault","yes","A component has fault=yes probability0.3. Alarm=yes probability is0.8 given fault=yes and0.1 given fault=no. Observe alarm=no. Find P(fault=yes | alarm=no)."),
        ("fault","yes","Fault=yes has prior0.2. Sensors a and b are conditionally independent given fault. P(a=yes|fault=yes)=0.8; P(a=yes|fault=no)=0.1. P(b=yes|fault=yes)=0.7; P(b=yes|fault=no)=0.2. Observe a=yes,b=yes. Find P(fault=yes | observations)."),
        ("fault","yes","Fault=yes has prior0.4. Sensors a and b are conditionally independent given fault. P(a=yes|fault=yes)=0.9; P(a=yes|fault=no)=0.2. P(b=yes|fault=yes)=0.8; P(b=yes|fault=no)=0.1. Observe a=yes,b=no. Find P(fault=yes | observations)."),
        ("a","yes","Independent root switches a and b have P(a=yes)=0.3,P(b=yes)=0.2. Output o depends on both: P(o=yes|a=yes,b=yes)=0.99; P(o=yes|a=yes,b=no)=0.8; P(o=yes|a=no,b=yes)=0.7; P(o=yes|a=no,b=no)=0.05. Observe o=yes,b=no. Find P(a=yes | observations)."),
        ("cause","yes","A chain has cause -> middle -> result and no other edges. P(cause=yes)=0.25. P(middle=yes|cause=yes)=0.8; P(middle=yes|cause=no)=0.1. P(result=yes|middle=yes)=0.9; P(result=yes|middle=no)=0.2. Observe result=yes; middle is unobserved. Find P(cause=yes | result=yes)."),
        ("a","yes","Independent root switches a and b have P(a=yes)=0.1,P(b=yes)=0.2. Output o depends on both: P(o=yes|a=yes,b=yes)=0.99; P(o=yes|a=yes,b=no)=0.9; P(o=yes|a=no,b=yes)=0.8; P(o=yes|a=no,b=no)=0.01. Observe o=yes,b=yes. Find P(a=yes | observations)."),
        ("mode","faulty","Mode has states fast,normal,faulty with respective prior probabilities0.1,0.7,0.2. Flag is yes with probability0.05 given fast,0.1 given normal,0.9 given faulty. Observe flag=yes. Find P(mode=faulty | flag=yes).")]
    [(;id="bayes_extract_"*lpad(i,2,'0'),query=x[1],target=x[2],request=x[3]) for (i,x) in enumerate(descriptions)]
end

const BAYES_EXTRACTION_CONTRACT="""
All scenarios are fictional discrete probability models, not medical advice.
Use exactly the stated conditional independences/edges. Binary nodes have
states no,yes. Missing binary complements are1 minus the given probability.
Unobserved nodes must be marginalized, NOT assumed yes or no. No simulation.

The addon network schema is one JSON object with EXACT keys nodes,query,observed.
nodes is an array of {name,states,parents,rows}. states is an ordered array of
state-name strings, parents is an ordered array of node-name strings. Each row
is {given,p}: given lists parent STATE NAMES in parent order, p lists numeric
probabilities in the node's state order. Roots have one row with given:[];
conditionals must list every parent-state combination exactly once. query is
the queried NODE NAME. observed maps observed node names to STATE NAME strings.
Do not include the target state or final answer as extra network fields.
No unknown fields, code, expressions, external imports, or hidden evidence.
All numeric JSON probabilities are decimal numbers, not fractions/percent text.
"""

function bayes_extraction_messages(task,arm)
    arm in ("direct_probability","bayes_network")||throw(ArgumentError("Unknown arm"))
    instruction=arm=="direct_probability" ?
        "Calculate the requested posterior yourself. Return ONLY JSON {\"probability\":NUMBER}, a numeric probability0:1 with at least6 decimal places when noninteger. Do NOT return a network." :
        "Translate the complete problem into the addon Bayesian network schema. Return ONLY that JSON network, NOT a posterior calculation. The exact solver will calculate the requested state's probability."
    ["system"=>"Follow the requested output interface exactly. No fences, commentary or reasoning text.",
        "user"=>BAYES_EXTRACTION_CONTRACT*"\nRequested interface:\n"*instruction*"\n\nProblem:\n"*task.request]
end
