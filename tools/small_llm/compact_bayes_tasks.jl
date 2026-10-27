isdefined(@__MODULE__,:bayes_extraction_tasks)||include("bayes_extraction_tasks.jl")

const COMPACT_BAYES_CONTRACT="""
All scenarios are fictional probability models, not medical advice.
Translate exactly the stated nodes and conditional-independence edges.
Unobserved variables are marginalized by the exact solver, NOT assumed yes/no.
Return one JSON object with EXACT keys nodes,query,observed. No code or prose.

nodes is an array of objects with required name,parents,table and optional states.
name is the node-name string; parents is the ordered list of parent-name strings.
Binary states default to ["no","yes"]; do NOT add states for binary nodes.
For other nodes specify the ordered state-name list explicitly in states.
table contains one entry per parent-state combination. Enumerate combinations
in parents' order using their state order, RIGHTMOST parent changing fastest.
For two binary parents [a,b], rows are (no,no),(no,yes),(yes,no),(yes,yes).
A root has parents:[] and table with exactly ONE entry.
For binary no,yes nodes each table entry is a NUMBER P(node=yes | parents),
NOT a pair. The packer computes P(no)=1-P(yes), so do not calculate complements.
For a non-binary node each entry is an array of probabilities in its state order.
No omitted rows, expressions, renormalization, inferred edges or extra fields.
query is the queried NODE NAME, not a state. observed maps observed node names
to STATE-NAME strings. Do not put the target state or posterior in extra fields.
The exact shared solver calculates the requested posterior from your network.
"""

function compact_bayes_messages(task)
    ["system"=>"Follow the requested output interface exactly. No fences, commentary or reasoning text.",
        "user"=>COMPACT_BAYES_CONTRACT*"\nProblem:\n"*task.request]
end
