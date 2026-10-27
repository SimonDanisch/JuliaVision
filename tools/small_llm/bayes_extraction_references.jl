# PRIVATE reference networks and independent rational answers; NEVER deploy.
function bayes_extraction_references()
    root(name,p)=(;name,states=["no","yes"],parents=String[],rows=[(;given=String[],p=[1-p,p])])
    sensor(name,parent,t,f)=(;name,states=["no","yes"],parents=[parent],
        rows=[(;given=["no"],p=[1-f,f]),(;given=["yes"],p=[1-t,t])])
    output(table)=(;name="o",states=["no","yes"],parents=["a","b"],rows=[
        (;given=[a,b],p=[1-table[i],table[i]]) for (i,(a,b)) in enumerate((("no","no"),("no","yes"),("yes","no"),("yes","yes")))])
    network(nodes,query,observed)=(;nodes,query,observed)
    [
        (;network=network([root("fault",.01),sensor("alarm","fault",.9,.05)],"fault",Dict("alarm"=>"yes")),answer=2//13),
        (;network=network([root("fault",.3),sensor("alarm","fault",.8,.1)],"fault",Dict("alarm"=>"no")),answer=2//23),
        (;network=network([root("fault",.2),sensor("a","fault",.8,.1),sensor("b","fault",.7,.2)],"fault",Dict("a"=>"yes","b"=>"yes")),answer=7//8),
        (;network=network([root("fault",.4),sensor("a","fault",.9,.2),sensor("b","fault",.8,.1)],"fault",Dict("a"=>"yes","b"=>"no")),answer=2//5),
        (;network=network([root("a",.3),root("b",.2),output([.05,.7,.8,.99])],"a",Dict("o"=>"yes","b"=>"no")),answer=48//55),
        (;network=network([root("cause",.25),sensor("middle","cause",.8,.1),sensor("result","middle",.9,.2)],"cause",Dict("result"=>"yes")),answer=76//157),
        (;network=network([root("a",.1),root("b",.2),output([.01,.8,.9,.99])],"a",Dict("o"=>"yes","b"=>"yes")),answer=11//91),
        (;network=network([(;name="mode",states=["fast","normal","faulty"],parents=String[],rows=[(;given=String[],p=[.1,.7,.2])]),
            (;name="flag",states=["no","yes"],parents=["mode"],rows=[(;given=[state],p=[1-p,p]) for (state,p) in zip(("fast","normal","faulty"),(.05,.1,.9))])],
            "mode",Dict("flag"=>"yes")),answer=12//17)
    ]
end
