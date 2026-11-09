"""Live integration check against an already running Bonsai API."""
import json,time,urllib.request

def request(messages,cache=True):
    payload={'model':'bonsai-27b','messages':messages,'max_tokens':16,'cache_prompt':cache}
    started=time.monotonic()
    req=urllib.request.Request('http://127.0.0.1:8092/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=240) as response:result=json.load(response)
    result['wall_seconds']=round(time.monotonic()-started,3)
    return result

def main():
    history=[{'role':'system','content':'Follow the request. Reply with only the requested word.'}, {'role':'user','content':'Return exactly the word amber.'}]
    first=request(history)
    history += [{'role':'assistant','content':first['choices'][0]['message']['content']}, {'role':'user','content':'Now return exactly the word violet.'}]
    cached=request(history)
    fresh=request(history,cache=False)
    assert cached['usage']['prompt_tokens_details']['cached_tokens']>0,cached
    assert fresh['usage']['prompt_tokens_details']['cached_tokens']==0,fresh
    assert cached['choices'][0]['message']==fresh['choices'][0]['message'],(cached,fresh)
    unrelated=request([{'role':'user','content':'Say exactly: hello.'}])
    assert unrelated['usage']['prompt_tokens_details']['cached_tokens']==0
    shortened=request([{'role':'system','content':'Follow the request. Reply with only the requested word.'}])
    assert shortened['usage']['prompt_tokens_details']['cached_tokens']==0
    for label,r in [('first',first),('cached continuation',cached),('fresh comparison',fresh),('unrelated reset',unrelated),('shortened reset',shortened)]:
        print(json.dumps({'case':label,'text':r['choices'][0]['message']['content'],'usage':r['usage'],'timings':r['timings'],'wall_seconds':r['wall_seconds']}),flush=True)
    print('PASS: cached continuation matches fresh output; unrelated and shortened histories reset safely',flush=True)
if __name__=='__main__':main()
