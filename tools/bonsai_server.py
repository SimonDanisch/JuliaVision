"""Local OpenAI-compatible text API for BonsaiRunner. No third-party Python packages.
Run: python3 tools/bonsai_server.py --project /path/to/VulkanDev
Greedy decoding only; tool calling and multimodal input are not supported.
"""
import argparse,json,queue,subprocess,threading,time,uuid
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument('--port',type=int,default=8092)
    args=parser.parse_args()
    worker=subprocess.Popen(['julia','--startup-file=no','--threads=2',f'--project={args.project}',str(Path(__file__).with_name('bonsai_worker.jl'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    events=queue.Queue();ready=threading.Event();lock=threading.Lock()
    def reader():
        for line in worker.stdout:
            if line.startswith('BONSAI_EVENT:'):
                event=json.loads(line[len('BONSAI_EVENT:'):])
                if event['type']=='ready':ready.set();print('Bonsai API ready',flush=True)
                elif ready.is_set():
                    if event['type']=='done':
                        print('Bonsai timing: '+json.dumps({k:event.get(k) for k in ('cached_tokens','completion_tokens','prefill_seconds','decode_seconds')}),flush=True)
                    events.put(event)
            else:print(line,end='',flush=True)
        ready.clear();events.put({'type':'error','message':'Inference worker exited'})
    threading.Thread(target=reader,daemon=True).start()
    class Handler(BaseHTTPRequestHandler):
        def sendjson(self,status,obj):
            data=json.dumps(obj).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path=='/health':return self.sendjson(200 if ready.is_set() else 503,{'status':'ok' if ready.is_set() else 'loading'})
            if self.path=='/v1/models':return self.sendjson(200,{'object':'list','data':[{'id':'bonsai-27b','object':'model','owned_by':'local'}]})
            self.sendjson(404,{'error':{'message':'Not found'}})
        def do_POST(self):
            if self.path!='/v1/chat/completions':return self.sendjson(404,{'error':{'message':'Not found'}})
            if not ready.is_set():return self.sendjson(503,{'error':{'message':'Bonsai is loading'}})
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=2_000_000:raise ValueError('Invalid request size')
                req=json.loads(self.rfile.read(size));messages=req['messages']
                for m in messages:
                    if isinstance(m.get('content'),list):
                        if any(part.get('type')!='text' for part in m['content']):raise ValueError('Only text input is supported')
                        m['content']='\n'.join(part['text'] for part in m['content'])
                if req.get('model','bonsai-27b')!='bonsai-27b':raise ValueError('Unknown model')
                if not messages or any(m.get('role') not in ('system','user','assistant') or not isinstance(m.get('content'),str) for m in messages):raise ValueError('Text-only system/user/assistant messages required')
                if req.get('tool_choice') not in (None,'auto','none'):raise ValueError('Tool calling is not supported')
                maximum=int(req.get('max_completion_tokens') or req.get('max_tokens') or 2048)
                if not 1<=maximum<=4096:raise ValueError('max_tokens must be between 1 and 4096')
            except (ValueError,KeyError,TypeError) as e:return self.sendjson(400,{'error':{'message':str(e)}})
            with lock:
                worker.stdin.write(json.dumps({'messages':messages,'max_tokens':maximum,'cache_prompt':bool(req.get('cache_prompt',True))})+'\n');worker.stdin.flush()
                first=events.get()
                if first['type']=='error':return self.sendjson(400,{'error':{'message':first['message']}})
                ident='chatcmpl-'+uuid.uuid4().hex;created=int(time.time());stream=bool(req.get('stream'));text='';disconnected=False
                if stream:
                    self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Cache-Control','no-cache');self.end_headers()
                def chunk(delta,reason=None,usage=None):
                    nonlocal disconnected
                    if disconnected:return
                    obj={'id':ident,'object':'chat.completion.chunk','created':created,'model':'bonsai-27b','choices':[{'index':0,'delta':delta,'finish_reason':reason}]}
                    if usage is not None:obj['usage']=usage
                    try:self.wfile.write(('data: '+json.dumps(obj)+'\n\n').encode());self.wfile.flush()
                    except (BrokenPipeError,ConnectionResetError):disconnected=True
                if stream:chunk({'role':'assistant','content':''})
                while True:
                    e=events.get()
                    if e['type']=='token':
                        text+=e['text']
                        if stream:chunk({'content':e['text']})
                    elif e['type'] in ('done','error'):break
                if e['type']=='error':
                    if stream:
                        if not disconnected:self.wfile.write(('data: '+json.dumps({'error':{'message':e['message']}})+'\n\n').encode())
                    else:self.sendjson(500,{'error':{'message':e['message']}})
                    return
                usage={'prompt_tokens':first['prompt_tokens'],'completion_tokens':e['completion_tokens'],'total_tokens':first['prompt_tokens']+e['completion_tokens'],'prompt_tokens_details':{'cached_tokens':first.get('cached_tokens',0)}}
                timings={k:e.get(k) for k in ('prefill_seconds','decode_seconds')}
                if stream:
                    chunk({},e['finish_reason'],usage)
                    if not disconnected:
                        try:self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
                        except (BrokenPipeError,ConnectionResetError):pass
                else:self.sendjson(200,{'id':ident,'object':'chat.completion','created':created,'model':'bonsai-27b','choices':[{'index':0,'message':{'role':'assistant','content':text},'finish_reason':e['finish_reason']}],'usage':usage,'timings':timings})
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Listening on http://127.0.0.1:{args.port}; waiting for Bonsai warmup',flush=True)
    try:server.serve_forever()
    finally:server.server_close();worker.terminate();worker.wait()
if __name__=='__main__':main()
