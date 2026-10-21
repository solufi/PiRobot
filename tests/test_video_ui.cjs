const fs=require('fs'),vm=require('vm'),assert=require('assert');
const path=require('path'),{execFileSync}=require('child_process');
const root=path.resolve(__dirname,'..');
const html=execFileSync('python3',['-c',`import ast,sys
from pathlib import Path
m=ast.parse(Path(sys.argv[1]).read_text())
h=next(ast.literal_eval(n.value) for n in m.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='HTML' for t in n.targets))
print(h)
`,path.join(root,'server/picar_server.py')],{encoding:'utf8'});
assert.equal((html.match(/id="updateBtn"/g)||[]).length,1);
assert(html.indexOf('id="settingsPanel"')<html.indexOf('id="updateBtn"'));
assert(html.indexOf('id="updateBtn"')<html.indexOf('id="caliPanel"'));
const source=html.split('<script>')[1].split('</script>')[0];
function setup(mode='success'){
 const events={},elements={},timers=new Map(),sent=[],requests=[];let serial=0,getCount=0,reloaded=false;
 function element(id){return {value:'0',style:{},classList:{toggle(){},add(){},remove(){}},addEventListener(k,fn){events[id+':'+k]=fn;},insertAdjacentHTML(){}};}
 const context={console,AbortController,Date,URL,URLSearchParams,navigator:{},FormData:class{},location:{protocol:'http:',host:'preview',reload(){reloaded=true;}},
 document:{getElementById(id){return elements[id]??=element(id);},querySelectorAll(){return [];},addEventListener(k,fn){(events["document:"+k]??=[]).push(fn);}},window:{addEventListener(){}},WebSocket:class{send(data){sent.push(JSON.parse(data));}},setInterval(){},
 setTimeout(fn,delay){const id=++serial;timers.set(id,{fn,delay});return id;},clearTimeout(id){timers.delete(id);},
 fetch:async(url,opts)=>{
   requests.push({url,opts});
   if(opts.method==='POST')return {ok:mode!=='reject',json:async()=>mode==='reject'?{error:'résolution invalide'}:{ok:true,restarting:true}};
   getCount++;
   if(getCount===1)throw Error('service restarting');
   return {ok:true,json:async()=>mode==='fallback'?{resolution:'640x480',error:'Résolution indisponible'}:{resolution:'1280x720'}};
 }};
 vm.createContext(context);vm.runInContext(source,context);
 return {events,elements,context,sent,requests,get timerCount(){return timers.size;},get reloaded(){return reloaded;},async next(){const [id,t]=[...timers.entries()].sort((a,b)=>a[1].delay-b[1].delay)[0]||[];assert(t,'Expected scheduled retry');timers.delete(id);await t.fn();}};
}
(async()=>{
 let t=setup();t.context.document.getElementById('videoResolution').value='1280x720';
 await t.events['videoApply:click']();assert(t.elements.videoApply.disabled);await t.next();assert(!t.reloaded);await t.next();await t.next();assert(t.reloaded);
 t=setup('fallback');t.context.document.getElementById('videoResolution').value='1280x720';await t.events['videoApply:click']();await t.next();await t.next();assert(!t.reloaded);assert(!t.elements.videoApply.disabled);assert.equal(t.elements.videoResolution.value,'640x480');
 t=setup('reject');t.context.document.getElementById('videoResolution').value='8k';await t.events['videoApply:click']();assert(!t.elements.videoApply.disabled);assert.equal(t.elements.videoStatus.innerText,'résolution invalide');
 t=setup();t.elements.videoStream.src='stale';t.events['videoStream:error']();await t.next();assert(t.elements.videoStream.src.startsWith('/video_frame?t='));
 t=setup();t.elements.videoStream.src='stale';vm.runInContext('ws.onopen()',t.context);assert(t.elements.videoStream.src.startsWith('/video_frame?t='));
 t=setup();vm.runInContext('wsReady=true',t.context);t.elements.pan.value='12';t.events['pan:input']();await t.next();assert.deepEqual(t.sent[0],{type:'camera',pan:12});
 t.elements.tilt.value='-12';t.events['tilt:input']();await t.next();assert.deepEqual(t.sent[1],{type:'camera',tilt:-12});
 t=setup();t.elements.pan.value='7';t.events['pan:input']();await t.next();assert.equal(t.requests[0].url,'/camera_settings');assert.equal(t.requests[0].opts.body,'pan=7');
 t=setup();vm.runInContext('wsReady=true',t.context);for(let i=1;i<=30;i++){t.elements.pan.value=String(i);t.events['pan:input']();}await t.next();assert.equal(t.sent.length,1);assert.equal(t.sent[0].pan,30);
 t=setup();let previous='';for(let i=0;i<50;i++){t.events['videoStream:load']();await t.next();assert.notEqual(t.elements.videoStream.src,previous);previous=t.elements.videoStream.src;assert.equal(t.timerCount,1);}
 t.context.document.hidden=true;for(const fn of t.events['document:visibilitychange'])fn();assert.equal(t.timerCount,0);
 console.log('Interface vidéo: changement, coupure réseau, reconnexion, repli et erreur vérifiés.');
})().catch(e=>{console.error(e);process.exit(1);});
