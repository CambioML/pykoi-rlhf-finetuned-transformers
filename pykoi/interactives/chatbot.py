import json

from random import randint

from typing import Dict, Any


class Chatbot:
    def __init__(self) -> None:
        """
        Initialize the component.
        """
        self.name = "Chatbot"
        self.iife_script = """<script>var Chatbot=function(){"use strict";var Q=document.createElement("style");Q.textContent=`:root{--green:#00ebc7;--red:#FF5470;--yellow:#fde24f;--black:#1b2d45;--darkBlue:#00214d;--darkGrey:#222;--grey:#bfbfbf;--lightGrey:#f2f4f6;--white:white;--primary:var(--yellow);--danger:var(--red);--background:var(--white);--textColor:var(--black);--lineColor:var(--grey);--cardBg:var(--white);--headerBackground:var(--white);--footerBackground:var(--green);--footerTextColor:var(--black);--headerTextColor:var(--black);--buttonColor:var(--primary);--buttonTextColor:var(--textColor);--borderBottom:solid 2px var(--primary);--line:solid 1px var(--lineColor);--headingFont:"Lato", monospace;--bodyFont:"Work Sans", sans-serif;--baseFontSize:100%;--h1:3.052em;--h2:2.441em;--h3:1.953em;--h4:1.563em;--h5:1.25em;--smallText:.8em;--shadow-s:0 1px 3px 0 rgb(0 0 0 / .1), 0 1px 2px -1px rgb(0 0 0 / .1);--shadow-md:0 4px 6px -1px rgb(0 0 0 / .1), 0 2px 4px -2px rgb(0 0 0 / .1);--shadow-lg:0 10px 15px -3px rgb(0 0 0 / .1), 0 4px 6px -4px rgb(0 0 0 / .1);--shadow-xl:0 20px 25px -5px rgb(0 0 0 / .1), 0 8px 10px -6px rgb(0 0 0 / .1);--containerPadding:2.5%;--headerHeight:3rem;--borderRadius:0px;--height:height: calc(100vh - var(--headerHeight))}.ranked-chat.svelte-f76u98.svelte-f76u98{height:100vh;display:grid;grid-template-columns:100%;grid-template-rows:80% 20%}.message.svelte-f76u98.svelte-f76u98{font-size:var(--smallText);padding-left:40px;padding-right:40px;margin:0 auto}.chat-input-holder.svelte-f76u98.svelte-f76u98{display:flex;flex-direction:column;align-items:center;padding:24px;width:100%;max-width:640px;margin:auto}.chat-input-textarea.svelte-f76u98.svelte-f76u98{background-color:var(--lightgrey);cursor:pointer;width:100%;border:var(--line);border-color:none;margin:12px;outline:none;padding:12px;color:var(--black);font-size:var(--baseFontSize);box-shadow:var(--shadow-md);flex:3;border-radius:0;border-right:0px}.chat-input-form.svelte-f76u98.svelte-f76u98{display:flex;width:100%}.btnyousend.svelte-f76u98.svelte-f76u98{border-radius:0;margin-top:12px;margin-bottom:12px;margin-left:-15px;background:var(--primary);color:var(--black);opacity:.5;transition:all .3s}.active.svelte-f76u98.svelte-f76u98{opacity:1}.bold.svelte-f76u98.svelte-f76u98{font-weight:700;font-size:var(--smallText);margin:0;padding:0}.chatbox.svelte-f76u98.svelte-f76u98{display:flex;flex-direction:column;justify-content:space-between;height:calc(100vh - var(--headerHeight));background-color:var(--white);box-sizing:border-box;width:95%;margin:auto;height:100%}.chat-log.svelte-f76u98.svelte-f76u98{flex:1;overflow-y:auto;padding:0 10px;box-sizing:border-box}.chat-message.svelte-f76u98.svelte-f76u98{background-color:var(--white);border-bottom:var(--line);box-sizing:border-box}.chat-message-center.svelte-f76u98.svelte-f76u98{display:flex;flex-direction:column;margin-left:auto;margin-right:auto;padding:12px;box-sizing:border-box}.message-content.svelte-f76u98.svelte-f76u98{display:flex;flex-direction:column;box-sizing:border-box}.message-content.svelte-f76u98 .question.svelte-f76u98{text-align:left;border:1px solid var(--grey);padding:5px;background-color:var(--lightGrey)}.message-content.svelte-f76u98 .answer.svelte-f76u98{display:inline-block;text-align:left;padding:10px;border:1px solid var(--black)}.message-content.svelte-f76u98 .answers.svelte-f76u98{display:grid;grid-template-columns:100%;gap:0%;width:100%;margin:auto}
`,document.head.appendChild(Q);function z(){}function J(e){return e()}function W(){return Object.create(null)}function F(e){e.forEach(J)}function A(e){return typeof e=="function"}function K(e,t){return e!=e?t==t:e!==t||e&&typeof e=="object"||typeof e=="function"}function le(e){return Object.keys(e).length===0}function ae(e,...t){if(e==null)return z;const n=e.subscribe(...t);return n.unsubscribe?()=>n.unsubscribe():n}function ie(e,t,n){e.$$.on_destroy.push(ae(t,n))}function U(e,t,n){return e.set(n),t}function ue(e){return e&&A(e.destroy)?e.destroy:z}function i(e,t){e.appendChild(t)}function X(e,t,n){e.insertBefore(t,n||null)}function D(e){e.parentNode&&e.parentNode.removeChild(e)}function x(e){return document.createElement(e)}function H(e){return document.createTextNode(e)}function q(){return H(" ")}function Y(e,t,n,o){return e.addEventListener(t,n,o),()=>e.removeEventListener(t,n,o)}function m(e,t,n){n==null?e.removeAttribute(t):e.getAttribute(t)!==n&&e.setAttribute(t,n)}function ce(e){return Array.from(e.childNodes)}function G(e,t){t=""+t,e.data!==t&&(e.data=t)}function Z(e,t){e.value=t==null?"":t}let O;function R(e){O=e}function fe(){if(!O)throw new Error("Function called outside component initialization");return O}function de(e){fe().$$.on_mount.push(e)}const B=[],ee=[];let L=[];const te=[],he=Promise.resolve();let P=!1;function pe(){P||(P=!0,he.then(ne))}function V(e){L.push(e)}const I=new Set;let M=0;function ne(){if(M!==0)return;const e=O;do{try{for(;M<B.length;){const t=B[M];M++,R(t),ge(t.$$)}}catch(t){throw B.length=0,M=0,t}for(R(null),B.length=0,M=0;ee.length;)ee.pop()();for(let t=0;t<L.length;t+=1){const n=L[t];I.has(n)||(I.add(n),n())}L.length=0}while(B.length);for(;te.length;)te.pop()();P=!1,I.clear(),R(e)}function ge(e){if(e.fragment!==null){e.update(),F(e.before_update);const t=e.dirty;e.dirty=[-1],e.fragment&&e.fragment.p(e.ctx,t),e.after_update.forEach(V)}}function be(e){const t=[],n=[];L.forEach(o=>e.indexOf(o)===-1?t.push(o):n.push(o)),n.forEach(o=>o()),L=t}const ve=new Set;function se(e,t){e&&e.i&&(ve.delete(e),e.i(t))}function me(e,t){e.d(1),t.delete(e.key)}function _e(e,t,n,o,l,d,u,f,c,s,g,y){let h=e.length,b=d.length,v=h;const p={};for(;v--;)p[e[v].key]=v;const k=[],E=new Map,C=new Map,w=[];for(v=b;v--;){const r=y(l,d,v),_=n(r);let S=u.get(_);S?o&&w.push(()=>S.p(r,t)):(S=s(_,r),S.c()),E.set(_,k[v]=S),_ in p&&C.set(_,Math.abs(v-p[_]))}const $=new Set,j=new Set;function a(r){se(r,1),r.m(f,g),u.set(r.key,r),g=r.first,b--}for(;h&&b;){const r=k[b-1],_=e[h-1],S=r.key,T=_.key;r===_?(g=r.first,h--,b--):E.has(T)?!u.has(S)||$.has(S)?a(r):j.has(T)?h--:C.get(S)>C.get(T)?(j.add(S),a(r)):($.add(T),h--):(c(_,u),h--)}for(;h--;){const r=e[h];E.has(r.key)||c(r,u)}for(;b;)a(k[b-1]);return F(w),k}function xe(e,t,n,o){const{fragment:l,after_update:d}=e.$$;l&&l.m(t,n),o||V(()=>{const u=e.$$.on_mount.map(J).filter(A);e.$$.on_destroy?e.$$.on_destroy.push(...u):F(u),e.$$.on_mount=[]}),d.forEach(V)}function ye(e,t){const n=e.$$;n.fragment!==null&&(be(n.after_update),F(n.on_destroy),n.fragment&&n.fragment.d(t),n.on_destroy=n.fragment=null,n.ctx=[])}function we(e,t){e.$$.dirty[0]===-1&&(B.push(e),pe(),e.$$.dirty.fill(0)),e.$$.dirty[t/31|0]|=1<<t%31}function ke(e,t,n,o,l,d,u,f=[-1]){const c=O;R(e);const s=e.$$={fragment:null,ctx:[],props:d,update:z,not_equal:l,bound:W(),on_mount:[],on_destroy:[],on_disconnect:[],before_update:[],after_update:[],context:new Map(t.context||(c?c.$$.context:[])),callbacks:W(),dirty:f,skip_bound:!1,root:t.target||c.$$.root};u&&u(s.root);let g=!1;if(s.ctx=n?n(e,t.props||{},(y,h,...b)=>{const v=b.length?b[0]:h;return s.ctx&&l(s.ctx[y],s.ctx[y]=v)&&(!s.skip_bound&&s.bound[y]&&s.bound[y](v),g&&we(e,y)),h}):[],s.update(),g=!0,F(s.before_update),s.fragment=o?o(s.ctx):!1,t.target){if(t.hydrate){const y=ce(t.target);s.fragment&&s.fragment.l(y),y.forEach(D)}else s.fragment&&s.fragment.c();t.intro&&se(e.$$.fragment),xe(e,t.target,t.anchor,t.customElement),ne()}R(c)}class Ce{$destroy(){ye(this,1),this.$destroy=z}$on(t,n){if(!A(n))return z;const o=this.$$.callbacks[t]||(this.$$.callbacks[t]=[]);return o.push(n),()=>{const l=o.indexOf(n);l!==-1&&o.splice(l,1)}}$set(t){this.$$set&&!le(t)&&(this.$$.skip_bound=!0,this.$$set(t),this.$$.skip_bound=!1)}}const N=[];function Ee(e,t=z){let n;const o=new Set;function l(f){if(K(e,f)&&(e=f,n)){const c=!N.length;for(const s of o)s[1](),N.push(s,e);if(c){for(let s=0;s<N.length;s+=2)N[s][0](N[s+1]);N.length=0}}}function d(f){l(f(e))}function u(f,c=z){const s=[f,c];return o.add(s),o.size===1&&(n=t(l)||z),f(e),()=>{o.delete(s),o.size===0&&n&&(n(),n=null)}}return{set:l,update:d,subscribe:u}}const qe="";function oe(e,t,n){const o=e.slice();return o[13]=t[n],o[15]=n,o}function re(e,t){let n,o,l,d,u,f,c,s,g,y=t[13].question+"",h,b,v,p,k,E,C,w=t[13].answer+"",$,j,a,r,_;return{key:e,first:null,c(){n=x("div"),o=x("div"),l=x("div"),d=q(),u=x("div"),f=x("div"),c=x("h5"),c.textContent="Question:",s=q(),g=x("p"),h=H(y),b=q(),v=x("div"),p=x("div"),k=x("h5"),k.textContent="Response:",E=q(),C=x("p"),$=H(w),j=q(),m(l,"class","avatar"),m(c,"class","bold svelte-f76u98"),m(f,"class","question svelte-f76u98"),m(k,"class","bold svelte-f76u98"),m(p,"class","answer svelte-f76u98"),m(v,"class","answers svelte-f76u98"),m(u,"class","message-content svelte-f76u98"),m(o,"class","chat-message-center svelte-f76u98"),m(n,"class","chat-message svelte-f76u98"),this.first=n},m(S,T){X(S,n,T),i(n,o),i(o,l),i(o,d),i(o,u),i(u,f),i(f,c),i(f,s),i(f,g),i(g,h),i(u,b),i(u,v),i(v,p),i(p,k),i(p,E),i(p,C),i(C,$),i(n,j),r||(_=ue(a=$e.call(null,n,t[15]===t[3].length-1)),r=!0)},p(S,T){t=S,T&8&&y!==(y=t[13].question+"")&&G(h,y),T&8&&w!==(w=t[13].answer+"")&&G($,w),a&&A(a.update)&&T&8&&a.update.call(null,t[15]===t[3].length-1)},d(S){S&&D(n),r=!1,_()}}}function Se(e){let t,n,o,l,d=[],u=new Map,f,c,s,g,y,h,b=(e[1]?e[2]:"Send")+"",v,p,k,E,C,w,$=e[3];const j=a=>a[15];for(let a=0;a<$.length;a+=1){let r=oe(e,$,a),_=j(r);u.set(_,d[a]=re(_,r))}return{c(){t=x("div"),n=x("div"),o=x("section"),l=x("div");for(let a=0;a<d.length;a+=1)d[a].c();f=q(),c=x("div"),s=x("form"),g=x("input"),y=q(),h=x("button"),v=H(b),k=q(),E=x("p"),E.textContent="Note - may produce inaccurate information.",m(l,"class","chat-log svelte-f76u98"),m(o,"class","chatbox svelte-f76u98"),m(g,"class","chat-input-textarea svelte-f76u98"),m(g,"placeholder","Type Question Here"),m(h,"class",p="btnyousend "+(e[0]===""?"":"active")+" svelte-f76u98"),m(h,"type","submit"),m(s,"class","chat-input-form svelte-f76u98"),m(E,"class","message svelte-f76u98"),m(c,"class","chat-input-holder svelte-f76u98"),m(n,"class","ranked-chat svelte-f76u98"),m(t,"class","ranked-feedback-container")},m(a,r){X(a,t,r),i(t,n),i(n,o),i(o,l);for(let _=0;_<d.length;_+=1)d[_]&&d[_].m(l,null);i(n,f),i(n,c),i(c,s),i(s,g),Z(g,e[0]),i(s,y),i(s,h),i(h,v),i(c,k),i(c,E),C||(w=[Y(g,"input",e[9]),Y(s,"submit",e[5])],C=!0)},p(a,[r]){r&8&&($=a[3],d=_e(d,r,j,1,a,$,u,l,me,re,null,oe)),r&1&&g.value!==a[0]&&Z(g,a[0]),r&6&&b!==(b=(a[1]?a[2]:"Send")+"")&&G(v,b),r&1&&p!==(p="btnyousend "+(a[0]===""?"":"active")+" svelte-f76u98")&&m(h,"class",p)},i:z,o:z,d(a){a&&D(t);for(let r=0;r<d.length;r+=1)d[r].d();C=!1,F(w)}}}function $e(e){setTimeout(()=>{e.scrollIntoView({behavior:"smooth"})},0)}function ze(e,t,n){let o,l,{feedback:d=!1}=t,{port:u="http://127.0.0.1:50315"}=t;const f=Ee([]);ie(e,f,p=>n(3,l=p));let c="",s="",g=!1;de(()=>{y()});async function y(){const C=(await(await fetch(`${u}/chat/qa_table/retrieve`)).json()).rows.map(w=>({id:w[0],question:w[1],answer:w[2],vote_status:w[3]}));U(f,l=[...C],l)}const h=async p=>{p.preventDefault(),c=s,n(0,s=""),n(1,g=!0);let k={id:l.length+1,question:c,answer:"Loading...",vote_status:"na"};U(f,l=[...l,k],l);const E=await fetch(`${u}/chat/${c}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:c})});if(E.ok){const C=await E.json();k.answer=C.answer,f.update(w=>(w[w.length-1]=k,w))}else{const C=await E.text();alert(C)}n(1,g=!1)};let b=0;setInterval(()=>{n(8,b=(b+1)%4)},200);function v(){s=this.value,n(0,s)}return e.$$set=p=>{"feedback"in p&&n(6,d=p.feedback),"port"in p&&n(7,u=p.port)},e.$$.update=()=>{e.$$.dirty&256&&n(2,o=".".repeat(b).padEnd(3))},[s,g,o,l,f,h,d,u,b,v]}class Te extends Ce{constructor(t){super(),ke(this,t,ze,Se,K,{feedback:6,port:7})}}return Te}();
</script>"""
        self.div_id = self.randomize_hash_id()
        # self.props = get_svelte_props(self.name)
        self.markup = ""

    def add_params(self, params: Dict[str, Any]) -> None:
        """
        Add parameters to the component.

        Parameters
        ----------
        params : dict
            The parameters to add to the component.
        """
        js_data = json.dumps(params, indent=0)
        self.markup = f"""
        <div id="{self.div_id}"></div>
        <script>
        (() => {{
            var data = {js_data};
            window.{self.name}_data = data;
            var {self.name}_inst = new {self.name}({{
                "target": document.getElementById("{self.div_id}"),
                "props": data
            }});
        }})();
        </script>
        """

    def randomize_hash_id(name: str) -> str:
        """
        Generate a random hash for div id for a component.

        Parameters
        ----------
        name : str
            The name of the component. E.g. 'BarChart'.

        Returns
        -------
        str
            The generated hash ID.
        """
        random_hex_id = hex(randint(10**8, 10**9))[2:]
        hashed_div_id = f"{name}-{random_hex_id}"
        return hashed_div_id

    def _repr_html_(self) -> str:
        """
        Return the component as an HTML string.
        """
        return f"""
        {self.iife_script}
        {self.markup}
        """

    def __call__(self, **kwargs: Any) -> "Chatbot":
        """
        Call the component with the given kwargs.

        Parameters
        ----------
        kwargs : any
            The kwargs to pass to the component.

        Returns
        -------
        PackagedComponent
            A python class representing the svelte component, renderable in Jupyter.
        """
        # render with given arguments
        self.add_params(kwargs)
        return self
