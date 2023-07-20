(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const s of document.querySelectorAll('link[rel="modulepreload"]'))r(s);new MutationObserver(s=>{for(const o of s)if(o.type==="childList")for(const l of o.addedNodes)l.tagName==="LINK"&&l.rel==="modulepreload"&&r(l)}).observe(document,{childList:!0,subtree:!0});function n(s){const o={};return s.integrity&&(o.integrity=s.integrity),s.referrerPolicy&&(o.referrerPolicy=s.referrerPolicy),s.crossOrigin==="use-credentials"?o.credentials="include":s.crossOrigin==="anonymous"?o.credentials="omit":o.credentials="same-origin",o}function r(s){if(s.ep)return;s.ep=!0;const o=n(s);fetch(s.href,o)}})();function C(){}const Be=e=>e;function G(e,t){for(const n in t)e[n]=t[n];return e}function Ae(e){return e()}function ae(){return Object.create(null)}function M(e){e.forEach(Ae)}function J(e){return typeof e=="function"}function Z(e,t){return e!=e?t==t:e!==t||e&&typeof e=="object"||typeof e=="function"}function Fe(e){return Object.keys(e).length===0}function Ce(e,...t){if(e==null)return C;const n=e.subscribe(...t);return n.unsubscribe?()=>n.unsubscribe():n}function Ee(e,t,n){e.$$.on_destroy.push(Ce(t,n))}function fe(e,t,n){return e.set(n),t}function He(e){return e&&J(e.destroy)?e.destroy:C}const Oe=typeof window<"u";let Ie=Oe?()=>window.performance.now():()=>Date.now(),Se=Oe?e=>requestAnimationFrame(e):C;const I=new Set;function Te(e){I.forEach(t=>{t.c(e)||(I.delete(t),t.f())}),I.size!==0&&Se(Te)}function Qe(e){let t;return I.size===0&&Se(Te),{promise:new Promise(n=>{I.add(t={c:e,f:n})}),abort(){I.delete(t)}}}function b(e,t){e.appendChild(t)}function S(e,t,n){e.insertBefore(t,n||null)}function O(e){e.parentNode&&e.parentNode.removeChild(e)}function Ne(e,t){for(let n=0;n<e.length;n+=1)e[n]&&e[n].d(t)}function z(e){return document.createElement(e)}function q(e){return document.createTextNode(e)}function L(){return q(" ")}function Ve(){return q("")}function X(e,t,n,r){return e.addEventListener(t,n,r),()=>e.removeEventListener(t,n,r)}function j(e,t,n){n==null?e.removeAttribute(t):e.getAttribute(t)!==n&&e.setAttribute(t,n)}function We(e){return Array.from(e.childNodes)}function Y(e,t){t=""+t,e.data!==t&&(e.data=t)}function de(e,t){e.value=t??""}let R;function W(e){R=e}function Re(){if(!R)throw new Error("Function called outside component initialization");return R}function Le(e){Re().$$.on_mount.push(e)}const H=[],_e=[];let Q=[];const pe=[],Je=Promise.resolve();let te=!1;function Ke(){te||(te=!0,Je.then(Me))}function ne(e){Q.push(e)}const ee=new Set;let B=0;function Me(){if(B!==0)return;const e=R;do{try{for(;B<H.length;){const t=H[B];B++,W(t),Ue(t.$$)}}catch(t){throw H.length=0,B=0,t}for(W(null),H.length=0,B=0;_e.length;)_e.pop()();for(let t=0;t<Q.length;t+=1){const n=Q[t];ee.has(n)||(ee.add(n),n())}Q.length=0}while(H.length);for(;pe.length;)pe.pop()();te=!1,ee.clear(),W(e)}function Ue(e){if(e.fragment!==null){e.update(),M(e.before_update);const t=e.dirty;e.dirty=[-1],e.fragment&&e.fragment.p(e.ctx,t),e.after_update.forEach(ne)}}function Ge(e){const t=[],n=[];Q.forEach(r=>e.indexOf(r)===-1?t.push(r):n.push(r)),n.forEach(r=>r()),Q=t}const U=new Set;let D;function oe(){D={r:0,c:[],p:D}}function se(){D.r||M(D.c),D=D.p}function E(e,t){e&&e.i&&(U.delete(e),e.i(t))}function P(e,t,n,r){if(e&&e.o){if(U.has(e))return;U.add(e),D.c.push(()=>{U.delete(e),r&&(n&&e.d(1),r())}),e.o(t)}else r&&r()}function Xe(e,t){e.d(1),t.delete(e.key)}function Ye(e,t,n,r,s,o,l,i,c,u,m,d){let _=e.length,p=o.length,g=_;const k={};for(;g--;)k[e[g].key]=g;const h=[],v=new Map,w=new Map,$=[];for(g=p;g--;){const y=d(s,o,g),T=n(y);let N=l.get(T);N?r&&$.push(()=>N.p(y,t)):(N=u(T,y),N.c()),v.set(T,h[g]=N),T in k&&w.set(T,Math.abs(g-k[T]))}const a=new Set,f=new Set;function A(y){E(y,1),y.m(i,m),l.set(y.key,y),m=y.first,p--}for(;_&&p;){const y=h[p-1],T=e[_-1],N=y.key,K=T.key;y===T?(m=y.first,_--,p--):v.has(K)?!l.has(N)||a.has(N)?A(y):f.has(K)?_--:w.get(N)>w.get(K)?(f.add(N),A(y)):(a.add(K),_--):(c(T,l),_--)}for(;_--;){const y=e[_];v.has(y.key)||c(y,l)}for(;p;)A(h[p-1]);return M($),h}function qe(e,t){const n={},r={},s={$$scope:1};let o=e.length;for(;o--;){const l=e[o],i=t[o];if(i){for(const c in l)c in i||(r[c]=1);for(const c in i)s[c]||(n[c]=i[c],s[c]=1);e[o]=i}else for(const c in l)s[c]=1}for(const l in r)l in n||(n[l]=void 0);return n}function Pe(e){return typeof e=="object"&&e!==null?e:{}}function De(e){e&&e.c()}function le(e,t,n,r){const{fragment:s,after_update:o}=e.$$;s&&s.m(t,n),r||ne(()=>{const l=e.$$.on_mount.map(Ae).filter(J);e.$$.on_destroy?e.$$.on_destroy.push(...l):M(l),e.$$.on_mount=[]}),o.forEach(ne)}function ie(e,t){const n=e.$$;n.fragment!==null&&(Ge(n.after_update),M(n.on_destroy),n.fragment&&n.fragment.d(t),n.on_destroy=n.fragment=null,n.ctx=[])}function Ze(e,t){e.$$.dirty[0]===-1&&(H.push(e),Ke(),e.$$.dirty.fill(0)),e.$$.dirty[t/31|0]|=1<<t%31}function ce(e,t,n,r,s,o,l,i=[-1]){const c=R;W(e);const u=e.$$={fragment:null,ctx:[],props:o,update:C,not_equal:s,bound:ae(),on_mount:[],on_destroy:[],on_disconnect:[],before_update:[],after_update:[],context:new Map(t.context||(c?c.$$.context:[])),callbacks:ae(),dirty:i,skip_bound:!1,root:t.target||c.$$.root};l&&l(u.root);let m=!1;if(u.ctx=n?n(e,t.props||{},(d,_,...p)=>{const g=p.length?p[0]:_;return u.ctx&&s(u.ctx[d],u.ctx[d]=g)&&(!u.skip_bound&&u.bound[d]&&u.bound[d](g),m&&Ze(e,d)),_}):[],u.update(),m=!0,M(u.before_update),u.fragment=r?r(u.ctx):!1,t.target){if(t.hydrate){const d=We(t.target);u.fragment&&u.fragment.l(d),d.forEach(O)}else u.fragment&&u.fragment.c();t.intro&&E(e.$$.fragment),le(e,t.target,t.anchor,t.customElement),Me()}W(c)}class ue{$destroy(){ie(this,1),this.$destroy=C}$on(t,n){if(!J(n))return C;const r=this.$$.callbacks[t]||(this.$$.callbacks[t]=[]);return r.push(n),()=>{const s=r.indexOf(n);s!==-1&&r.splice(s,1)}}$set(t){this.$$set&&!Fe(t)&&(this.$$.skip_bound=!0,this.$$set(t),this.$$.skip_bound=!1)}}const F=[];function xe(e,t){return{subscribe:x(e,t).subscribe}}function x(e,t=C){let n;const r=new Set;function s(i){if(Z(e,i)&&(e=i,n)){const c=!F.length;for(const u of r)u[1](),F.push(u,e);if(c){for(let u=0;u<F.length;u+=2)F[u][0](F[u+1]);F.length=0}}}function o(i){s(i(e))}function l(i,c=C){const u=[i,c];return r.add(u),r.size===1&&(n=t(s)||C),i(e),()=>{r.delete(u),r.size===0&&n&&(n(),n=null)}}return{set:s,update:o,subscribe:l}}function et(e,t,n){const r=!Array.isArray(e),s=r?[e]:e,o=t.length<2;return xe(n,l=>{let i=!1;const c=[];let u=0,m=C;const d=()=>{if(u)return;m();const p=t(r?c[0]:c,l);o?l(p):m=J(p)?p:C},_=s.map((p,g)=>Ce(p,k=>{c[g]=k,u&=~(1<<g),i&&d()},()=>{u|=1<<g}));return i=!0,d(),function(){M(_),m(),i=!1}})}function he(e){return Object.prototype.toString.call(e)==="[object Date]"}function re(e,t){if(e===t||e!==e)return()=>e;const n=typeof e;if(n!==typeof t||Array.isArray(e)!==Array.isArray(t))throw new Error("Cannot interpolate values of different type");if(Array.isArray(e)){const r=t.map((s,o)=>re(e[o],s));return s=>r.map(o=>o(s))}if(n==="object"){if(!e||!t)throw new Error("Object cannot be null");if(he(e)&&he(t)){e=e.getTime(),t=t.getTime();const o=t-e;return l=>new Date(e+l*o)}const r=Object.keys(t),s={};return r.forEach(o=>{s[o]=re(e[o],t[o])}),o=>{const l={};return r.forEach(i=>{l[i]=s[i](o)}),l}}if(n==="number"){const r=t-e;return s=>e+s*r}throw new Error(`Cannot interpolate ${n} values`)}function tt(e,t={}){const n=x(e);let r,s=e;function o(l,i){if(e==null)return n.set(e=l),Promise.resolve();s=l;let c=r,u=!1,{delay:m=0,duration:d=400,easing:_=Be,interpolate:p=re}=G(G({},t),i);if(d===0)return c&&(c.abort(),c=null),n.set(e=s),Promise.resolve();const g=Ie()+m;let k;return r=Qe(h=>{if(h<g)return!0;u||(k=p(e,l),typeof d=="function"&&(d=d(e,l)),u=!0),c&&(c.abort(),c=null);const v=h-g;return v>d?(n.set(e=l),!1):(n.set(e=k(_(v/d))),!0)}),r.promise}return{set:o,update:(l,i)=>o(l(s,e),i),subscribe:n.subscribe}}const V=x([]),me=[{na:0,good:0,bad:0}];et(V,e=>{let t={...me[0]};return e.forEach(n=>{t[n.vote]++}),console.log("newTally",t),[{...t}]},me);const nt=["Who","What","How","Why","Where","Does","Can","N/A"];tt(nt.map(e=>({question:e,count:0})));function ge(e,t,n){const r=e.slice();return r[13]=t[n],r[15]=n,r}function ye(e){let t,n,r,s,o;function l(){return e[8](e[15])}function i(){return e[9](e[15])}return{c(){t=z("button"),t.textContent="👍",n=L(),r=z("button"),r.textContent="👎",j(t,"class","small-button thumbs-up svelte-ysjhkz"),j(r,"class","small-button thumbs-down svelte-ysjhkz")},m(c,u){S(c,t,u),S(c,n,u),S(c,r,u),s||(o=[X(t,"click",l),X(r,"click",i)],s=!0)},p(c,u){e=c},d(c){c&&O(t),c&&O(n),c&&O(r),s=!1,M(o)}}}function be(e,t){let n,r,s,o,l,i,c,u=t[13].question+"",m,d,_,p,g=t[13].answer+"",k,h,v,w,$,a,f=t[0]&&ye(t);return{key:e,first:null,c(){n=z("div"),r=z("div"),s=z("div"),o=L(),l=z("div"),i=z("p"),c=q("Q: "),m=q(u),d=L(),_=z("p"),p=q("A: "),k=q(g),h=L(),f&&f.c(),v=L(),j(s,"class","avatar svelte-ysjhkz"),j(i,"class","svelte-ysjhkz"),j(_,"class","svelte-ysjhkz"),j(l,"class","message-content svelte-ysjhkz"),j(r,"class","chat-message-center svelte-ysjhkz"),j(n,"class","chat-message svelte-ysjhkz"),this.first=n},m(A,y){S(A,n,y),b(n,r),b(r,s),b(r,o),b(r,l),b(l,i),b(i,c),b(i,m),b(l,d),b(l,_),b(_,p),b(_,k),b(l,h),f&&f.m(l,null),b(n,v),$||(a=He(w=st.call(null,n,t[15]===t[1].length-1)),$=!0)},p(A,y){t=A,y&2&&u!==(u=t[13].question+"")&&Y(m,u),y&2&&g!==(g=t[13].answer+"")&&Y(k,g),t[0]?f?f.p(t,y):(f=ye(t),f.c(),f.m(l,null)):f&&(f.d(1),f=null),w&&J(w.update)&&y&2&&w.update.call(null,t[15]===t[1].length-1)},d(A){A&&O(n),f&&f.d(),$=!1,a()}}}function ot(e){let t,n,r=[],s=new Map,o,l,i,c,u,m,d=(e[3]?e[4]:"Send")+"",_,p,g,k,h,v,w=e[1];const $=a=>a[15];for(let a=0;a<w.length;a+=1){let f=ge(e,w,a),A=$(f);s.set(A,r[a]=be(A,f))}return{c(){t=z("section"),n=z("div");for(let a=0;a<r.length;a+=1)r[a].c();o=L(),l=z("div"),i=z("form"),c=z("input"),u=L(),m=z("button"),_=q(d),g=L(),k=z("p"),k.textContent="Note - may produce inaccurate information.",j(n,"class","chat-log svelte-ysjhkz"),j(c,"class","chat-input-textarea svelte-ysjhkz"),j(c,"placeholder","Type Message Here"),j(m,"class",p="btnyousend "+(e[2]===""?"":"active")+" svelte-ysjhkz"),j(m,"type","submit"),j(i,"class","chat-input-form svelte-ysjhkz"),j(k,"class","message svelte-ysjhkz"),j(l,"class","chat-input-holder svelte-ysjhkz"),j(t,"class","chatbox svelte-ysjhkz")},m(a,f){S(a,t,f),b(t,n);for(let A=0;A<r.length;A+=1)r[A]&&r[A].m(n,null);b(t,o),b(t,l),b(l,i),b(i,c),de(c,e[2]),b(i,u),b(i,m),b(m,_),b(l,g),b(l,k),h||(v=[X(c,"input",e[10]),X(i,"submit",e[5])],h=!0)},p(a,[f]){f&67&&(w=a[1],r=Ye(r,f,$,1,a,w,s,n,Xe,be,null,ge)),f&4&&c.value!==a[2]&&de(c,a[2]),f&24&&d!==(d=(a[3]?a[4]:"Send")+"")&&Y(_,d),f&4&&p!==(p="btnyousend "+(a[2]===""?"":"active")+" svelte-ysjhkz")&&j(m,"class",p)},i:C,o:C,d(a){a&&O(t);for(let f=0;f<r.length;f+=1)r[f].d();h=!1,M(v)}}}function st(e){setTimeout(()=>{e.scrollIntoView({behavior:"smooth"})},0)}function rt(e,t,n){let r,s;Ee(e,V,h=>n(1,s=h));let{feedback:o=!1}=t,l="",i="",c=!1;Le(()=>{u()});async function u(){console.log("fetching data from db");const $=(await(await fetch("http://127.0.0.1:5000/chat/qa_table/retrieve")).json()).rows.map(a=>({id:a[0],question:a[1],answer:a[2],vote_status:a[3]}));fe(V,s=[...$],s)}const m=async h=>{h.preventDefault(),l=i,n(2,i=""),n(3,c=!0);let v={id:s.length+1,question:l,answer:"Loading...",vote_status:"na"};console.log("adding to log here"),fe(V,s=[...s,v],s);const w=await fetch(`/chat/${l}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({prompt:l})});if(w.ok){const $=await w.json();v.answer=$.answer,V.update(a=>(a[a.length-1]=v,a))}else{const $=await w.text();alert($)}n(3,c=!1)};let d=0;setInterval(()=>{n(7,d=(d+1)%4)},200);async function _(h,v){const w=s[v];w.vote=h;const $={id:v+1,vote_status:h};console.log($),console.log($);const a=await fetch("http://127.0.0.1:5000/chat/qa_table/update",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify($)});if(a.ok)console.log("update worked!",a);else{const f=await a.text();alert(f)}}const p=h=>_("up",h),g=h=>_("down",h);function k(){i=this.value,n(2,i)}return e.$$set=h=>{"feedback"in h&&n(0,o=h.feedback)},e.$$.update=()=>{e.$$.dirty&128&&n(4,r=".".repeat(d).padEnd(3)),e.$$.dirty&2&&console.log("chat updated!",s)},[o,s,i,c,r,m,_,d,p,g,k]}class lt extends ue{constructor(t){super(),ce(this,t,rt,ot,Z,{feedback:0})}}function ve(e,t,n){const r=e.slice();return r[2]=t[n],r}function we(e){let t,n=e[2]+"",r,s;return{c(){t=z("option"),r=q(n),t.__value=s=e[2],t.value=t.__value},m(o,l){S(o,t,l),b(t,r)},p(o,l){l&1&&n!==(n=o[2]+"")&&Y(r,n),l&1&&s!==(s=o[2])&&(t.__value=s,t.value=t.__value)},d(o){o&&O(t)}}}function it(e){let t,n=e[0],r=[];for(let s=0;s<n.length;s+=1)r[s]=we(ve(e,n,s));return{c(){t=z("select");for(let s=0;s<r.length;s+=1)r[s].c()},m(s,o){S(s,t,o);for(let l=0;l<r.length;l+=1)r[l]&&r[l].m(t,null)},p(s,[o]){if(o&1){n=s[0];let l;for(l=0;l<n.length;l+=1){const i=ve(s,n,l);r[l]?r[l].p(i,o):(r[l]=we(i),r[l].c(),r[l].m(t,null))}for(;l<r.length;l+=1)r[l].d(1);r.length=n.length}},i:C,o:C,d(s){s&&O(t),Ne(r,s)}}}function ct(e,t,n){let{data_endpoint:r}=t,s=[];return Le(async()=>{const o=await fetch(`/data/${r}`);n(0,s=await o.json())}),e.$$set=o=>{"data_endpoint"in o&&n(1,r=o.data_endpoint)},[s,r]}class ut extends ue{constructor(t){super(),ce(this,t,ct,it,Z,{data_endpoint:1})}}function ke(e,t,n){const r=e.slice();return r[2]=t[n],r}function $e(e){let t,n;const r=[e[2].props];let s={};for(let o=0;o<r.length;o+=1)s=G(s,r[o]);return t=new lt({props:s}),{c(){De(t.$$.fragment)},m(o,l){le(t,o,l),n=!0},p(o,l){const i=l&1?qe(r,[Pe(o[2].props)]):{};t.$set(i)},i(o){n||(E(t.$$.fragment,o),n=!0)},o(o){P(t.$$.fragment,o),n=!1},d(o){ie(t,o)}}}function je(e){let t,n;const r=[e[2].props];let s={};for(let o=0;o<r.length;o+=1)s=G(s,r[o]);return t=new ut({props:s}),{c(){De(t.$$.fragment)},m(o,l){le(t,o,l),n=!0},p(o,l){const i=l&1?qe(r,[Pe(o[2].props)]):{};t.$set(i)},i(o){n||(E(t.$$.fragment,o),n=!0)},o(o){P(t.$$.fragment,o),n=!1},d(o){ie(t,o)}}}function ze(e){let t,n,r,s=e[2].svelte_component==="Chatbot"&&$e(e),o=e[2].svelte_component==="Dropdown"&&je(e);return{c(){s&&s.c(),t=L(),o&&o.c(),n=Ve()},m(l,i){s&&s.m(l,i),S(l,t,i),o&&o.m(l,i),S(l,n,i),r=!0},p(l,i){l[2].svelte_component==="Chatbot"?s?(s.p(l,i),i&1&&E(s,1)):(s=$e(l),s.c(),E(s,1),s.m(t.parentNode,t)):s&&(oe(),P(s,1,1,()=>{s=null}),se()),l[2].svelte_component==="Dropdown"?o?(o.p(l,i),i&1&&E(o,1)):(o=je(l),o.c(),E(o,1),o.m(n.parentNode,n)):o&&(oe(),P(o,1,1,()=>{o=null}),se())},i(l){r||(E(s),E(o),r=!0)},o(l){P(s),P(o),r=!1},d(l){s&&s.d(l),l&&O(t),o&&o.d(l),l&&O(n)}}}function at(e){let t,n,r,s=e[0],o=[];for(let i=0;i<s.length;i+=1)o[i]=ze(ke(e,s,i));const l=i=>P(o[i],1,1,()=>{o[i]=null});return{c(){for(let i=0;i<o.length;i+=1)o[i].c();t=L(),n=z("p"),n.textContent="Made with CambioML",j(n,"class","footer-logo svelte-bw62g5")},m(i,c){for(let u=0;u<o.length;u+=1)o[u]&&o[u].m(i,c);S(i,t,c),S(i,n,c),r=!0},p(i,[c]){if(c&1){s=i[0];let u;for(u=0;u<s.length;u+=1){const m=ke(i,s,u);o[u]?(o[u].p(m,c),E(o[u],1)):(o[u]=ze(m),o[u].c(),E(o[u],1),o[u].m(t.parentNode,t))}for(oe(),u=s.length;u<o.length;u+=1)l(u);se()}},i(i){if(!r){for(let c=0;c<s.length;c+=1)E(o[c]);r=!0}},o(i){o=o.filter(Boolean);for(let c=0;c<o.length;c+=1)P(o[c]);r=!1},d(i){Ne(o,i),i&&O(t),i&&O(n)}}}function ft(e,t,n){let r;const s=x([]);return Ee(e,s,o=>n(0,r=o)),fetch("/components").then(o=>o.json()).then(o=>{s.set(o)}),[r,s]}class dt extends ue{constructor(t){super(),ce(this,t,ft,at,Z,{})}}new dt({target:document.getElementById("app")});
