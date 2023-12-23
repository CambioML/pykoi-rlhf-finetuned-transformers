"""Module for the compiled Svelte Barchart interactive."""
import json
from random import randint
from typing import Any, Dict


class Barchart:
    def __init__(self) -> None:
        """
        Initialize the component.
        """
        self.name = "Barchart"
        self.iife_script = """<script>var Barchart=function(){"use strict";var Me=document.createElement("style");Me.textContent=`@import"https://fonts.googleapis.com/css?family=Work+Sans:400|Lato:400|Inconsolata:400";.svelte-1vd4qdl{font-family:Lato}#bar-chart-holder.svelte-1vd4qdl{height:100%;width:100%}rect.svelte-1vd4qdl:hover{stroke:red;stroke-width:5}.axis-text.svelte-1vd4qdl{font-size:12px}.axis-line.svelte-1vd4qdl{stroke-width:3;stroke:#000;fill:none}
`,document.head.appendChild(Me);function z(){}function ke(e){return e()}function Ne(){return Object.create(null)}function W(e){e.forEach(ke)}function Re(e){return typeof e=="function"}function mn(e,n){return e!=e?n==n:e!==n||e&&typeof e=="object"||typeof e=="function"}function xn(e){return Object.keys(e).length===0}function R(e,n){e.appendChild(n)}function B(e,n,t){e.insertBefore(n,t||null)}function S(e){e.parentNode&&e.parentNode.removeChild(e)}function re(e,n){for(let t=0;t<e.length;t+=1)e[t]&&e[t].d(n)}function bn(e){return document.createElement(e)}function $(e){return document.createElementNS("http://www.w3.org/2000/svg",e)}function X(e){return document.createTextNode(e)}function Ae(){return X("")}function x(e,n,t){t==null?e.removeAttribute(n):e.getAttribute(n)!==t&&e.setAttribute(n,t)}function pn(e){return Array.from(e.childNodes)}function yn(e,n){n=""+n,e.data!==n&&(e.data=n)}let ie;function L(e){ie=e}const j=[],$e=[];let P=[];const qe=[],wn=Promise.resolve();let ae=!1;function _n(){ae||(ae=!0,wn.then(Ee))}function fe(e){P.push(e)}const oe=new Set;let I=0;function Ee(){if(I!==0)return;const e=ie;do{try{for(;I<j.length;){const n=j[I];I++,L(n),vn(n.$$)}}catch(n){throw j.length=0,I=0,n}for(L(null),j.length=0,I=0;$e.length;)$e.pop()();for(let n=0;n<P.length;n+=1){const t=P[n];oe.has(t)||(oe.add(t),t())}P.length=0}while(j.length);for(;qe.length;)qe.pop()();ae=!1,oe.clear(),L(e)}function vn(e){if(e.fragment!==null){e.update(),W(e.before_update);const n=e.dirty;e.dirty=[-1],e.fragment&&e.fragment.p(e.ctx,n),e.after_update.forEach(fe)}}function Mn(e){const n=[],t=[];P.forEach(r=>e.indexOf(r)===-1?n.push(r):t.push(r)),t.forEach(r=>r()),P=n}const kn=new Set;function Nn(e,n){e&&e.i&&(kn.delete(e),e.i(n))}function Rn(e,n,t,r){const{fragment:i,after_update:f}=e.$$;i&&i.m(n,t),r||fe(()=>{const a=e.$$.on_mount.map(ke).filter(Re);e.$$.on_destroy?e.$$.on_destroy.push(...a):W(a),e.$$.on_mount=[]}),f.forEach(fe)}function An(e,n){const t=e.$$;t.fragment!==null&&(Mn(t.after_update),W(t.on_destroy),t.fragment&&t.fragment.d(n),t.on_destroy=t.fragment=null,t.ctx=[])}function $n(e,n){e.$$.dirty[0]===-1&&(j.push(e),_n(),e.$$.dirty.fill(0)),e.$$.dirty[n/31|0]|=1<<n%31}function qn(e,n,t,r,i,f,a,o=[-1]){const u=ie;L(e);const l=e.$$={fragment:null,ctx:[],props:f,update:z,not_equal:i,bound:Ne(),on_mount:[],on_destroy:[],on_disconnect:[],before_update:[],after_update:[],context:new Map(n.context||(u?u.$$.context:[])),callbacks:Ne(),dirty:o,skip_bound:!1,root:n.target||u.$$.root};a&&a(l.root);let h=!1;if(l.ctx=t?t(e,n.props||{},(c,s,...p)=>{const m=p.length?p[0]:s;return l.ctx&&i(l.ctx[c],l.ctx[c]=m)&&(!l.skip_bound&&l.bound[c]&&l.bound[c](m),h&&$n(e,c)),s}):[],l.update(),h=!0,W(l.before_update),l.fragment=r?r(l.ctx):!1,n.target){if(n.hydrate){const c=pn(n.target);l.fragment&&l.fragment.l(c),c.forEach(S)}else l.fragment&&l.fragment.c();n.intro&&Nn(e.$$.fragment),Rn(e,n.target,n.anchor,n.customElement),Ee()}L(u)}class En{$destroy(){An(this,1),this.$destroy=z}$on(n,t){if(!Re(t))return z;const r=this.$$.callbacks[n]||(this.$$.callbacks[n]=[]);return r.push(t),()=>{const i=r.indexOf(t);i!==-1&&r.splice(i,1)}}$set(n){this.$$set&&!xn(n)&&(this.$$.skip_bound=!0,this.$$set(n),this.$$.skip_bound=!1)}}function Sn(e,n){let t;if(n===void 0)for(const r of e)r!=null&&(t<r||t===void 0&&r>=r)&&(t=r);else{let r=-1;for(let i of e)(i=n(i,++r,e))!=null&&(t<i||t===void 0&&i>=i)&&(t=i)}return t}function Se(e,n){return e<n?-1:e>n?1:e>=n?0:NaN}function je(e){let n=e,t=e;e.length===1&&(n=(a,o)=>e(a)-o,t=jn(e));function r(a,o,u,l){for(u==null&&(u=0),l==null&&(l=a.length);u<l;){const h=u+l>>>1;t(a[h],o)<0?u=h+1:l=h}return u}function i(a,o,u,l){for(u==null&&(u=0),l==null&&(l=a.length);u<l;){const h=u+l>>>1;t(a[h],o)>0?l=h:u=h+1}return u}function f(a,o,u,l){u==null&&(u=0),l==null&&(l=a.length);const h=r(a,o,u,l-1);return h>u&&n(a[h-1],o)>-n(a[h],o)?h-1:h}return{left:r,center:f,right:i}}function jn(e){return(n,t)=>Se(e(n),t)}function Pn(e){return e===null?NaN:+e}const In=je(Se).right;je(Pn).center;const Cn=In;var le=Math.sqrt(50),ue=Math.sqrt(10),se=Math.sqrt(2);function Hn(e,n,t){var r,i=-1,f,a,o;if(n=+n,e=+e,t=+t,e===n&&t>0)return[e];if((r=n<e)&&(f=e,e=n,n=f),(o=Pe(e,n,t))===0||!isFinite(o))return[];if(o>0){let u=Math.round(e/o),l=Math.round(n/o);for(u*o<e&&++u,l*o>n&&--l,a=new Array(f=l-u+1);++i<f;)a[i]=(u+i)*o}else{o=-o;let u=Math.round(e*o),l=Math.round(n*o);for(u/o<e&&++u,l/o>n&&--l,a=new Array(f=l-u+1);++i<f;)a[i]=(u+i)/o}return r&&a.reverse(),a}function Pe(e,n,t){var r=(n-e)/Math.max(0,t),i=Math.floor(Math.log(r)/Math.LN10),f=r/Math.pow(10,i);return i>=0?(f>=le?10:f>=ue?5:f>=se?2:1)*Math.pow(10,i):-Math.pow(10,-i)/(f>=le?10:f>=ue?5:f>=se?2:1)}function Fn(e,n,t){var r=Math.abs(n-e)/Math.max(0,t),i=Math.pow(10,Math.floor(Math.log(r)/Math.LN10)),f=r/i;return f>=le?i*=10:f>=ue?i*=5:f>=se&&(i*=2),n<e?-i:i}function On(e,n,t){e=+e,n=+n,t=(i=arguments.length)<2?(n=e,e=0,1):i<3?1:+t;for(var r=-1,i=Math.max(0,Math.ceil((n-e)/t))|0,f=new Array(i);++r<i;)f[r]=e+r*t;return f}function ce(e,n){switch(arguments.length){case 0:break;case 1:this.range(e);break;default:this.range(n).domain(e);break}return this}const Ie=Symbol("implicit");function he(){var e=new Map,n=[],t=[],r=Ie;function i(f){var a=f+"",o=e.get(a);if(!o){if(r!==Ie)return r;e.set(a,o=n.push(f))}return t[(o-1)%t.length]}return i.domain=function(f){if(!arguments.length)return n.slice();n=[],e=new Map;for(const a of f){const o=a+"";e.has(o)||e.set(o,n.push(a))}return i},i.range=function(f){return arguments.length?(t=Array.from(f),i):t.slice()},i.unknown=function(f){return arguments.length?(r=f,i):r},i.copy=function(){return he(n,t).unknown(r)},ce.apply(i,arguments),i}function Ce(){var e=he().unknown(void 0),n=e.domain,t=e.range,r=0,i=1,f,a,o=!1,u=0,l=0,h=.5;delete e.unknown;function c(){var s=n().length,p=i<r,m=p?i:r,d=p?r:i;f=(d-m)/Math.max(1,s-u+l*2),o&&(f=Math.floor(f)),m+=(d-m-f*(s-u))*h,a=f*(1-u),o&&(m=Math.round(m),a=Math.round(a));var y=On(s).map(function(g){return m+f*g});return t(p?y.reverse():y)}return e.domain=function(s){return arguments.length?(n(s),c()):n()},e.range=function(s){return arguments.length?([r,i]=s,r=+r,i=+i,c()):[r,i]},e.rangeRound=function(s){return[r,i]=s,r=+r,i=+i,o=!0,c()},e.bandwidth=function(){return a},e.step=function(){return f},e.round=function(s){return arguments.length?(o=!!s,c()):o},e.padding=function(s){return arguments.length?(u=Math.min(1,l=+s),c()):u},e.paddingInner=function(s){return arguments.length?(u=Math.min(1,s),c()):u},e.paddingOuter=function(s){return arguments.length?(l=+s,c()):l},e.align=function(s){return arguments.length?(h=Math.max(0,Math.min(1,s)),c()):h},e.copy=function(){return Ce(n(),[r,i]).round(o).paddingInner(u).paddingOuter(l).align(h)},ce.apply(c(),arguments)}function de(e,n,t){e.prototype=n.prototype=t,t.constructor=e}function He(e,n){var t=Object.create(e.prototype);for(var r in n)t[r]=n[r];return t}function D(){}var T=.7,U=1/T,C="\\s*([+-]?\\d+)\\s*",G="\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)\\s*",k="\\s*([+-]?\\d*\\.?\\d+(?:[eE][+-]?\\d+)?)%\\s*",zn=/^#([0-9a-f]{3,8})$/,Bn=new RegExp("^rgb\\("+[C,C,C]+"\\)$"),Ln=new RegExp("^rgb\\("+[k,k,k]+"\\)$"),Dn=new RegExp("^rgba\\("+[C,C,C,G]+"\\)$"),Tn=new RegExp("^rgba\\("+[k,k,k,G]+"\\)$"),Gn=new RegExp("^hsl\\("+[G,k,k]+"\\)$"),Vn=new RegExp("^hsla\\("+[G,k,k,G]+"\\)$"),Fe={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074};de(D,V,{copy:function(e){return Object.assign(new this.constructor,this,e)},displayable:function(){return this.rgb().displayable()},hex:Oe,formatHex:Oe,formatHsl:Wn,formatRgb:ze,toString:ze});function Oe(){return this.rgb().formatHex()}function Wn(){return Ge(this).formatHsl()}function ze(){return this.rgb().formatRgb()}function V(e){var n,t;return e=(e+"").trim().toLowerCase(),(n=zn.exec(e))?(t=n[1].length,n=parseInt(n[1],16),t===6?Be(n):t===3?new w(n>>8&15|n>>4&240,n>>4&15|n&240,(n&15)<<4|n&15,1):t===8?Y(n>>24&255,n>>16&255,n>>8&255,(n&255)/255):t===4?Y(n>>12&15|n>>8&240,n>>8&15|n>>4&240,n>>4&15|n&240,((n&15)<<4|n&15)/255):null):(n=Bn.exec(e))?new w(n[1],n[2],n[3],1):(n=Ln.exec(e))?new w(n[1]*255/100,n[2]*255/100,n[3]*255/100,1):(n=Dn.exec(e))?Y(n[1],n[2],n[3],n[4]):(n=Tn.exec(e))?Y(n[1]*255/100,n[2]*255/100,n[3]*255/100,n[4]):(n=Gn.exec(e))?Te(n[1],n[2]/100,n[3]/100,1):(n=Vn.exec(e))?Te(n[1],n[2]/100,n[3]/100,n[4]):Fe.hasOwnProperty(e)?Be(Fe[e]):e==="transparent"?new w(NaN,NaN,NaN,0):null}function Be(e){return new w(e>>16&255,e>>8&255,e&255,1)}function Y(e,n,t,r){return r<=0&&(e=n=t=NaN),new w(e,n,t,r)}function Xn(e){return e instanceof D||(e=V(e)),e?(e=e.rgb(),new w(e.r,e.g,e.b,e.opacity)):new w}function ge(e,n,t,r){return arguments.length===1?Xn(e):new w(e,n,t,r==null?1:r)}function w(e,n,t,r){this.r=+e,this.g=+n,this.b=+t,this.opacity=+r}de(w,ge,He(D,{brighter:function(e){return e=e==null?U:Math.pow(U,e),new w(this.r*e,this.g*e,this.b*e,this.opacity)},darker:function(e){return e=e==null?T:Math.pow(T,e),new w(this.r*e,this.g*e,this.b*e,this.opacity)},rgb:function(){return this},displayable:function(){return-.5<=this.r&&this.r<255.5&&-.5<=this.g&&this.g<255.5&&-.5<=this.b&&this.b<255.5&&0<=this.opacity&&this.opacity<=1},hex:Le,formatHex:Le,formatRgb:De,toString:De}));function Le(){return"#"+me(this.r)+me(this.g)+me(this.b)}function De(){var e=this.opacity;return e=isNaN(e)?1:Math.max(0,Math.min(1,e)),(e===1?"rgb(":"rgba(")+Math.max(0,Math.min(255,Math.round(this.r)||0))+", "+Math.max(0,Math.min(255,Math.round(this.g)||0))+", "+Math.max(0,Math.min(255,Math.round(this.b)||0))+(e===1?")":", "+e+")")}function me(e){return e=Math.max(0,Math.min(255,Math.round(e)||0)),(e<16?"0":"")+e.toString(16)}function Te(e,n,t,r){return r<=0?e=n=t=NaN:t<=0||t>=1?e=n=NaN:n<=0&&(e=NaN),new N(e,n,t,r)}function Ge(e){if(e instanceof N)return new N(e.h,e.s,e.l,e.opacity);if(e instanceof D||(e=V(e)),!e)return new N;if(e instanceof N)return e;e=e.rgb();var n=e.r/255,t=e.g/255,r=e.b/255,i=Math.min(n,t,r),f=Math.max(n,t,r),a=NaN,o=f-i,u=(f+i)/2;return o?(n===f?a=(t-r)/o+(t<r)*6:t===f?a=(r-n)/o+2:a=(n-t)/o+4,o/=u<.5?f+i:2-f-i,a*=60):o=u>0&&u<1?0:a,new N(a,o,u,e.opacity)}function Un(e,n,t,r){return arguments.length===1?Ge(e):new N(e,n,t,r==null?1:r)}function N(e,n,t,r){this.h=+e,this.s=+n,this.l=+t,this.opacity=+r}de(N,Un,He(D,{brighter:function(e){return e=e==null?U:Math.pow(U,e),new N(this.h,this.s,this.l*e,this.opacity)},darker:function(e){return e=e==null?T:Math.pow(T,e),new N(this.h,this.s,this.l*e,this.opacity)},rgb:function(){var e=this.h%360+(this.h<0)*360,n=isNaN(e)||isNaN(this.s)?0:this.s,t=this.l,r=t+(t<.5?t:1-t)*n,i=2*t-r;return new w(xe(e>=240?e-240:e+120,i,r),xe(e,i,r),xe(e<120?e+240:e-120,i,r),this.opacity)},displayable:function(){return(0<=this.s&&this.s<=1||isNaN(this.s))&&0<=this.l&&this.l<=1&&0<=this.opacity&&this.opacity<=1},formatHsl:function(){var e=this.opacity;return e=isNaN(e)?1:Math.max(0,Math.min(1,e)),(e===1?"hsl(":"hsla(")+(this.h||0)+", "+(this.s||0)*100+"%, "+(this.l||0)*100+"%"+(e===1?")":", "+e+")")}}));function xe(e,n,t){return(e<60?n+(t-n)*e/60:e<180?t:e<240?n+(t-n)*(240-e)/60:n)*255}const be=e=>()=>e;function Yn(e,n){return function(t){return e+t*n}}function Zn(e,n,t){return e=Math.pow(e,t),n=Math.pow(n,t)-e,t=1/t,function(r){return Math.pow(e+r*n,t)}}function Jn(e){return(e=+e)==1?Ve:function(n,t){return t-n?Zn(n,t,e):be(isNaN(n)?t:n)}}function Ve(e,n){var t=n-e;return t?Yn(e,t):be(isNaN(e)?n:e)}const We=function e(n){var t=Jn(n);function r(i,f){var a=t((i=ge(i)).r,(f=ge(f)).r),o=t(i.g,f.g),u=t(i.b,f.b),l=Ve(i.opacity,f.opacity);return function(h){return i.r=a(h),i.g=o(h),i.b=u(h),i.opacity=l(h),i+""}}return r.gamma=e,r}(1);function Kn(e,n){n||(n=[]);var t=e?Math.min(n.length,e.length):0,r=n.slice(),i;return function(f){for(i=0;i<t;++i)r[i]=e[i]*(1-f)+n[i]*f;return r}}function Qn(e){return ArrayBuffer.isView(e)&&!(e instanceof DataView)}function et(e,n){var t=n?n.length:0,r=e?Math.min(t,e.length):0,i=new Array(r),f=new Array(t),a;for(a=0;a<r;++a)i[a]=we(e[a],n[a]);for(;a<t;++a)f[a]=n[a];return function(o){for(a=0;a<r;++a)f[a]=i[a](o);return f}}function nt(e,n){var t=new Date;return e=+e,n=+n,function(r){return t.setTime(e*(1-r)+n*r),t}}function Z(e,n){return e=+e,n=+n,function(t){return e*(1-t)+n*t}}function tt(e,n){var t={},r={},i;(e===null||typeof e!="object")&&(e={}),(n===null||typeof n!="object")&&(n={});for(i in n)i in e?t[i]=we(e[i],n[i]):r[i]=n[i];return function(f){for(i in t)r[i]=t[i](f);return r}}var pe=/[-+]?(?:\d+\.?\d*|\.?\d+)(?:[eE][-+]?\d+)?/g,ye=new RegExp(pe.source,"g");function rt(e){return function(){return e}}function it(e){return function(n){return e(n)+""}}function at(e,n){var t=pe.lastIndex=ye.lastIndex=0,r,i,f,a=-1,o=[],u=[];for(e=e+"",n=n+"";(r=pe.exec(e))&&(i=ye.exec(n));)(f=i.index)>t&&(f=n.slice(t,f),o[a]?o[a]+=f:o[++a]=f),(r=r[0])===(i=i[0])?o[a]?o[a]+=i:o[++a]=i:(o[++a]=null,u.push({i:a,x:Z(r,i)})),t=ye.lastIndex;return t<n.length&&(f=n.slice(t),o[a]?o[a]+=f:o[++a]=f),o.length<2?u[0]?it(u[0].x):rt(n):(n=u.length,function(l){for(var h=0,c;h<n;++h)o[(c=u[h]).i]=c.x(l);return o.join("")})}function we(e,n){var t=typeof n,r;return n==null||t==="boolean"?be(n):(t==="number"?Z:t==="string"?(r=V(n))?(n=r,We):at:n instanceof V?We:n instanceof Date?nt:Qn(n)?Kn:Array.isArray(n)?et:typeof n.valueOf!="function"&&typeof n.toString!="function"||isNaN(n)?tt:Z)(e,n)}function ft(e,n){return e=+e,n=+n,function(t){return Math.round(e*(1-t)+n*t)}}function ot(e){return function(){return e}}function lt(e){return+e}var Xe=[0,1];function H(e){return e}function _e(e,n){return(n-=e=+e)?function(t){return(t-e)/n}:ot(isNaN(n)?NaN:.5)}function ut(e,n){var t;return e>n&&(t=e,e=n,n=t),function(r){return Math.max(e,Math.min(n,r))}}function st(e,n,t){var r=e[0],i=e[1],f=n[0],a=n[1];return i<r?(r=_e(i,r),f=t(a,f)):(r=_e(r,i),f=t(f,a)),function(o){return f(r(o))}}function ct(e,n,t){var r=Math.min(e.length,n.length)-1,i=new Array(r),f=new Array(r),a=-1;for(e[r]<e[0]&&(e=e.slice().reverse(),n=n.slice().reverse());++a<r;)i[a]=_e(e[a],e[a+1]),f[a]=t(n[a],n[a+1]);return function(o){var u=Cn(e,o,1,r)-1;return f[u](i[u](o))}}function ht(e,n){return n.domain(e.domain()).range(e.range()).interpolate(e.interpolate()).clamp(e.clamp()).unknown(e.unknown())}function dt(){var e=Xe,n=Xe,t=we,r,i,f,a=H,o,u,l;function h(){var s=Math.min(e.length,n.length);return a!==H&&(a=ut(e[0],e[s-1])),o=s>2?ct:st,u=l=null,c}function c(s){return s==null||isNaN(s=+s)?f:(u||(u=o(e.map(r),n,t)))(r(a(s)))}return c.invert=function(s){return a(i((l||(l=o(n,e.map(r),Z)))(s)))},c.domain=function(s){return arguments.length?(e=Array.from(s,lt),h()):e.slice()},c.range=function(s){return arguments.length?(n=Array.from(s),h()):n.slice()},c.rangeRound=function(s){return n=Array.from(s),t=ft,h()},c.clamp=function(s){return arguments.length?(a=s?!0:H,h()):a!==H},c.interpolate=function(s){return arguments.length?(t=s,h()):t},c.unknown=function(s){return arguments.length?(f=s,c):f},function(s,p){return r=s,i=p,h()}}function gt(){return dt()(H,H)}function mt(e){return Math.abs(e=Math.round(e))>=1e21?e.toLocaleString("en").replace(/,/g,""):e.toString(10)}function J(e,n){if((t=(e=n?e.toExponential(n-1):e.toExponential()).indexOf("e"))<0)return null;var t,r=e.slice(0,t);return[r.length>1?r[0]+r.slice(2):r,+e.slice(t+1)]}function F(e){return e=J(Math.abs(e)),e?e[1]:NaN}function xt(e,n){return function(t,r){for(var i=t.length,f=[],a=0,o=e[0],u=0;i>0&&o>0&&(u+o+1>r&&(o=Math.max(1,r-u)),f.push(t.substring(i-=o,i+o)),!((u+=o+1)>r));)o=e[a=(a+1)%e.length];return f.reverse().join(n)}}function bt(e){return function(n){return n.replace(/[0-9]/g,function(t){return e[+t]})}}var pt=/^(?:(.)?([<>=^]))?([+\-( ])?([$#])?(0)?(\d+)?(,)?(\.\d+)?(~)?([a-z%])?$/i;function K(e){if(!(n=pt.exec(e)))throw new Error("invalid format: "+e);var n;return new ve({fill:n[1],align:n[2],sign:n[3],symbol:n[4],zero:n[5],width:n[6],comma:n[7],precision:n[8]&&n[8].slice(1),trim:n[9],type:n[10]})}K.prototype=ve.prototype;function ve(e){this.fill=e.fill===void 0?" ":e.fill+"",this.align=e.align===void 0?">":e.align+"",this.sign=e.sign===void 0?"-":e.sign+"",this.symbol=e.symbol===void 0?"":e.symbol+"",this.zero=!!e.zero,this.width=e.width===void 0?void 0:+e.width,this.comma=!!e.comma,this.precision=e.precision===void 0?void 0:+e.precision,this.trim=!!e.trim,this.type=e.type===void 0?"":e.type+""}ve.prototype.toString=function(){return this.fill+this.align+this.sign+this.symbol+(this.zero?"0":"")+(this.width===void 0?"":Math.max(1,this.width|0))+(this.comma?",":"")+(this.precision===void 0?"":"."+Math.max(0,this.precision|0))+(this.trim?"~":"")+this.type};function yt(e){e:for(var n=e.length,t=1,r=-1,i;t<n;++t)switch(e[t]){case".":r=i=t;break;case"0":r===0&&(r=t),i=t;break;default:if(!+e[t])break e;r>0&&(r=0);break}return r>0?e.slice(0,r)+e.slice(i+1):e}var Ue;function wt(e,n){var t=J(e,n);if(!t)return e+"";var r=t[0],i=t[1],f=i-(Ue=Math.max(-8,Math.min(8,Math.floor(i/3)))*3)+1,a=r.length;return f===a?r:f>a?r+new Array(f-a+1).join("0"):f>0?r.slice(0,f)+"."+r.slice(f):"0."+new Array(1-f).join("0")+J(e,Math.max(0,n+f-1))[0]}function Ye(e,n){var t=J(e,n);if(!t)return e+"";var r=t[0],i=t[1];return i<0?"0."+new Array(-i).join("0")+r:r.length>i+1?r.slice(0,i+1)+"."+r.slice(i+1):r+new Array(i-r.length+2).join("0")}const Ze={"%":(e,n)=>(e*100).toFixed(n),b:e=>Math.round(e).toString(2),c:e=>e+"",d:mt,e:(e,n)=>e.toExponential(n),f:(e,n)=>e.toFixed(n),g:(e,n)=>e.toPrecision(n),o:e=>Math.round(e).toString(8),p:(e,n)=>Ye(e*100,n),r:Ye,s:wt,X:e=>Math.round(e).toString(16).toUpperCase(),x:e=>Math.round(e).toString(16)};function Je(e){return e}var Ke=Array.prototype.map,Qe=["y","z","a","f","p","n","\xB5","m","","k","M","G","T","P","E","Z","Y"];function _t(e){var n=e.grouping===void 0||e.thousands===void 0?Je:xt(Ke.call(e.grouping,Number),e.thousands+""),t=e.currency===void 0?"":e.currency[0]+"",r=e.currency===void 0?"":e.currency[1]+"",i=e.decimal===void 0?".":e.decimal+"",f=e.numerals===void 0?Je:bt(Ke.call(e.numerals,String)),a=e.percent===void 0?"%":e.percent+"",o=e.minus===void 0?"\u2212":e.minus+"",u=e.nan===void 0?"NaN":e.nan+"";function l(c){c=K(c);var s=c.fill,p=c.align,m=c.sign,d=c.symbol,y=c.zero,g=c.width,_=c.comma,q=c.precision,cn=c.trim,v=c.type;v==="n"?(_=!0,v="g"):Ze[v]||(q===void 0&&(q=12),cn=!0,v="g"),(y||s==="0"&&p==="=")&&(y=!0,s="0",p="=");var St=d==="$"?t:d==="#"&&/[boxX]/.test(v)?"0"+v.toLowerCase():"",jt=d==="$"?r:/[%p]/.test(v)?a:"",hn=Ze[v],Pt=/[defgprs%]/.test(v);q=q===void 0?6:/[gprs]/.test(v)?Math.max(1,Math.min(21,q)):Math.max(0,Math.min(20,q));function dn(b){var E=St,M=jt,O,gn,ee;if(v==="c")M=hn(b)+M,b="";else{b=+b;var ne=b<0||1/b<0;if(b=isNaN(b)?u:hn(Math.abs(b),q),cn&&(b=yt(b)),ne&&+b==0&&m!=="+"&&(ne=!1),E=(ne?m==="("?m:o:m==="-"||m==="("?"":m)+E,M=(v==="s"?Qe[8+Ue/3]:"")+M+(ne&&m==="("?")":""),Pt){for(O=-1,gn=b.length;++O<gn;)if(ee=b.charCodeAt(O),48>ee||ee>57){M=(ee===46?i+b.slice(O+1):b.slice(O))+M,b=b.slice(0,O);break}}}_&&!y&&(b=n(b,1/0));var te=E.length+b.length+M.length,A=te<g?new Array(g-te+1).join(s):"";switch(_&&y&&(b=n(A+b,A.length?g-M.length:1/0),A=""),p){case"<":b=E+b+M+A;break;case"=":b=E+A+b+M;break;case"^":b=A.slice(0,te=A.length>>1)+E+b+M+A.slice(te);break;default:b=A+E+b+M;break}return f(b)}return dn.toString=function(){return c+""},dn}function h(c,s){var p=l((c=K(c),c.type="f",c)),m=Math.max(-8,Math.min(8,Math.floor(F(s)/3)))*3,d=Math.pow(10,-m),y=Qe[8+m/3];return function(g){return p(d*g)+y}}return{format:l,formatPrefix:h}}var Q,en,nn;vt({thousands:",",grouping:[3],currency:["$",""]});function vt(e){return Q=_t(e),en=Q.format,nn=Q.formatPrefix,Q}function Mt(e){return Math.max(0,-F(Math.abs(e)))}function kt(e,n){return Math.max(0,Math.max(-8,Math.min(8,Math.floor(F(n)/3)))*3-F(Math.abs(e)))}function Nt(e,n){return e=Math.abs(e),n=Math.abs(n)-e,Math.max(0,F(n)-F(e))+1}function Rt(e,n,t,r){var i=Fn(e,n,t),f;switch(r=K(r==null?",f":r),r.type){case"s":{var a=Math.max(Math.abs(e),Math.abs(n));return r.precision==null&&!isNaN(f=kt(i,a))&&(r.precision=f),nn(r,a)}case"":case"e":case"g":case"p":case"r":{r.precision==null&&!isNaN(f=Nt(i,Math.max(Math.abs(e),Math.abs(n))))&&(r.precision=f-(r.type==="e"));break}case"f":case"%":{r.precision==null&&!isNaN(f=Mt(i))&&(r.precision=f-(r.type==="%")*2);break}}return en(r)}function At(e){var n=e.domain;return e.ticks=function(t){var r=n();return Hn(r[0],r[r.length-1],t==null?10:t)},e.tickFormat=function(t,r){var i=n();return Rt(i[0],i[i.length-1],t==null?10:t,r)},e.nice=function(t){t==null&&(t=10);var r=n(),i=0,f=r.length-1,a=r[i],o=r[f],u,l,h=10;for(o<a&&(l=a,a=o,o=l,l=i,i=f,f=l);h-- >0;){if(l=Pe(a,o,t),l===u)return r[i]=a,r[f]=o,n(r);if(l>0)a=Math.floor(a/l)*l,o=Math.ceil(o/l)*l;else if(l<0)a=Math.ceil(a*l)/l,o=Math.floor(o*l)/l;else break;u=l}return e},e}function tn(){var e=gt();return e.copy=function(){return ht(e,tn())},ce.apply(e,arguments),At(e)}const Ct="";function rn(e,n,t){const r=e.slice();return r[11]=n[t],r}function an(e,n,t){const r=e.slice();return r[14]=n[t],r}function fn(e,n,t){const r=e.slice();return r[14]=n[t],r}function on(e){let n,t,r=e[14]+"",i,f;return{c(){n=$("g"),t=$("text"),i=X(r),x(t,"class","axis-text svelte-1vd4qdl"),x(t,"x","-5"),x(t,"y","0"),x(t,"text-anchor","end"),x(n,"transform",f=`translate(${e[8].left} ${e[6](e[14])+e[6].bandwidth()/2})`),x(n,"class","svelte-1vd4qdl")},m(a,o){B(a,n,o),R(n,t),R(t,i)},p(a,o){o&64&&f!==(f=`translate(${a[8].left} ${a[6](a[14])+a[6].bandwidth()/2})`)&&x(n,"transform",f)},d(a){a&&S(n)}}}function ln(e){let n,t,r=e[14]+"",i,f;return{c(){n=$("g"),t=$("text"),i=X(r),x(t,"class","axis-text svelte-1vd4qdl"),x(t,"y","15"),x(t,"text-anchor","middle"),x(n,"transform",f=`translate(${e[5](e[14])}, ${e[3]-e[8].bottom})`),x(n,"class","svelte-1vd4qdl")},m(a,o){B(a,n,o),R(n,t),R(t,i)},p(a,o){o&32&&r!==(r=a[14]+"")&&yn(i,r),o&40&&f!==(f=`translate(${a[5](a[14])}, ${a[3]-a[8].bottom})`)&&x(n,"transform",f)},d(a){a&&S(n)}}}function un(e){let n,t,r,i,f,a,o=e[11].avgRank.toFixed(2)+"",u,l,h;return{c(){n=$("rect"),a=$("text"),u=X(o),x(n,"y",t=e[6](e[11].model)),x(n,"x",e[8].left),x(n,"width",r=e[5](e[11].avgRank)-e[8].left),x(n,"height",i=e[6].bandwidth()),x(n,"fill",f=e[4](e[11].model)),x(n,"class","svelte-1vd4qdl"),x(a,"class","label-text svelte-1vd4qdl"),x(a,"y",l=e[6](e[11].model)+e[6].bandwidth()/2),x(a,"x",h=e[5](e[11].avgRank)+5),x(a,"text-anchor","start"),x(a,"dominant-baseline","middle")},m(c,s){B(c,n,s),B(c,a,s),R(a,u)},p(c,s){s&64&&t!==(t=c[6](c[11].model))&&x(n,"y",t),s&32&&r!==(r=c[5](c[11].avgRank)-c[8].left)&&x(n,"width",r),s&64&&i!==(i=c[6].bandwidth())&&x(n,"height",i),s&16&&f!==(f=c[4](c[11].model))&&x(n,"fill",f),s&64&&l!==(l=c[6](c[11].model)+c[6].bandwidth()/2)&&x(a,"y",l),s&32&&h!==(h=c[5](c[11].avgRank)+5)&&x(a,"x",h)},d(c){c&&S(n),c&&S(a)}}}function $t(e){let n,t,r,i,f,a,o,u,l=e[7].map(sn),h=[];for(let d=0;d<l.length;d+=1)h[d]=on(fn(e,l,d));let c=e[5].ticks(),s=[];for(let d=0;d<c.length;d+=1)s[d]=ln(an(e,c,d));let p=e[7],m=[];for(let d=0;d<p.length;d+=1)m[d]=un(rn(e,p,d));return{c(){n=bn("div"),t=$("svg");for(let d=0;d<h.length;d+=1)h[d].c();r=Ae();for(let d=0;d<s.length;d+=1)s[d].c();i=Ae();for(let d=0;d<m.length;d+=1)m[d].c();f=$("line"),x(f,"class","axis-line svelte-1vd4qdl"),x(f,"x1",e[8].left),x(f,"x2",a=e[2]-e[8].right),x(f,"y1",o=e[3]-e[8].bottom),x(f,"y2",u=e[3]-e[8].bottom),x(t,"width",e[1]),x(t,"height",e[0]),x(t,"class","svelte-1vd4qdl"),x(n,"id","bar-chart-holder"),x(n,"width",e[1]),x(n,"height",e[0]),x(n,"class","svelte-1vd4qdl")},m(d,y){B(d,n,y),R(n,t);for(let g=0;g<h.length;g+=1)h[g]&&h[g].m(t,null);R(t,r);for(let g=0;g<s.length;g+=1)s[g]&&s[g].m(t,null);R(t,i);for(let g=0;g<m.length;g+=1)m[g]&&m[g].m(t,null);R(t,f)},p(d,[y]){if(y&448){l=d[7].map(sn);let g;for(g=0;g<l.length;g+=1){const _=fn(d,l,g);h[g]?h[g].p(_,y):(h[g]=on(_),h[g].c(),h[g].m(t,r))}for(;g<h.length;g+=1)h[g].d(1);h.length=l.length}if(y&296){c=d[5].ticks();let g;for(g=0;g<c.length;g+=1){const _=an(d,c,g);s[g]?s[g].p(_,y):(s[g]=ln(_),s[g].c(),s[g].m(t,i))}for(;g<s.length;g+=1)s[g].d(1);s.length=c.length}if(y&496){p=d[7];let g;for(g=0;g<p.length;g+=1){const _=rn(d,p,g);m[g]?m[g].p(_,y):(m[g]=un(_),m[g].c(),m[g].m(t,f))}for(;g<m.length;g+=1)m[g].d(1);m.length=p.length}y&4&&a!==(a=d[2]-d[8].right)&&x(f,"x2",a),y&8&&o!==(o=d[3]-d[8].bottom)&&x(f,"y1",o),y&8&&u!==(u=d[3]-d[8].bottom)&&x(f,"y2",u),y&2&&x(t,"width",d[1]),y&1&&x(t,"height",d[0]),y&2&&x(n,"width",d[1]),y&1&&x(n,"height",d[0])},i:z,o:z,d(d){d&&S(n),re(h,d),re(s,d),re(m,d)}}}const sn=e=>e.model;function qt(e,n,t){let r,i,f,a,o,{data:u}=n,{height:l=400}=n,{width:h=700}=n;const c=u.reduce((m,d)=>(m[d.model]?(m[d.model].sum+=d.latency,m[d.model].count++):m[d.model]={sum:d.latency,count:1},m),{}),s=Object.keys(c).map(m=>({model:m,avgRank:c[m].sum/c[m].count}));let p={top:20,bottom:20,left:175,right:20};return e.$$set=m=>{"data"in m&&t(9,u=m.data),"height"in m&&t(0,l=m.height),"width"in m&&t(1,h=m.width)},e.$$.update=()=>{e.$$.dirty&2&&t(2,r=h-p.left-p.right),e.$$.dirty&1&&t(3,i=l-p.top-p.bottom),e.$$.dirty&8&&t(6,f=Ce().rangeRound([p.top,i-p.bottom]).padding(.05).domain(s.map(m=>m.model))),e.$$.dirty&4&&t(5,a=tn().rangeRound([p.left,r-p.right]).domain([0,Sn(s,m=>m.avgRank)]))},t(4,o=he().domain(s.map(m=>m.model)).range(["#FF5470","#1B2D45","#00EBC7","#FDE24F"])),[l,h,r,i,o,a,f,s,p,u]}class Et extends En{constructor(n){super(),qn(this,n,qt,$t,mn,{data:9,height:0,width:1})}}return Et}();
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

    def __call__(self, **kwargs: Any) -> "Barchart":
        """
        Call the component with the given kwargs.

        Parameters
        ----------
        kwargs : any
            The kwargs to pass to the component.

        Returns
        -------
        Barchart
            A python class representing the svelte component, renderable in Jupyter.
        """
        # render with given arguments
        self.add_params(kwargs)
        return self
