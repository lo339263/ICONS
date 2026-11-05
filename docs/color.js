"use strict";
const Color = (() => {
  // CIE 1964 10-degree xy coordinates, normalized to Y = 100.
  // Source: Colour illuminants/chromaticity_coordinates dataset.
  const xyWhite = (x,y) => [100*x/y,100,100*(1-x-y)/y].map(v=>Math.round(v*1000)/1000);
  const whitepoints = {D65:xyWhite(.31382,.33100)};
  // solve(LMS, [1,1,1]), normalized to Y=100; matches util.py.
  const modelWhite = [95.0470287776504,100,108.883115044617];
  const srgbWhite = [95.047,100,108.883];
  const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
  const mul = (m,v) => m.map(r=>dot(r,v));
  const clamp = v => Math.max(0,Math.min(1,v));
  const angle = (a,b) => (Math.atan2(b,a)*180/Math.PI+360)%360;
  const CAT = [[.401288,.650173,-.051461],[-.250268,1.204414,.045854],[-.002079,.048952,.953127]];
  const CAT_INV = [[1.8620678550872325,-1.0112546305316843,.14918677544445175],[.3875265432361372,.6214474419314753,-.008973985167612518],[-.01584149884933386,-.03412293802851557,1.0499644368778496]];
  const LMS = [[.400227052325915,.707583738918745,-.080809279274243],[-.226301727628935,1.165334160925190,.045699378859811],[0,0,.918416046041877]];
  const IAB = [[65.57377049,32.78688525,1.63934426],[430,-470,40],[49,49,-98]];
  const matrix = [[3.2406,-1.5372,-.4986],[-.9689,1.8758,.0415],[.0557,-.204,1.057]];
  const f = t => t > 216/24389 ? Math.cbrt(t) : (24389/27*t+16)/116;
  const fi = t => t > 6/29 ? t**3 : 3*(6/29)**2*(t-4/29);
  function labToXYZ([L,a,b],w) {const y=(L+16)/116; return [y+a/500,y,y-b/200].map((v,i)=>fi(v)*w[i]);}
  function xyzToLab(x,w) {const [a,b,c]=x.map((v,i)=>f(v/w[i]));return [116*b-16,500*(a-b),200*(b-c)];}
  function adapt(x,w) {
    const src=mul(CAT,w), ref=mul(CAT,modelWhite);
    // Full adaptation preserves the supplied whitepoint's neutral axis.
    return mul(CAT_INV,mul(CAT,x).map((v,i)=>v*w[1]*ref[i]/(100*src[i])));
  }
  function appearance(x,w,la,yb,surround) {
    const c={avg:.52,dim:.50,dark:.39}[surround];
    const adapted=adapt(x,w);
    let [I,a,b]=mul(IAB,mul(LMS,adapted.map(v=>v/100)).map(v=>Math.sign(v)*Math.abs(v)**.43));
    if(Math.hypot(a,b)<1e-10){a=0;b=0;}
    const C=Math.log1p(.0447*Math.hypot(a,b))/.0252, J=100*(I/100)**(c*(1.48+Math.sqrt(yb/w[1]))), h=angle(a,b);
    const hi=[16.5987,80.2763,157.779,219.7174,376.5987], ei=[.7,.6,1.2,.9,.7], hm=h<hi[0]?h+360:h;
    let i=0; while(i<3&&hm>=hi[i+1])i++;
    const t=(hm-hi[i])/ei[i], u=(hi[i+1]-hm)/ei[i+1], H=100*i+100*t/(t+u);
    const D=1.3*Math.sqrt((100-J)**2+1.6*C*C);
    const n=H%400, q=Math.floor(n/100), ncs=n%100===0?["R","Y","G","B"][q]:["Y","G","B","R"][q]+Math.round(100-n%100)+["R","Y","G","B"][q];
    const depth=["Very pale","Pale","Mid depth","Deep","Very Deep"][Math.min(4,Math.floor(D/20))];
    return {JCh:[J,C,h],H,D,ncs,depth,sucs:[I,C*Math.cos(h*Math.PI/180),C*Math.sin(h*Math.PI/180)]};
  }
  // Cholesky factorization keeps the Gaussian calculation compact and stable.
  const factors=GAUSSIAN.covariances.map(m=>{
    const L=Array.from({length:3},()=>[0,0,0]);
    for(let i=0;i<3;i++)for(let j=0;j<=i;j++){
      let v=m[i][j];for(let k=0;k<j;k++)v-=L[i][k]*L[j][k];
      L[i][j]=i===j?Math.sqrt(v):v/L[j][j];
    } return L;
  });
  function predict(lab) {
    const scores=GAUSSIAN.means.map((mean,k)=>{
      const L=factors[k], z=[0,0,0];
      for(let i=0;i<3;i++){let v=lab[i]-mean[i];for(let j=0;j<i;j++)v-=L[i][j]*z[j];z[i]=v/L[i][i];}
      return Math.log(GAUSSIAN.priors[k])-.5*dot(z,z)-L.reduce((s,r,i)=>s+Math.log(r[i]),0);
    });
    const weights=scores.map(v=>Math.exp(v-Math.max(...scores))), total=weights.reduce((a,b)=>a+b);
    return GAUSSIAN.classes.map((name,i)=>({name,probability:weights[i]/total})).sort((a,b)=>b.probability-a.probability);
  }
  function calculate(mode,values,white=whitepoints.D65,la=64,yb=20,surround="avg") {
    if(!["lab","xyz"].includes(mode)||values.length!==3||white.length!==3||![...values,...white,la,yb].every(Number.isFinite))throw Error("Enter three finite values for the colour and reference white.");
    if(white.some(v=>v<=0)||la<0||yb<=0||!["avg","dim","dark"].includes(surround))throw Error("White-point values and Yb must be positive; La must be nonnegative; select a valid surround.");
    if(mode==="lab"&&(values[0]<0||values[0]>100))throw Error("L* must be between 0 and 100.");
    if(mode==="xyz"&&values.some(v=>v<0))throw Error("XYZ must be nonnegative.");
    const xyz=mode==="lab"?labToXYZ(values,white):values.slice(), lab=mode==="lab"?values.slice():xyzToLab(xyz,white);
    const ap=appearance(xyz,white,la,yb,surround);
    if(!ap.JCh.every(Number.isFinite))throw Error("Cannot calculate a finite colour appearance. Check the coordinates.");
    // Full CAT16 adaptation to D65 is for the screen preview only.
    const previewXYZ=mul(CAT_INV,mul(CAT,xyz).map((v,i)=>v*mul(CAT,srgbWhite)[i]/mul(CAT,white)[i]));
    const rgb=mul(matrix,previewXYZ.map(v=>v/100)).map(v=>v<=.0031308?12.92*v:1.055*v**(1/2.4)-.055);
    const clipped=rgb.map(clamp), hex="#"+clipped.map(v=>Math.round(v*255).toString(16).padStart(2,"0")).join("");
    const ranked=predict(lab), sum=xyz.reduce((a,b)=>a+b), C=Math.hypot(lab[1],lab[2]);
    const name=ranked[0].name, uname=ap.JCh.map(v=>String(Math.floor(v)).padStart(3,"0")).join("")+ap.ncs+ap.depth.replaceAll(" ","")+name;
    return {name,confidence:ranked[0].probability,ranked,lab,xyz,white,La:la,Yb:yb,surround,LCh:[lab[0],C,angle(lab[1],lab[2])],
      xyY:sum?[xyz[0]/sum,xyz[1]/sum,xyz[1]]:[null,null,0],...ap,rgb:clipped,hex,outOfGamut:rgb.some(v=>v<-.001||v>1.001),u_name:uname};
  }
  return {calculate,predict,whitepoints,labToXYZ,xyzToLab};
})();
