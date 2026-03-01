#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

DH=[(0.000,0.333,0.0),(0.000,0.000,-math.pi/2),(0.000,0.316,math.pi/2),
    (0.0825,0.000,math.pi/2),(-0.0825,0.384,-math.pi/2),(0.000,0.000,math.pi/2),(0.088,0.000,math.pi/2)]

def dh(a,d,alpha,theta):
    ca,sa,ct,st=math.cos(alpha),math.sin(alpha),math.cos(theta),math.sin(theta)
    return np.array([[ct,-st*ca,st*sa],[st,ct*ca,-ct*sa],[0,sa,ca]]),np.array([a*ct,a*st,d])

def compute_fk(q):
    pos=[np.zeros(3)];ori=[np.eye(3)]
    for i,(a,d,alpha) in enumerate(DH):
        R,t=dh(a,d,alpha,q[i]);pos.append(pos[-1]+ori[-1]@t);ori.append(ori[-1]@R)
    return pos,ori

def make_perp(n):
    v=np.array([1,0,0]) if abs(n[0])<0.9 else np.array([0,1,0])
    p=np.cross(n,v);p/=np.linalg.norm(p);return p,np.cross(n,p)

def closest_on_seg(P,Q,x):
    seg=Q-P;l2=np.dot(seg,seg)+1e-12
    t=np.clip(np.dot(x-P,seg)/l2,0,1)
    return P+t*seg

def compute_contacts(fkp,lcs,obs):
    """全リンク×全障害物の最近点ペアを返す。最悪ペアも返す。"""
    contacts=[]
    for lc in lcs:
        P=np.array(fkp[lc['joint_i']]);Q=np.array(fkp[lc['joint_j']]);r=lc['radius']
        for o in obs:
            c=np.array(o['pos'])
            p_rob=closest_on_seg(P,Q,c)
            if o['type']=='sphere':
                d=np.linalg.norm(p_rob-c);robs=o['radius']
                n=(p_rob-c)/(d+1e-12);p_obs=c+robs*n;p_rob2=p_rob-r*n
                sd=d-r-robs
            elif o['type']=='capsule':
                w,x,y,z=o.get('quat',[1,0,0,0])
                R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                ax_=R@np.array([0,0,1]);hl=o['half_length']
                p_obs_ax=closest_on_seg(c-hl*ax_,c+hl*ax_,p_rob)
                d=np.linalg.norm(p_rob-p_obs_ax);robs=o['radius']
                n=(p_rob-p_obs_ax)/(d+1e-12);p_obs=p_obs_ax+robs*n;p_rob2=p_rob-r*n
                sd=d-r-robs
            elif o['type']=='box':
                w,x,y,z=o.get('quat',[1,0,0,0])
                R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                he=np.array(o['half_extents']);d_=p_rob-np.array(o['pos'])
                local=R.T@d_;clamped=np.clip(local,-he,he);p_obs=np.array(o['pos'])+R@clamped
                d=np.linalg.norm(p_rob-p_obs);n=(p_rob-p_obs)/(d+1e-12)
                p_rob2=p_rob-r*n;sd=d-r
            else:
                continue
            contacts.append(dict(sd=sd,p_obs=p_obs,p_rob=p_rob2,n=n))
    contacts.sort(key=lambda c:c['sd'])
    return contacts

def draw_capsule(ax,p,q,r,col,alpha=0.90,N=16):
    axis=q-p;L=np.linalg.norm(axis)
    if L<1e-6:return
    n=axis/L;u,v=make_perp(n)
    th=np.linspace(0,2*math.pi,N,endpoint=False)
    ring=np.array([math.cos(t)*u+math.sin(t)*v for t in th])
    sides=[[p+r*ring[i],p+r*ring[(i+1)%N],q+r*ring[(i+1)%N],q+r*ring[i]] for i in range(N)]
    pc=Poly3DCollection(sides,alpha=alpha,linewidth=0);pc.set_facecolor(col);pc.set_edgecolor('none');ax.add_collection3d(pc)
    for c in [p,q]:
        cc=Poly3DCollection([[c+r*ring[i],c+r*ring[(i+1)%N],c] for i in range(N)],alpha=alpha,linewidth=0)
        cc.set_facecolor(col);cc.set_edgecolor('none');ax.add_collection3d(cc)

def draw_sphere(ax,pos,r,col,a=0.50):
    u=np.linspace(0,2*math.pi,24);v=np.linspace(0,math.pi,14)
    ax.plot_surface(pos[0]+r*np.outer(np.cos(u),np.sin(v)),
                    pos[1]+r*np.outer(np.sin(u),np.sin(v)),
                    pos[2]+r*np.outer(np.ones_like(u),np.cos(v)),
                    color=col,alpha=a,linewidth=0)

def draw_box(ax,pos,qw,he,col):
    w,x,y,z=qw
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    C=np.array([(R@(np.array(s)*np.array(he)))+pos
                for s in [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                           [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]])
    faces=[[C[0],C[1],C[2],C[3]],[C[4],C[5],C[6],C[7]],
           [C[0],C[1],C[5],C[4]],[C[2],C[3],C[7],C[6]],
           [C[1],C[2],C[6],C[5]],[C[0],C[3],C[7],C[4]]]
    fc=Poly3DCollection(faces,alpha=0.15,linewidth=0);fc.set_facecolor(col);fc.set_edgecolor('none');ax.add_collection3d(fc)
    for a,b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        ax.plot3D(*zip(C[a],C[b]),color=col,alpha=1.0,lw=2.5)

def draw_obs_capsule(ax,pos,qw,r,hl,col):
    w,x,y,z=qw
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    ax_=R@np.array([0,0,1])
    draw_capsule(ax,np.array(pos)-hl*ax_,np.array(pos)+hl*ax_,r,col,0.50)

def sd_color(sd,ds):
    return '#e74c3c' if sd<0 else ('#f39c12' if sd<ds else '#2ecc71')

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--csv',required=True);pa.add_argument('--scene',required=True)
    pa.add_argument('--output',default='out/plan.gif');pa.add_argument('--fps',type=int,default=6)
    pa.add_argument('--dpi',type=int,default=140);pa.add_argument('--elev',type=float,default=28)
    pa.add_argument('--azim',type=float,default=-50)
    args=pa.parse_args()

    wps=np.loadtxt(args.csv,delimiter=',',skiprows=1)
    if wps.ndim==1:wps=wps[np.newaxis,:]
    with open(args.scene) as f:sc=json.load(f)
    obs=sc.get('obstacles',[]);lcs=sc['robot']['link_capsules'];ds=sc['planning']['d_safe']

    all_fk=[compute_fk(q) for q in wps]
    all_contacts=[compute_contacts(f[0],lcs,obs) for f in all_fk]
    all_sd=[c[0]['sd'] if c else 1e9 for c in all_contacts]
    ee=np.array([f[0][-1] for f in all_fk])

    fd=Path(args.output).parent/'frames';fd.mkdir(parents=True,exist_ok=True)
    print(f"[render] {len(wps)} frames | fps={args.fps} | dpi={args.dpi}")

    frame_paths=[]
    for fi,((fkp,_),contacts,s) in enumerate(zip(all_fk,all_contacts,all_sd)):
        fig=plt.figure(figsize=(11,8),facecolor='#0d0d1f')
        ax=fig.add_subplot(111,projection='3d');ax.set_facecolor('#0d0d1f')
        ax.view_init(elev=args.elev,azim=args.azim)
        ax.set_xlim(-0.2,0.9);ax.set_ylim(-0.7,0.7);ax.set_zlim(0.0,1.3)
        for sp in [ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:
            sp.fill=False;sp.set_edgecolor('#2a2a4a')
        ax.tick_params(colors='#8888aa',labelsize=8)
        ax.set_xlabel('X (m)',color='#8888aa',labelpad=6)
        ax.set_ylabel('Y (m)',color='#8888aa',labelpad=6)
        ax.set_zlabel('Z (m)',color='#8888aa',labelpad=6)

        # EE trail
        past=ee[:fi+1]
        if len(past)>1:
            for k in range(len(past)-1):
                a_t=0.15+0.75*(k/max(len(past)-1,1))
                ax.plot([past[k,0],past[k+1,0]],[past[k,1],past[k+1,1]],
                        [past[k,2],past[k+1,2]],'-',color='#a29bfe',lw=2.0,alpha=a_t)
        ax.scatter(*ee[fi],c='#fd79a8',s=120,zorder=15,depthshade=False,
                   edgecolors='white',linewidths=1.2)

        # d_safe バッファ球（障害物ごと）
        for o in obs:
            if o['type']=='sphere':
                draw_sphere(ax,o['pos'],o['radius']+ds,'#ffffff',0.05)

        # 障害物
        for o in obs:
            t=o['type']
            if t=='sphere':draw_sphere(ax,o['pos'],o['radius'],'#0984e3',0.55)
            elif t=='box':draw_box(ax,o['pos'],o.get('quat',[1,0,0,0]),o['half_extents'],'#74b9ff')
            elif t=='capsule':draw_obs_capsule(ax,o['pos'],o.get('quat',[1,0,0,0]),o['radius'],o['half_length'],'#0984e3')

        # ロボット
        rc=sd_color(s,ds)
        for lc in lcs:
            P=np.array(fkp[lc['joint_i']]);Q=np.array(fkp[lc['joint_j']])
            draw_capsule(ax,P,Q,lc['radius'],rc,alpha=0.90)
            ax.plot3D(*zip(P,Q),color='white',lw=2.0,alpha=1.0,zorder=6)
        pts=np.array(fkp)
        ax.scatter(pts[:,0],pts[:,1],pts[:,2],c='white',s=55,zorder=11,depthshade=False)

        # ★ 最近点ペアを線で描画（上位3ペア）
        for i,ct in enumerate(contacts[:3]):
            pr=ct['p_rob'];po=ct['p_obs'];sd_=ct['sd']
            lc_=sd_color(sd_,ds)
            lw_=3.0 if i==0 else 1.5
            alpha_=1.0 if i==0 else 0.4
            # 距離線
            ax.plot3D(*zip(pr,po),color=lc_,lw=lw_,alpha=alpha_,
                      linestyle='--',zorder=20)
            # witness点
            ax.scatter(*pr,c=lc_,s=60 if i==0 else 30,zorder=21,depthshade=False)
            ax.scatter(*po,c=lc_,s=60 if i==0 else 30,zorder=21,
                       marker='x',depthshade=False)
            # 最近ペアにsd値をラベル
            if i==0:
                mid=(np.array(pr)+np.array(po))/2
                ax.text(mid[0],mid[1],mid[2],f" {sd_:.3f}m",
                        color=lc_,fontsize=9,fontweight='bold',zorder=22)

        # タイトル
        st="⚠ COLLISION" if s<0 else("△ NEAR" if s<ds else "✓ SAFE")
        rc2=sd_color(s,ds)
        pct=(fi+1)/len(wps)*100
        fig.text(0.5,0.97,
                 f"Step {fi+1:02d}/{len(wps)}   │   min_sd={s:+.4f}m   │   {st}   │   {pct:.0f}%",
                 ha='center',va='top',color='white',fontsize=13,fontweight='bold',
                 bbox=dict(facecolor=rc2,alpha=0.25,edgecolor=rc2,boxstyle='round,pad=0.5'))

        # プログレスバー
        bar=fig.add_axes([0.12,0.04,0.78,0.018])
        bar.set_xlim(0,1);bar.set_ylim(0,1);bar.set_facecolor('#222244')
        bar.set_xticks([]);bar.set_yticks([])
        bar.barh(0.5,(fi+1)/len(wps),height=1.0,color=rc2,alpha=0.85)
        bar.text(0.5,0.5,f"{pct:.0f}%",ha='center',va='center',
                 color='white',fontsize=9,fontweight='bold')

        patches=[
            mpatches.Patch(color='#2ecc71',label=f'✓ SAFE  sd≥{ds}m'),
            mpatches.Patch(color='#f39c12',label=f'△ NEAR  0≤sd<{ds}m'),
            mpatches.Patch(color='#e74c3c',label='⚠ COLLISION  sd<0'),
            mpatches.Patch(color='#0984e3',label='Obstacle'),
            mpatches.Patch(color='#a29bfe',label='EE path'),
            plt.Line2D([0],[0],color='#2ecc71',lw=2,ls='--',label='Closest distance'),
        ]
        ax.legend(handles=patches,loc='upper left',fontsize=8.5,
                  framealpha=0.40,facecolor='#1a1a35',edgecolor='#444466',labelcolor='white')

        fig.subplots_adjust(top=0.93,bottom=0.08,left=0.02,right=0.98)
        fp2=fd/f"frame_{fi:04d}.png"
        fig.savefig(fp2,dpi=args.dpi,bbox_inches='tight',facecolor=fig.get_facecolor())
        plt.close(fig);frame_paths.append(fp2)
        if (fi+1)%10==0:print(f"  {fi+1}/{len(wps)}")

    from PIL import Image
    imgs=[Image.open(f) for f in frame_paths]
    ms=int(1000/args.fps);dur=[ms]*len(imgs)
    dur[0]=ms*4;dur[-1]=ms*6
    imgs[0].save(args.output,save_all=True,append_images=imgs[1:],loop=0,duration=dur,optimize=False)
    print(f"[render] ✓ {args.output}  ({len(imgs)} frames, {args.fps} fps)")

if __name__=='__main__':main()
