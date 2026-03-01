#!/usr/bin/env python3
import argparse, json, math
import numpy as np
import plotly.graph_objects as go

DH=[(0.000,0.333,0.0),(0.000,0.000,-math.pi/2),(0.000,0.316,math.pi/2),
    (0.0825,0.000,math.pi/2),(-0.0825,0.384,-math.pi/2),(0.000,0.000,math.pi/2),(0.088,0.000,math.pi/2)]

def dh(a,d,alpha,theta):
    ca,sa,ct,st=math.cos(alpha),math.sin(alpha),math.cos(theta),math.sin(theta)
    return np.array([[ct,-st*ca,st*sa],[st,ct*ca,-ct*sa],[0,sa,ca]]),np.array([a*ct,a*st,d])

def compute_fk(q):
    pos=[np.zeros(3)];ori=[np.eye(3)]
    for i,(a,d,alpha) in enumerate(DH):
        R,t=dh(a,d,alpha,q[i]);pos.append(pos[-1]+ori[-1]@t);ori.append(ori[-1]@R)
    return pos

def make_perp(n):
    v=np.array([1,0,0]) if abs(n[0])<0.9 else np.array([0,1,0])
    p=np.cross(n,v);p/=np.linalg.norm(p);return p,np.cross(n,p)

def capsule_mesh(p,q,r,N=12):
    axis=q-p;L=np.linalg.norm(axis)
    if L<1e-6:return None
    n=axis/L;u,v=make_perp(n)
    th=np.linspace(0,2*math.pi,N,endpoint=False)
    ring=np.array([math.cos(t)*u+math.sin(t)*v for t in th])
    verts=[]
    for ring_pt in ring:
        verts.append(p+r*ring_pt)
    for ring_pt in ring:
        verts.append(q+r*ring_pt)
    verts=np.array(verts)
    i_idx,j_idx,k_idx=[],[],[]
    for i in range(N):
        j=(i+1)%N
        # side quad → 2 triangles
        i_idx+=[i,i];j_idx+=[j,N+j];k_idx+=[N+i,N+i]
        i_idx+=[i,N+j];j_idx+=[j,N+i];k_idx+=[N+j,N+j]
    # caps
    cp=len(verts);verts=np.vstack([verts,p,q])
    for i in range(N):
        j=(i+1)%N
        i_idx+=[cp,j,cp+1];j_idx+=[i,N+i,N+j];k_idx+=[j,N+j,N+i]
    return verts,np.array(i_idx),np.array(j_idx),np.array(k_idx)

def sphere_mesh(pos,r,Nu=16,Nv=10):
    u=np.linspace(0,2*math.pi,Nu);v=np.linspace(0,math.pi,Nv)
    x=pos[0]+r*np.outer(np.cos(u),np.sin(v))
    y=pos[1]+r*np.outer(np.sin(u),np.sin(v))
    z=pos[2]+r*np.outer(np.ones_like(u),np.cos(v))
    return x.flatten(),y.flatten(),z.flatten()

def closest_on_seg(P,Q,x):
    seg=Q-P;l2=np.dot(seg,seg)+1e-12
    t=np.clip(np.dot(x-P,seg)/l2,0,1);return P+t*seg

def compute_min_sd(fkp,lcs,obs):
    best=1e9;bp_rob=bp_obs=None
    for lc in lcs:
        P=np.array(fkp[lc['joint_i']]);Q=np.array(fkp[lc['joint_j']]);r=lc['radius']
        for o in obs:
            c=np.array(o['pos']);p_rob=closest_on_seg(P,Q,c)
            if o['type']=='sphere':
                d=np.linalg.norm(p_rob-c);robs=o['radius']
                n=(p_rob-c)/(d+1e-12);p_obs=c+robs*n;p_rob2=p_rob-r*n;sd=d-r-robs
            elif o['type']=='capsule':
                w,x,y,z=o.get('quat',[1,0,0,0])
                R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                ax_=R@np.array([0,0,1]);hl=o['half_length']
                p_obs_ax=closest_on_seg(c-hl*ax_,c+hl*ax_,p_rob)
                d=np.linalg.norm(p_rob-p_obs_ax);robs=o['radius']
                n=(p_rob-p_obs_ax)/(d+1e-12);p_obs=p_obs_ax+robs*n;p_rob2=p_rob-r*n;sd=d-r-robs
            elif o['type']=='box':
                w,x,y,z=o.get('quat',[1,0,0,0])
                R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
                he=np.array(o['half_extents']);d_=p_rob-np.array(o['pos'])
                local=R.T@d_;clamped=np.clip(local,-he,he)
                p_obs=np.array(o['pos'])+R@clamped
                d=np.linalg.norm(p_rob-p_obs);n=(p_rob-p_obs)/(d+1e-12)
                p_rob2=p_rob-r*n;sd=d-r
            else:
                continue
            if sd<best:best=sd;bp_rob=p_rob2;bp_obs=p_obs
    return best,bp_rob,bp_obs

def sd_color(sd,ds):
    if sd<0:return 'red'
    if sd<ds:return 'orange'
    return 'limegreen'

def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--csv',required=True);pa.add_argument('--scene',required=True)
    pa.add_argument('--output',default='out/viewer.html')
    args=pa.parse_args()

    wps=np.loadtxt(args.csv,delimiter=',',skiprows=1)
    if wps.ndim==1:wps=wps[np.newaxis,:]
    with open(args.scene) as f:sc=json.load(f)
    obs=sc.get('obstacles',[]);lcs=sc['robot']['link_capsules'];ds=sc['planning']['d_safe']

    all_fk=[compute_fk(q) for q in wps]
    all_sd=[compute_min_sd(fk,lcs,obs) for fk in all_fk]
    ee=np.array([fk[-1] for fk in all_fk])

    frames=[]
    for fi,(fkp,(s,bp_rob,bp_obs)) in enumerate(zip(all_fk,all_sd)):
        traces=[]
        rc=sd_color(s,ds)

        # EE trail
        traces.append(go.Scatter3d(
            x=ee[:fi+1,0],y=ee[:fi+1,1],z=ee[:fi+1,2],
            mode='lines',line=dict(color='mediumpurple',width=3),
            name='EE path',showlegend=(fi==0)))

        # EE dot
        traces.append(go.Scatter3d(
            x=[ee[fi,0]],y=[ee[fi,1]],z=[ee[fi,2]],
            mode='markers',marker=dict(size=8,color='hotpink',
            symbol='circle',line=dict(color='white',width=1)),
            name='End-effector',showlegend=(fi==0)))

        # d_safe buffer spheres
        for o in obs:
            if o['type']=='sphere':
                x,y,z=sphere_mesh(o['pos'],o['radius']+ds)
                traces.append(go.Mesh3d(x=x,y=y,z=z,alphahull=0,
                    color='white',opacity=0.06,name='d_safe buffer',
                    showlegend=(fi==0),showscale=False))

        # Obstacles
        for o in obs:
            t=o['type']
            if t=='sphere':
                x,y,z=sphere_mesh(o['pos'],o['radius'])
                traces.append(go.Mesh3d(x=x,y=y,z=z,alphahull=0,
                    color='steelblue',opacity=0.55,name='Obstacle',
                    showlegend=(fi==0),showscale=False))
            elif t=='box':
                w,x,y,z2=o.get('quat',[1,0,0,0])
                R=np.array([[1-2*(y*y+z2*z2),2*(x*y-z2*w),2*(x*z2+y*w)],
                            [2*(x*y+z2*w),1-2*(x*x+z2*z2),2*(y*z2-x*w)],
                            [2*(x*z2-y*w),2*(y*z2+x*w),1-2*(x*x+y*y)]])
                he=np.array(o['half_extents'])
                signs=[[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                       [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]]
                C=np.array([(R@(np.array(s)*he))+np.array(o['pos']) for s in signs])
                edges=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
                for a,b in edges:
                    traces.append(go.Scatter3d(
                        x=[C[a,0],C[b,0]],y=[C[a,1],C[b,1]],z=[C[a,2],C[b,2]],
                        mode='lines',line=dict(color='cornflowerblue',width=4),
                        showlegend=False))
                traces.append(go.Mesh3d(
                    x=C[:,0],y=C[:,1],z=C[:,2],
                    i=[0,0,0,4,4,4],j=[1,2,3,5,6,7],k=[2,3,1,6,7,5],
                    color='steelblue',opacity=0.15,showlegend=False,showscale=False))
            elif t=='capsule':
                w,x,y,z2=o.get('quat',[1,0,0,0])
                R=np.array([[1-2*(y*y+z2*z2),2*(x*y-z2*w),2*(x*z2+y*w)],
                            [2*(x*y+z2*w),1-2*(x*x+z2*z2),2*(y*z2-x*w)],
                            [2*(x*z2-y*w),2*(y*z2+x*w),1-2*(x*x+y*y)]])
                ax_=R@np.array([0,0,1]);hl=o['half_length']
                cp=np.array(o['pos'])
                res=capsule_mesh(cp-hl*ax_,cp+hl*ax_,o['radius'])
                if res:
                    v,ii,jj,kk=res
                    traces.append(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],
                        i=ii,j=jj,k=kk,color='steelblue',opacity=0.55,
                        showlegend=False,showscale=False))

        # Robot capsules
        for lc in lcs:
            P=np.array(fkp[lc['joint_i']]);Q=np.array(fkp[lc['joint_j']])
            res=capsule_mesh(P,Q,lc['radius'])
            if res:
                v,ii,jj,kk=res
                traces.append(go.Mesh3d(x=v[:,0],y=v[:,1],z=v[:,2],
                    i=ii,j=jj,k=kk,color=rc,opacity=0.90,
                    name='Robot',showlegend=False,showscale=False))
            traces.append(go.Scatter3d(
                x=[P[0],Q[0]],y=[P[1],Q[1]],z=[P[2],Q[2]],
                mode='lines',line=dict(color='white',width=3),showlegend=False))

        # Joint dots
        pts=np.array(fkp)
        traces.append(go.Scatter3d(
            x=pts[:,0],y=pts[:,1],z=pts[:,2],
            mode='markers',marker=dict(size=5,color='white'),
            name='Joints',showlegend=False))

        # Closest distance line
        if bp_rob is not None and bp_obs is not None:
            traces.append(go.Scatter3d(
                x=[bp_rob[0],bp_obs[0]],y=[bp_rob[1],bp_obs[1]],z=[bp_rob[2],bp_obs[2]],
                mode='lines+markers',
                line=dict(color=rc,width=5,dash='dash'),
                marker=dict(size=6,color=rc),
                name=f'min_sd={s:.4f}m',showlegend=True))
            mid=(bp_rob+bp_obs)/2
            traces.append(go.Scatter3d(
                x=[mid[0]],y=[mid[1]],z=[mid[2]],
                mode='text',text=[f'{s:.3f}m'],
                textfont=dict(color=rc,size=14),
                showlegend=False))

        st='✓ SAFE' if s>=ds else ('△ NEAR' if s>=0 else '⚠ COLLISION')
        frames.append(go.Frame(
            data=traces,
            name=str(fi),
            layout=go.Layout(title=dict(
                text=f'Step {fi+1}/{len(wps)}  │  min_sd={s:+.4f}m  │  {st}',
                font=dict(size=16,color=rc),x=0.5))))

    # Initial frame
    init_traces=frames[0].data

    fig=go.Figure(
        data=init_traces,
        frames=frames,
        layout=go.Layout(
            title=dict(text=f'Step 1/{len(wps)}',font=dict(size=16),x=0.5),
            paper_bgcolor='#0d0d1f',
            plot_bgcolor='#0d0d1f',
            scene=dict(
                bgcolor='#0d0d1f',
                xaxis=dict(range=[-0.25,0.9],title='X (m)',color='#aaaacc',gridcolor='#2a2a4a'),
                yaxis=dict(range=[-0.7,0.7],title='Y (m)',color='#aaaacc',gridcolor='#2a2a4a'),
                zaxis=dict(range=[0,1.3],title='Z (m)',color='#aaaacc',gridcolor='#2a2a4a'),
                aspectmode='manual',
                aspectratio=dict(x=1.15,y=1.15,z=1.0),
                camera=dict(eye=dict(x=1.4,y=-1.4,z=0.8))
            ),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                y=0.02,x=0.5,xanchor='center',
                buttons=[
                    dict(label='▶ Play',method='animate',
                         args=[None,dict(frame=dict(duration=int(1000/6),redraw=True),
                                         fromcurrent=True,mode='immediate')]),
                    dict(label='⏸ Pause',method='animate',
                         args=[[None],dict(frame=dict(duration=0,redraw=False),
                                            mode='immediate')])
                ])],
            sliders=[dict(
                steps=[dict(method='animate',args=[[str(i)],
                    dict(mode='immediate',frame=dict(duration=0,redraw=True))],
                    label=str(i+1)) for i in range(len(wps))],
                active=0,y=0.06,x=0.05,len=0.9,
                currentvalue=dict(prefix='Step: ',font=dict(color='white')),
                font=dict(color='white'),
                bgcolor='#222244',bordercolor='#444466'
            )],
            font=dict(color='white'),
            legend=dict(bgcolor='#1a1a35',bordercolor='#444466',font=dict(color='white'))
        )
    )

    fig.write_html(args.output,include_plotlyjs='cdn')
    print(f"[viewer] ✓ saved → {args.output}")
    print(f"[viewer]   ブラウザで開いてマウスで自由回転できます")

if __name__=='__main__':main()
