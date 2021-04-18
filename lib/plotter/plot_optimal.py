from ..planner import Frenet
from ..trajectory import SplineTrajectory2D
from matplotlib import pyplot as plt
import numpy as np

def plot_4_paths_lst(path_lst1: [[Frenet]], path_lst2: [[Frenet]], path_lst3: [[Frenet]], path_lst4: [[Frenet]], sp: SplineTrajectory2D,target: Frenet, three:bool=True):
    max_cd, max_cv, max_ctot, max_ct = 1e6, 1e6, 1e6, 1e6
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    for plst1,plst2,plst3 in zip(path_lst1,path_lst2,path_lst3):
        min_cd = max_cd
        min_cv = max_cv
        min_ct = max_ct
        min_path_cd = None
        min_path_cv = None
        min_path_ct = None
        for path1,path2, path3 in zip(plst1, plst2, plst3):

            if path1.cd < min_cd:
                min_cd = path1.cd
                min_path_cd = path1
            
            if path2.cv < min_cv:
                min_cv = path2.cv
                min_path_cv = path2

            if path3.ct < min_ct:
                min_ct= path3.ct
                min_path_ct = path3
            
            color = 'k'
            if path1.cd > 30:
                color = 'gray'
            if path2.cv > 30:
                color = 'gray'
            if path3.ct > 30:
                color = 'gray'

            ax1.plot(path1.t, path1.d, color, linewidth= 0.5)#c=cm.gnuplot(path.cd / 25))
            ax1.set_xlabel("t/s")
            ax1.set_ylabel("d/m")
            ax2.plot(path2.t, path2.dot_s, color, linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            ax2.set_xlabel("t/s")
            ax2.set_ylabel("$\dot{s}$/m/s")
            ax3.plot(path3.t, path3.s, color, linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            ax3.set_xlabel("t/s")
            ax3.set_ylabel("s/m")
            
            if three:
                ax1.plot(path1.t[0], path1.d[0], "og")
                ax2.plot(path2.t[0], path2.dot_s[0], "og")
                ax3.plot(path3.t[0], path3.s[0], "og")
            
        ax1.plot(min_path_cd.t, min_path_cd.d, '-g', linewidth=2)
        ax2.plot(min_path_cv.t, min_path_cv.dot_s, '-g', linewidth=2)
        ax3.plot(min_path_ct.t, min_path_ct.s, '-g', linewidth=2)
        ax3.plot(target.t, target.s, '-b', linewidth=2)

    for plst in path_lst4:
        min_ctot = max_ctot
        min_path = None
        s = list(np.arange(0, sp.s[-1], 0.1))
        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = sp.compute_pt(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))
            rk.append(sp.compute_curvature(i_s))
        ax4.plot(rx, ry, '-k', linewidth=2)
        for path in plst:
            if path.ctot < min_ctot:
                min_ctot = path.ctot
                min_path = path
            color = 'gray'
            good_lim = 7
            gray_lim = 6
            yellow_lim = 1.4
            if path.ctot > good_lim:
                continue
            if path.ctot > gray_lim:
                color = 'gray'
            if path.ctot < gray_lim and path.ctot > yellow_lim:
                color = 'red'
            if path.ctot < yellow_lim and path.ctot > 0:
                color = 'yellow'
            path_x = [x for i,(x,y,d) in enumerate(zip(path.x,path.y, path.d)) if x<=21.5 and d>0]
            path_y = [y for i,(x,y,d) in enumerate(zip(path.x,path.y, path.d)) if x<=21.5 and d>0]
            ax4.plot(path_x, path_y, color, linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            path_x = [path_x[-1]]+[x for i,(x,y,d) in enumerate(zip(path.x,path.y,path.d)) if (x>=21.5 or d<0) and not path.ctot>gray_lim-4]
            path_y = [path_y[-1]]+[y for i,(x,y,d) in enumerate(zip(path.x,path.y,path.d)) if (x>=21.5 or d<0)and not path.ctot>gray_lim-4]
            ax4.plot(path_x, path_y, 'k', linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            path_x = [path_x[-1]]+[x for i,(x,y,d) in enumerate(zip(path.x,path.y,path.d)) if (x>=21.5 or d<0) and path.ctot>gray_lim-4]
            path_y = [path_y[-1]]+[y for i,(x,y,d) in enumerate(zip(path.x,path.y,path.d)) if (x>=21.5 or d<0)and path.ctot>gray_lim-4]
            ax4.plot(path_x, path_y, 'gray', linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            ax4.set_xlabel("x/m")
            ax4.set_ylabel("y/m")
            ax4.set_xlim(-3,42)
            ax4.set_ylim(-14,3)
        min_path_x = [x for x in min_path.x if x<=19]
        min_path_y = [y for x,y in zip(min_path.x,min_path.y) if x<=19]
        ax4.plot(min_path_x, min_path_y, '-g', linewidth=2) 
        min_path_x = [min_path_x[-1]]+[x for x in min_path.x if x>=19]
        min_path_y = [min_path_y[-1]]+[y for x,y in zip(min_path.x,min_path.y) if x>=19]
        ax4.plot(min_path_x, min_path_y, 'gray', linewidth=2) 
        
    return fig

def plot_3_paths_lst(path_lst1: [[Frenet]], path_lst2: [[Frenet]], path_lst4: [[Frenet]], sp: SplineTrajectory2D, three:bool=True):
    max_cd, max_cv, max_ctot = 1e6, 1e6, 1e6
    fig, (ax1, ax2, ax4) = plt.subplots(3)
    for plst1,plst2 in zip(path_lst1,path_lst2):
        min_cd = max_cd
        min_cv = max_cv
        min_path_cd = None
        min_path_cv = None
        for path1,path2 in zip(plst1, plst2):

            if path1.cd < min_cd:
                min_cd = path1.cd
                min_path_cd = path1
            
            if path2.cv < min_cv:
                min_cv = path2.cv
                min_path_cv = path2

            
            color = 'k'
            if path1.cd > 30:
                color = 'gray'
            if path2.cv > 30:
                color = 'gray'
           

            ax1.plot(path1.t, path1.d, color, linewidth= 0.5)#c=cm.gnuplot(path.cd / 25))
            ax1.set_xlabel("t/s")
            ax1.set_ylabel("d/m")
            ax2.plot(path2.t, path2.dot_s, color, linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            ax2.set_xlabel("t/s")
            ax2.set_ylabel("$\dot{s}$/m/s")
           
            
            if three:
                ax1.plot(path1.t[0], path1.d[0], "og")
                ax2.plot(path2.t[0], path2.dot_s[0], "og")
            
        ax1.plot(min_path_cd.t, min_path_cd.d, '-g', linewidth=2)
        ax2.plot(min_path_cv.t, min_path_cv.dot_s, '-g', linewidth=2)

    for plst in path_lst4:
        min_ctot = max_ctot
        min_path = None
        s = list(np.arange(0, sp.s[-1], 0.1))
        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = sp.compute_pt(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))
            rk.append(sp.compute_curvature(i_s))
        ax4.plot(rx, ry, '-k', linewidth=2)
        for path in plst:
            if path.ctot < min_ctot:
                min_ctot = path.ctot
                min_path = path
            color = 'gray'
            good_lim = 7
            gray_lim = 6
            yellow_lim = 1.4
            if path.ctot > good_lim:
                continue
            if path.ctot > gray_lim:
                color = 'gray'
            if path.ctot < gray_lim and path.ctot > yellow_lim:
                color = 'red'
            if path.ctot < yellow_lim and path.ctot > 0:
                color = 'yellow'
            path_x = [x for i,(x,y,d) in enumerate(zip(path.x,path.y, path.d)) if x<=21.5 and d>0]
            path_y = [y for i,(x,y,d) in enumerate(zip(path.x,path.y, path.d)) if x<=21.5 and d>0]
            ax4.plot(path_x, path_y, color, linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            path_x = [path_x[-1]]+[x for i,(x,y,d) in enumerate(zip(path.x,path.y,path.d)) if (x>=21.5 or d<0) and not path.ctot>gray_lim-4]
            path_y = [path_y[-1]]+[y for i,(x,y,d) in enumerate(zip(path.x,path.y,path.d)) if (x>=21.5 or d<0)and not path.ctot>gray_lim-4]
            ax4.plot(path_x, path_y, 'k', linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            path_x = [path_x[-1]]+[x for i,(x,y,d) in enumerate(zip(path.x,path.y,path.d)) if (x>=21.5 or d<0) and path.ctot>gray_lim-4]
            path_y = [path_y[-1]]+[y for i,(x,y,d) in enumerate(zip(path.x,path.y,path.d)) if (x>=21.5 or d<0)and path.ctot>gray_lim-4]
            ax4.plot(path_x, path_y, 'gray', linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            ax4.set_xlabel("x/m")
            ax4.set_ylabel("y/m")
            ax4.set_xlim(-3,42)
            ax4.set_ylim(-14,3)
        min_path_x = [x for x in min_path.x if x<=19]
        min_path_y = [y for x,y in zip(min_path.x,min_path.y) if x<=19]
        ax4.plot(min_path_x, min_path_y, '-g', linewidth=2) 
        min_path_x = [min_path_x[-1]]+[x for x in min_path.x if x>=19]
        min_path_y = [min_path_y[-1]]+[y for x,y in zip(min_path.x,min_path.y) if x>=19]
        ax4.plot(min_path_x, min_path_y, 'gray', linewidth=2) 
        
    return fig


def plot_lateral(path_lst1: [[Frenet]],three:bool=True):
    max_cd = 1e6
    fig, ax1 = plt.subplots(1)
    for plst1 in path_lst1:
        min_cd = max_cd
        min_path_cd = None
        for path1 in plst1:

            if path1.cd < min_cd:
                min_cd = path1.cd
                min_path_cd = path1
            
            color = 'k'
            if path1.cd > 30:
                color = 'gray'
        
            ax1.plot(path1.t, path1.d, color, linewidth= 0.5)#c=cm.gnuplot(path.cd / 25))
            ax1.set_xlabel("t/s")
            ax1.set_ylabel("d/m")
          
            if three:
                ax1.plot(path1.t[0], path1.d[0], "og")
            
        ax1.plot(min_path_cd.t, min_path_cd.d, '-g', linewidth=2) 
    return fig

def plot_long( path_lst2: [[Frenet]], three:bool=True):
    max_cv = 1e6
    fig, ax2 = plt.subplots(1)
    for plst2 in path_lst2:
        
        min_cv = max_cv
        min_path_cv = None
        for path2 in plst2:

            if path2.cv < min_cv:
                min_cv = path2.cv
                min_path_cv = path2

            
            color = 'k'
            if path2.cv > 30:
                color = 'gray'

            ax2.plot(path2.t, path2.dot_s, color, linewidth =0.5)#c=cm.gnuplot(path.cv / 25))
            ax2.set_xlabel("t/s")
            ax2.set_ylabel("$\dot{s}$/m/s")
           
            
            if three:
                ax2.plot(path2.t[0], path2.dot_s[0], "og")
            
        ax2.plot(min_path_cv.t, min_path_cv.dot_s, '-g', linewidth=2)
    return fig

def plot_target(path_lst3: [[Frenet]],target: Frenet, three:bool=True):
    max_ct = 1e6
    fig, ax3 = plt.subplots(1)
    for plst3 in path_lst3:
        min_ct = max_ct
        min_path_ct = None
        for path3 in plst3:

            if path3.ct < min_ct:
                min_ct= path3.ct
                min_path_ct = path3
            
            color = 'k'
            if path3.ct > 30:
                color = 'gray'

            ax3.plot(path3.t, path3.s, color, linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            ax3.set_xlabel("t/s")
            ax3.set_ylabel("s/m")
            
            if three:
    
                ax3.plot(path3.t[0], path3.s[0], "og")
            
        ax3.plot(min_path_ct.t, min_path_ct.s, '-g', linewidth=2)
        ax3.plot(target.t, target.s, '-b', linewidth=2)
        
    return fig

def plot_xy(path_lst4: [[Frenet]],sp: SplineTrajectory2D,three:bool=True):
    max_ctot = 1e6
    fig, ax4 = plt.subplots(1)

    for plst in path_lst4:
        min_ctot = max_ctot
        min_path = None
        s = list(np.arange(0, sp.s[-1], 0.1))
        rx, ry, ryaw, rk = [], [], [], []
        for i_s in s:
            ix, iy = sp.compute_pt(i_s)
            rx.append(ix)
            ry.append(iy)
            ryaw.append(sp.calc_yaw(i_s))
            rk.append(sp.compute_curvature(i_s))
        ax4.plot(rx, ry, '-k', linewidth=2)
        for path in plst:
            if path.ctot < min_ctot:
                min_ctot = path.ctot
                min_path = path
            color = 'gray'
            good_lim = 7
            gray_lim = 6
            yellow_lim = 1.4
            if path.ctot > good_lim:
                continue
            if path.ctot > gray_lim:
                color = 'gray'
            if path.ctot < gray_lim and path.ctot > yellow_lim:
                color = 'red'
            if path.ctot < yellow_lim and path.ctot > 0:
                color = 'yellow'
            path_x = [x for i,(x,y,d) in enumerate(zip(path.x,path.y, path.d)) if x<=21.5 and d>0]
            path_y = [y for i,(x,y,d) in enumerate(zip(path.x,path.y, path.d)) if x<=21.5 and d>0]
            ax4.plot(path_x, path_y, color, linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            path_x = [path_x[-1]]+[x for i,(x,y,d) in enumerate(zip(path.x,path.y,path.d)) if (x>=21.5 or d<0) and not path.ctot>gray_lim-4]
            path_y = [path_y[-1]]+[y for i,(x,y,d) in enumerate(zip(path.x,path.y,path.d)) if (x>=21.5 or d<0)and not path.ctot>gray_lim-4]
            ax4.plot(path_x, path_y, 'k', linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            path_x = [path_x[-1]]+[x for i,(x,y,d) in enumerate(zip(path.x,path.y,path.d)) if (x>=21.5 or d<0) and path.ctot>gray_lim-4]
            path_y = [path_y[-1]]+[y for i,(x,y,d) in enumerate(zip(path.x,path.y,path.d)) if (x>=21.5 or d<0)and path.ctot>gray_lim-4]
            ax4.plot(path_x, path_y, 'gray', linewidth = 0.5)#c=cm.gnuplot(path.cv / 25))
            ax4.set_xlabel("x/m")
            ax4.set_ylabel("y/m")
            ax4.set_xlim(-3,42)
            ax4.set_ylim(-14,3)
        min_path_x = [x for x in min_path.x if x<=19]
        min_path_y = [y for x,y in zip(min_path.x,min_path.y) if x<=19]
        ax4.plot(min_path_x, min_path_y, '-g', linewidth=2) 
        min_path_x = [min_path_x[-1]]+[x for x in min_path.x if x>=19]
        min_path_y = [min_path_y[-1]]+[y for x,y in zip(min_path.x,min_path.y) if x>=19]
        ax4.plot(min_path_x, min_path_y, 'gray', linewidth=2) 
        
    return fig