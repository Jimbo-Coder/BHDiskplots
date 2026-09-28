#!/usr/bin/python
############################################################################
import matplotlib
#from matplotlib import rc
matplotlib.use('Agg')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pylab,scipy,commands
from pylab import *
from scipy.interpolate import griddata
from multiprocessing import Pool
from scipy import optimize
from contextlib import closing
from os import makedirs 
from os import path
from scipy import stats
import os
from sys import exit
 
## USER SPECS ##
# column format: 1:it   2:tl 3:rl 4:c 5:ml      6:ix 7:iy 8:iz  9:time  10:x 11:y 12:z  13:data
cols = [8,9,10,12]

rc("font",size=18)
# rc('text', usetex=True)

scratch = os.environ["SCRATCH"] + "/"
home = os.environ["HOME"] 
plot_loc = home + "/NS_NS_sims/plotting_scripts/"

## INCLUDE IO LIBRARY ## 
execfile(home+"/py/carpio-v3.py")

## USER SPECS ##
out_freq = 256

MSun_in_ms = 0.0049272004672325466
MSun_in_km = 1.4771375391303936

sim_names = ["bhtD2.0_cAJS0.80_000_000_q1.85_l3.65_r0.40_sol_20",
"bhtD2.0_cAJS0.80_045_000_q1.85_l4.85_r0.40_sol_25",
"bhtD2.0_cAJS0.80_045_000_q1.85_l4.85_r0.40_sol_33",
"bhtD2.0_cAJS0.80_090_000_q1.85_l3.85_r0.40_sol_25",
"bhtD2.0_cAJS0.80_180_000_q1.85_l4.40_r0.40_sol_11"]

M_ADM_dict = {"bhtD2.0_cAJS0.80_000_000_q1.85_l3.65_r0.40_sol_20" : 0.133522731924614,
"bhtD2.0_cAJS0.80_045_000_q1.85_l4.85_r0.40_sol_25" : 0.120635535567938,
"bhtD2.0_cAJS0.80_045_000_q1.85_l4.85_r0.40_sol_33" : 0.527871401527185,
"bhtD2.0_cAJS0.80_045_000_q1.85_l4.85_r0.40_sol_33_magnetised" : 0.527871401527185,
"bhtD2.0_cAJS0.80_090_000_q1.85_l3.85_r0.40_sol_25" : 0.102797742130745,
"bhtD2.0_cAJS0.80_180_000_q1.85_l4.40_r0.40_sol_11" : 0.11404112653795
}

P_c_dict = {"bhtD2.0_cAJS0.80_000_000_q1.85_l3.65_r0.40_sol_20" : 24.2983890,
"bhtD2.0_cAJS0.80_045_000_q1.85_l4.85_r0.40_sol_25" : 24.1766323,
"bhtD2.0_cAJS0.80_045_000_q1.85_l4.85_r0.40_sol_33" : 44.2335367,
"bhtD2.0_cAJS0.80_090_000_q1.85_l3.85_r0.40_sol_25" : 25.1016434,
"bhtD2.0_cAJS0.80_180_000_q1.85_l4.40_r0.40_sol_11" : 32.3451007
}

M_BH = 0.05

rho_0 = 2.2362990124E-004

def get_AH_location(data_dir,user_iter):
  AH_data = np.genfromtxt(data_dir + "/beta100/BH_diagnostics.ah1.gp")
  iteration = 128*user_iter
  this_row = -1
  for i in range(0,AH_data.shape[0]):
    if (AH_data[i,0] == iteration):
      this_row = i
      break
  x_centroid = AH_data[this_row,2]
  y_centroid = AH_data[this_row,3]
  z_centroid = AH_data[this_row,4]
  BH_x_vel = (AH_data[this_row+1,2]-AH_data[this_row,2])/(AH_data[this_row+1,1]-AH_data[this_row,1])
  BH_y_vel = (AH_data[this_row+1,3]-AH_data[this_row,3])/(AH_data[this_row+1,1]-AH_data[this_row,1])
  return ([x_centroid,y_centroid,z_centroid],[BH_x_vel,BH_y_vel])

def plot_AH(sim_name,user_iter,data_dir,ax,M_ADM,linecolor,BH_POS):
  #
  iteration = user_iter*out_freq
  print "iteration = ", iteration
  AH_data_path = data_dir + "/beta100/h.t{:d}.ah1.gp".format(iteration)
  AH_xy = []
  #try:
  with open(AH_data_path,"r") as f:
    for line in f:
      li=line.strip()
      if not li.startswith("#"):
        data_line = line.split()
        if (len(data_line)==6):
          z = float(data_line[5])
          if (np.abs(z-BH_POS[2])<0.0005):
            x = float(data_line[3])
            y = float(data_line[4])
            AH_xy.append([x,y])
  AH_arr = np.array(AH_xy)
  x_max = np.max(AH_arr[:,0])
  x_min = np.min(AH_arr[:,0])
  y_max = np.max(AH_arr[:,1])
  y_min = np.min(AH_arr[:,1])
  BH_pos = [0.5*(x_max+x_min),0.5*(y_max+y_min)]
  angle_pos = np.arctan2(AH_arr[:,1]-BH_pos[1],AH_arr[:,0]-BH_pos[1])
  sorted_AH_coords = AH_arr[angle_pos.argsort()]
  x_plot = np.append(sorted_AH_coords[:,0],sorted_AH_coords[0,0]) - BH_POS[0]
  y_plot = np.append(sorted_AH_coords[:,1],sorted_AH_coords[0,1]) - BH_POS[1]
  #
  ax.plot(x_plot/M_BH,y_plot/M_BH,color=linecolor,linestyle="-",linewidth=1)
  print "plotted AH!"
  #except:
  #  pass

def make_plot(sim_name,movie_dir,user_iter):
  #movie_dir = home + "/NS_NS_sims/" + sim_name + "/" + movie_name
  save_path = movie_dir+"/frame_"+string.zfill(user_iter,5)+".png"
  data_dir = scratch + "/BH_massiveDisk/" + sim_name
  #if (path.exists(save_path)==False):
  if (True):
    if (True):
      M_ADM = float(M_ADM_dict[sim_name])
      print "M_ADM = ", M_ADM
      print "user_iter = ", user_iter
      ## READ IN SCALAR DATA ##
      print "READING DENSITY..."
      print("reading from ",data_dir+"/beta100/rho_b.xy.asc")
      t, x, y, rho = readslice(data_dir+"/beta100/rho_b.xy.asc", user_iter=user_iter,cols=cols,out_freq=out_freq)
      #ell = readslice(data_dir+"/beta100/temp19.xy.asc", user_iter=user_iter,cols=[cols[-1]],out_freq=out_freq)
      smallb2 = readslice(data_dir+"/beta100/smallb2.xy.asc", user_iter=user_iter,cols=[cols[-1]],out_freq=out_freq)

      T_int = unique(t)
      time_now = T_int[0]

      AH_data_path_now = data_dir + "/beta100/h.t{:d}.ah1.gp".format(user_iter*128)
      BH_POS, BH_VEL = get_AH_location(data_dir,user_iter)

      #x *= MSun_in_km #M_ADM
      #y *= MSun_in_km #M_ADM
      #x /= M_ADM
      #y /= M_ADM
      x /= M_BH
      y /= M_BH

      # get coordinates to interpolate onto
      X_int = unique(x)
      Y_int = unique(y)
      # print "X_int = ", X_int
      X_int, Y_int = meshgrid(X_int,Y_int)

      ## interpolate data onto grid 
      inter_rho  =  scipy.interpolate.griddata((x,y),rho,(X_int,Y_int))
      inter_b2  =  scipy.interpolate.griddata((x,y),smallb2,(X_int,Y_int))
      #mon_file = scratch + sim_name + "/bhns.mon"
      #rho_0 = np.genfromtxt(mon_file)[0,8]

      ## PLOT ##
      print "PLOTTING GRAPH...."
      fig, ax = plt.subplots(1,2,gridspec_kw=dict(left=0.05,right=0.925,top=0.85,bottom=0.15,wspace=0.2,hspace=0.1,width_ratios=[1.0,1.0]))
      # gridspec_kw=dict(left=0.15,right=0.85,wspace=0.3,hspace=0.3)
      label_font = 10
      title_font = 10
      tick_label_font = 10
      legend_font = 8
      fig.set_size_inches(7,3.375)

      ax_rho = ax[0]
      ax_b2 = ax[1]

      XY_extent = 5.0
      XMIN = -XY_extent #*M_ADM # [-40.*M_ADM,-10*M_ADM,-2.*M_ADM][["total","BH","NS"].index(PLOT)]
      XMAX = XY_extent #*M_ADM  # [+40.*M_ADM, 10*M_ADM,+2.*M_ADM][["total","BH","NS"].index(PLOT)]
      YMIN = -XY_extent #*M_ADM   # [ 0.*M_ADM, 0*M_ADM, 0.*M_ADM][["total","BH","NS"].index(PLOT)]
      YMAX = XY_extent #*M_ADM # [+40.*M_ADM, 10*M_ADM,+2.*M_ADM][["total","BH","NS"].index(PLOT)]    
      
      X_int = X_int - BH_POS[0]/M_BH
      Y_int = Y_int - BH_POS[1]/M_BH

      # plot the density
      #pc_rho = ax_rho.pcolormesh(X_int,Y_int,inter_rho/rho_0,rasterized=True,cmap=cm.nipy_spectral,vmin=0,vmax=3.0) # This colormap looks cool B-]
      #rho_min = 10**(-3.0)
      print "plotting density 2D plot"
      rho_min = 10**(-2.0)
      #pc_rho = ax_rho.pcolormesh(X_int,Y_int,np.abs(inter_rho/rho_0),rasterized=True,cmap=cm.nipy_spectral,norm=matplotlib.colors.LogNorm(vmin=rho_min,vmax=10.0)) # This colormap looks cool B-]
      pc_rho = ax_rho.pcolormesh(X_int,Y_int,inter_rho/rho_0,rasterized=True,cmap=cm.inferno,vmin=0,vmax=5.0) # This colormap looks cool B-]
      divider_rho = make_axes_locatable(ax_rho)
      cax_rho = divider_rho.append_axes("right", size="5%", pad=0.0)
      cbar_rho=colorbar(pc_rho, cax=cax_rho)
      ax_rho.set_title(r'$\rho_0/\rho_{0_{\mathrm{max},t=0}}$',fontsize = title_font)
      # cbar_rho.ax.set_ylabel(r'$\rho_0/\rho_{0_{\mathrm{max},t=0}}$',fontsize = label_font,color='black',rotation=90)
      ax_rho.set_xlim(XMIN,XMAX)
      ax_rho.set_ylim(YMIN,YMAX)
      ax_rho.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
      ax_rho.tick_params(axis='x', labelsize=tick_label_font)
      ax_rho.tick_params(axis='y', labelsize=tick_label_font)
      cbar_rho.ax.tick_params(labelsize=tick_label_font) 
      ax_rho.set_aspect('equal', adjustable='box', anchor='C')
      ax_rho.set_xlabel(r"$x/M_{\mathrm{BH}}$",fontsize=label_font)
      ax_rho.set_ylabel(r"$y/M_{\mathrm{BH}}$",fontsize=label_font)            
      plot_AH(sim_name,user_iter,data_dir,ax_rho,M_ADM,"r",BH_POS)

      # plot the Omega
      print "plotting Omega 2D plot"
      pc_b2 = ax_b2.pcolormesh(X_int,Y_int,np.sqrt(abs(inter_b2))/(2*rho_0),rasterized=True,cmap=cm.jet,vmin=0,vmax=100.0) # This colormap looks cool B-]
      divider_b2 = make_axes_locatable(ax_b2)
      cax_b2 = divider_b2.append_axes("right", size="5%", pad=0.0)
      cbar_b2=colorbar(pc_b2, cax=cax_b2)
      ax_b2.set_title(r'$\sqrt{b^2}/(2\rho_0)$',fontsize = title_font)
      #cbar_b2.ax.set_ylabel(r'$\Omega M$',fontsize = label_font,color='black',rotation=90)
      ax_b2.set_xlim(XMIN,XMAX)
      ax_b2.set_ylim(YMIN,YMAX)
      ax_b2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
      ax_b2.tick_params(axis='x', labelsize=tick_label_font)
      ax_b2.tick_params(axis='y', labelsize=tick_label_font, labelleft=False)
      cbar_b2.ax.tick_params(labelsize=tick_label_font) 
      ax_b2.set_aspect('equal', adjustable='box', anchor='C')
      ax_b2.set_xlabel(r"$x/M_{\mathrm{BH}}$",fontsize=label_font)
      plot_AH(sim_name,user_iter,data_dir,ax_b2,M_ADM,"k",BH_POS)

      #fig.set_title(sim_name[0:2] + M_ADM_label + "   $t - t_{\mathrm{merger}} =$"+'%.3f' % t_post_merger_in_ms + " ms",color='black',fontsize=label_font)
      angle_in_deg = float(sim_name[17:20])
      padding = "   "
      label_text = r"$ \theta_{\mathrm{BH}} = $"+" ${:.0f}\%$".format(angle_in_deg)+padding+r"$\quad M/M_{{\mathrm{{BH}}}}=$" + "${:.2f}$".format(M_ADM/M_BH)
      plt.suptitle(label_text + "\n$t/M =$"+'%.3f' % (time_now/M_ADM),color='black',fontsize=title_font)
      ## DUMP TO DISK ##
      #plt.tight_layout()
      savefig(save_path, dpi=400)
      print "saved " + save_path
      plt.close()
      #exit()
    #except:
    #  print "failed to make user-iter ", user_iter
  else:
    print save_path, " already exists"

sim_names = ["bhtD2.0_cAJS0.80_045_000_q1.85_l4.85_r0.40_sol_33_magnetised"]

for sname in sim_names:
  print "Making xy b2 rho plot for ", sname
  movie_name = sname + "_rho_b2_xy"
  movie_dir = home + "/BH_disk/scripts/" + movie_name
  try:
    makedirs(movie_dir)
  except:
    pass

  don_file = scratch + "BH_massiveDisk/" + sname + "/bhns.don"
  don_data = np.genfromtxt(don_file)
  N_iter = 0
  dt = don_data[11,0] - don_data[10,0]
  print("dt = ", dt)
  N_iter = int(np.max(don_data[:,0])/dt)

  def make_this_plot(user_iter):
    make_plot(sname,movie_dir,user_iter)

  it_max = N_iter
  make_this_plot(N_iter-1)

print("Program finished!")
