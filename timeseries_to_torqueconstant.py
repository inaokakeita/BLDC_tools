import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal,optimize

skipcsvtop = 6 #csvの頭何行読み飛ばすか
anglegain = 2*math.pi/3.3 #a6500センサの電圧出力前提 0~3.3Vで1周
#角度センサローパスパラメータ
filter_order = 5
cutoff = 0.01 #ナイキスト周波数のn倍

filename = "slow"
#オシロch接続 ch1:u ch2:v ch3:w ch4:a5600out
timeseries = np.loadtxt("datas/"+filename+".csv",delimiter=",",skiprows=skipcsvtop)

#逆起波形をplot
fig,ax = plt.subplots(3,2,figsize=(10,10))
ax[0,0].plot(timeseries[:,0],timeseries[:,1:4])
ax[0,0].set_ylabel("Counter-Electromotive Voltage[V]")
ax[0,0].set_xlabel("Time[s]")
ax[0,0].legend(["U","V","W"])

#a5600センサの角度信号処理 速度計算
time = timeseries[:,0]
fs = 1/((time[-1]-time[0])/len(time))
angle_rad = anglegain * timeseries[:,4]
vel =np.diff(angle_rad)*(1.0/np.diff(time))
#ノイジーなのでlpf
b,a = signal.butter(filter_order,cutoff)
lp_vel = signal.filtfilt(b,a,vel,method="gust") 
ax[1,0].plot(time,angle_rad)
ax[1,0].set_ylabel("Mecha Angle[rad]")
ax[1,0].set_xlabel("Time[s]")
ax[2,0].plot(time[:len(time)-1],lp_vel)
# ax[2,0].plot(time[:len(time)-1],vel)
ax[2,0].set_ylabel("velocity[rad/s]")
ax[2,0].set_xlabel("Time[s]")

#各相の逆起定数を計算(励磁なし)
vu = timeseries[1:,1]
vv = timeseries[1:,2]
vw = timeseries[1:,3]
u_cemv = vu/lp_vel
v_cemv = vv/lp_vel
w_cemv = vw/lp_vel
angle_rad = angle_rad[:-1]
ax[0,1].plot(angle_rad,u_cemv)
ax[0,1].plot(angle_rad,v_cemv)
ax[0,1].plot(angle_rad,w_cemv)
ax[0,1].set_ylabel("Counter-EMF Constant(raw)[V/rad]")
ax[0,1].set_xlabel("Mecha Angle[rad]")

#UVW相から電気角を計算（特に何かに使うわけではない）
alpha = u_cemv -0.5 * v_cemv -0.5 * w_cemv
beta = math.sqrt(2)/2 * v_cemv - math.sqrt(2)/2 * w_cemv
elecangle = np.arctan2(alpha,beta)
# ax[1,1].plot(angle_rad,elecangle)
# ax[1,1].set_ylabel("Elec Angle[rad]")
# ax[1,1].set_xlabel("Angle[rad]")

#励磁込みのトルク定数を計算する
#U相をsin波とフィッティングして位相と振幅を割り出す
def fit_func(x, a, b, c, d):
    return a * np.sin(b*(x*np.pi*2 - c)) + d
popt, pcov = optimize.curve_fit(fit_func, angle_rad[::10], u_cemv[::10])
ax[1,1].plot(angle_rad, fit_func(angle_rad,*popt))
ax[1,1].plot(angle_rad[:len(angle_rad)],u_cemv)
ax[1,1].set_xlabel("Mecha Angle[rad]")
ax[1,1].legend(["U","fitting"])
#ドライブ信号を生成　U相の電気角から90度オフセット 角相間は120度
u_drive = fit_func(angle_rad,1,popt[1],popt[2],0)
v_drive = fit_func(angle_rad,1,popt[1],popt[2]+2*math.pi/3,0)
w_drive = fit_func(angle_rad,1,popt[1],popt[2]+2*math.pi/3*2,0)
u_torque = u_drive * u_cemv
v_torque = v_drive * v_cemv
w_torque = w_drive * w_cemv
torque = u_torque + v_torque + w_torque
ax[2,1].plot(angle_rad, u_torque)
ax[2,1].plot(angle_rad, v_torque)
ax[2,1].plot(angle_rad, w_torque)
ax[2,1].plot(angle_rad, torque)
ax[2,1].set_ylabel("Torque Constant[Nm/A]")
ax[2,1].set_xlabel("Mecha Angle[rad]")
ax[2,1].legend(["U","V","W","3phase"],loc="lower right")

fig.savefig(filename+".png")

fig2,ax2 = plt.subplots(figsize=(8,5))
ax2.plot(angle_rad, u_torque)
ax2.plot(angle_rad, v_torque)
ax2.plot(angle_rad, w_torque)
ax2.plot(angle_rad, torque)
ax2.set_ylabel("Torque Constant[Nm/A]")
ax2.set_xlabel("Mecha Angle[rad]")
ax2.legend(["U","V","W","3phase"],loc="lower right")
fig2.savefig(filename+"-TorqueConstant"+".png")