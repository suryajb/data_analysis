from analysis_lib import *

# filepath = "Y:\Josh\Data\\test_anomalous"
filepath = "Y:\Josh\Data\85"
a = file_obj(filepath)
length = len(a.shgfiles)
a.plot_ir_single(yautoscale=False,ylogscale=True,xrange=[1480,1620],save=True,sx=6,sy=length*0.8)
# a.plot_shg_single(yautoscale=True,xrange=[1480,1620],save=True,sx=6,sy=length*0.8)
# a.plot_shg_multi(yautoscale=False,xrange=[1577.3,1577.34],save=True,sx=6,sy=length*3)
# a.plot_ir_multi(yautoscale=True,xrange=[1480,1620],save=False,sx=8,sy=6)
# a.plot_ir_shg_multi(ir_ylogscale=True,xrange=[1480,1620],shg_yautoscale=False,title=False, save=True)
# plt.show()
# a.Qfit(radius=50,n_mode=4,save=False)
# a.plot_shg_peaks(save=True)


plt.show()