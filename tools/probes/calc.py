import math
# canopy visibility: lens at 0.98 m, need elevation to tree top H at horizontal distance r_surface
for H in (1.2,2.0,3.0):
  for r in (1.2,1.5,2.0,2.5):
    print("top",H,"r",r,"elev_deg",round(math.degrees(math.atan2(H-0.98,r)),1),"base_dep_deg",round(math.degrees(math.atan2(0.98,r)),1))
# mm/px across horizontal at distance D for HFOV and width
for name,hf,w in (("pi3w_1080",102,1920),("pi3w_2304",102,2304),("sim_320",100,320),("sim_640",100,640),("d405_1280",87,1280)):
  for D in (0.15,0.3,1.0,1.5,2.0):
    print(name,D,"mm/px",round(2*D*math.tan(math.radians(hf/2))/w*1000,2))
# orbit time per radius at speeds
for v in (0.18,0.34):
  for R in (1.5,2.0,2.5,3.0):
    print("v",v,"R",R,"s",round(2*math.pi*R/v,1),"deg_per_s",round(math.degrees(v/R),2))
# storage: 3 cams 1080p H.265 ~4 Mbit/s each (assumption), JPEG 2fps 1.5MB (platform figure)
for t in (60,90,150):
  print("t",t,"h265_3cam_MB",round(3*4e6*t/8/1e6,1),"jpeg2fps_3cam_MB",round(3*2*t*1.5,1))
# sim render budget: 3 cams 320x240 at 5 Hz using ~36 ms/pass -> ms per s
print("sim 3cam@5Hz ms/s",3*5*36, "3cam@2Hz",3*2*36)
# sim storage 320x240 JPEG ~12KB at 5fps 3 cams 150 s
print("sim MB",round(3*5*150*12/1024,1))
