from opensimplex import OpenSimplex
import pybullet as p
import time
import numpy as np
import random

p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

def create_opensimplex_map(amplitude, feature_size=24):
    
    num = random.randint(0,1000)
    a = np.zeros((128, 128))    
    simplex = OpenSimplex()
    for x in range(128):
        for y in range(0, 128):
            a[x,y] = simplex.noise2d((x+num) /feature_size,(y+num)/feature_size)*amplitude
    return a
    
a = create_opensimplex_map(0.35, feature_size=22)
heightfield = a.reshape(-1)

terrainShape = p.createCollisionShape(shapeType = p.GEOM_HEIGHTFIELD, 
                                        meshScale=[0.05,0.05,1],
                                        numHeightfieldRows= 128, 
                                        numHeightfieldColumns=128,
                                        heightfieldData = heightfield,
                                        heightfieldTextureScaling=128)

terrain  = p.createMultiBody(0, terrainShape)
p.changeVisualShape(terrain, -1, textureUniqueId = -1)
p.resetBasePositionAndOrientation(terrain,[0,0,0], [0,0,0,1])

destination = 'hills.txt'
np.savetxt(destination, heightfield, delimiter=',' , fmt='%1.5f')

# WHITE
p.changeVisualShape(terrain, -1, rgbaColor=[1,1,1,1])
# GREY
p.changeVisualShape(terrain, -1, rgbaColor=[0.9,0.9,0.9,1])
# MARS
# p.changeVisualShape(terrain, -1, rgbaColor=[0.9,0.5,0.05,1])

# texUid = p.loadTexture("/grid1.png")
# p.changeVisualShape(terrain,-1,textureUniqueId=texUid)

while (p.isConnected()):
    keys = p.getKeyboardEvents()
    time.sleep(0.01)

# amplitude = 0.15

# for i in range(4):
#     feature_size = 50
#     for j in range(4):
#         destination = 'heightmaps/hills/' + str(i+1)+str(j+1) +'.txt'
#         np.savetxt(destination,
#                    (create_opensimplex_map(amplitude, feature_size)).reshape(-1), 
#                    delimiter=',', 
#                    fmt='%1.5f')
#         feature_size -= 4
#     amplitude += 0.05