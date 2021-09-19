import pybullet as p
import time
import numpy as np

p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

def create_steps_map(amplitude,step):
    a = np.random.random( [128//step, 128//step] )*amplitude
    return a.repeat( step, axis=0 ).repeat( step, axis=1 )
    
    
a = create_steps_map(0.08,4)
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

destination = 'steps.txt'
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

# np.savetxt('ae2_heightmaps/steps.txt', a, delimiter=',', fmt='%1.5f')

# amplitude = 0.05

# for i in range(4):
#     step = 16
#     for j in range(4):
#         destination = 'heightmaps/steps/' + str(i+1)+str(j+1) +'.txt'
#         np.savetxt(destination,
#                    (create_steps_map(amplitude, step)).reshape(-1), 
#                    delimiter=',', 
#                    fmt='%1.5f')
#         step = int(step/2)
#     amplitude += 0.01