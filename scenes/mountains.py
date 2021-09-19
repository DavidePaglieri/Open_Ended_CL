from fractal import (
    generate_perlin_noise_2d, generate_fractal_noise_2d
)
import pybullet as p
import time
import numpy as np

p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

def create_fractal_map(amplitude, persistence):
    return generate_fractal_noise_2d(shape=(128,128), 
                                res=(2,2), 
                                octaves=6,
                                amplitude=amplitude,
                                # tileable=(True, True),
                                persistence=persistence)
    
a = create_fractal_map(0.4, 0.6)
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

destination = 'mountains.txt'
np.savetxt(destination, heightfield, delimiter=',' , fmt='%1.5f')

# WHITE
# p.changeVisualShape(terrain, -1, rgbaColor=[1,1,1,1])
# GREY
p.changeVisualShape(terrain, -1, rgbaColor=[0.9,0.9,0.9,1])
# MARS
# p.changeVisualShape(terrain, -1, rgbaColor=[0.9,0.5,0.05,1])

# texUid = p.loadTexture("/grid1.png")
# p.changeVisualShape(terrain,-1,textureUniqueId=texUid)

while (p.isConnected()):
    keys = p.getKeyboardEvents()
    time.sleep(0.01)

# amplitude = 0.075

# for i in range(4):
#     persistence = 0.3
#     for j in range(4):
#         destination = 'heightmaps/mountains/' + str(i+1)+str(j+1) +'.txt'
#         np.savetxt(destination,
#                    (create_fractal_map(amplitude, persistence)).reshape(-1), 
#                    delimiter=',', 
#                    fmt='%1.5f')
#         persistence += 0.1
#     amplitude += 0.05
