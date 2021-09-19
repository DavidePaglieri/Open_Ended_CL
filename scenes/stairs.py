import numpy as np
import pybullet as p
import time

p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)

def create_stairs_map(width, height):
    
    a = np.zeros((1,128))
    inc = 0
    i = 0
    while i<128:
        for j in range(i, i+1+width,1):
            if j>=128:
                break
            a[0,j] = inc
        i += 1+width
        if not 46<i<64:
            inc += height

    return a.repeat(128, axis=0)
    
a = create_stairs_map(8,0.12)
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

# width = 15
# height = 0.01

# for i in range(4):
#     height = 0.02
#     for j in range(4):
#         destination = 'heightmaps/stairs/' + str(i+1)+str(j+1) +'.txt'
#         np.savetxt(destination,
#                    (create_stairs_map(width, height)).reshape(-1), 
#                    delimiter=',', 
#                    fmt='%1.5f')
#         height += 0.01
#     width -= 4