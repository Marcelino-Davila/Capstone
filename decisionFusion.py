from enum import Enum, auto

class obejectClassification(Enum):
    medalTarget = auto()
    plasticTarget = auto()
    subSurfaceTarget = auto()
    naturalClutter = auto()
    naturalObjects = auto()

class scanResult:
    def __init__(self,decision,xPosition,yPosition,zPosition,confidence,image,description="None"):
        self.decision = decision
        self.xPosition = xPosition
        self.yPosition = yPosition
        self.zPosition = zPosition
        self.confidence = confidence
        self.description = description
        self.image = image

class fusionCore:
    def __init__(self):
        self.radar = []
        self.lidar = []
        self.downRGB = []
        self.sideRGB = []
        self.downLWIR = []
        self.sideLWIR = []

    def makeDecision(self,radar,lidar,downRGB,sideRGB,downLWIR,sideLWIR):
        self.radar.append(radar)
        self.lidar.append(lidar)
        self.downRGB.append(downRGB)
        self.sideRGB.append(sideRGB)
        self.downLWIR.append(downLWIR)
        self.sideLWIR.append(sideLWIR)
        