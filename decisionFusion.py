from enum import Enum, auto
import json
import modalities.detection.RADAR as RD
import imgAutomator as IA
from pathlib import Path

lidarPath = r"data\ASPIRE_forDistro\3 LIDAR\lidar_point_cloud_2024_07_31.mat"
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
    def __init__(self,xRange=(0,1199),yRange=(0,1199)):
        self.xRange = xRange
        self.yRange = yRange
        self.electronicDevices = [] # RADAR readings 
        self.topologyAnomoly = [] # LiDAR readings   
         
        self.visibleObjectDown = [] # RGB object detection
        self.visibleObjectSide = [] # RGB object detection
        self.linearObjectDown = [] #scan for linear target
        self.linearObjectSide = [] #scan for linear target
        self.hotObjectsDown = [] # LWIR readings
        self.hotObjectsSide = [] # LWIR readings
    
    def scanArea(self):
        print("RADAR")
        #self.electronicDevices = RD.radarDetection((0,1199),(0,1199))
        print("downRGB")
        #self.visibleObjectDown.append(IA.scan("downRGB",-2,9999,-1,99999))
        print("sideRGB")
        #self.visibleObjectSide.append(IA.scan("sideRGB",-2,9999,-1,99999))
        print("downLWIR")
        self.hotObjectsDown.append(IA.scan("downLWIR",-2,9999,-1,99999))
        print("sideLWIR")
        self.hotObjectsDown.append(IA.scan("sideLWIR",-2,9999,-1,99999))
        print("downLinear")
        #self.linearObjectDown.append(IA.scan("wireDown",-2,9999,-1,99999))
        print("sideLinear")
        #self.linearObjectDown.append(IA.scan("wireSide",-2,9999,-1,99999))

        scan_data = {
            #"visibleObjectDown": self.visibleObjectDown,
            #"visibleObjectSide": self.visibleObjectSide,
            "hotObjectsDown": self.hotObjectsDown,
            "hotObjectsSide": self.hotObjectsSide,
            #"linearObjectDown": self.linearObjectDown,
            #"linearObjectSide": self.linearObjectSide
        }
        output_path = Path("scan_results")
        output_path.mkdir(exist_ok=True)
        with open(output_path / "LWIR_results.json", "w") as f:
            #json.dump({"electronicDevices": self.electronicDevices}, f, indent=4)
            json.dump(scan_data, f, indent=4, default=str) 
core = fusionCore()
core.scanArea()
