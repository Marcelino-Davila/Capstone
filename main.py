import decisionFusion as DF
import fusion
import imgSorter as IS
import lasCreater as LC

LC.createLas()
IS.sort()
core = DF.fusionCore((0,1199),(0,1199))
core.scanArea()
fusion.fuseMods()

