#Main file
import fileReader as ef

groundTruth = ef.file("xlsx","D:\\capstoneRoot\\data\\ASPIRE_forDistro\\0 Ground Truth\\ground_truth.xlsx")
LWIRSideLooking = ef.file("csv","D:\\capstoneRoot\\data\\ASPIRE_forDistro\\2 Sidelooking\\LWIR\\2024_07_31_aspire_sar_75_LWIR.csv")
LWIRDownLooking = ef.file("csv","D:\\capstoneRoot\\data\\ASPIRE_forDistro\\1 Downlooking\\LWIR\\2024_07_30_aspire_LWIR.csv")
RGBSideLooking = ef.file("csv","D:\\capstoneRoot\\data\\ASPIRE_forDistro\\2 Sidelooking\\RGB\\2024_07_31_aspire_sar_75_RGB.csv")
RGBDownlooking = ef.file("csv","D:\\capstoneRoot\\data\\ASPIRE_forDistro\\1 Downlooking\\RGB\\2024_07_30_aspire_RGB.csv")
#LIDARPointCloud = ef.file("mat","D:\\capstoneRoot\\data\\ASPIRE_forDistro\\3 LIDAR\\lidar_point_cloud_2024_07_31.mat")
#LIDARProfile = ef.file("mat","D:\\capstoneRoot\\data\\ASPIRE_forDistro\\3 LIDAR\\lidar_profile_2024_07_31.mat")
picture = ef.file("img","D:\\capstoneRoot\\data\\ASPIRE_forDistro\\3 LIDAR\\LIDAR_profile_2024_07_31.png")
