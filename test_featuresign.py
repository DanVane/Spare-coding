import l1_featuresign



#
# A = np.array([[1,2,3],[4,5,6],[7,8,9]])
# y = np.array([1,5,5])
# gamma = 0.2
# X = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
#
# act=  np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
#
# theta = np.array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,
#         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
# act_idx1 = np.array([46])

# AtA = np.dot(np.transpose(AA),AA)
# Aty = np.dot(np.transpose(AA),y)
#
# fobj_featuresign(X, A, y, AtA, Aty, gamma)
#
# ret = l1ls_featuresign(A, y, gamma, X)


# ls_featuresign_sub(A, y, AtA, Aty, gamma, np.array([0,0,0]))

# compute_FS_step(X, AA, y, AtA, Aty, theta, act, act_idx1, gamma)

import matplotlib.pyplot as plt
import cv2
# A = cv2.imread("/Users/stevenydc/Documents/3rd year/Math 191/test2.jpg", cv2.CV_LOAD_IMAGE_GRAYSCALE)
# AA = A[:,:,0].astype(float)
# plt.imshow(A,'gray')

from scipy import io
A= io.loadmat("/Users/stevenydc/Documents/3rd year/Math 191/test.mat")
A = A['A']
A=A.astype(float)
gamma = 0.2
y = np.array([35,648,565,363,754,188,416,310,686,579,713,612,473,668,56,371,77,224,563,522,742,691,29,3,479,495,465,627,639,552,554,605,792,670,165,663,618,169,671,500,501,722,562,287,600,458,277,621,285,736,498,675,455,488,741,52,270,710,402,433,251,464,641,477,505,254,578,26,11,240,331,441,738,71,553,497,142,136,634,414,117,515,155,658,519,69,32,230,708,635,15,93,358,572,14,410,617,530,528,795,583,728,203,599,694,329,721,749,20,662,569,472,676,118,430,179,221,766,780,7,356,61,544,133,395,130,323,616,242,372,447,768,227,731,248,100,784,587,601,799,476,86,490,107,540,170,649,619,405,66,776,132,771,462,338,595,393,397,407,444,346,72,324,467,206,239,392,377,654,194,364,692,348,256,128,297,304,631,213,423,231,264,342,502,576,436,334,160,779,85,645,398,443,326,209,139,192,42,369,777,685,610,55,727,156,102,427,674,426,611,212,311,493,422,753,47,327,448,186,159,119,400,588,114,431,113,336,97,546,330,163,357,543,153,266,1,555,362,261,714,513,793,471,548,245,10,90,292,131,68,233,67,719,135,390,602,613,633,237,706,550,482,171,651,370,284,707,24,597,325,314,143,260,734,79,438,126,271,677,45,162,647,322,168,484,593,412,161,298,539,195,496,628,620,704,43,492,301,521,269,374,740,788,243,229,273,178,380,466,92,98,683,783,96,537,103,157,303,210,249,84,720,790,151,796,514,756,379,365,636,222,28,316,65,483,382,140,767,5,606,432,529,508,460,442,215,524,172,672,673,403,418,236,428,120,177,451,748,27,656,746,147,366,394,76,278,54,586,630,146,59,283,564,518,598,184,789,456,289,567,669,34,452,705,547,534,781,341,51,437,124,696,475,413,689,144,625,764,718,216,197,265,317,127,315,0,535,355,726,122,733,570,755,556,772,115,123,345,542,454,511,112,541,594,411,759,23,125,680,353,6,787,561,732,557,604,272,509,763,520,62,373,276,78,506,778,53,16,149,786,575,208,585,571,470,510,205,659,352,319,83,623,560,263,46,252,219,399,389,775,516,559,512,409,306,4,525,545,110,19,679,280,376,794,105,218,752,693,527,632,747,259,485,494,22,533,253,439,487,13,757,425,765,294,596,361,607,141,220,762,279,340,255,204,712,257,198,384,417,798,480,453,702,36,401,406,70,30,74,785,684,375,549,302,744,577,116,347,558,640,584,695,729,228,517,415,750,262,725,12,44,770,715,48,225,305,282,217,743,538,145,199,739,758,281,681,64,745,167,109,678,154,354,175,368,526,701,31,223,226,385,214,75,459,660,591,137,211,95,589,241,574,531,646,615,590,180,408,667,150,201,291,106,682,478,351,737,293,258,108,419,573,173,711,295,17,328,383,275,614,449,687,9,474,99,312,461,629,609,381,333,50,21,396,387,769,440,196,797,182,664,690,469,60,773,152,344,650,286,58,566,63,82,274,592,247,481,735,207,181,57,523,49,468,420,183,643,386,391,642,504,350,463,378,700,709,626,698,101,187,33,300,435,339,335,716,88,250,624,200,191,134,603,424,349,8,665,318,638,332,121,343,25,41,581,290,321,582,688,580,138,189,434,666,37,267,608,234,129,307,717,499,491,2,791,507,238,404,760,246,308,724,244,320,503,761,421,536,176,39,299,235,644,637,388,661,489,445,457,782,288,164,429,94,190,774,486,723,551,730,73,653,360,268,232,367,296,202,81,18,568,699,166,193,313,89,104,40,359,158,185,655,309,111,657,80,337,532,751,697,38,174,652,87,148,446,622,450,703,91]).astype(float)
# y = np.array([54,25,2,5,44,27,23,47,14,51,9,41,30,59,61,6,3,26,63,8,19,10,17,33,18,46,42,55,52,7,12,37,13,21,22,34,48,60,53,57,35,24,28,16,15,50,32,1,4,40,39,58,0,45,56,43,36,31,29,38,11,62,20,49])
ret = l1ls_featuresign(A, y, gamma)



