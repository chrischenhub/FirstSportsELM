import matplotlib.pyplot as plt
import re
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go

def FineTunePlot():
    log_data = """
    step 30000: train loss 3.3268, val loss 3.3133
    iter 30000: loss 3.2638, time 6373.73ms, mfu -100.00%
    iter 30001: loss 3.3235, time 720.81ms, mfu -100.00%
    iter 30002: loss 2.9508, time 720.63ms, mfu -100.00%
    iter 30003: loss 2.6891, time 722.29ms, mfu -100.00%
    iter 30004: loss 2.9367, time 719.92ms, mfu -100.00%
    step 30005: train loss 2.7771, val loss 2.7546
    iter 30005: loss 2.8480, time 5752.30ms, mfu 0.78%
    iter 30006: loss 2.7321, time 725.75ms, mfu 1.32%
    iter 30007: loss 2.8254, time 725.78ms, mfu 1.81%
    iter 30008: loss 2.4610, time 724.46ms, mfu 2.25%
    iter 30009: loss 2.5669, time 744.65ms, mfu 2.62%
    step 30010: train loss 2.5848, val loss 2.5951
    iter 30010: loss 2.5468, time 5871.26ms, mfu 2.44%
    iter 30011: loss 2.4531, time 729.40ms, mfu 2.81%
    iter 30012: loss 2.5904, time 728.42ms, mfu 3.15%
    iter 30013: loss 2.3354, time 726.90ms, mfu 3.45%
    iter 30014: loss 2.4935, time 731.01ms, mfu 3.72%
    step 30015: train loss 2.4741, val loss 2.5135
    iter 30015: loss 2.6163, time 5927.21ms, mfu 3.42%
    iter 30016: loss 2.4686, time 732.89ms, mfu 3.69%
    iter 30017: loss 2.2625, time 726.54ms, mfu 3.94%
    iter 30018: loss 2.7362, time 727.00ms, mfu 4.16%
    iter 30019: loss 2.6094, time 726.53ms, mfu 4.37%
    step 30020: train loss 2.3996, val loss 2.4430
    iter 30020: loss 2.1769, time 5875.08ms, mfu 4.00%
    iter 30021: loss 2.4245, time 738.02ms, mfu 4.21%
    iter 30022: loss 2.3560, time 738.84ms, mfu 4.40%
    iter 30023: loss 2.4100, time 741.18ms, mfu 4.56%
    iter 30024: loss 2.2091, time 728.89ms, mfu 4.72%
    step 30025: train loss 2.3068, val loss 2.3987
    iter 30025: loss 2.3732, time 5808.67ms, mfu 4.33%
    iter 30026: loss 2.3404, time 725.78ms, mfu 4.51%
    iter 30027: loss 2.3611, time 730.73ms, mfu 4.68%
    iter 30028: loss 2.2114, time 728.92ms, mfu 4.83%
    iter 30029: loss 2.3799, time 726.67ms, mfu 4.96%
    step 30030: train loss 2.2434, val loss 2.3404
    saving checkpoint iter:30 to ../model
    iter 30030: loss 2.2437, time 7320.17ms, mfu 4.53%
    iter 30031: loss 2.4950, time 725.90ms, mfu 4.69%
    iter 30032: loss 2.0229, time 724.36ms, mfu 4.84%
    iter 30033: loss 2.2752, time 740.07ms, mfu 4.96%
    iter 30034: loss 2.0294, time 749.49ms, mfu 5.07%
    step 30035: train loss 2.1831, val loss 2.3236
    saving checkpoint iter:35 to ../model
    iter 30035: loss 2.1437, time 7178.74ms, mfu 4.62%
    iter 30036: loss 1.9993, time 755.46ms, mfu 4.75%
    iter 30037: loss 2.2311, time 739.97ms, mfu 4.89%
    iter 30038: loss 2.1377, time 737.89ms, mfu 5.01%
    iter 30039: loss 2.2075, time 725.98ms, mfu 5.12%
    step 30040: train loss 2.1168, val loss 2.3069
    saving checkpoint iter:40 to ../model
    iter 30040: loss 2.3586, time 7126.27ms, mfu 4.67%
    iter 30041: loss 2.1217, time 728.93ms, mfu 4.82%
    iter 30042: loss 2.0624, time 727.07ms, mfu 4.96%
    iter 30043: loss 2.3698, time 727.53ms, mfu 5.08%
    iter 30044: loss 2.1500, time 728.06ms, mfu 5.19%
    step 30045: train loss 2.0894, val loss 2.2409
    saving checkpoint iter:45 to ../model
    iter 30045: loss 2.1259, time 7177.92ms, mfu 4.73%
    iter 30046: loss 1.8538, time 758.61ms, mfu 4.85%
    iter 30047: loss 2.0962, time 733.43ms, mfu 4.98%
    iter 30048: loss 2.0663, time 729.10ms, mfu 5.09%
    iter 30049: loss 2.0708, time 729.91ms, mfu 5.20%
    step 30050: train loss 2.0037, val loss 2.2285
    saving checkpoint iter:50 to ../model
    iter 30050: loss 1.9051, time 7534.67ms, mfu 4.74%
    iter 30051: loss 2.0264, time 744.24ms, mfu 4.87%
    iter 30052: loss 1.9985, time 729.46ms, mfu 5.00%
    iter 30053: loss 1.7295, time 731.46ms, mfu 5.11%
    iter 30054: loss 1.8658, time 730.91ms, mfu 5.21%
    step 30055: train loss 1.9482, val loss 2.2376
    iter 30055: loss 1.7447, time 5847.18ms, mfu 4.77%
    iter 30056: loss 2.0283, time 737.93ms, mfu 4.90%
    iter 30057: loss 1.8879, time 735.51ms, mfu 5.02%
    iter 30058: loss 1.9657, time 729.71ms, mfu 5.13%
    iter 30059: loss 1.8924, time 734.19ms, mfu 5.23%
    step 30060: train loss 1.8967, val loss 2.1771
    saving checkpoint iter:60 to ../model
    iter 30060: loss 1.8944, time 12484.47ms, mfu 4.74%
    iter 30061: loss 1.9044, time 729.40ms, mfu 4.89%
    iter 30062: loss 2.0202, time 732.19ms, mfu 5.01%
    iter 30063: loss 1.8825, time 731.51ms, mfu 5.12%
    iter 30064: loss 1.8217, time 728.10ms, mfu 5.23%
    step 30065: train loss 1.8517, val loss 2.1567
    saving checkpoint iter:65 to ../model
    iter 30065: loss 1.8038, time 7145.80ms, mfu 4.77%
    iter 30066: loss 1.7137, time 727.47ms, mfu 4.91%
    iter 30067: loss 1.7463, time 727.46ms, mfu 5.03%
    iter 30068: loss 1.6276, time 727.42ms, mfu 5.15%
    iter 30069: loss 1.8804, time 728.96ms, mfu 5.25%
    step 30070: train loss 1.7836, val loss 2.1242
    saving checkpoint iter:70 to ../model
    iter 30070: loss 1.8292, time 7257.39ms, mfu 4.79%
    iter 30071: loss 1.8558, time 727.53ms, mfu 4.92%
    iter 30072: loss 1.5984, time 725.96ms, mfu 5.05%
    iter 30073: loss 2.0902, time 727.43ms, mfu 5.16%
    iter 30074: loss 1.6039, time 727.68ms, mfu 5.26%
    step 30075: train loss 1.7388, val loss 2.1052
    saving checkpoint iter:75 to ../model
    iter 30075: loss 1.7362, time 7126.41ms, mfu 4.80%
    iter 30076: loss 1.5794, time 728.01ms, mfu 4.94%
    iter 30077: loss 1.8709, time 740.04ms, mfu 5.05%
    iter 30078: loss 1.7640, time 740.88ms, mfu 5.15%
    iter 30079: loss 1.6376, time 737.64ms, mfu 5.24%
    step 30080: train loss 1.6713, val loss 2.0892
    saving checkpoint iter:80 to ../model
    iter 30080: loss 1.4982, time 7319.71ms, mfu 4.78%
    iter 30081: loss 1.7552, time 745.21ms, mfu 4.90%
    iter 30082: loss 1.6608, time 737.00ms, mfu 5.02%
    iter 30083: loss 1.7032, time 736.06ms, mfu 5.13%
    iter 30084: loss 1.7192, time 739.88ms, mfu 5.22%
    step 30085: train loss 1.6528, val loss 2.0511
    saving checkpoint iter:85 to ../model
    iter 30085: loss 1.6055, time 7273.07ms, mfu 4.76%
    iter 30086: loss 1.6744, time 737.33ms, mfu 4.90%
    iter 30087: loss 1.5528, time 741.22ms, mfu 5.01%
    iter 30088: loss 1.3920, time 740.25ms, mfu 5.12%
    iter 30089: loss 1.7297, time 734.97ms, mfu 5.22%
    step 30090: train loss 1.5873, val loss 2.0329
    saving checkpoint iter:90 to ../model
    iter 30090: loss 1.5023, time 7208.09ms, mfu 4.76%
    iter 30091: loss 1.5419, time 738.03ms, mfu 4.89%
    iter 30092: loss 1.5987, time 733.89ms, mfu 5.01%
    iter 30093: loss 1.7212, time 739.03ms, mfu 5.12%
    iter 30094: loss 1.5323, time 736.65ms, mfu 5.22%
    step 30095: train loss 1.5445, val loss 2.0086
    saving checkpoint iter:95 to ../model
    iter 30095: loss 1.6120, time 7228.75ms, mfu 4.76%
    iter 30096: loss 1.4834, time 759.63ms, mfu 4.87%
    iter 30097: loss 1.5308, time 749.56ms, mfu 4.98%
    iter 30098: loss 1.8503, time 747.69ms, mfu 5.09%
    iter 30099: loss 1.6519, time 741.19ms, mfu 5.18%
    step 30100: train loss 1.4790, val loss 1.9892
    saving checkpoint iter:100 to ../model
    iter 30100: loss 1.6191, time 7492.28ms, mfu 4.72%
    iter 30101: loss 1.3800, time 748.95ms, mfu 4.85%
    iter 30102: loss 1.6648, time 753.36ms, mfu 4.96%
    iter 30103: loss 1.3657, time 740.03ms, mfu 5.07%
    iter 30104: loss 1.5832, time 740.93ms, mfu 5.17%
    step 30105: train loss 1.4351, val loss 1.9763
    saving checkpoint iter:105 to ../model
    iter 30105: loss 1.3684, time 7329.89ms, mfu 4.71%
    iter 30106: loss 1.5597, time 734.90ms, mfu 4.85%
    iter 30107: loss 1.4045, time 738.99ms, mfu 4.98%
    iter 30108: loss 1.3864, time 731.76ms, mfu 5.09%
    iter 30109: loss 1.6928, time 734.85ms, mfu 5.19%
    step 30110: train loss 1.3784, val loss 1.9357
    saving checkpoint iter:110 to ../model
    iter 30110: loss 1.2639, time 7222.40ms, mfu 4.74%
    iter 30111: loss 1.2935, time 740.34ms, mfu 4.87%
    iter 30112: loss 1.4359, time 732.07ms, mfu 5.00%
    iter 30113: loss 1.2420, time 729.62ms, mfu 5.11%
    iter 30114: loss 1.1214, time 736.41ms, mfu 5.21%
    step 30115: train loss 1.3555, val loss 1.9098
    saving checkpoint iter:115 to ../model
    iter 30115: loss 1.2006, time 7227.52ms, mfu 4.75%
    iter 30116: loss 1.3583, time 729.62ms, mfu 4.89%
    iter 30117: loss 1.2892, time 740.55ms, mfu 5.01%
    iter 30118: loss 1.2130, time 747.82ms, mfu 5.11%
    iter 30119: loss 1.1647, time 728.07ms, mfu 5.21%
    step 30120: train loss 1.2740, val loss 1.8676
    saving checkpoint iter:120 to ../model
    iter 30120: loss 1.3765, time 7240.21ms, mfu 4.75%
    iter 30121: loss 1.4050, time 740.19ms, mfu 4.88%
    iter 30122: loss 1.1894, time 737.52ms, mfu 5.00%
    iter 30123: loss 1.2681, time 745.12ms, mfu 5.11%
    iter 30124: loss 1.5953, time 731.47ms, mfu 5.21%
    step 30125: train loss 1.2620, val loss 1.8782
    iter 30125: loss 1.0364, time 5867.36ms, mfu 4.77%
    iter 30126: loss 1.4937, time 736.13ms, mfu 4.90%
    iter 30127: loss 1.3429, time 732.24ms, mfu 5.02%
    iter 30128: loss 1.1869, time 733.92ms, mfu 5.13%
    iter 30129: loss 1.2386, time 737.87ms, mfu 5.23%
    step 30130: train loss 1.2311, val loss 1.8559
    saving checkpoint iter:130 to ../model
    iter 30130: loss 1.1230, time 7263.66ms, mfu 4.77%
    iter 30131: loss 0.9943, time 732.52ms, mfu 4.90%
    iter 30132: loss 1.3382, time 730.56ms, mfu 5.03%
    iter 30133: loss 1.3217, time 735.26ms, mfu 5.13%
    iter 30134: loss 1.2386, time 734.06ms, mfu 5.23%
    step 30135: train loss 1.1852, val loss 1.8354
    saving checkpoint iter:135 to ../model
    iter 30135: loss 1.1029, time 7200.29ms, mfu 4.77%
    iter 30136: loss 1.0426, time 731.44ms, mfu 4.91%
    iter 30137: loss 0.8245, time 728.12ms, mfu 5.03%
    iter 30138: loss 1.2839, time 730.32ms, mfu 5.14%
    iter 30139: loss 1.3225, time 731.72ms, mfu 5.24%
    step 30140: train loss 1.1209, val loss 1.7714
    saving checkpoint iter:140 to ../model
    iter 30140: loss 1.0725, time 7211.02ms, mfu 4.78%
    iter 30141: loss 1.3578, time 737.03ms, mfu 4.91%
    iter 30142: loss 1.0714, time 734.03ms, mfu 5.03%
    iter 30143: loss 1.2134, time 735.79ms, mfu 5.14%
    iter 30144: loss 1.2811, time 739.95ms, mfu 5.23%
    step 30145: train loss 1.0713, val loss 1.7633
    saving checkpoint iter:145 to ../model
    iter 30145: loss 0.9141, time 7230.23ms, mfu 4.77%
    iter 30146: loss 1.0154, time 742.34ms, mfu 4.90%
    iter 30147: loss 1.0599, time 731.80ms, mfu 5.02%
    iter 30148: loss 0.7711, time 736.09ms, mfu 5.13%
    iter 30149: loss 0.9923, time 736.39ms, mfu 5.23%
    step 30150: train loss 1.0449, val loss 1.7372
    saving checkpoint iter:150 to ../model
    iter 30150: loss 1.0665, time 7233.41ms, mfu 4.77%
    iter 30151: loss 1.0119, time 730.45ms, mfu 4.90%
    iter 30152: loss 0.7667, time 733.32ms, mfu 5.03%
    iter 30153: loss 0.9975, time 736.11ms, mfu 5.13%
    iter 30154: loss 1.4793, time 746.63ms, mfu 5.22%
    step 30155: train loss 0.9773, val loss 1.7639
    iter 30155: loss 0.8695, time 5847.44ms, mfu 4.78%
    iter 30156: loss 0.9042, time 735.83ms, mfu 4.91%
    iter 30157: loss 0.8293, time 746.08ms, mfu 5.02%
    iter 30158: loss 1.0286, time 753.93ms, mfu 5.11%
    iter 30159: loss 1.2572, time 731.71ms, mfu 5.21%
    step 30160: train loss 0.9427, val loss 1.7177
    saving checkpoint iter:160 to ../model
    iter 30160: loss 0.8212, time 7329.62ms, mfu 4.75%
    iter 30161: loss 0.9859, time 745.74ms, mfu 4.88%
    iter 30162: loss 0.7323, time 744.75ms, mfu 5.00%
    iter 30163: loss 0.8101, time 743.45ms, mfu 5.10%
    iter 30164: loss 0.6342, time 746.89ms, mfu 5.19%
    step 30165: train loss 0.8782, val loss 1.7120
    saving checkpoint iter:165 to ../model
    iter 30165: loss 1.1389, time 7353.69ms, mfu 4.73%
    iter 30166: loss 0.8907, time 740.28ms, mfu 4.87%
    iter 30167: loss 0.9108, time 737.99ms, mfu 4.99%
    iter 30168: loss 0.6041, time 743.46ms, mfu 5.09%
    iter 30169: loss 0.9960, time 739.60ms, mfu 5.19%
    step 30170: train loss 0.8965, val loss 1.6707
    saving checkpoint iter:170 to ../model
    iter 30170: loss 0.6852, time 7235.67ms, mfu 4.73%
    iter 30171: loss 1.0472, time 732.93ms, mfu 4.87%
    iter 30172: loss 0.9642, time 727.54ms, mfu 5.00%
    iter 30173: loss 0.9601, time 743.43ms, mfu 5.11%
    iter 30174: loss 0.5586, time 746.00ms, mfu 5.20%
    step 30175: train loss 0.8369, val loss 1.6511
    saving checkpoint iter:175 to ../model
    iter 30175: loss 0.8504, time 7273.72ms, mfu 4.74%
    iter 30176: loss 0.7648, time 743.25ms, mfu 4.87%
    iter 30177: loss 0.5661, time 735.44ms, mfu 4.99%
    iter 30178: loss 0.8062, time 738.74ms, mfu 5.10%
    iter 30179: loss 0.6502, time 738.12ms, mfu 5.20%
    step 30180: train loss 0.8113, val loss 1.6448
    saving checkpoint iter:180 to ../model
    iter 30180: loss 0.7237, time 7313.86ms, mfu 4.74%
    iter 30181: loss 0.8398, time 760.81ms, mfu 4.86%
    iter 30182: loss 1.0733, time 739.97ms, mfu 4.98%
    iter 30183: loss 0.7762, time 729.55ms, mfu 5.09%
    iter 30184: loss 0.5699, time 765.94ms, mfu 5.17%
    step 30185: train loss 0.7547, val loss 1.6087
    saving checkpoint iter:185 to ../model
    iter 30185: loss 0.8090, time 8067.84ms, mfu 4.71%
    iter 30186: loss 0.7726, time 730.10ms, mfu 4.85%
    iter 30187: loss 0.8009, time 731.03ms, mfu 4.98%
    iter 30188: loss 0.7843, time 733.88ms, mfu 5.10%
    iter 30189: loss 0.6593, time 730.71ms, mfu 5.20%
    step 30190: train loss 0.7175, val loss 1.6225
    iter 30190: loss 0.6148, time 5813.93ms, mfu 4.76%
    iter 30191: loss 0.7376, time 732.66ms, mfu 4.89%
    iter 30192: loss 0.9104, time 729.78ms, mfu 5.02%
    iter 30193: loss 0.9252, time 729.96ms, mfu 5.13%
    iter 30194: loss 0.6398, time 733.55ms, mfu 5.23%
    step 30195: train loss 0.7070, val loss 1.5686
    saving checkpoint iter:195 to ../model
    iter 30195: loss 0.5951, time 7196.73ms, mfu 4.77%
    iter 30196: loss 0.9564, time 731.57ms, mfu 4.91%
    iter 30197: loss 0.7297, time 731.72ms, mfu 5.03%
    iter 30198: loss 0.8111, time 733.98ms, mfu 5.14%
    iter 30199: loss 0.5247, time 730.23ms, mfu 5.24%
    step 30200: train loss 0.6288, val loss 1.5493
    saving checkpoint iter:200 to ../model
    iter 30200: loss 0.5281, time 7151.85ms, mfu 4.78%
    iter 30201: loss 0.6851, time 733.26ms, mfu 4.91%
    iter 30202: loss 0.7848, time 729.64ms, mfu 5.04%
    iter 30203: loss 0.7905, time 731.38ms, mfu 5.15%
    iter 30204: loss 0.7114, time 731.82ms, mfu 5.25%
    step 30205: train loss 0.6202, val loss 1.5057
    saving checkpoint iter:205 to ../model
    iter 30205: loss 0.5821, time 7144.20ms, mfu 4.78%
    iter 30206: loss 0.5489, time 728.92ms, mfu 4.92%
    iter 30207: loss 0.6309, time 733.05ms, mfu 5.04%
    iter 30208: loss 0.7083, time 731.69ms, mfu 5.15%
    iter 30209: loss 0.6347, time 734.33ms, mfu 5.25%
    step 30210: train loss 0.6018, val loss 1.5004
    saving checkpoint iter:210 to ../model
    iter 30210: loss 0.2972, time 7181.91ms, mfu 4.78%
    iter 30211: loss 0.5472, time 733.92ms, mfu 4.92%
    iter 30212: loss 0.4290, time 733.69ms, mfu 5.04%
    iter 30213: loss 0.3347, time 730.11ms, mfu 5.15%
    iter 30214: loss 0.7689, time 732.45ms, mfu 5.25%
    step 30215: train loss 0.5545, val loss 1.4551
    saving checkpoint iter:215 to ../model
    iter 30215: loss 0.5060, time 7199.55ms, mfu 4.78%
    iter 30216: loss 0.6880, time 731.71ms, mfu 4.92%
    iter 30217: loss 0.5519, time 728.40ms, mfu 5.04%
    iter 30218: loss 0.4901, time 732.20ms, mfu 5.15%
    iter 30219: loss 0.4493, time 733.65ms, mfu 5.25%
    step 30220: train loss 0.5498, val loss 1.4628
    iter 30220: loss 0.4424, time 5816.01ms, mfu 4.80%
    iter 30221: loss 0.4947, time 727.96ms, mfu 4.94%
    iter 30222: loss 0.5255, time 732.17ms, mfu 5.06%
    iter 30223: loss 0.5176, time 731.39ms, mfu 5.17%
    iter 30224: loss 0.5298, time 730.30ms, mfu 5.26%
    step 30225: train loss 0.5112, val loss 1.4632
    iter 30225: loss 0.6169, time 5821.45ms, mfu 4.81%
    iter 30226: loss 0.4410, time 732.89ms, mfu 4.94%
    iter 30227: loss 0.5253, time 729.98ms, mfu 5.07%
    iter 30228: loss 0.3432, time 729.90ms, mfu 5.17%
    iter 30229: loss 0.4821, time 730.01ms, mfu 5.27%
    step 30230: train loss 0.4854, val loss 1.4288
    saving checkpoint iter:230 to ../model
    iter 30230: loss 0.4087, time 7163.98ms, mfu 4.81%
    iter 30231: loss 0.7423, time 733.36ms, mfu 4.94%
    iter 30232: loss 0.3685, time 731.00ms, mfu 5.06%
    iter 30233: loss 0.6375, time 727.78ms, mfu 5.17%
    iter 30234: loss 0.4405, time 733.65ms, mfu 5.26%
    step 30235: train loss 0.4694, val loss 1.3710
    saving checkpoint iter:235 to ../model
    iter 30235: loss 0.5556, time 7170.27ms, mfu 4.80%
    iter 30236: loss 0.4910, time 730.11ms, mfu 4.94%
    iter 30237: loss 0.3843, time 732.05ms, mfu 5.05%
    iter 30238: loss 0.4867, time 727.55ms, mfu 5.17%
    iter 30239: loss 0.3070, time 734.34ms, mfu 5.26%
    step 30240: train loss 0.4458, val loss 1.3857
    iter 30240: loss 0.5191, time 5816.49ms, mfu 4.81%
    iter 30241: loss 0.5119, time 729.08ms, mfu 4.95%
    iter 30242: loss 0.5499, time 730.37ms, mfu 5.07%
    iter 30243: loss 0.5617, time 729.78ms, mfu 5.17%
    iter 30244: loss 0.5282, time 731.85ms, mfu 5.27%
    step 30245: train loss 0.4119, val loss 1.3663
    saving checkpoint iter:245 to ../model
    iter 30245: loss 0.4893, time 7168.26ms, mfu 4.81%
    iter 30246: loss 0.4040, time 730.22ms, mfu 4.94%
    iter 30247: loss 0.4247, time 731.56ms, mfu 5.06%
    iter 30248: loss 0.2806, time 732.64ms, mfu 5.17%
    iter 30249: loss 0.3851, time 727.29ms, mfu 5.27%
    step 30250: train loss 0.4038, val loss 1.3360
    saving checkpoint iter:250 to ../model
    iter 30250: loss 0.5003, time 7178.37ms, mfu 4.80%
    iter 30251: loss 0.2816, time 730.29ms, mfu 4.94%
    iter 30252: loss 0.4015, time 731.36ms, mfu 5.06%
    iter 30253: loss 0.4386, time 734.94ms, mfu 5.16%
    iter 30254: loss 0.3541, time 729.04ms, mfu 5.26%
    step 30255: train loss 0.3814, val loss 1.3175
    saving checkpoint iter:255 to ../model
    iter 30255: loss 0.3733, time 8540.36ms, mfu 4.79%
    iter 30256: loss 0.2950, time 731.64ms, mfu 4.92%
    iter 30257: loss 0.4622, time 730.42ms, mfu 5.05%
    iter 30258: loss 0.3631, time 731.62ms, mfu 5.15%
    iter 30259: loss 0.4237, time 725.91ms, mfu 5.26%
    step 30260: train loss 0.3737, val loss 1.3074
    saving checkpoint iter:260 to ../model
    iter 30260: loss 0.2535, time 7134.07ms, mfu 4.79%
    iter 30261: loss 0.3342, time 733.41ms, mfu 4.93%
    iter 30262: loss 0.3809, time 730.47ms, mfu 5.05%
    iter 30263: loss 0.2660, time 728.02ms, mfu 5.16%
    iter 30264: loss 0.3487, time 731.98ms, mfu 5.26%
    step 30265: train loss 0.3460, val loss 1.2722
    saving checkpoint iter:265 to ../model
    iter 30265: loss 0.3751, time 7210.42ms, mfu 4.79%
    iter 30266: loss 0.2951, time 732.26ms, mfu 4.93%
    iter 30267: loss 0.2764, time 725.75ms, mfu 5.05%
    iter 30268: loss 0.2697, time 731.64ms, mfu 5.16%
    iter 30269: loss 0.2731, time 736.32ms, mfu 5.25%
    step 30270: train loss 0.3341, val loss 1.2842
    iter 30270: loss 0.2556, time 5837.75ms, mfu 4.81%
    iter 30271: loss 0.3196, time 731.82ms, mfu 4.94%
    iter 30272: loss 0.3316, time 732.05ms, mfu 5.06%
    iter 30273: loss 0.2164, time 730.55ms, mfu 5.17%
    iter 30274: loss 0.3064, time 733.36ms, mfu 5.26%
    step 30275: train loss 0.3268, val loss 1.2593
    saving checkpoint iter:275 to ../model
    iter 30275: loss 0.2999, time 7180.14ms, mfu 4.80%
    iter 30276: loss 0.3840, time 728.23ms, mfu 4.94%
    iter 30277: loss 0.3636, time 734.02ms, mfu 5.05%
    iter 30278: loss 0.3295, time 731.55ms, mfu 5.16%
    iter 30279: loss 0.3018, time 730.39ms, mfu 5.26%
    step 30280: train loss 0.3015, val loss 1.2345
    saving checkpoint iter:280 to ../model
    iter 30280: loss 0.3712, time 7181.67ms, mfu 4.80%
    iter 30281: loss 0.3475, time 730.92ms, mfu 4.93%
    iter 30282: loss 0.3878, time 733.01ms, mfu 5.05%
    iter 30283: loss 0.2848, time 729.19ms, mfu 5.16%
    iter 30284: loss 0.3418, time 733.27ms, mfu 5.26%
    step 30285: train loss 0.3059, val loss 1.2325
    saving checkpoint iter:285 to ../model
    iter 30285: loss 0.2778, time 8689.59ms, mfu 4.78%
    iter 30286: loss 0.2733, time 728.18ms, mfu 4.92%
    iter 30287: loss 0.3063, time 726.98ms, mfu 5.05%
    iter 30288: loss 0.2493, time 729.17ms, mfu 5.16%
    iter 30289: loss 0.2600, time 727.93ms, mfu 5.26%
    step 30290: train loss 0.2878, val loss 1.1935
    saving checkpoint iter:290 to ../model
    iter 30290: loss 0.2906, time 7200.44ms, mfu 4.79%
    iter 30291: loss 0.2787, time 729.80ms, mfu 4.93%
    iter 30292: loss 0.2134, time 729.89ms, mfu 5.05%
    iter 30293: loss 0.2536, time 725.60ms, mfu 5.17%
    iter 30294: loss 0.2898, time 730.21ms, mfu 5.26%
    step 30295: train loss 0.2820, val loss 1.2440
    iter 30295: loss 0.2426, time 5822.00ms, mfu 4.81%
    iter 30296: loss 0.2967, time 733.71ms, mfu 4.94%
    iter 30297: loss 0.2596, time 730.95ms, mfu 5.06%
    iter 30298: loss 0.2331, time 733.30ms, mfu 5.17%
    iter 30299: loss 0.2681, time 728.02ms, mfu 5.27%
    step 30300: train loss 0.2736, val loss 1.1575
    saving checkpoint iter:300 to ../model
    iter 30300: loss 0.1791, time 7182.01ms, mfu 4.81%
    iter 30301: loss 0.2525, time 733.87ms, mfu 4.94%
    iter 30302: loss 0.2097, time 729.77ms, mfu 5.06%
    iter 30303: loss 0.2319, time 730.31ms, mfu 5.17%
    iter 30304: loss 0.2418, time 735.44ms, mfu 5.26%
    step 30305: train loss 0.2534, val loss 1.1883
    iter 30305: loss 0.2736, time 5822.21ms, mfu 4.81%
    iter 30306: loss 0.2608, time 732.16ms, mfu 4.94%
    iter 30307: loss 0.1746, time 733.82ms, mfu 5.06%
    iter 30308: loss 0.2700, time 731.80ms, mfu 5.17%
    iter 30309: loss 0.2314, time 729.74ms, mfu 5.27%
    step 30310: train loss 0.2469, val loss 1.1280
    saving checkpoint iter:310 to ../model
    iter 30310: loss 0.2439, time 7214.84ms, mfu 4.80%
    iter 30311: loss 0.1958, time 730.58ms, mfu 4.94%
    iter 30312: loss 0.2698, time 729.94ms, mfu 5.06%
    iter 30313: loss 0.4190, time 729.97ms, mfu 5.17%
    iter 30314: loss 0.2244, time 731.74ms, mfu 5.26%
    step 30315: train loss 0.2446, val loss 1.1252
    saving checkpoint iter:315 to ../model
    iter 30315: loss 0.2271, time 8882.26ms, mfu 4.79%
    iter 30316: loss 0.2412, time 729.15ms, mfu 4.92%
    iter 30317: loss 0.3992, time 728.20ms, mfu 5.05%
    iter 30318: loss 0.2712, time 728.66ms, mfu 5.16%
    iter 30319: loss 0.2898, time 729.40ms, mfu 5.26%
    step 30320: train loss 0.2290, val loss 1.1333
    iter 30320: loss 0.1777, time 5820.06ms, mfu 4.81%
    iter 30321: loss 0.2567, time 731.02ms, mfu 4.94%
    iter 30322: loss 0.2471, time 730.60ms, mfu 5.06%
    iter 30323: loss 0.1827, time 731.96ms, mfu 5.17%
    iter 30324: loss 0.2002, time 730.40ms, mfu 5.27%
    step 30325: train loss 0.2165, val loss 1.1374
    iter 30325: loss 0.2024, time 5820.04ms, mfu 4.82%
    iter 30326: loss 0.2027, time 732.00ms, mfu 4.95%
    iter 30327: loss 0.2145, time 733.77ms, mfu 5.07%
    iter 30328: loss 0.2143, time 729.80ms, mfu 5.17%
    iter 30329: loss 0.1616, time 733.92ms, mfu 5.27%
    step 30330: train loss 0.2263, val loss 1.1134
    saving checkpoint iter:330 to ../model
    iter 30330: loss 0.2035, time 7237.29ms, mfu 4.80%
    iter 30331: loss 0.1972, time 736.04ms, mfu 4.93%
    iter 30332: loss 0.2817, time 732.98ms, mfu 5.05%
    iter 30333: loss 0.2197, time 730.36ms, mfu 5.16%
    iter 30334: loss 0.1867, time 729.49ms, mfu 5.26%
    step 30335: train loss 0.2182, val loss 1.0755
    saving checkpoint iter:335 to ../model
    iter 30335: loss 0.1663, time 7251.24ms, mfu 4.80%
    iter 30336: loss 0.2132, time 730.93ms, mfu 4.93%
    iter 30337: loss 0.2528, time 730.96ms, mfu 5.05%
    iter 30338: loss 0.2939, time 729.30ms, mfu 5.16%
    iter 30339: loss 0.2743, time 731.89ms, mfu 5.26%
    step 30340: train loss 0.2136, val loss 1.1121
    iter 30340: loss 0.2575, time 5819.67ms, mfu 4.81%
    iter 30341: loss 0.2265, time 733.94ms, mfu 4.94%
    iter 30342: loss 0.1698, time 728.22ms, mfu 5.06%
    iter 30343: loss 0.2449, time 733.65ms, mfu 5.17%
    iter 30344: loss 0.1518, time 728.28ms, mfu 5.27%
    step 30345: train loss 0.2053, val loss 1.1077
    iter 30345: loss 0.3925, time 5881.88ms, mfu 4.82%
    iter 30346: loss 0.2387, time 729.94ms, mfu 4.95%
    iter 30347: loss 0.1973, time 732.51ms, mfu 5.07%
    iter 30348: loss 0.1911, time 731.57ms, mfu 5.18%
    iter 30349: loss 0.1662, time 733.94ms, mfu 5.27%
    step 30350: train loss 0.2046, val loss 1.1183
    iter 30350: loss 0.1998, time 5823.06ms, mfu 4.82%
    iter 30351: loss 0.2055, time 732.74ms, mfu 4.95%
    iter 30352: loss 0.1832, time 732.25ms, mfu 5.07%
    iter 30353: loss 0.1752, time 728.05ms, mfu 5.18%
    iter 30354: loss 0.2187, time 732.06ms, mfu 5.27%
    step 30355: train loss 0.2001, val loss 1.0708
    saving checkpoint iter:355 to ../model
    iter 30355: loss 0.1863, time 12736.14ms, mfu 4.78%
    iter 30356: loss 0.2032, time 727.92ms, mfu 4.92%
    iter 30357: loss 0.2249, time 727.85ms, mfu 5.04%
    iter 30358: loss 0.2752, time 730.10ms, mfu 5.15%
    iter 30359: loss 0.1798, time 729.57ms, mfu 5.25%
    step 30360: train loss 0.1911, val loss 1.1138
    iter 30360: loss 0.2119, time 5827.74ms, mfu 4.81%"""

    pattern = re.compile(r"step (\d+): train loss ([\d\.]+), val loss ([\d\.]+)")

    # Extracting the losses
    steps, train_losses, val_losses = [], [], []
    for match in re.finditer(pattern, log_data):
        steps.append(int(match.group(1)))
        train_losses.append(float(match.group(2)))
        val_losses.append(float(match.group(3)))

    # Plotting the losses
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label='Train Loss')
    plt.plot(steps, val_losses, label='Validation Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def ScoreDistribution():
    rubric_0 = pd.read_csv('../data/rubric_0.csv')
    rubric_1 = pd.read_csv('../data/rubric_1.csv')
    rubric_2 = pd.read_csv('../data/rubric_2.csv')
    rubric_3 = pd.read_csv('../data/rubric_3.csv')
    rubric_4 = pd.read_csv('../data/rubric_4.csv')

    rubrics = [rubric_0, rubric_1, rubric_2, rubric_3, rubric_4]

    plt.figure(figsize=(15, 15))

    Titles = ['Accuracy and Factuality',
             'Relevance and Comprehension',
             'Effectiveness',
             'Grammar and Clarity',
             'Creativity and Novelty']
    
    for i, rubric in enumerate(rubrics):
        plt.subplot(3, 2, i+1)
        sns.boxplot(data=rubric)
        plt.title(Titles[i])
        plt.ylabel('Scores')
        plt.xlabel('Models')

    plt.show()

def radar():
    Titles = ['Accuracy and Factuality',
             'Relevance and Comprehension',
             'Effectiveness',
             'Grammar and Clarity',
             'Creativity and Novelty']
    
    rubric_0 = pd.read_csv('../data/rubric_0.csv')
    rubric_1 = pd.read_csv('../data/rubric_1.csv')
    rubric_2 = pd.read_csv('../data/rubric_2.csv')
    rubric_3 = pd.read_csv('../data/rubric_3.csv')
    rubric_4 = pd.read_csv('../data/rubric_4.csv')


    llama = [rubric_0.mean()[0], 
             rubric_1.mean()[0],
             rubric_2.mean()[0],
             rubric_3.mean()[0],
             rubric_4.mean()[0]]
    
    default = [rubric_0.mean()[1],
               rubric_1.mean()[1],
               rubric_2.mean()[1],
               rubric_3.mean()[1],
               rubric_4.mean()[1]]
    
    randombot = [rubric_0.mean()[2],
                 rubric_1.mean()[2],
                 rubric_2.mean()[2],
                 rubric_3.mean()[2],
                 rubric_4.mean()[2]]
    
    target = [rubric_0.mean()[3],
              rubric_1.mean()[3],
              rubric_2.mean()[3],
              rubric_3.mean()[3],
              rubric_4.mean()[3]]

    fig = go.Figure()

    '''fig.add_trace(go.Scatterpolar(
        r=llama,
        theta=Titles,
        fill='toself',
        name='LlaMa2'
    ))
'''
    fig.add_trace(go.Scatterpolar(
        r=default,
        theta=Titles,
        fill='toself',
        name='Default'
    ))

    fig.add_trace(go.Scatterpolar(
        r=randombot,
        theta=Titles,
        fill='toself',
        name='RandomChatBot'
    ))

    fig.add_trace(go.Scatterpolar(
        r=target,
        theta=Titles,
        fill='toself',
        name='Target'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 2.5])
        ),
        title='Comparison of Model Scores',
        template='plotly',
        showlegend=True
    )

    fig.show()

if __name__ == '__main__': 
    #ScoreDistribution()
    radar()
