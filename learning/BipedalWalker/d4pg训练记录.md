C:\Users\admin\.conda\envs\pytorch-gym\python.exe K:\Projects\python\DQN\learning\BipedalWalker\train_d4pg.py -n d4pg 
DDPGActor(
  (net): Sequential(
    (0): Linear(in_features=24, out_features=400, bias=True)
    (1): ReLU()
    (2): Linear(in_features=400, out_features=300, bias=True)
    (3): ReLU()
    (4): Linear(in_features=300, out_features=4, bias=True)
    (5): Tanh()
  )
)
D4PGCritic(
  (obs_net): Sequential(
    (0): Linear(in_features=24, out_features=400, bias=True)
    (1): ReLU()
  )
  (out_net): Sequential(
    (0): Linear(in_features=404, out_features=300, bias=True)
    (1): ReLU()
    (2): Linear(in_features=300, out_features=51, bias=True)
  )
)
1105: done 10 episodes, mean reward -102.993, speed 1084.90 f/s
2175: done 18 episodes, mean reward -105.997, speed 983.54 f/s
3672: done 25 episodes, mean reward -107.104, speed 1199.94 f/s
5676: done 32 episodes, mean reward -108.471, speed 1187.16 f/s
6902: done 44 episodes, mean reward -107.871, speed 1189.97 f/s
8270: done 47 episodes, mean reward -108.998, speed 1179.50 f/s
K:\Projects\python\DQN\learning\BipedalWalker\train_d4pg.py:59: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  dones_mask = dones_mask_t.cpu().numpy().astype(np.bool)
Test done in 0.80 sec, reward -92.329, steps 102
10021: done 51 episodes, mean reward -108.812, speed 638.52 f/s
10169: done 52 episodes, mean reward -108.814, speed 57.13 f/s
10254: done 53 episodes, mean reward -109.103, speed 73.83 f/s
10456: done 54 episodes, mean reward -109.300, speed 69.31 f/s
10578: done 55 episodes, mean reward -109.382, speed 66.71 f/s
10686: done 56 episodes, mean reward -109.444, speed 65.54 f/s
10802: done 57 episodes, mean reward -109.511, speed 68.23 f/s
10933: done 58 episodes, mean reward -109.581, speed 68.48 f/s
Test done in 1.23 sec, reward -116.071, steps 130
11064: done 59 episodes, mean reward -109.657, speed 40.46 f/s
11203: done 60 episodes, mean reward -109.721, speed 63.35 f/s
11345: done 61 episodes, mean reward -109.788, speed 67.02 f/s
11449: done 62 episodes, mean reward -109.826, speed 65.61 f/s
11551: done 63 episodes, mean reward -109.867, speed 65.02 f/s
11629: done 64 episodes, mean reward -109.895, speed 61.33 f/s
11742: done 65 episodes, mean reward -109.950, speed 62.97 f/s
11868: done 66 episodes, mean reward -110.007, speed 52.40 f/s
Test done in 0.86 sec, reward -116.203, steps 132
12025: done 68 episodes, mean reward -110.205, speed 46.43 f/s
12193: done 70 episodes, mean reward -110.392, speed 61.44 f/s
12323: done 71 episodes, mean reward -110.442, speed 66.93 f/s
12466: done 72 episodes, mean reward -110.498, speed 66.27 f/s
12598: done 73 episodes, mean reward -110.535, speed 63.18 f/s
12723: done 74 episodes, mean reward -110.577, speed 58.01 f/s
12858: done 75 episodes, mean reward -110.619, speed 67.40 f/s
Test done in 1.04 sec, reward -116.047, steps 130
13006: done 76 episodes, mean reward -110.686, speed 48.09 f/s
13117: done 78 episodes, mean reward -110.815, speed 65.50 f/s
13251: done 79 episodes, mean reward -110.862, speed 64.75 f/s
13362: done 80 episodes, mean reward -110.884, speed 67.95 f/s
13503: done 82 episodes, mean reward -111.001, speed 54.72 f/s
13673: done 84 episodes, mean reward -111.138, speed 62.55 f/s
13851: done 86 episodes, mean reward -111.290, speed 69.62 f/s
13928: done 87 episodes, mean reward -111.286, speed 63.04 f/s
13994: done 88 episodes, mean reward -111.288, speed 61.14 f/s
Test done in 0.89 sec, reward -115.556, steps 123
14033: done 89 episodes, mean reward -111.370, speed 26.22 f/s
14164: done 90 episodes, mean reward -111.394, speed 63.41 f/s
14341: done 92 episodes, mean reward -111.507, speed 57.27 f/s
14503: done 93 episodes, mean reward -111.565, speed 68.61 f/s
14669: done 95 episodes, mean reward -111.687, speed 69.28 f/s
14784: done 96 episodes, mean reward -111.696, speed 69.42 f/s
Test done in 0.55 sec, reward -99.203, steps 79
Test done in 8.11 sec, reward -128.947, steps 1164
16384: done 97 episodes, mean reward -111.984, speed 49.22 f/s
16586: done 98 episodes, mean reward -111.925, speed 71.50 f/s
16662: done 99 episodes, mean reward -111.984, speed 73.46 f/s
16778: done 101 episodes, mean reward -112.148, speed 67.84 f/s
16917: done 102 episodes, mean reward -112.394, speed 68.55 f/s
Test done in 0.75 sec, reward -121.772, steps 106
17122: done 104 episodes, mean reward -112.584, speed 54.15 f/s
17366: done 105 episodes, mean reward -112.375, speed 60.69 f/s
Test done in 11.58 sec, reward -103.071, steps 1600
18281: done 106 episodes, mean reward -113.003, speed 34.38 f/s
18373: done 108 episodes, mean reward -113.191, speed 70.51 f/s
18465: done 110 episodes, mean reward -113.174, speed 64.88 f/s
18554: done 112 episodes, mean reward -113.230, speed 70.01 f/s
18674: done 114 episodes, mean reward -113.324, speed 66.85 f/s
Test done in 12.50 sec, reward -105.400, steps 1600
19383: done 116 episodes, mean reward -113.508, speed 29.22 f/s
19506: done 118 episodes, mean reward -113.287, speed 69.62 f/s
19628: done 119 episodes, mean reward -113.238, speed 68.06 f/s
19894: done 121 episodes, mean reward -113.164, speed 63.30 f/s
Test done in 0.61 sec, reward -105.297, steps 63
20815: done 122 episodes, mean reward -113.596, speed 63.18 f/s
20934: done 123 episodes, mean reward -113.854, speed 58.82 f/s
Test done in 1.19 sec, reward -103.474, steps 143
21208: done 124 episodes, mean reward -113.783, speed 45.83 f/s
21300: done 125 episodes, mean reward -113.675, speed 68.34 f/s
21502: done 127 episodes, mean reward -113.895, speed 64.77 f/s
21649: done 128 episodes, mean reward -113.984, speed 56.38 f/s
21755: done 129 episodes, mean reward -113.939, speed 67.91 f/s
21870: done 130 episodes, mean reward -113.858, speed 70.05 f/s
Test done in 0.51 sec, reward -100.521, steps 69
22032: done 131 episodes, mean reward -113.972, speed 47.16 f/s
22112: done 132 episodes, mean reward -113.375, speed 63.25 f/s
22218: done 133 episodes, mean reward -113.424, speed 62.69 f/s
22302: done 134 episodes, mean reward -113.198, speed 68.82 f/s
22445: done 135 episodes, mean reward -113.226, speed 60.22 f/s
22524: done 136 episodes, mean reward -113.218, speed 58.05 f/s
22690: done 138 episodes, mean reward -113.078, speed 66.68 f/s
22791: done 139 episodes, mean reward -112.984, speed 64.24 f/s
Test done in 9.16 sec, reward -122.813, steps 1149
Test done in 11.51 sec, reward -123.027, steps 1600
24391: done 140 episodes, mean reward -113.056, speed 35.60 f/s
Test done in 12.44 sec, reward -91.659, steps 1600
Best reward updated: -92.329 -> -91.659
25991: done 141 episodes, mean reward -112.938, speed 44.17 f/s
Test done in 13.58 sec, reward -109.117, steps 1600
Test done in 0.72 sec, reward -110.093, steps 97
27591: done 142 episodes, mean reward -112.487, speed 38.30 f/s
27670: done 143 episodes, mean reward -112.435, speed 62.39 f/s
27761: done 144 episodes, mean reward -112.435, speed 65.89 f/s
27834: done 145 episodes, mean reward -111.953, speed 69.09 f/s
27925: done 146 episodes, mean reward -112.093, speed 60.98 f/s
Test done in 0.48 sec, reward -110.791, steps 73
28001: done 147 episodes, mean reward -112.022, speed 49.64 f/s
28088: done 148 episodes, mean reward -112.054, speed 68.16 f/s
28267: done 149 episodes, mean reward -112.233, speed 60.07 f/s
28339: done 150 episodes, mean reward -112.390, speed 64.78 f/s
28420: done 151 episodes, mean reward -112.191, speed 55.35 f/s
28477: done 152 episodes, mean reward -112.236, speed 54.02 f/s
28539: done 153 episodes, mean reward -112.148, speed 54.61 f/s
28650: done 154 episodes, mean reward -111.958, speed 52.34 f/s
28747: done 155 episodes, mean reward -111.916, speed 55.77 f/s
28847: done 156 episodes, mean reward -111.805, speed 56.33 f/s
Test done in 1.12 sec, reward -106.788, steps 144
29203: done 157 episodes, mean reward -111.815, speed 47.33 f/s
29386: done 158 episodes, mean reward -111.784, speed 52.51 f/s
29568: done 159 episodes, mean reward -111.681, speed 44.17 f/s
29725: done 160 episodes, mean reward -111.610, speed 55.97 f/s
Test done in 7.89 sec, reward -90.747, steps 897
Best reward updated: -91.659 -> -90.747
30069: done 161 episodes, mean reward -111.723, speed 24.49 f/s
30348: done 162 episodes, mean reward -111.751, speed 53.94 f/s
30472: done 163 episodes, mean reward -111.667, speed 56.18 f/s
30729: done 164 episodes, mean reward -111.719, speed 52.43 f/s
30847: done 165 episodes, mean reward -111.630, speed 43.94 f/s
Test done in 5.54 sec, reward -130.415, steps 600
Test done in 6.01 sec, reward -66.946, steps 660
Best reward updated: -90.747 -> -66.946
32171: done 166 episodes, mean reward -111.876, speed 36.86 f/s
Test done in 14.44 sec, reward -60.038, steps 1600
Best reward updated: -66.946 -> -60.038
33136: done 167 episodes, mean reward -111.779, speed 29.98 f/s
33880: done 168 episodes, mean reward -111.625, speed 55.26 f/s
Test done in 13.33 sec, reward -59.809, steps 1469
Best reward updated: -60.038 -> -59.809
34338: done 169 episodes, mean reward -111.323, speed 21.09 f/s
34440: done 170 episodes, mean reward -111.194, speed 54.87 f/s
34490: done 171 episodes, mean reward -111.276, speed 49.80 f/s
Test done in 14.23 sec, reward -64.737, steps 1600
35178: done 172 episodes, mean reward -111.134, speed 25.56 f/s
35422: done 173 episodes, mean reward -110.905, speed 54.59 f/s
35650: done 174 episodes, mean reward -110.723, speed 53.80 f/s
35749: done 175 episodes, mean reward -110.682, speed 53.43 f/s
35837: done 176 episodes, mean reward -110.504, speed 53.80 f/s
35995: done 177 episodes, mean reward -110.174, speed 51.66 f/s
Test done in 2.92 sec, reward -86.790, steps 330
36307: done 178 episodes, mean reward -109.954, speed 35.45 f/s
36386: done 179 episodes, mean reward -109.795, speed 52.37 f/s
36564: done 180 episodes, mean reward -109.609, speed 53.69 f/s
36635: done 181 episodes, mean reward -109.479, speed 56.52 f/s
36742: done 182 episodes, mean reward -109.635, speed 52.68 f/s
36875: done 183 episodes, mean reward -109.383, speed 51.57 f/s
Test done in 7.35 sec, reward -75.054, steps 806
37295: done 184 episodes, mean reward -108.978, speed 27.89 f/s
37368: done 185 episodes, mean reward -108.777, speed 55.81 f/s
37608: done 186 episodes, mean reward -108.652, speed 54.38 f/s
Test done in 10.78 sec, reward -68.338, steps 1197
38030: done 187 episodes, mean reward -108.548, speed 22.90 f/s
Test done in 13.39 sec, reward -61.762, steps 1484
39630: done 188 episodes, mean reward -107.689, speed 37.44 f/s
Test done in 14.40 sec, reward -62.631, steps 1600
Test done in 13.78 sec, reward -67.486, steps 1455
41230: done 189 episodes, mean reward -106.907, speed 27.40 f/s
Test done in 9.46 sec, reward -84.021, steps 1010
42830: done 190 episodes, mean reward -106.559, speed 40.67 f/s
42960: done 191 episodes, mean reward -106.346, speed 52.92 f/s
Test done in 14.75 sec, reward -73.889, steps 1600
Test done in 3.68 sec, reward -96.691, steps 386
44560: done 192 episodes, mean reward -106.087, speed 32.37 f/s
Test done in 14.82 sec, reward -86.432, steps 1600
Test done in 10.42 sec, reward -83.822, steps 1138
46160: done 193 episodes, mean reward -105.681, speed 29.21 f/s
46341: done 194 episodes, mean reward -105.422, speed 51.86 f/s
Test done in 12.72 sec, reward -59.695, steps 1309
Best reward updated: -59.809 -> -59.695
47397: done 195 episodes, mean reward -105.641, speed 31.51 f/s
47578: done 196 episodes, mean reward -105.546, speed 52.24 f/s
47662: done 197 episodes, mean reward -105.179, speed 49.34 f/s
Test done in 15.31 sec, reward -44.779, steps 1600
Best reward updated: -59.695 -> -44.779
Test done in 12.94 sec, reward -63.619, steps 1460
49262: done 198 episodes, mean reward -104.542, speed 27.47 f/s
49361: done 199 episodes, mean reward -104.325, speed 55.78 f/s
49420: done 200 episodes, mean reward -104.159, speed 51.45 f/s
Test done in 12.59 sec, reward -44.156, steps 1311
Best reward updated: -44.779 -> -44.156
50079: done 201 episodes, mean reward -103.971, speed 26.12 f/s
50454: done 202 episodes, mean reward -103.732, speed 46.97 f/s
50761: done 203 episodes, mean reward -103.474, speed 48.54 f/s
Test done in 7.42 sec, reward -58.934, steps 822
51265: done 204 episodes, mean reward -103.338, speed 30.51 f/s
51331: done 205 episodes, mean reward -103.302, speed 51.04 f/s
51419: done 206 episodes, mean reward -102.651, speed 52.47 f/s
51849: done 207 episodes, mean reward -102.350, speed 53.68 f/s
Test done in 1.91 sec, reward -88.828, steps 204
52157: done 208 episodes, mean reward -102.096, speed 40.43 f/s
52406: done 209 episodes, mean reward -101.902, speed 53.03 f/s
Test done in 6.66 sec, reward -36.945, steps 743
Best reward updated: -44.156 -> -36.945
53012: done 210 episodes, mean reward -101.494, speed 33.14 f/s
53177: done 211 episodes, mean reward -101.329, speed 53.30 f/s
53478: done 212 episodes, mean reward -101.037, speed 53.70 f/s
53695: done 213 episodes, mean reward -100.799, speed 53.57 f/s
53926: done 214 episodes, mean reward -100.592, speed 52.63 f/s
Test done in 9.88 sec, reward -60.110, steps 1048
54262: done 215 episodes, mean reward -100.432, speed 19.99 f/s
54505: done 216 episodes, mean reward -100.049, speed 51.33 f/s
54690: done 217 episodes, mean reward -99.888, speed 46.37 f/s
Test done in 2.57 sec, reward -82.691, steps 278
55045: done 218 episodes, mean reward -99.535, speed 35.82 f/s
55282: done 219 episodes, mean reward -99.399, speed 50.71 f/s
55564: done 220 episodes, mean reward -99.212, speed 47.17 f/s
55666: done 221 episodes, mean reward -99.180, speed 45.39 f/s
55820: done 222 episodes, mean reward -98.436, speed 51.46 f/s
55881: done 223 episodes, mean reward -98.231, speed 52.44 f/s
55963: done 224 episodes, mean reward -98.250, speed 54.22 f/s
Test done in 2.68 sec, reward -80.114, steps 269
56073: done 225 episodes, mean reward -98.232, speed 21.82 f/s
56401: done 226 episodes, mean reward -97.990, speed 45.34 f/s
56735: done 227 episodes, mean reward -97.482, speed 49.98 f/s
Test done in 1.71 sec, reward -83.126, steps 179
57012: done 228 episodes, mean reward -97.107, speed 37.04 f/s
57144: done 229 episodes, mean reward -96.970, speed 51.07 f/s
57313: done 230 episodes, mean reward -96.740, speed 48.79 f/s
57463: done 231 episodes, mean reward -96.355, speed 50.86 f/s
57630: done 232 episodes, mean reward -96.266, speed 54.13 f/s
57825: done 233 episodes, mean reward -96.143, speed 53.03 f/s
57947: done 234 episodes, mean reward -96.010, speed 52.89 f/s
Test done in 1.24 sec, reward -88.840, steps 127
58140: done 235 episodes, mean reward -95.825, speed 39.48 f/s
58251: done 236 episodes, mean reward -95.764, speed 55.08 f/s
58404: done 237 episodes, mean reward -95.586, speed 51.83 f/s
58546: done 238 episodes, mean reward -95.800, speed 50.33 f/s
58602: done 239 episodes, mean reward -95.769, speed 49.58 f/s
58793: done 240 episodes, mean reward -95.535, speed 48.96 f/s
Test done in 1.00 sec, reward -90.664, steps 115
59044: done 241 episodes, mean reward -95.430, speed 38.02 f/s
59119: done 242 episodes, mean reward -95.929, speed 46.90 f/s
59203: done 243 episodes, mean reward -95.797, speed 55.15 f/s
59313: done 244 episodes, mean reward -95.682, speed 49.88 f/s
59588: done 245 episodes, mean reward -95.472, speed 51.16 f/s
59704: done 246 episodes, mean reward -95.465, speed 55.06 f/s
59816: done 247 episodes, mean reward -95.394, speed 52.97 f/s
59903: done 248 episodes, mean reward -95.307, speed 50.54 f/s
Test done in 0.61 sec, reward -111.062, steps 61
60042: done 249 episodes, mean reward -95.286, speed 42.03 f/s
60214: done 250 episodes, mean reward -95.003, speed 53.67 f/s
60463: done 251 episodes, mean reward -94.658, speed 51.21 f/s
60594: done 252 episodes, mean reward -94.403, speed 53.13 f/s
60778: done 253 episodes, mean reward -94.098, speed 47.60 f/s
60878: done 254 episodes, mean reward -94.021, speed 42.52 f/s
60955: done 255 episodes, mean reward -94.122, speed 39.23 f/s
Test done in 1.07 sec, reward -91.133, steps 110
61072: done 256 episodes, mean reward -94.100, speed 33.30 f/s
61193: done 257 episodes, mean reward -93.840, speed 48.71 f/s
61290: done 258 episodes, mean reward -93.620, speed 47.21 f/s
61472: done 259 episodes, mean reward -93.408, speed 50.74 f/s
Test done in 1.43 sec, reward -89.221, steps 141
62078: done 260 episodes, mean reward -92.818, speed 42.99 f/s
62288: done 261 episodes, mean reward -92.393, speed 51.30 f/s
62470: done 262 episodes, mean reward -92.044, speed 52.45 f/s
62572: done 263 episodes, mean reward -91.889, speed 49.59 f/s
62696: done 264 episodes, mean reward -91.582, speed 53.67 f/s
62825: done 265 episodes, mean reward -91.650, speed 54.48 f/s
62950: done 266 episodes, mean reward -91.166, speed 51.21 f/s
Test done in 1.18 sec, reward -90.550, steps 131
63053: done 267 episodes, mean reward -90.947, speed 32.26 f/s
63181: done 268 episodes, mean reward -90.822, speed 51.99 f/s
63290: done 269 episodes, mean reward -90.857, speed 51.23 f/s
63402: done 270 episodes, mean reward -90.794, speed 46.26 f/s
63560: done 271 episodes, mean reward -90.486, speed 50.81 f/s
63733: done 272 episodes, mean reward -90.343, speed 51.05 f/s
63897: done 273 episodes, mean reward -90.321, speed 45.51 f/s
Test done in 1.45 sec, reward -85.942, steps 161
64020: done 274 episodes, mean reward -90.273, speed 32.21 f/s
64174: done 275 episodes, mean reward -90.269, speed 54.08 f/s
64285: done 276 episodes, mean reward -90.203, speed 53.57 f/s
64398: done 277 episodes, mean reward -90.234, speed 52.56 f/s
64512: done 278 episodes, mean reward -90.209, speed 54.62 f/s
64596: done 279 episodes, mean reward -90.177, speed 52.81 f/s
64701: done 280 episodes, mean reward -90.178, speed 56.21 f/s
64850: done 281 episodes, mean reward -89.997, speed 51.81 f/s
64958: done 282 episodes, mean reward -89.639, speed 54.11 f/s
Test done in 3.27 sec, reward -85.974, steps 370
65262: done 283 episodes, mean reward -89.504, speed 32.90 f/s
65441: done 284 episodes, mean reward -89.650, speed 52.66 f/s
65615: done 285 episodes, mean reward -89.804, speed 53.97 f/s
65752: done 286 episodes, mean reward -89.625, speed 54.36 f/s
Test done in 1.36 sec, reward -90.494, steps 142
66041: done 287 episodes, mean reward -89.375, speed 43.22 f/s
66247: done 288 episodes, mean reward -90.021, speed 53.25 f/s
66348: done 289 episodes, mean reward -90.515, speed 54.05 f/s
66427: done 290 episodes, mean reward -90.867, speed 51.69 f/s
66549: done 291 episodes, mean reward -90.845, speed 56.33 f/s
66672: done 292 episodes, mean reward -90.873, speed 51.64 f/s
66800: done 293 episodes, mean reward -91.006, speed 53.67 f/s
66996: done 294 episodes, mean reward -90.912, speed 52.12 f/s
Test done in 1.32 sec, reward -89.594, steps 136
67231: done 295 episodes, mean reward -90.358, speed 41.62 f/s
67482: done 296 episodes, mean reward -90.056, speed 52.58 f/s
67611: done 297 episodes, mean reward -89.926, speed 55.08 f/s
67814: done 298 episodes, mean reward -90.269, speed 52.31 f/s
Test done in 2.05 sec, reward -79.677, steps 223
68243: done 299 episodes, mean reward -90.202, speed 42.64 f/s
68398: done 300 episodes, mean reward -90.080, speed 54.08 f/s
68521: done 301 episodes, mean reward -89.918, speed 53.78 f/s
68730: done 302 episodes, mean reward -89.808, speed 53.04 f/s
Test done in 2.30 sec, reward -74.396, steps 239
69010: done 303 episodes, mean reward -89.812, speed 36.76 f/s
Test done in 2.67 sec, reward -69.768, steps 291
70165: done 304 episodes, mean reward -88.958, speed 47.26 f/s
70258: done 305 episodes, mean reward -88.909, speed 56.23 f/s
70432: done 306 episodes, mean reward -88.786, speed 53.35 f/s
70585: done 307 episodes, mean reward -88.967, speed 52.32 f/s
70766: done 308 episodes, mean reward -88.978, speed 53.73 f/s
70822: done 309 episodes, mean reward -89.079, speed 50.55 f/s
Test done in 4.40 sec, reward -54.848, steps 477
71143: done 310 episodes, mean reward -89.066, speed 30.44 f/s
71356: done 311 episodes, mean reward -88.896, speed 50.68 f/s
71541: done 312 episodes, mean reward -88.878, speed 51.18 f/s
71713: done 313 episodes, mean reward -89.330, speed 51.18 f/s
71856: done 314 episodes, mean reward -89.198, speed 51.08 f/s
71948: done 315 episodes, mean reward -89.265, speed 48.75 f/s
Test done in 1.60 sec, reward -77.803, steps 169
72051: done 316 episodes, mean reward -89.291, speed 26.34 f/s
72223: done 317 episodes, mean reward -89.204, speed 51.60 f/s
72375: done 318 episodes, mean reward -89.230, speed 50.78 f/s
72789: done 319 episodes, mean reward -89.099, speed 49.41 f/s
72931: done 320 episodes, mean reward -89.000, speed 51.46 f/s
Test done in 2.51 sec, reward -70.318, steps 267
73068: done 321 episodes, mean reward -88.954, speed 26.14 f/s
73258: done 322 episodes, mean reward -88.763, speed 51.83 f/s
73455: done 323 episodes, mean reward -88.515, speed 53.44 f/s
73540: done 324 episodes, mean reward -88.309, speed 51.31 f/s
Test done in 1.45 sec, reward -80.588, steps 154
74032: done 325 episodes, mean reward -88.070, speed 46.38 f/s
74171: done 326 episodes, mean reward -88.101, speed 53.83 f/s
74357: done 327 episodes, mean reward -88.244, speed 52.52 f/s
74503: done 328 episodes, mean reward -88.457, speed 52.79 f/s
74701: done 329 episodes, mean reward -88.267, speed 51.96 f/s
74907: done 330 episodes, mean reward -88.285, speed 51.74 f/s
Test done in 3.56 sec, reward -62.673, steps 399
75105: done 331 episodes, mean reward -88.202, speed 26.74 f/s
75267: done 332 episodes, mean reward -88.158, speed 53.94 f/s
75921: done 333 episodes, mean reward -87.902, speed 52.20 f/s
Test done in 1.43 sec, reward -82.626, steps 140
76091: done 334 episodes, mean reward -87.798, speed 36.75 f/s
76296: done 335 episodes, mean reward -87.636, speed 52.40 f/s
76825: done 336 episodes, mean reward -87.121, speed 52.22 f/s
76968: done 337 episodes, mean reward -87.129, speed 51.43 f/s
Test done in 4.70 sec, reward -31.639, steps 519
Best reward updated: -36.945 -> -31.639
77136: done 338 episodes, mean reward -86.738, speed 21.07 f/s
77298: done 339 episodes, mean reward -86.581, speed 53.35 f/s
77392: done 340 episodes, mean reward -86.632, speed 51.80 f/s
77588: done 341 episodes, mean reward -86.727, speed 51.42 f/s
77756: done 342 episodes, mean reward -86.481, speed 51.86 f/s
77891: done 343 episodes, mean reward -86.310, speed 53.57 f/s
Test done in 1.56 sec, reward -82.756, steps 175
78360: done 344 episodes, mean reward -85.973, speed 44.75 f/s
78510: done 345 episodes, mean reward -85.981, speed 51.63 f/s
78653: done 346 episodes, mean reward -85.620, speed 53.75 f/s
78907: done 347 episodes, mean reward -85.327, speed 53.79 f/s
Test done in 3.29 sec, reward -57.562, steps 364
79084: done 348 episodes, mean reward -85.189, speed 26.29 f/s
79207: done 349 episodes, mean reward -84.899, speed 51.67 f/s
79336: done 350 episodes, mean reward -84.877, speed 54.35 f/s
79468: done 351 episodes, mean reward -84.969, speed 51.33 f/s
79852: done 352 episodes, mean reward -84.772, speed 53.36 f/s
Test done in 13.93 sec, reward 95.298, steps 1502
Best reward updated: -31.639 -> 95.298
80070: done 353 episodes, mean reward -84.656, speed 11.99 f/s
80686: done 354 episodes, mean reward -84.411, speed 47.66 f/s
80833: done 355 episodes, mean reward -84.110, speed 45.61 f/s
80988: done 356 episodes, mean reward -84.053, speed 51.63 f/s
Test done in 1.39 sec, reward -86.451, steps 145
81137: done 357 episodes, mean reward -84.056, speed 32.98 f/s
81247: done 358 episodes, mean reward -84.014, speed 50.78 f/s
81984: done 359 episodes, mean reward -83.619, speed 47.03 f/s
Test done in 2.07 sec, reward -80.093, steps 197
82156: done 360 episodes, mean reward -84.013, speed 31.75 f/s
82345: done 361 episodes, mean reward -84.010, speed 49.20 f/s
82516: done 362 episodes, mean reward -84.049, speed 50.93 f/s
82657: done 363 episodes, mean reward -83.981, speed 51.21 f/s
82795: done 364 episodes, mean reward -83.974, speed 43.38 f/s
82936: done 365 episodes, mean reward -83.690, speed 51.79 f/s
Test done in 9.36 sec, reward 39.320, steps 1027
83129: done 366 episodes, mean reward -83.568, speed 14.66 f/s
83252: done 367 episodes, mean reward -83.491, speed 51.79 f/s
83538: done 368 episodes, mean reward -83.258, speed 50.26 f/s
83777: done 369 episodes, mean reward -82.984, speed 53.09 f/s
83913: done 370 episodes, mean reward -82.954, speed 51.23 f/s
Test done in 2.57 sec, reward -66.305, steps 267
84087: done 371 episodes, mean reward -82.832, speed 30.25 f/s
84207: done 372 episodes, mean reward -82.887, speed 49.19 f/s
84302: done 373 episodes, mean reward -82.948, speed 52.15 f/s
84653: done 374 episodes, mean reward -82.754, speed 50.13 f/s
84735: done 375 episodes, mean reward -82.668, speed 52.22 f/s
84876: done 376 episodes, mean reward -82.594, speed 48.90 f/s
Test done in 2.81 sec, reward -61.926, steps 305
85815: done 377 episodes, mean reward -81.688, speed 41.96 f/s
85932: done 378 episodes, mean reward -81.682, speed 44.18 f/s
Test done in 5.23 sec, reward -14.879, steps 553
86474: done 379 episodes, mean reward -81.062, speed 31.79 f/s
86594: done 380 episodes, mean reward -80.995, speed 49.51 f/s
86698: done 381 episodes, mean reward -80.992, speed 53.17 f/s
86867: done 382 episodes, mean reward -80.828, speed 53.45 f/s
Test done in 14.60 sec, reward 157.045, steps 1600
Best reward updated: 95.298 -> 157.045
87375: done 383 episodes, mean reward -80.344, speed 20.94 f/s
87505: done 384 episodes, mean reward -80.282, speed 54.20 f/s
87754: done 385 episodes, mean reward -79.830, speed 51.30 f/s
Test done in 6.46 sec, reward 22.096, steps 721
88475: done 386 episodes, mean reward -79.291, speed 34.19 f/s
88618: done 387 episodes, mean reward -79.382, speed 50.93 f/s
88784: done 388 episodes, mean reward -79.321, speed 50.58 f/s
88878: done 389 episodes, mean reward -79.393, speed 43.95 f/s
Test done in 7.19 sec, reward 14.138, steps 724
89026: done 390 episodes, mean reward -79.106, speed 14.50 f/s
89143: done 391 episodes, mean reward -78.987, speed 45.76 f/s
89564: done 392 episodes, mean reward -78.802, speed 46.89 f/s
89654: done 393 episodes, mean reward -78.829, speed 47.83 f/s
89783: done 394 episodes, mean reward -78.832, speed 49.05 f/s
89922: done 395 episodes, mean reward -78.852, speed 49.77 f/s
Test done in 2.35 sec, reward -64.255, steps 248
90128: done 396 episodes, mean reward -78.873, speed 33.11 f/s
90341: done 397 episodes, mean reward -78.708, speed 51.47 f/s
90499: done 398 episodes, mean reward -78.782, speed 49.77 f/s
90643: done 399 episodes, mean reward -78.719, speed 48.13 f/s
90854: done 400 episodes, mean reward -78.610, speed 49.12 f/s
Test done in 1.72 sec, reward -77.736, steps 184
91004: done 401 episodes, mean reward -78.638, speed 32.43 f/s
91212: done 402 episodes, mean reward -78.522, speed 51.17 f/s
91534: done 403 episodes, mean reward -78.267, speed 51.73 f/s
Test done in 2.07 sec, reward -73.527, steps 199
92264: done 404 episodes, mean reward -78.351, speed 44.36 f/s
92383: done 405 episodes, mean reward -78.264, speed 50.48 f/s
92626: done 406 episodes, mean reward -78.117, speed 51.23 f/s
92954: done 407 episodes, mean reward -77.821, speed 51.44 f/s
Test done in 10.29 sec, reward 67.721, steps 1125
93092: done 408 episodes, mean reward -77.802, speed 10.53 f/s
93670: done 409 episodes, mean reward -77.175, speed 51.05 f/s
93784: done 410 episodes, mean reward -77.383, speed 50.61 f/s
93946: done 411 episodes, mean reward -77.414, speed 52.16 f/s
Test done in 5.47 sec, reward -14.910, steps 582
94027: done 412 episodes, mean reward -77.586, speed 11.43 f/s
94150: done 413 episodes, mean reward -77.125, speed 51.00 f/s
94470: done 414 episodes, mean reward -76.940, speed 51.80 f/s
94785: done 415 episodes, mean reward -76.561, speed 51.69 f/s
Test done in 8.16 sec, reward 33.791, steps 884
95680: done 416 episodes, mean reward -75.442, speed 35.38 f/s
Test done in 11.73 sec, reward 114.981, steps 1311
96260: done 417 episodes, mean reward -75.094, speed 25.11 f/s
96350: done 418 episodes, mean reward -75.254, speed 49.03 f/s
96679: done 419 episodes, mean reward -75.123, speed 46.38 f/s
96954: done 420 episodes, mean reward -75.090, speed 50.99 f/s
Test done in 7.17 sec, reward 12.444, steps 770
97281: done 421 episodes, mean reward -74.959, speed 23.69 f/s
97482: done 422 episodes, mean reward -74.935, speed 45.04 f/s
97806: done 423 episodes, mean reward -74.830, speed 51.41 f/s
Test done in 10.27 sec, reward 70.792, steps 1107
98500: done 424 episodes, mean reward -74.100, speed 29.40 f/s
98595: done 425 episodes, mean reward -74.254, speed 51.54 f/s
98921: done 426 episodes, mean reward -74.060, speed 52.65 f/s
Test done in 7.30 sec, reward 15.395, steps 752
99046: done 427 episodes, mean reward -74.042, speed 12.53 f/s
99378: done 428 episodes, mean reward -73.796, speed 47.51 f/s
99838: done 429 episodes, mean reward -73.533, speed 50.39 f/s
Test done in 9.69 sec, reward 57.913, steps 1034
100050: done 430 episodes, mean reward -73.624, speed 15.35 f/s
Test done in 11.06 sec, reward 84.851, steps 1201
101168: done 431 episodes, mean reward -72.637, speed 33.20 f/s
101330: done 432 episodes, mean reward -72.588, speed 50.77 f/s
101687: done 433 episodes, mean reward -72.559, speed 53.69 f/s
101766: done 434 episodes, mean reward -72.777, speed 50.97 f/s
Test done in 12.80 sec, reward 121.468, steps 1411
102922: done 435 episodes, mean reward -71.811, speed 33.12 f/s
Test done in 6.96 sec, reward 7.785, steps 763
103810: done 436 episodes, mean reward -71.508, speed 36.82 f/s
Test done in 8.38 sec, reward 12.986, steps 796
Test done in 11.19 sec, reward 110.499, steps 1179
105028: done 437 episodes, mean reward -70.307, speed 26.36 f/s
105090: done 438 episodes, mean reward -70.502, speed 46.04 f/s
105497: done 439 episodes, mean reward -70.174, speed 48.77 f/s
105663: done 440 episodes, mean reward -70.055, speed 50.84 f/s
Test done in 12.64 sec, reward 122.340, steps 1325
106765: done 441 episodes, mean reward -69.080, speed 31.96 f/s
106914: done 442 episodes, mean reward -68.990, speed 52.81 f/s
Test done in 6.54 sec, reward 7.746, steps 728
107461: done 443 episodes, mean reward -68.628, speed 32.09 f/s
Test done in 9.80 sec, reward 89.243, steps 1067
108156: done 444 episodes, mean reward -68.210, speed 30.10 f/s
108472: done 445 episodes, mean reward -68.129, speed 51.99 f/s
Test done in 13.13 sec, reward 109.100, steps 1403
Test done in 10.09 sec, reward 75.696, steps 1053
110072: done 446 episodes, mean reward -65.433, speed 27.60 f/s
110517: done 447 episodes, mean reward -65.167, speed 47.32 f/s
Test done in 14.34 sec, reward 146.718, steps 1492
Test done in 14.23 sec, reward 170.045, steps 1502
Best reward updated: 157.045 -> 170.045
112117: done 448 episodes, mean reward -62.501, speed 25.95 f/s
112315: done 449 episodes, mean reward -62.529, speed 48.02 f/s
112625: done 450 episodes, mean reward -62.380, speed 45.29 f/s
Test done in 14.28 sec, reward 165.753, steps 1495
113658: done 451 episodes, mean reward -61.310, speed 28.95 f/s
113867: done 452 episodes, mean reward -61.396, speed 39.14 f/s
Test done in 13.55 sec, reward 157.524, steps 1363
Test done in 13.76 sec, reward 210.212, steps 1508
Best reward updated: 170.045 -> 210.212
115384: done 453 episodes, mean reward -59.711, speed 26.10 f/s
Test done in 10.99 sec, reward 82.346, steps 1148
116509: done 454 episodes, mean reward -58.509, speed 33.40 f/s
116913: done 455 episodes, mean reward -58.056, speed 50.05 f/s
Test done in 14.78 sec, reward 226.449, steps 1600
Best reward updated: 210.212 -> 226.449
Test done in 14.35 sec, reward 188.309, steps 1569
118477: done 456 episodes, mean reward -56.117, speed 26.15 f/s
Test done in 16.14 sec, reward 229.503, steps 1600
Best reward updated: 226.449 -> 229.503
119739: done 457 episodes, mean reward -54.679, speed 29.87 f/s
Test done in 15.51 sec, reward 246.486, steps 1600
Best reward updated: 229.503 -> 246.486
Test done in 14.95 sec, reward 237.894, steps 1600
121115: done 458 episodes, mean reward -53.088, speed 23.77 f/s
Test done in 14.87 sec, reward 265.493, steps 1599
Best reward updated: 246.486 -> 265.493
122250: done 459 episodes, mean reward -52.228, speed 30.57 f/s
122850: done 460 episodes, mean reward -51.516, speed 50.59 f/s
Test done in 15.06 sec, reward 216.847, steps 1600
123856: done 461 episodes, mean reward -50.498, speed 28.69 f/s
Test done in 14.75 sec, reward 248.244, steps 1600
124241: done 462 episodes, mean reward -50.253, speed 17.24 f/s
124468: done 463 episodes, mean reward -50.148, speed 47.00 f/s
Test done in 14.65 sec, reward 232.722, steps 1570
Test done in 14.90 sec, reward 238.803, steps 1600
126068: done 464 episodes, mean reward -47.385, speed 26.19 f/s
Test done in 14.41 sec, reward 234.698, steps 1600
127668: done 465 episodes, mean reward -44.522, speed 35.30 f/s
Test done in 14.70 sec, reward 216.344, steps 1532
Test done in 15.06 sec, reward 240.524, steps 1600
129268: done 466 episodes, mean reward -41.344, speed 25.47 f/s
Test done in 12.16 sec, reward 178.222, steps 1352
130344: done 467 episodes, mean reward -40.082, speed 32.70 f/s
Test done in 12.05 sec, reward 199.318, steps 1331
131084: done 468 episodes, mean reward -39.651, speed 28.51 f/s
131481: done 469 episodes, mean reward -39.512, speed 51.52 f/s
131989: done 470 episodes, mean reward -38.770, speed 52.93 f/s
Test done in 14.64 sec, reward 262.494, steps 1600
Test done in 14.84 sec, reward 226.489, steps 1600
133452: done 471 episodes, mean reward -37.095, speed 25.09 f/s
133681: done 472 episodes, mean reward -36.861, speed 51.62 f/s
Test done in 14.37 sec, reward 277.085, steps 1592
Best reward updated: 265.493 -> 277.085
134932: done 473 episodes, mean reward -35.110, speed 32.64 f/s
Test done in 14.24 sec, reward 218.436, steps 1500
135067: done 474 episodes, mean reward -35.297, speed 8.04 f/s
Test done in 12.94 sec, reward 190.941, steps 1394
136667: done 475 episodes, mean reward -32.003, speed 34.84 f/s
Test done in 15.10 sec, reward 258.407, steps 1600
137568: done 476 episodes, mean reward -30.872, speed 26.47 f/s
Test done in 14.39 sec, reward 243.342, steps 1528
138664: done 477 episodes, mean reward -30.736, speed 28.60 f/s
Test done in 14.93 sec, reward 246.742, steps 1600
Test done in 14.87 sec, reward 268.592, steps 1600
140264: done 478 episodes, mean reward -27.561, speed 25.38 f/s
140460: done 479 episodes, mean reward -28.239, speed 45.65 f/s
140765: done 480 episodes, mean reward -28.026, speed 44.52 f/s
Test done in 14.78 sec, reward 230.664, steps 1486
141323: done 481 episodes, mean reward -27.333, speed 20.65 f/s
Test done in 15.10 sec, reward 251.543, steps 1600
142923: done 482 episodes, mean reward -24.176, speed 33.32 f/s
Test done in 15.28 sec, reward 270.908, steps 1600
Test done in 15.10 sec, reward 274.930, steps 1595
144523: done 483 episodes, mean reward -21.460, speed 24.81 f/s
144616: done 484 episodes, mean reward -21.571, speed 50.20 f/s
144917: done 485 episodes, mean reward -21.535, speed 48.84 f/s
Test done in 13.98 sec, reward 245.869, steps 1451
Test done in 14.17 sec, reward 287.197, steps 1473
Best reward updated: 277.085 -> 287.197
146517: done 486 episodes, mean reward -18.559, speed 25.51 f/s
146650: done 487 episodes, mean reward -18.574, speed 46.91 f/s
146725: done 488 episodes, mean reward -18.657, speed 48.82 f/s
Test done in 15.25 sec, reward 282.478, steps 1504
Test done in 13.76 sec, reward 253.741, steps 1448
148223: done 489 episodes, mean reward -16.372, speed 24.62 f/s
Test done in 14.99 sec, reward 256.297, steps 1577
149171: done 490 episodes, mean reward -15.071, speed 27.11 f/s
149789: done 491 episodes, mean reward -14.252, speed 48.18 f/s
Test done in 15.14 sec, reward 285.377, steps 1548
150418: done 492 episodes, mean reward -13.624, speed 22.19 f/s
150940: done 493 episodes, mean reward -12.932, speed 43.45 f/s
Test done in 13.37 sec, reward 233.733, steps 1413
151349: done 494 episodes, mean reward -12.455, speed 18.59 f/s
151489: done 495 episodes, mean reward -12.423, speed 47.99 f/s
152000: done 496 episodes, mean reward -11.989, speed 47.84 f/s
Test done in 14.10 sec, reward 257.861, steps 1509
152389: done 497 episodes, mean reward -11.673, speed 17.28 f/s
152586: done 498 episodes, mean reward -11.642, speed 49.71 f/s
Test done in 13.06 sec, reward 228.542, steps 1415
153651: done 499 episodes, mean reward -10.187, speed 30.10 f/s
153833: done 500 episodes, mean reward -10.281, speed 42.68 f/s
Test done in 15.30 sec, reward 264.370, steps 1598
154368: done 501 episodes, mean reward -9.708, speed 19.56 f/s
154458: done 502 episodes, mean reward -9.870, speed 44.16 f/s
Test done in 13.40 sec, reward 177.619, steps 1357
155391: done 503 episodes, mean reward -8.792, speed 27.33 f/s
Test done in 14.68 sec, reward 253.471, steps 1573
156374: done 504 episodes, mean reward -8.116, speed 27.13 f/s
156666: done 505 episodes, mean reward -7.899, speed 47.08 f/s
Test done in 13.99 sec, reward 210.646, steps 1469
157696: done 506 episodes, mean reward -6.696, speed 28.76 f/s
Test done in 15.48 sec, reward 266.030, steps 1600
158514: done 507 episodes, mean reward -5.869, speed 25.22 f/s
158678: done 508 episodes, mean reward -5.882, speed 51.09 f/s
Test done in 13.66 sec, reward 226.780, steps 1457
159511: done 509 episodes, mean reward -5.348, speed 27.64 f/s
Test done in 9.47 sec, reward 111.725, steps 1026
160435: done 510 episodes, mean reward -4.156, speed 32.75 f/s
Test done in 12.87 sec, reward 196.351, steps 1375
161007: done 511 episodes, mean reward -3.545, speed 23.17 f/s
161294: done 512 episodes, mean reward -3.213, speed 48.04 f/s
161646: done 513 episodes, mean reward -2.748, speed 45.30 f/s
Test done in 15.36 sec, reward 282.818, steps 1575
162199: done 514 episodes, mean reward -2.318, speed 20.52 f/s
Test done in 15.02 sec, reward 278.200, steps 1600
163657: done 515 episodes, mean reward -0.323, speed 31.31 f/s
163917: done 516 episodes, mean reward -1.189, speed 45.17 f/s
163986: done 517 episodes, mean reward -1.702, speed 44.93 f/s
Test done in 15.11 sec, reward 262.574, steps 1600
164775: done 518 episodes, mean reward -0.594, speed 24.62 f/s
164922: done 519 episodes, mean reward -0.835, speed 47.36 f/s
Test done in 15.15 sec, reward 265.632, steps 1600
165761: done 520 episodes, mean reward 0.250, speed 25.33 f/s
Test done in 15.13 sec, reward 271.071, steps 1600
Test done in 15.11 sec, reward 255.992, steps 1600
167324: done 521 episodes, mean reward 2.396, speed 24.88 f/s
167416: done 522 episodes, mean reward 2.241, speed 45.20 f/s
167770: done 523 episodes, mean reward 2.395, speed 49.89 f/s
Test done in 15.05 sec, reward 260.062, steps 1600
Test done in 16.00 sec, reward 275.031, steps 1600
169370: done 524 episodes, mean reward 5.011, speed 24.64 f/s
Test done in 15.47 sec, reward 278.585, steps 1597
170285: done 525 episodes, mean reward 6.395, speed 26.22 f/s
170496: done 526 episodes, mean reward 6.313, speed 47.24 f/s
Test done in 13.66 sec, reward 222.843, steps 1426
171311: done 527 episodes, mean reward 7.378, speed 25.45 f/s
Test done in 14.60 sec, reward 239.772, steps 1493
172534: done 528 episodes, mean reward 8.919, speed 29.96 f/s
Test done in 15.15 sec, reward 265.786, steps 1600
173004: done 529 episodes, mean reward 9.021, speed 17.23 f/s
Test done in 13.81 sec, reward 282.307, steps 1483
174604: done 530 episodes, mean reward 12.377, speed 33.42 f/s
Test done in 15.02 sec, reward 276.831, steps 1600
Test done in 15.98 sec, reward 273.350, steps 1600
176204: done 531 episodes, mean reward 14.511, speed 24.56 f/s
Test done in 15.47 sec, reward 260.090, steps 1569
177643: done 532 episodes, mean reward 16.580, speed 30.66 f/s
Test done in 15.75 sec, reward 272.146, steps 1600
178842: done 533 episodes, mean reward 18.175, speed 29.06 f/s
178991: done 534 episodes, mean reward 18.321, speed 39.52 f/s
Test done in 15.23 sec, reward 271.415, steps 1600
Test done in 15.73 sec, reward 280.070, steps 1600
180063: done 535 episodes, mean reward 18.718, speed 20.25 f/s
180210: done 536 episodes, mean reward 17.976, speed 48.07 f/s
Test done in 15.82 sec, reward 283.983, steps 1586
181810: done 537 episodes, mean reward 20.145, speed 32.46 f/s
181875: done 538 episodes, mean reward 20.143, speed 46.66 f/s
Test done in 15.49 sec, reward 278.295, steps 1598
Test done in 15.04 sec, reward 284.207, steps 1554
183185: done 539 episodes, mean reward 21.857, speed 22.31 f/s
Test done in 15.11 sec, reward 280.969, steps 1598
184785: done 540 episodes, mean reward 25.162, speed 32.20 f/s
Test done in 15.27 sec, reward 289.613, steps 1549
Best reward updated: 287.197 -> 289.613
185283: done 541 episodes, mean reward 24.737, speed 19.12 f/s
185692: done 542 episodes, mean reward 25.181, speed 49.21 f/s
185807: done 543 episodes, mean reward 24.816, speed 47.64 f/s
Test done in 15.51 sec, reward 277.712, steps 1600
186024: done 544 episodes, mean reward 24.308, speed 10.83 f/s
186916: done 545 episodes, mean reward 25.420, speed 48.09 f/s
186994: done 546 episodes, mean reward 22.604, speed 45.44 f/s
Test done in 15.33 sec, reward 273.587, steps 1600
187124: done 547 episodes, mean reward 22.284, speed 7.21 f/s
Test done in 15.68 sec, reward 283.615, steps 1599
188023: done 548 episodes, mean reward 20.854, speed 25.92 f/s
188087: done 549 episodes, mean reward 20.738, speed 44.46 f/s
188954: done 550 episodes, mean reward 21.808, speed 46.60 f/s
Test done in 15.45 sec, reward 277.248, steps 1600
189056: done 551 episodes, mean reward 20.733, speed 5.78 f/s
189176: done 552 episodes, mean reward 20.682, speed 46.55 f/s
Test done in 15.13 sec, reward 287.546, steps 1559
190776: done 553 episodes, mean reward 22.097, speed 32.42 f/s
Test done in 15.59 sec, reward 267.451, steps 1600
Test done in 15.45 sec, reward 286.630, steps 1589
192376: done 554 episodes, mean reward 24.172, speed 24.68 f/s
192429: done 555 episodes, mean reward 23.600, speed 47.82 f/s
192566: done 556 episodes, mean reward 21.711, speed 49.80 f/s
192809: done 557 episodes, mean reward 20.495, speed 48.91 f/s
Test done in 14.17 sec, reward 285.661, steps 1531
193420: done 558 episodes, mean reward 19.647, speed 23.06 f/s
Test done in 14.81 sec, reward 286.950, steps 1588
194129: done 559 episodes, mean reward 19.434, speed 24.34 f/s
194851: done 560 episodes, mean reward 19.773, speed 49.05 f/s
Test done in 13.28 sec, reward 286.936, steps 1457
Test done in 14.14 sec, reward 285.622, steps 1538
196451: done 561 episodes, mean reward 22.303, speed 26.64 f/s
196615: done 562 episodes, mean reward 22.132, speed 49.50 f/s
196955: done 563 episodes, mean reward 22.229, speed 49.92 f/s
Test done in 14.66 sec, reward 280.450, steps 1600
197046: done 564 episodes, mean reward 19.445, speed 5.51 f/s
Test done in 15.00 sec, reward 283.437, steps 1582
198646: done 565 episodes, mean reward 19.996, speed 33.74 f/s
Test done in 14.29 sec, reward 286.448, steps 1544
199406: done 566 episodes, mean reward 17.752, speed 25.48 f/s
199893: done 567 episodes, mean reward 17.124, speed 49.17 f/s
Test done in 14.90 sec, reward 284.283, steps 1533
Test done in 14.98 sec, reward 283.078, steps 1564
201151: done 568 episodes, mean reward 18.404, speed 22.34 f/s
201877: done 569 episodes, mean reward 19.130, speed 46.17 f/s
Test done in 14.28 sec, reward 287.380, steps 1498
Test done in 14.13 sec, reward 281.580, steps 1509
203377: done 570 episodes, mean reward 20.851, speed 25.37 f/s
Test done in 14.87 sec, reward 283.765, steps 1528
204899: done 571 episodes, mean reward 21.406, speed 32.56 f/s
Test done in 15.08 sec, reward 279.751, steps 1599
Test done in 14.67 sec, reward 278.570, steps 1600
206499: done 572 episodes, mean reward 24.574, speed 25.96 f/s
206678: done 573 episodes, mean reward 23.032, speed 51.46 f/s
Test done in 14.57 sec, reward 277.927, steps 1600
Test done in 15.73 sec, reward 277.045, steps 1600
208278: done 574 episodes, mean reward 26.455, speed 25.10 f/s
208437: done 575 episodes, mean reward 23.410, speed 23.95 f/s
Test done in 16.45 sec, reward 278.754, steps 1600
209348: done 576 episodes, mean reward 23.497, speed 17.51 f/s
Test done in 15.08 sec, reward 272.430, steps 1600
210481: done 577 episodes, mean reward 24.192, speed 29.64 f/s
Test done in 14.84 sec, reward 266.948, steps 1600
211187: done 578 episodes, mean reward 21.866, speed 24.85 f/s
Test done in 37.34 sec, reward 276.910, steps 1600
212787: done 579 episodes, mean reward 25.198, speed 20.29 f/s
Test done in 16.63 sec, reward 282.115, steps 1598
Test done in 15.83 sec, reward 274.399, steps 1600
214387: done 580 episodes, mean reward 28.376, speed 22.79 f/s
Test done in 15.08 sec, reward 256.232, steps 1600
215244: done 581 episodes, mean reward 28.994, speed 25.39 f/s
Test done in 17.79 sec, reward 286.078, steps 1574
216448: done 582 episodes, mean reward 27.402, speed 24.47 f/s
Test done in 15.60 sec, reward 264.420, steps 1600
Test done in 14.74 sec, reward 285.777, steps 1583
218048: done 583 episodes, mean reward 27.481, speed 25.05 f/s
Test done in 15.27 sec, reward 269.132, steps 1600
219648: done 584 episodes, mean reward 30.844, speed 33.15 f/s
Test done in 14.98 sec, reward 271.870, steps 1600
Test done in 14.64 sec, reward 271.986, steps 1600
221191: done 585 episodes, mean reward 32.903, speed 25.02 f/s
Test done in 14.90 sec, reward 278.403, steps 1598
222020: done 586 episodes, mean reward 30.522, speed 26.02 f/s
222478: done 587 episodes, mean reward 31.103, speed 49.15 f/s
Test done in 14.66 sec, reward 256.220, steps 1575
Test done in 14.86 sec, reward 282.561, steps 1591
224078: done 588 episodes, mean reward 34.513, speed 25.77 f/s
Test done in 14.85 sec, reward 276.834, steps 1600
225678: done 589 episodes, mean reward 35.760, speed 33.63 f/s
Test done in 15.09 sec, reward 256.205, steps 1600
Test done in 14.75 sec, reward 275.619, steps 1598
227278: done 590 episodes, mean reward 37.776, speed 25.24 f/s
Test done in 14.97 sec, reward 265.215, steps 1600
228878: done 591 episodes, mean reward 40.399, speed 33.15 f/s
Test done in 15.05 sec, reward 275.237, steps 1600
229335: done 592 episodes, mean reward 40.018, speed 18.69 f/s
229460: done 593 episodes, mean reward 39.426, speed 50.43 f/s
Test done in 15.25 sec, reward 279.742, steps 1597
230183: done 594 episodes, mean reward 39.971, speed 24.12 f/s
Test done in 14.78 sec, reward 271.332, steps 1599
231783: done 595 episodes, mean reward 43.219, speed 33.91 f/s
Test done in 15.38 sec, reward 274.011, steps 1599
232309: done 596 episodes, mean reward 43.373, speed 20.01 f/s
232795: done 597 episodes, mean reward 43.611, speed 49.07 f/s
Test done in 15.12 sec, reward 275.344, steps 1600
233304: done 598 episodes, mean reward 44.281, speed 19.81 f/s
Test done in 15.61 sec, reward 266.679, steps 1600
234904: done 599 episodes, mean reward 46.125, speed 31.40 f/s
Test done in 15.39 sec, reward 265.471, steps 1600
Test done in 15.18 sec, reward 261.870, steps 1600
236504: done 600 episodes, mean reward 49.371, speed 24.78 f/s
Test done in 14.74 sec, reward 276.284, steps 1584
Test done in 15.12 sec, reward 280.402, steps 1568
238104: done 601 episodes, mean reward 52.141, speed 24.78 f/s
Test done in 15.86 sec, reward 269.995, steps 1600
239082: done 602 episodes, mean reward 53.479, speed 26.41 f/s
Test done in 15.61 sec, reward 272.323, steps 1600
240682: done 603 episodes, mean reward 55.650, speed 31.75 f/s
Test done in 15.40 sec, reward 272.248, steps 1597
Test done in 15.29 sec, reward 262.420, steps 1600
242282: done 604 episodes, mean reward 57.576, speed 24.32 f/s
Test done in 15.83 sec, reward 272.171, steps 1600
243882: done 605 episodes, mean reward 60.678, speed 31.15 f/s
Test done in 15.26 sec, reward 266.769, steps 1600
244124: done 606 episodes, mean reward 59.526, speed 11.68 f/s
244236: done 607 episodes, mean reward 58.557, speed 47.10 f/s
Test done in 15.17 sec, reward 259.603, steps 1600
245836: done 608 episodes, mean reward 61.761, speed 32.66 f/s
245944: done 609 episodes, mean reward 60.728, speed 49.92 f/s
Test done in 15.90 sec, reward 266.094, steps 1600
Test done in 15.16 sec, reward 272.357, steps 1600
247359: done 610 episodes, mean reward 61.702, speed 23.27 f/s
247727: done 611 episodes, mean reward 61.396, speed 47.91 f/s
Test done in 15.53 sec, reward 260.008, steps 1600
Test done in 16.14 sec, reward 253.756, steps 1600
249327: done 612 episodes, mean reward 64.372, speed 23.87 f/s
Test done in 15.50 sec, reward 260.596, steps 1600
250927: done 613 episodes, mean reward 67.161, speed 32.20 f/s
Test done in 15.48 sec, reward 244.641, steps 1600
Test done in 15.57 sec, reward 257.389, steps 1600
252527: done 614 episodes, mean reward 69.785, speed 24.56 f/s
Test done in 15.12 sec, reward 268.545, steps 1600
Test done in 15.38 sec, reward 265.898, steps 1600
254127: done 615 episodes, mean reward 70.733, speed 24.68 f/s
Test done in 15.68 sec, reward 250.518, steps 1600
255727: done 616 episodes, mean reward 73.750, speed 32.39 f/s
Test done in 15.12 sec, reward 256.245, steps 1600
256840: done 617 episodes, mean reward 75.435, speed 29.02 f/s
Test done in 15.81 sec, reward 250.298, steps 1600
257615: done 618 episodes, mean reward 75.500, speed 23.53 f/s
Test done in 15.61 sec, reward 261.769, steps 1600
258786: done 619 episodes, mean reward 77.110, speed 28.49 f/s
Test done in 15.78 sec, reward 244.164, steps 1600
259375: done 620 episodes, mean reward 76.631, speed 19.76 f/s
Test done in 16.05 sec, reward 263.138, steps 1600
260975: done 621 episodes, mean reward 77.630, speed 31.18 f/s
Test done in 14.85 sec, reward 244.630, steps 1600
261084: done 622 episodes, mean reward 77.706, speed 6.35 f/s
261392: done 623 episodes, mean reward 77.540, speed 47.27 f/s
Test done in 15.02 sec, reward 244.066, steps 1600
262992: done 624 episodes, mean reward 77.389, speed 33.10 f/s
Test done in 15.03 sec, reward 255.682, steps 1600
263268: done 625 episodes, mean reward 76.099, speed 13.26 f/s
Test done in 14.84 sec, reward 247.457, steps 1600
264575: done 626 episodes, mean reward 77.595, speed 31.59 f/s
Test done in 14.13 sec, reward 214.442, steps 1486
Test done in 15.34 sec, reward 232.111, steps 1600
266175: done 627 episodes, mean reward 79.630, speed 25.59 f/s
Test done in 15.51 sec, reward 246.157, steps 1600
267775: done 628 episodes, mean reward 81.177, speed 31.63 f/s
Test done in 15.74 sec, reward 259.687, steps 1600
268272: done 629 episodes, mean reward 81.192, speed 18.63 f/s
Test done in 16.16 sec, reward 246.212, steps 1600
269872: done 630 episodes, mean reward 81.121, speed 31.18 f/s
Test done in 15.44 sec, reward 256.490, steps 1600
270461: done 631 episodes, mean reward 78.619, speed 20.58 f/s
Test done in 15.25 sec, reward 274.250, steps 1600
Test done in 14.85 sec, reward 264.073, steps 1600
272061: done 632 episodes, mean reward 79.711, speed 24.70 f/s
Test done in 14.79 sec, reward 249.011, steps 1600
273661: done 633 episodes, mean reward 81.074, speed 33.71 f/s
Test done in 14.85 sec, reward 266.507, steps 1600
Test done in 15.00 sec, reward 250.393, steps 1600
275261: done 634 episodes, mean reward 84.274, speed 25.73 f/s
275357: done 635 episodes, mean reward 82.684, speed 49.41 f/s
275647: done 636 episodes, mean reward 82.938, speed 47.91 f/s
Test done in 16.86 sec, reward 259.951, steps 1600
Test done in 14.90 sec, reward 256.769, steps 1600
277247: done 637 episodes, mean reward 82.751, speed 24.81 f/s
Test done in 14.79 sec, reward 266.126, steps 1600
278847: done 638 episodes, mean reward 86.117, speed 33.77 f/s
Test done in 14.73 sec, reward 257.692, steps 1600
Test done in 14.50 sec, reward 266.641, steps 1600
280447: done 639 episodes, mean reward 87.367, speed 26.02 f/s
Test done in 14.59 sec, reward 269.632, steps 1600
Test done in 15.33 sec, reward 241.259, steps 1600
282047: done 640 episodes, mean reward 87.396, speed 25.32 f/s
Test done in 15.46 sec, reward 269.694, steps 1600
283647: done 641 episodes, mean reward 90.246, speed 33.23 f/s
Test done in 14.94 sec, reward 277.300, steps 1600
Test done in 14.61 sec, reward 267.275, steps 1600
285247: done 642 episodes, mean reward 93.175, speed 25.59 f/s
285986: done 643 episodes, mean reward 94.027, speed 48.64 f/s
Test done in 14.88 sec, reward 257.036, steps 1600
Test done in 14.93 sec, reward 274.693, steps 1600
287586: done 644 episodes, mean reward 97.237, speed 25.52 f/s
Test done in 14.68 sec, reward 272.311, steps 1600
288719: done 645 episodes, mean reward 97.511, speed 30.43 f/s
Test done in 14.48 sec, reward 275.520, steps 1600
Test done in 14.79 sec, reward 269.232, steps 1600
290319: done 646 episodes, mean reward 101.042, speed 26.11 f/s
Test done in 14.88 sec, reward 268.208, steps 1600
291919: done 647 episodes, mean reward 104.342, speed 34.13 f/s
Test done in 14.31 sec, reward 271.130, steps 1600
292004: done 648 episodes, mean reward 102.950, speed 5.29 f/s
Test done in 14.52 sec, reward 280.820, steps 1600
293604: done 649 episodes, mean reward 106.478, speed 34.37 f/s
Test done in 14.60 sec, reward 268.337, steps 1600
Test done in 14.66 sec, reward 270.839, steps 1600
295204: done 650 episodes, mean reward 108.430, speed 26.10 f/s
295281: done 651 episodes, mean reward 108.417, speed 47.81 f/s
295670: done 652 episodes, mean reward 108.742, speed 50.05 f/s
295978: done 653 episodes, mean reward 105.838, speed 49.98 f/s
Test done in 14.82 sec, reward 282.721, steps 1599
Test done in 14.66 sec, reward 265.330, steps 1600
297578: done 654 episodes, mean reward 105.868, speed 25.99 f/s
Test done in 14.44 sec, reward 273.015, steps 1600
Test done in 14.48 sec, reward 284.596, steps 1594
299178: done 655 episodes, mean reward 109.353, speed 26.20 f/s
299871: done 656 episodes, mean reward 110.300, speed 49.97 f/s
Test done in 14.43 sec, reward 277.702, steps 1600
300055: done 657 episodes, mean reward 110.247, speed 10.09 f/s
300194: done 658 episodes, mean reward 109.492, speed 52.82 f/s
300720: done 660 episodes, mean reward 107.833, speed 50.20 f/s
300835: done 661 episodes, mean reward 104.278, speed 49.40 f/s
Test done in 6.15 sec, reward 45.315, steps 672
Test done in 12.97 sec, reward 233.846, steps 1449
302053: done 662 episodes, mean reward 105.916, speed 27.89 f/s
302139: done 663 episodes, mean reward 105.599, speed 51.21 f/s
Test done in 14.52 sec, reward 266.316, steps 1600
303739: done 664 episodes, mean reward 108.887, speed 34.15 f/s
Test done in 14.43 sec, reward 262.360, steps 1600
304021: done 665 episodes, mean reward 105.727, speed 14.13 f/s
Test done in 14.55 sec, reward 265.723, steps 1600
305606: done 666 episodes, mean reward 106.924, speed 34.09 f/s
305782: done 667 episodes, mean reward 106.286, speed 50.35 f/s
Test done in 14.50 sec, reward 259.520, steps 1600
306104: done 668 episodes, mean reward 104.573, speed 15.37 f/s
306784: done 669 episodes, mean reward 104.331, speed 50.05 f/s
Test done in 14.66 sec, reward 262.640, steps 1600
307469: done 670 episodes, mean reward 102.653, speed 24.33 f/s
Test done in 14.36 sec, reward 262.018, steps 1600
Test done in 14.61 sec, reward 255.905, steps 1600
309069: done 671 episodes, mean reward 103.632, speed 26.37 f/s
Test done in 14.58 sec, reward 254.417, steps 1600
310434: done 672 episodes, mean reward 102.197, speed 32.58 f/s
Test done in 14.75 sec, reward 257.168, steps 1600
Test done in 14.77 sec, reward 257.321, steps 1600
312034: done 673 episodes, mean reward 105.214, speed 25.95 f/s
312448: done 674 episodes, mean reward 102.308, speed 48.53 f/s
Test done in 14.84 sec, reward 281.453, steps 1600
Test done in 14.90 sec, reward 262.887, steps 1600
314048: done 675 episodes, mean reward 105.581, speed 25.39 f/s
Test done in 14.76 sec, reward 255.686, steps 1577
315648: done 676 episodes, mean reward 107.685, speed 33.57 f/s
315737: done 677 episodes, mean reward 105.894, speed 48.40 f/s
Test done in 14.99 sec, reward 272.340, steps 1600
Test done in 14.77 sec, reward 267.015, steps 1600
317337: done 678 episodes, mean reward 108.468, speed 25.45 f/s
Test done in 14.76 sec, reward 266.382, steps 1600
318937: done 679 episodes, mean reward 108.575, speed 33.41 f/s
Test done in 14.90 sec, reward 266.810, steps 1600
Test done in 14.88 sec, reward 258.167, steps 1600
320537: done 680 episodes, mean reward 108.543, speed 25.53 f/s
Test done in 14.69 sec, reward 264.080, steps 1600
Test done in 14.85 sec, reward 273.930, steps 1598
322137: done 681 episodes, mean reward 110.476, speed 25.61 f/s
Test done in 14.79 sec, reward 260.529, steps 1600
323737: done 682 episodes, mean reward 112.187, speed 33.53 f/s
323917: done 683 episodes, mean reward 108.935, speed 49.05 f/s
Test done in 14.68 sec, reward 284.742, steps 1586
324863: done 684 episodes, mean reward 106.960, speed 27.64 f/s
Test done in 15.01 sec, reward 271.920, steps 1600
325243: done 685 episodes, mean reward 104.804, speed 16.62 f/s
Test done in 14.86 sec, reward 266.919, steps 1600
326150: done 686 episodes, mean reward 104.915, speed 26.96 f/s
Test done in 14.18 sec, reward 254.244, steps 1516
327750: done 687 episodes, mean reward 107.634, speed 33.90 f/s
Test done in 14.97 sec, reward 267.419, steps 1600
328515: done 688 episodes, mean reward 105.225, speed 24.80 f/s
328678: done 689 episodes, mean reward 101.978, speed 47.51 f/s
Test done in 14.96 sec, reward 263.184, steps 1600
329294: done 690 episodes, mean reward 99.428, speed 22.23 f/s
Test done in 15.03 sec, reward 278.469, steps 1600
330894: done 691 episodes, mean reward 99.274, speed 33.53 f/s
Test done in 15.04 sec, reward 278.376, steps 1600
331402: done 692 episodes, mean reward 99.537, speed 19.85 f/s
Test done in 14.70 sec, reward 285.899, steps 1599
332620: done 693 episodes, mean reward 101.207, speed 30.51 f/s
Test done in 15.00 sec, reward 268.812, steps 1600
333572: done 694 episodes, mean reward 101.458, speed 26.99 f/s
333961: done 695 episodes, mean reward 98.521, speed 47.83 f/s
Test done in 14.75 sec, reward 291.130, steps 1598
Best reward updated: 289.613 -> 291.130
Test done in 14.93 sec, reward 273.433, steps 1600
335316: done 696 episodes, mean reward 99.781, speed 23.46 f/s
335989: done 697 episodes, mean reward 99.962, speed 48.02 f/s
Test done in 14.82 sec, reward 282.172, steps 1594
336374: done 698 episodes, mean reward 99.643, speed 16.95 f/s
Test done in 15.10 sec, reward 274.045, steps 1600
337974: done 699 episodes, mean reward 99.489, speed 33.13 f/s
Test done in 14.91 sec, reward 272.660, steps 1600
338298: done 700 episodes, mean reward 96.529, speed 14.97 f/s
Test done in 15.09 sec, reward 270.951, steps 1600
339898: done 701 episodes, mean reward 96.389, speed 32.92 f/s
Test done in 14.77 sec, reward 263.503, steps 1600
Test done in 14.71 sec, reward 266.400, steps 1600
341498: done 702 episodes, mean reward 98.312, speed 25.42 f/s
Test done in 15.25 sec, reward 285.528, steps 1600
Test done in 15.13 sec, reward 282.958, steps 1600
343098: done 703 episodes, mean reward 98.092, speed 24.98 f/s
343305: done 704 episodes, mean reward 94.870, speed 49.19 f/s
Test done in 14.75 sec, reward 276.954, steps 1600
344816: done 705 episodes, mean reward 93.594, speed 32.98 f/s
Test done in 14.76 sec, reward 280.652, steps 1600
Test done in 14.89 sec, reward 283.425, steps 1598
346416: done 706 episodes, mean reward 96.616, speed 25.51 f/s
Test done in 14.95 sec, reward 258.525, steps 1600
347261: done 707 episodes, mean reward 97.610, speed 25.70 f/s
Test done in 14.87 sec, reward 253.052, steps 1600
348861: done 708 episodes, mean reward 97.570, speed 33.12 f/s
Test done in 15.33 sec, reward 282.843, steps 1600
349230: done 709 episodes, mean reward 97.798, speed 15.78 f/s
Traceback (most recent call last):
  File "K:\Projects\python\DQN\learning\BipedalWalker\train_d4pg.py", line 185, in <module>
    proj_distr_v = distr_projection(last_distr_v, rewards_v, dones_mask,
  File "K:\Projects\python\DQN\learning\BipedalWalker\train_d4pg.py", line 93, in distr_projection
    proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]
KeyboardInterrupt

Process finished with exit code -1073741510 (0xC000013A: interrupted by Ctrl+C)
