已经到达极限奖励处，奖励可以接近300

C:\Users\admin\.conda\envs\pytorch-gym\python.exe K:\Projects\python\DQN\learning\BipedalWalker\train_ddpg.py -n bipedalwalker_ddpg 
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
DDPGCritic(
  (obs_net): Sequential(
    (0): Linear(in_features=24, out_features=400, bias=True)
    (1): ReLU()
  )
  (out_net): Sequential(
    (0): Linear(in_features=404, out_features=300, bias=True)
    (1): ReLU()
    (2): Linear(in_features=300, out_features=1, bias=True)
  )
)
1943: done 5 episodes, mean reward -117.338, speed 1328.00 f/s
3215: done 18 episodes, mean reward -110.139, speed 1225.34 f/s
4872: done 20 episodes, mean reward -113.336, speed 1155.14 f/s
6217: done 26 episodes, mean reward -113.121, speed 1282.62 f/s
7566: done 39 episodes, mean reward -112.109, speed 1235.53 f/s
9313: done 46 episodes, mean reward -112.913, speed 1173.55 f/s
Test done in 0.72 sec, reward -92.664, steps 102
10012: done 53 episodes, mean reward -112.857, speed 506.20 f/s


Test done in 11.02 sec, reward -180.944, steps 1600
11612: done 54 episodes, mean reward -113.448, speed 54.20 f/s
Test done in 11.33 sec, reward -180.907, steps 1600
Test done in 10.92 sec, reward -180.877, steps 1600
13212: done 55 episodes, mean reward -114.076, speed 38.92 f/s
13590: done 56 episodes, mean reward -114.546, speed 88.79 f/s
13723: done 57 episodes, mean reward -114.590, speed 88.31 f/s
13892: done 58 episodes, mean reward -114.625, speed 91.35 f/s
13997: done 59 episodes, mean reward -114.530, speed 85.84 f/s
Test done in 0.72 sec, reward -112.991, steps 109
14183: done 60 episodes, mean reward -114.561, speed 67.64 f/s
Test done in 11.40 sec, reward -180.947, steps 1600
15783: done 61 episodes, mean reward -115.254, speed 53.13 f/s
Test done in 11.09 sec, reward -180.916, steps 1600
Test done in 10.93 sec, reward -180.787, steps 1600
17383: done 62 episodes, mean reward -115.905, speed 39.37 f/s
17473: done 64 episodes, mean reward -116.246, speed 79.45 f/s
17566: done 66 episodes, mean reward -116.547, speed 85.41 f/s
17659: done 68 episodes, mean reward -116.845, speed 85.08 f/s
17753: done 70 episodes, mean reward -117.097, speed 82.35 f/s
17846: done 72 episodes, mean reward -117.245, speed 89.52 f/s
17943: done 74 episodes, mean reward -117.414, speed 82.89 f/s
Test done in 0.37 sec, reward -127.707, steps 45
18044: done 76 episodes, mean reward -117.620, speed 63.02 f/s
18200: done 77 episodes, mean reward -117.587, speed 85.96 f/s
18304: done 78 episodes, mean reward -117.515, speed 89.02 f/s
18399: done 80 episodes, mean reward -117.748, speed 86.77 f/s
Test done in 0.35 sec, reward -127.758, steps 45
19999: done 81 episodes, mean reward -118.281, speed 83.76 f/s
Test done in 0.55 sec, reward -127.764, steps 45
20132: done 82 episodes, mean reward -118.222, speed 59.12 f/s
Test done in 0.35 sec, reward -127.756, steps 45
21656: done 84 episodes, mean reward -120.030, speed 79.47 f/s
21751: done 86 episodes, mean reward -120.126, speed 83.31 f/s
21847: done 88 episodes, mean reward -120.182, speed 86.69 f/s
21938: done 90 episodes, mean reward -120.283, speed 87.30 f/s
Test done in 0.37 sec, reward -125.091, steps 45
22033: done 92 episodes, mean reward -120.354, speed 65.71 f/s
22127: done 94 episodes, mean reward -120.405, speed 85.23 f/s
22219: done 96 episodes, mean reward -120.444, speed 91.86 f/s
22533: done 98 episodes, mean reward -120.523, speed 82.36 f/s
22622: done 100 episodes, mean reward -120.565, speed 84.08 f/s
22709: done 102 episodes, mean reward -120.829, speed 79.04 f/s
22807: done 104 episodes, mean reward -120.704, speed 81.70 f/s
22893: done 106 episodes, mean reward -120.704, speed 75.75 f/s
22977: done 108 episodes, mean reward -121.099, speed 79.00 f/s
Test done in 0.40 sec, reward -123.308, steps 44
23083: done 109 episodes, mean reward -121.263, speed 58.17 f/s
23171: done 111 episodes, mean reward -121.667, speed 78.88 f/s
23266: done 113 episodes, mean reward -121.402, speed 84.66 f/s
23354: done 115 episodes, mean reward -121.786, speed 86.31 f/s
23441: done 117 episodes, mean reward -122.118, speed 83.96 f/s
23625: done 120 episodes, mean reward -121.724, speed 88.09 f/s
23747: done 123 episodes, mean reward -122.213, speed 89.04 f/s
23834: done 125 episodes, mean reward -122.222, speed 81.05 f/s
23918: done 127 episodes, mean reward -122.336, speed 82.97 f/s
Test done in 0.33 sec, reward -123.313, steps 44
24050: done 129 episodes, mean reward -122.444, speed 68.47 f/s
24137: done 131 episodes, mean reward -122.721, speed 83.55 f/s
24268: done 134 episodes, mean reward -123.029, speed 88.00 f/s
24356: done 136 episodes, mean reward -123.326, speed 85.23 f/s
24477: done 138 episodes, mean reward -123.376, speed 84.55 f/s
24559: done 140 episodes, mean reward -123.648, speed 80.31 f/s
24656: done 142 episodes, mean reward -123.656, speed 89.42 f/s
24744: done 144 episodes, mean reward -123.843, speed 86.69 f/s
24830: done 146 episodes, mean reward -123.632, speed 84.04 f/s
24917: done 148 episodes, mean reward -123.825, speed 81.22 f/s
Test done in 0.35 sec, reward -123.253, steps 44
25022: done 150 episodes, mean reward -124.005, speed 65.98 f/s
25163: done 152 episodes, mean reward -124.028, speed 87.37 f/s
25249: done 154 episodes, mean reward -123.848, speed 83.17 f/s
25337: done 156 episodes, mean reward -123.388, speed 84.06 f/s
25426: done 158 episodes, mean reward -123.481, speed 85.79 f/s
25535: done 160 episodes, mean reward -123.600, speed 83.01 f/s
25624: done 162 episodes, mean reward -122.910, speed 82.48 f/s
25712: done 164 episodes, mean reward -122.821, speed 86.20 f/s
25810: done 166 episodes, mean reward -122.567, speed 82.84 f/s
25983: done 168 episodes, mean reward -122.418, speed 76.85 f/s
Test done in 0.67 sec, reward -123.477, steps 43
26027: done 169 episodes, mean reward -122.377, speed 29.91 f/s
26115: done 171 episodes, mean reward -122.265, speed 81.09 f/s
26203: done 173 episodes, mean reward -122.272, speed 87.61 f/s
26308: done 175 episodes, mean reward -122.172, speed 84.78 f/s
26413: done 177 episodes, mean reward -122.075, speed 88.33 f/s
26617: done 179 episodes, mean reward -122.096, speed 86.82 f/s
26813: done 182 episodes, mean reward -121.789, speed 90.18 f/s
26900: done 184 episodes, mean reward -120.285, speed 85.44 f/s
26999: done 186 episodes, mean reward -120.101, speed 88.13 f/s
Test done in 0.35 sec, reward -123.789, steps 47
27119: done 187 episodes, mean reward -120.018, speed 71.66 f/s
27212: done 189 episodes, mean reward -119.999, speed 86.03 f/s
27305: done 191 episodes, mean reward -119.919, speed 86.30 f/s
27398: done 193 episodes, mean reward -119.917, speed 84.88 f/s
27528: done 195 episodes, mean reward -119.967, speed 74.30 f/s
27618: done 197 episodes, mean reward -119.948, speed 84.59 f/s
27742: done 199 episodes, mean reward -119.806, speed 80.46 f/s
27830: done 201 episodes, mean reward -119.795, speed 87.19 f/s
27932: done 202 episodes, mean reward -119.700, speed 92.05 f/s
Test done in 0.47 sec, reward -123.749, steps 46
28019: done 203 episodes, mean reward -119.687, speed 59.44 f/s
28148: done 205 episodes, mean reward -119.588, speed 88.42 f/s
28254: done 207 episodes, mean reward -119.417, speed 81.97 f/s
28352: done 209 episodes, mean reward -119.229, speed 84.82 f/s
28453: done 211 episodes, mean reward -118.939, speed 85.41 f/s
28564: done 213 episodes, mean reward -118.678, speed 86.61 f/s
28660: done 215 episodes, mean reward -118.378, speed 87.86 f/s
28769: done 217 episodes, mean reward -118.129, speed 85.26 f/s
28869: done 219 episodes, mean reward -117.841, speed 87.79 f/s
28972: done 221 episodes, mean reward -117.602, speed 85.55 f/s
Test done in 0.34 sec, reward -107.378, steps 48
29073: done 223 episodes, mean reward -117.422, speed 66.98 f/s
29171: done 225 episodes, mean reward -117.162, speed 88.57 f/s
29274: done 227 episodes, mean reward -116.925, speed 87.79 f/s
29370: done 229 episodes, mean reward -116.657, speed 89.19 f/s
29479: done 231 episodes, mean reward -116.404, speed 88.97 f/s
29576: done 233 episodes, mean reward -116.110, speed 83.58 f/s
29672: done 235 episodes, mean reward -115.794, speed 84.10 f/s
29793: done 237 episodes, mean reward -115.646, speed 85.01 f/s
29895: done 239 episodes, mean reward -115.376, speed 90.28 f/s
29993: done 241 episodes, mean reward -115.090, speed 84.56 f/s
Test done in 0.35 sec, reward -108.076, steps 48
30093: done 243 episodes, mean reward -114.902, speed 67.03 f/s
30195: done 245 episodes, mean reward -114.677, speed 90.35 f/s
30294: done 247 episodes, mean reward -114.357, speed 84.94 f/s
30391: done 249 episodes, mean reward -114.065, speed 86.26 f/s
30490: done 251 episodes, mean reward -113.850, speed 83.42 f/s
30591: done 253 episodes, mean reward -113.703, speed 83.65 f/s
30697: done 255 episodes, mean reward -113.424, speed 81.55 f/s
30812: done 257 episodes, mean reward -113.232, speed 87.10 f/s
30920: done 259 episodes, mean reward -113.076, speed 83.95 f/s
Test done in 0.36 sec, reward -109.633, steps 49
31023: done 261 episodes, mean reward -112.827, speed 64.81 f/s
31124: done 263 episodes, mean reward -112.516, speed 81.71 f/s
31224: done 265 episodes, mean reward -112.288, speed 86.65 f/s
31331: done 267 episodes, mean reward -112.038, speed 90.77 f/s
31437: done 269 episodes, mean reward -111.886, speed 86.88 f/s
31558: done 271 episodes, mean reward -111.651, speed 88.88 f/s
31662: done 273 episodes, mean reward -111.353, speed 80.28 f/s
31762: done 275 episodes, mean reward -111.167, speed 82.59 f/s
31856: done 277 episodes, mean reward -110.957, speed 88.85 f/s
31980: done 279 episodes, mean reward -110.733, speed 87.74 f/s
Test done in 0.37 sec, reward -108.980, steps 50
32055: done 280 episodes, mean reward -110.581, speed 59.65 f/s
32158: done 282 episodes, mean reward -110.240, speed 87.99 f/s
32260: done 284 episodes, mean reward -109.952, speed 84.98 f/s
32368: done 286 episodes, mean reward -109.872, speed 86.67 f/s
32471: done 288 episodes, mean reward -109.601, speed 86.98 f/s
32572: done 290 episodes, mean reward -109.284, speed 81.89 f/s
32664: done 292 episodes, mean reward -109.003, speed 78.57 f/s
32762: done 294 episodes, mean reward -108.702, speed 83.67 f/s
32860: done 296 episodes, mean reward -108.332, speed 83.45 f/s
32967: done 298 episodes, mean reward -108.164, speed 89.58 f/s
Test done in 0.41 sec, reward -107.768, steps 53
33027: done 299 episodes, mean reward -107.998, speed 52.85 f/s
33121: done 301 episodes, mean reward -107.732, speed 82.52 f/s
33247: done 303 episodes, mean reward -107.618, speed 85.73 f/s
33389: done 305 episodes, mean reward -107.410, speed 91.70 f/s
33509: done 307 episodes, mean reward -107.359, speed 86.16 f/s
33612: done 309 episodes, mean reward -107.377, speed 85.73 f/s
33708: done 311 episodes, mean reward -107.379, speed 88.86 f/s
33831: done 313 episodes, mean reward -107.409, speed 83.76 f/s
33951: done 315 episodes, mean reward -107.421, speed 88.74 f/s
Test done in 0.40 sec, reward -106.303, steps 54
34045: done 317 episodes, mean reward -107.408, speed 63.09 f/s
34139: done 319 episodes, mean reward -107.448, speed 87.27 f/s
34235: done 321 episodes, mean reward -107.514, speed 87.72 f/s
34367: done 323 episodes, mean reward -107.447, speed 84.34 f/s
34473: done 325 episodes, mean reward -107.406, speed 85.24 f/s
34580: done 327 episodes, mean reward -107.323, speed 89.91 f/s
34688: done 329 episodes, mean reward -107.293, speed 84.44 f/s
34788: done 331 episodes, mean reward -107.174, speed 84.26 f/s
34906: done 333 episodes, mean reward -107.102, speed 83.46 f/s
Test done in 0.48 sec, reward -104.841, steps 56
35065: done 335 episodes, mean reward -107.018, speed 68.01 f/s
35191: done 337 episodes, mean reward -106.886, speed 82.44 f/s
35313: done 339 episodes, mean reward -106.812, speed 88.22 f/s
35435: done 341 episodes, mean reward -106.745, speed 86.88 f/s
35559: done 343 episodes, mean reward -106.668, speed 83.51 f/s
35686: done 345 episodes, mean reward -106.544, speed 92.80 f/s
35803: done 347 episodes, mean reward -106.500, speed 85.49 f/s
35898: done 349 episodes, mean reward -106.466, speed 86.90 f/s
Test done in 0.41 sec, reward -104.277, steps 56
36012: done 351 episodes, mean reward -106.413, speed 66.14 f/s
36117: done 353 episodes, mean reward -106.329, speed 86.87 f/s
36226: done 355 episodes, mean reward -106.261, speed 84.34 f/s
36351: done 357 episodes, mean reward -106.093, speed 87.08 f/s
36483: done 359 episodes, mean reward -105.898, speed 83.95 f/s
36585: done 361 episodes, mean reward -105.847, speed 86.86 f/s
36688: done 363 episodes, mean reward -105.770, speed 89.01 f/s
36792: done 365 episodes, mean reward -105.693, speed 85.39 f/s
36895: done 367 episodes, mean reward -105.608, speed 86.74 f/s
Test done in 0.44 sec, reward -105.394, steps 52
37006: done 369 episodes, mean reward -105.422, speed 64.91 f/s
37122: done 371 episodes, mean reward -105.257, speed 85.66 f/s
37228: done 373 episodes, mean reward -105.185, speed 88.43 f/s
37320: done 375 episodes, mean reward -105.151, speed 84.06 f/s
37414: done 377 episodes, mean reward -105.165, speed 90.12 f/s
37548: done 379 episodes, mean reward -105.026, speed 88.82 f/s
37696: done 381 episodes, mean reward -104.915, speed 87.23 f/s
37790: done 383 episodes, mean reward -104.915, speed 79.22 f/s
37917: done 385 episodes, mean reward -104.807, speed 80.91 f/s
Test done in 0.40 sec, reward -102.656, steps 53
38055: done 387 episodes, mean reward -104.683, speed 66.15 f/s
38177: done 389 episodes, mean reward -104.622, speed 85.38 f/s
38296: done 391 episodes, mean reward -104.571, speed 86.62 f/s
38434: done 393 episodes, mean reward -104.496, speed 85.68 f/s
38528: done 395 episodes, mean reward -104.477, speed 87.37 f/s
38656: done 397 episodes, mean reward -104.396, speed 85.03 f/s
38784: done 399 episodes, mean reward -104.322, speed 90.03 f/s
38877: done 401 episodes, mean reward -104.281, speed 84.57 f/s
38975: done 402 episodes, mean reward -104.260, speed 87.82 f/s
Test done in 0.45 sec, reward -104.165, steps 57
39033: done 403 episodes, mean reward -104.183, speed 50.55 f/s
39140: done 405 episodes, mean reward -104.082, speed 83.18 f/s
39254: done 407 episodes, mean reward -103.966, speed 88.13 f/s
39362: done 409 episodes, mean reward -103.836, speed 84.30 f/s
39474: done 411 episodes, mean reward -103.699, speed 85.43 f/s
39585: done 413 episodes, mean reward -103.595, speed 85.28 f/s
39686: done 415 episodes, mean reward -103.563, speed 85.08 f/s
39821: done 417 episodes, mean reward -103.458, speed 86.00 f/s
39975: done 419 episodes, mean reward -103.341, speed 84.13 f/s
Test done in 0.47 sec, reward -103.267, steps 60
40036: done 420 episodes, mean reward -103.270, speed 51.99 f/s
40148: done 422 episodes, mean reward -103.238, speed 87.85 f/s
40261: done 424 episodes, mean reward -103.220, speed 88.02 f/s
40372: done 425 episodes, mean reward -103.207, speed 88.06 f/s
40464: done 427 episodes, mean reward -103.276, speed 81.90 f/s
40570: done 429 episodes, mean reward -103.302, speed 85.46 f/s
40669: done 431 episodes, mean reward -103.330, speed 84.47 f/s
40788: done 433 episodes, mean reward -103.296, speed 85.35 f/s
40930: done 435 episodes, mean reward -103.276, speed 87.55 f/s
Test done in 0.45 sec, reward -103.763, steps 58
41023: done 437 episodes, mean reward -103.344, speed 61.65 f/s
41120: done 439 episodes, mean reward -103.393, speed 90.40 f/s
41275: done 441 episodes, mean reward -103.346, speed 87.09 f/s
41385: done 443 episodes, mean reward -103.335, speed 81.61 f/s
41488: done 445 episodes, mean reward -103.338, speed 86.47 f/s
41603: done 447 episodes, mean reward -103.268, speed 86.01 f/s
41703: done 449 episodes, mean reward -103.203, speed 83.46 f/s
41801: done 451 episodes, mean reward -103.173, speed 87.92 f/s
41902: done 453 episodes, mean reward -103.131, speed 87.49 f/s
Test done in 0.43 sec, reward -102.580, steps 54
42046: done 455 episodes, mean reward -103.063, speed 68.85 f/s
42166: done 457 episodes, mean reward -103.093, speed 88.52 f/s
42299: done 459 episodes, mean reward -103.143, speed 87.85 f/s
42402: done 461 episodes, mean reward -103.119, speed 87.69 f/s
42551: done 463 episodes, mean reward -103.097, speed 86.74 f/s
42667: done 465 episodes, mean reward -103.142, speed 85.93 f/s
42795: done 467 episodes, mean reward -103.163, speed 84.90 f/s
42889: done 469 episodes, mean reward -103.284, speed 80.25 f/s
42981: done 471 episodes, mean reward -103.387, speed 74.78 f/s
Test done in 0.38 sec, reward -105.320, steps 51
43041: done 472 episodes, mean reward -103.406, speed 52.74 f/s
43146: done 474 episodes, mean reward -103.356, speed 85.03 f/s
43272: done 476 episodes, mean reward -103.264, speed 84.84 f/s
43396: done 478 episodes, mean reward -103.310, speed 86.35 f/s
43491: done 480 episodes, mean reward -103.398, speed 86.98 f/s
43600: done 482 episodes, mean reward -103.396, speed 91.84 f/s
43709: done 484 episodes, mean reward -103.293, speed 85.53 f/s
43818: done 486 episodes, mean reward -103.288, speed 80.68 f/s
43929: done 488 episodes, mean reward -103.280, speed 88.85 f/s
Test done in 0.41 sec, reward -104.351, steps 56
44039: done 490 episodes, mean reward -103.277, speed 64.04 f/s
44140: done 492 episodes, mean reward -103.215, speed 84.36 f/s
44250: done 494 episodes, mean reward -103.175, speed 89.34 f/s
44344: done 496 episodes, mean reward -103.156, speed 87.92 f/s
44456: done 498 episodes, mean reward -103.174, speed 87.87 f/s
44655: done 500 episodes, mean reward -103.159, speed 85.49 f/s
44770: done 502 episodes, mean reward -103.161, speed 86.04 f/s
44866: done 504 episodes, mean reward -103.261, speed 84.37 f/s
44993: done 506 episodes, mean reward -103.325, speed 85.87 f/s
Test done in 0.41 sec, reward -104.503, steps 56
45066: done 507 episodes, mean reward -103.339, speed 56.56 f/s
45217: done 509 episodes, mean reward -103.341, speed 85.07 f/s
45303: done 510 episodes, mean reward -103.351, speed 83.17 f/s
45397: done 512 episodes, mean reward -103.457, speed 91.54 f/s
45513: done 513 episodes, mean reward -103.464, speed 86.81 f/s
45627: done 515 episodes, mean reward -103.448, speed 86.73 f/s
45734: done 517 episodes, mean reward -103.515, speed 84.60 f/s
45864: done 519 episodes, mean reward -103.530, speed 84.21 f/s
45963: done 521 episodes, mean reward -103.610, speed 89.67 f/s
Test done in 0.42 sec, reward -103.243, steps 53
46012: done 522 episodes, mean reward -103.590, speed 47.52 f/s
46115: done 524 episodes, mean reward -103.571, speed 86.62 f/s
46221: done 526 episodes, mean reward -103.535, speed 83.16 f/s
46331: done 528 episodes, mean reward -103.482, speed 85.14 f/s
46445: done 530 episodes, mean reward -103.406, speed 85.17 f/s
46552: done 532 episodes, mean reward -103.358, speed 84.62 f/s
46660: done 534 episodes, mean reward -103.335, speed 89.65 f/s
46764: done 536 episodes, mean reward -103.288, speed 90.03 f/s
46896: done 538 episodes, mean reward -103.243, speed 84.21 f/s
Test done in 0.42 sec, reward -103.524, steps 60
47032: done 540 episodes, mean reward -103.249, speed 69.71 f/s
47125: done 542 episodes, mean reward -103.318, speed 83.78 f/s
47217: done 544 episodes, mean reward -103.414, speed 84.72 f/s
47334: done 546 episodes, mean reward -103.455, speed 86.52 f/s
47426: done 548 episodes, mean reward -103.529, speed 87.15 f/s
47523: done 550 episodes, mean reward -103.624, speed 91.08 f/s
47639: done 552 episodes, mean reward -103.656, speed 84.16 f/s
47759: done 554 episodes, mean reward -103.712, speed 85.57 f/s
47890: done 556 episodes, mean reward -103.742, speed 84.50 f/s
Test done in 0.41 sec, reward -105.266, steps 54
48011: done 558 episodes, mean reward -103.692, speed 61.60 f/s
48139: done 560 episodes, mean reward -103.711, speed 81.38 f/s
48235: done 562 episodes, mean reward -103.766, speed 79.42 f/s
48349: done 564 episodes, mean reward -103.760, speed 85.99 f/s
48453: done 566 episodes, mean reward -103.733, speed 88.71 f/s
48556: done 568 episodes, mean reward -103.662, speed 87.86 f/s
48661: done 570 episodes, mean reward -103.594, speed 85.10 f/s
48775: done 572 episodes, mean reward -103.538, speed 84.75 f/s
48883: done 574 episodes, mean reward -103.485, speed 85.67 f/s
49000: done 576 episodes, mean reward -103.489, speed 86.23 f/s
Test done in 0.46 sec, reward -105.318, steps 56
49054: done 577 episodes, mean reward -103.451, speed 49.59 f/s
49149: done 579 episodes, mean reward -103.449, speed 86.43 f/s
49243: done 581 episodes, mean reward -103.501, speed 83.47 f/s
49372: done 583 episodes, mean reward -103.526, speed 85.66 f/s
49480: done 585 episodes, mean reward -103.623, speed 87.01 f/s
49578: done 587 episodes, mean reward -103.732, speed 90.17 f/s
49672: done 589 episodes, mean reward -103.839, speed 83.62 f/s
49786: done 591 episodes, mean reward -103.898, speed 88.44 f/s
49881: done 593 episodes, mean reward -103.972, speed 87.31 f/s
Test done in 0.40 sec, reward -103.009, steps 58
50006: done 595 episodes, mean reward -103.997, speed 67.76 f/s
50098: done 597 episodes, mean reward -104.039, speed 84.50 f/s
50197: done 599 episodes, mean reward -104.130, speed 84.37 f/s
50295: done 601 episodes, mean reward -104.144, speed 83.37 f/s
50396: done 603 episodes, mean reward -104.175, speed 87.40 f/s
50510: done 605 episodes, mean reward -104.090, speed 82.62 f/s
50616: done 607 episodes, mean reward -104.114, speed 85.77 f/s
50724: done 609 episodes, mean reward -104.140, speed 87.97 f/s
50819: done 611 episodes, mean reward -104.149, speed 87.57 f/s
50926: done 613 episodes, mean reward -104.119, speed 86.20 f/s
Test done in 0.43 sec, reward -103.645, steps 60
51033: done 615 episodes, mean reward -104.120, speed 63.14 f/s
51130: done 617 episodes, mean reward -104.117, speed 91.30 f/s
51279: done 619 episodes, mean reward -104.124, speed 88.81 f/s
51385: done 621 episodes, mean reward -104.111, speed 87.85 f/s
51484: done 623 episodes, mean reward -104.173, speed 85.86 f/s
51600: done 625 episodes, mean reward -104.221, speed 87.59 f/s
51739: done 627 episodes, mean reward -104.264, speed 86.22 f/s
51854: done 629 episodes, mean reward -104.336, speed 84.96 f/s
51988: done 631 episodes, mean reward -104.366, speed 87.01 f/s
Test done in 0.50 sec, reward -104.686, steps 61
52037: done 632 episodes, mean reward -104.421, speed 45.47 f/s
52192: done 634 episodes, mean reward -104.447, speed 89.49 f/s
52317: done 636 episodes, mean reward -104.521, speed 90.21 f/s
52461: done 638 episodes, mean reward -104.505, speed 88.54 f/s
52585: done 640 episodes, mean reward -104.531, speed 85.75 f/s
52688: done 642 episodes, mean reward -104.553, speed 89.54 f/s
52799: done 644 episodes, mean reward -104.478, speed 87.58 f/s
52907: done 646 episodes, mean reward -104.451, speed 90.00 f/s
Test done in 0.49 sec, reward -106.505, steps 59
53022: done 648 episodes, mean reward -104.387, speed 63.48 f/s
53142: done 650 episodes, mean reward -104.311, speed 72.42 f/s
53258: done 652 episodes, mean reward -104.274, speed 79.66 f/s
53354: done 654 episodes, mean reward -104.302, speed 83.75 f/s
53468: done 656 episodes, mean reward -104.301, speed 84.10 f/s
53583: done 658 episodes, mean reward -104.271, speed 81.37 f/s
53706: done 660 episodes, mean reward -104.296, speed 83.18 f/s
53807: done 662 episodes, mean reward -104.342, speed 87.40 f/s
53930: done 664 episodes, mean reward -104.416, speed 84.62 f/s
Test done in 0.44 sec, reward -105.077, steps 58
54003: done 665 episodes, mean reward -104.433, speed 57.31 f/s
54154: done 667 episodes, mean reward -104.508, speed 82.50 f/s
54255: done 669 episodes, mean reward -104.601, speed 90.01 f/s
54385: done 671 episodes, mean reward -104.672, speed 87.43 f/s
54491: done 673 episodes, mean reward -104.819, speed 87.68 f/s
54633: done 675 episodes, mean reward -104.913, speed 85.30 f/s
54743: done 677 episodes, mean reward -105.023, speed 84.88 f/s
54851: done 679 episodes, mean reward -105.085, speed 86.27 f/s
54935: done 680 episodes, mean reward -105.035, speed 79.57 f/s
Test done in 0.44 sec, reward -107.393, steps 57
55036: done 682 episodes, mean reward -105.119, speed 63.84 f/s
55155: done 684 episodes, mean reward -105.057, speed 85.40 f/s
55275: done 686 episodes, mean reward -105.096, speed 88.87 f/s
55392: done 688 episodes, mean reward -105.032, speed 84.72 f/s
55515: done 690 episodes, mean reward -105.036, speed 83.27 f/s
55625: done 692 episodes, mean reward -104.991, speed 81.02 f/s
55735: done 694 episodes, mean reward -105.011, speed 93.00 f/s
55856: done 696 episodes, mean reward -104.981, speed 86.81 f/s
55969: done 698 episodes, mean reward -104.939, speed 85.85 f/s
Test done in 0.41 sec, reward -109.321, steps 63
56030: done 699 episodes, mean reward -104.929, speed 56.26 f/s
56138: done 701 episodes, mean reward -104.944, speed 88.27 f/s
56273: done 703 episodes, mean reward -105.000, speed 83.87 f/s
56413: done 705 episodes, mean reward -105.139, speed 88.84 f/s
56498: done 706 episodes, mean reward -105.196, speed 84.37 f/s
56620: done 708 episodes, mean reward -105.256, speed 82.72 f/s
56785: done 710 episodes, mean reward -105.354, speed 89.23 f/s
56952: done 712 episodes, mean reward -105.429, speed 84.76 f/s
Test done in 0.42 sec, reward -110.064, steps 56
57007: done 713 episodes, mean reward -105.521, speed 52.32 f/s
57113: done 715 episodes, mean reward -105.635, speed 88.50 f/s
57232: done 717 episodes, mean reward -105.769, speed 86.92 f/s
57341: done 719 episodes, mean reward -105.940, speed 90.18 f/s
57455: done 721 episodes, mean reward -106.050, speed 86.08 f/s
57575: done 723 episodes, mean reward -106.135, speed 85.53 f/s
57702: done 725 episodes, mean reward -106.221, speed 83.55 f/s
57817: done 727 episodes, mean reward -106.318, speed 88.48 f/s
57939: done 729 episodes, mean reward -106.423, speed 83.81 f/s
Test done in 0.53 sec, reward -110.954, steps 64
58056: done 731 episodes, mean reward -106.573, speed 62.20 f/s
58183: done 733 episodes, mean reward -106.648, speed 82.79 f/s
58315: done 735 episodes, mean reward -106.738, speed 73.04 f/s
58438: done 737 episodes, mean reward -106.845, speed 78.76 f/s
58551: done 739 episodes, mean reward -106.951, speed 81.13 f/s
58669: done 741 episodes, mean reward -107.039, speed 80.25 f/s
58814: done 743 episodes, mean reward -107.126, speed 84.51 f/s
58984: done 745 episodes, mean reward -107.235, speed 85.83 f/s
Test done in 0.46 sec, reward -111.521, steps 59
59064: done 746 episodes, mean reward -107.287, speed 58.17 f/s
59183: done 748 episodes, mean reward -107.503, speed 88.67 f/s
59282: done 749 episodes, mean reward -107.561, speed 87.20 f/s
59405: done 751 episodes, mean reward -107.757, speed 85.87 f/s
59537: done 753 episodes, mean reward -107.936, speed 86.10 f/s
59658: done 755 episodes, mean reward -108.098, speed 81.37 f/s
59807: done 757 episodes, mean reward -108.221, speed 85.60 f/s
59950: done 759 episodes, mean reward -108.371, speed 89.76 f/s
Test done in 0.44 sec, reward -111.836, steps 62
60011: done 760 episodes, mean reward -108.415, speed 55.38 f/s
60145: done 762 episodes, mean reward -108.447, speed 85.64 f/s
60269: done 764 episodes, mean reward -108.502, speed 87.27 f/s
60388: done 766 episodes, mean reward -108.547, speed 87.47 f/s
60511: done 768 episodes, mean reward -108.599, speed 84.63 f/s
60630: done 770 episodes, mean reward -108.615, speed 86.58 f/s
60756: done 772 episodes, mean reward -108.607, speed 84.32 f/s
60878: done 774 episodes, mean reward -108.618, speed 84.95 f/s
60988: done 776 episodes, mean reward -108.662, speed 85.46 f/s
Test done in 0.45 sec, reward -110.226, steps 63
61049: done 777 episodes, mean reward -108.671, speed 53.79 f/s
61170: done 779 episodes, mean reward -108.636, speed 92.57 f/s
61289: done 781 episodes, mean reward -108.708, speed 88.24 f/s
61412: done 783 episodes, mean reward -108.779, speed 84.58 f/s
61553: done 784 episodes, mean reward -108.866, speed 87.93 f/s
61668: done 786 episodes, mean reward -108.954, speed 87.00 f/s
61797: done 788 episodes, mean reward -109.070, speed 86.80 f/s
61922: done 790 episodes, mean reward -109.192, speed 85.15 f/s
Test done in 0.44 sec, reward -108.910, steps 59
62060: done 792 episodes, mean reward -109.311, speed 66.70 f/s
62196: done 794 episodes, mean reward -109.425, speed 86.25 f/s
62316: done 796 episodes, mean reward -109.543, speed 88.94 f/s
62418: done 798 episodes, mean reward -109.678, speed 85.71 f/s
62522: done 800 episodes, mean reward -109.812, speed 89.60 f/s
62640: done 802 episodes, mean reward -109.905, speed 89.01 f/s
62738: done 804 episodes, mean reward -109.893, speed 87.69 f/s
62844: done 806 episodes, mean reward -109.917, speed 89.01 f/s
62956: done 808 episodes, mean reward -109.928, speed 87.38 f/s
Test done in 0.35 sec, reward -111.025, steps 49
63060: done 810 episodes, mean reward -109.955, speed 67.42 f/s
63162: done 812 episodes, mean reward -109.972, speed 88.59 f/s
63272: done 814 episodes, mean reward -109.931, speed 86.57 f/s
63393: done 816 episodes, mean reward -109.870, speed 76.83 f/s
63496: done 818 episodes, mean reward -109.815, speed 77.53 f/s
63602: done 820 episodes, mean reward -109.797, speed 76.19 f/s
63708: done 822 episodes, mean reward -109.744, speed 87.07 f/s
63855: done 824 episodes, mean reward -109.735, speed 83.40 f/s
63981: done 826 episodes, mean reward -109.747, speed 87.35 f/s
Test done in 0.42 sec, reward -111.195, steps 61
64046: done 827 episodes, mean reward -109.758, speed 54.33 f/s
64163: done 829 episodes, mean reward -109.818, speed 87.53 f/s
64321: done 831 episodes, mean reward -109.766, speed 85.35 f/s
64441: done 833 episodes, mean reward -109.826, speed 83.53 f/s
64524: done 834 episodes, mean reward -109.812, speed 79.98 f/s
64638: done 836 episodes, mean reward -109.859, speed 83.77 f/s
64801: done 838 episodes, mean reward -109.829, speed 87.85 f/s
64915: done 840 episodes, mean reward -109.868, speed 89.32 f/s
Test done in 0.47 sec, reward -110.367, steps 60
65044: done 842 episodes, mean reward -109.919, speed 63.67 f/s
65161: done 844 episodes, mean reward -109.886, speed 81.53 f/s
65287: done 846 episodes, mean reward -109.834, speed 83.30 f/s
65401: done 848 episodes, mean reward -109.718, speed 87.81 f/s
65514: done 850 episodes, mean reward -109.680, speed 77.91 f/s
65633: done 852 episodes, mean reward -109.598, speed 78.88 f/s
65761: done 854 episodes, mean reward -109.506, speed 79.56 f/s
65880: done 856 episodes, mean reward -109.423, speed 89.81 f/s
65994: done 858 episodes, mean reward -109.370, speed 90.64 f/s
Test done in 0.41 sec, reward -109.698, steps 55
66051: done 859 episodes, mean reward -109.368, speed 52.84 f/s
66172: done 861 episodes, mean reward -109.307, speed 83.88 f/s
66294: done 863 episodes, mean reward -109.302, speed 83.51 f/s
66415: done 865 episodes, mean reward -109.333, speed 77.41 f/s
66570: done 867 episodes, mean reward -109.283, speed 81.69 f/s
66716: done 869 episodes, mean reward -109.267, speed 84.55 f/s
66860: done 871 episodes, mean reward -109.315, speed 81.40 f/s
66993: done 873 episodes, mean reward -109.377, speed 81.68 f/s
Test done in 0.45 sec, reward -108.443, steps 57
67049: done 874 episodes, mean reward -109.397, speed 48.06 f/s
67188: done 876 episodes, mean reward -109.399, speed 78.09 f/s
67325: done 878 episodes, mean reward -109.434, speed 78.48 f/s
67397: done 879 episodes, mean reward -109.467, speed 66.91 f/s
67550: done 881 episodes, mean reward -109.484, speed 70.36 f/s
67668: done 883 episodes, mean reward -109.462, speed 75.82 f/s
67797: done 885 episodes, mean reward -109.355, speed 79.79 f/s
67915: done 887 episodes, mean reward -109.371, speed 79.41 f/s
67998: done 888 episodes, mean reward -109.341, speed 82.20 f/s
Test done in 12.65 sec, reward -176.533, steps 1600
Test done in 12.08 sec, reward -173.408, steps 1600
69598: done 889 episodes, mean reward -109.751, speed 35.18 f/s
69765: done 891 episodes, mean reward -109.804, speed 81.13 f/s
69856: done 893 episodes, mean reward -109.801, speed 78.08 f/s
69939: done 895 episodes, mean reward -109.834, speed 74.43 f/s
Test done in 2.55 sec, reward -125.299, steps 359
Test done in 3.01 sec, reward -125.156, steps 382
71539: done 896 episodes, mean reward -110.149, speed 61.23 f/s
71679: done 898 episodes, mean reward -110.286, speed 77.12 f/s
Test done in 11.24 sec, reward -162.251, steps 1600
Test done in 10.97 sec, reward -161.480, steps 1600
73279: done 899 episodes, mean reward -110.657, speed 37.25 f/s
73414: done 901 episodes, mean reward -110.806, speed 80.02 f/s
73559: done 903 episodes, mean reward -110.957, speed 77.91 f/s
73694: done 905 episodes, mean reward -111.132, speed 78.65 f/s
Test done in 2.66 sec, reward -128.980, steps 363
Test done in 1.48 sec, reward -116.391, steps 196
75294: done 906 episodes, mean reward -111.539, speed 66.42 f/s
Test done in 11.48 sec, reward -178.285, steps 1600
76894: done 907 episodes, mean reward -111.929, speed 49.95 f/s
Test done in 11.30 sec, reward -176.354, steps 1600
Test done in 2.32 sec, reward -122.791, steps 352
78494: done 908 episodes, mean reward -112.402, speed 48.72 f/s
Test done in 0.40 sec, reward -107.319, steps 48
Test done in 11.26 sec, reward -178.431, steps 1600
80149: done 910 episodes, mean reward -112.835, speed 51.97 f/s
Test done in 11.07 sec, reward -178.109, steps 1600
81749: done 911 episodes, mean reward -113.247, speed 49.29 f/s
Test done in 2.68 sec, reward -120.337, steps 351
Test done in 0.29 sec, reward -108.087, steps 41
83349: done 912 episodes, mean reward -113.662, speed 67.37 f/s
Test done in 10.20 sec, reward -171.413, steps 1444
84949: done 913 episodes, mean reward -114.133, speed 52.75 f/s
Test done in 11.19 sec, reward -178.675, steps 1600
Test done in 2.55 sec, reward -121.300, steps 352
86549: done 914 episodes, mean reward -114.558, speed 47.61 f/s
Test done in 11.21 sec, reward -180.541, steps 1600
Test done in 11.18 sec, reward -179.048, steps 1600
88149: done 915 episodes, mean reward -114.982, speed 37.94 f/s
Test done in 11.00 sec, reward -181.495, steps 1600
89749: done 916 episodes, mean reward -115.453, speed 52.04 f/s
Test done in 11.33 sec, reward -180.475, steps 1600
Test done in 10.81 sec, reward -180.603, steps 1600
91349: done 917 episodes, mean reward -115.907, speed 37.99 f/s
Test done in 12.07 sec, reward -183.587, steps 1600
92949: done 918 episodes, mean reward -116.350, speed 48.16 f/s
Test done in 11.47 sec, reward -183.548, steps 1600
Test done in 11.46 sec, reward -179.631, steps 1600
94549: done 919 episodes, mean reward -116.803, speed 36.43 f/s
Test done in 2.43 sec, reward -157.860, steps 374
Test done in 8.31 sec, reward -173.259, steps 1221
96149: done 920 episodes, mean reward -117.290, speed 53.51 f/s
Test done in 11.37 sec, reward -181.539, steps 1600
97749: done 921 episodes, mean reward -117.748, speed 51.62 f/s
Test done in 10.65 sec, reward -180.876, steps 1600
Test done in 10.94 sec, reward -180.681, steps 1600
99404: done 923 episodes, mean reward -118.151, speed 39.64 f/s
Test done in 11.32 sec, reward -180.936, steps 1600
Test done in 10.99 sec, reward -181.717, steps 1600
101004: done 924 episodes, mean reward -118.628, speed 37.50 f/s
Test done in 11.40 sec, reward -180.975, steps 1600
102604: done 925 episodes, mean reward -119.053, speed 50.91 f/s
Test done in 11.19 sec, reward -180.392, steps 1600
Test done in 12.57 sec, reward -181.026, steps 1600
104204: done 926 episodes, mean reward -119.512, speed 36.56 f/s
Test done in 11.31 sec, reward -180.825, steps 1600
105804: done 927 episodes, mean reward -119.934, speed 50.73 f/s
Test done in 9.19 sec, reward -163.037, steps 1298
Test done in 10.97 sec, reward -178.311, steps 1600
107404: done 928 episodes, mean reward -120.321, speed 40.16 f/s
Test done in 12.18 sec, reward -180.234, steps 1600
Test done in 10.83 sec, reward -181.098, steps 1600
109004: done 929 episodes, mean reward -120.701, speed 36.30 f/s
Test done in 11.08 sec, reward -180.686, steps 1600
110652: done 931 episodes, mean reward -121.124, speed 53.33 f/s
Test done in 10.82 sec, reward -179.865, steps 1600
Test done in 10.67 sec, reward -180.517, steps 1600
112252: done 932 episodes, mean reward -121.538, speed 38.87 f/s
Test done in 11.67 sec, reward -180.606, steps 1600
113852: done 933 episodes, mean reward -121.952, speed 49.02 f/s
Test done in 11.28 sec, reward -180.705, steps 1600
Test done in 11.11 sec, reward -180.768, steps 1600
115452: done 934 episodes, mean reward -122.369, speed 37.56 f/s
Test done in 12.13 sec, reward -179.722, steps 1600
Test done in 11.41 sec, reward -181.052, steps 1600
117052: done 935 episodes, mean reward -122.786, speed 36.77 f/s
Test done in 10.87 sec, reward -180.627, steps 1600
118652: done 936 episodes, mean reward -123.193, speed 51.67 f/s
Test done in 10.77 sec, reward -176.696, steps 1600
Test done in 11.70 sec, reward -180.545, steps 1600
120252: done 937 episodes, mean reward -123.595, speed 37.48 f/s
Test done in 10.96 sec, reward -175.977, steps 1600
121852: done 938 episodes, mean reward -124.044, speed 51.64 f/s
Test done in 11.51 sec, reward -169.893, steps 1600
Test done in 10.65 sec, reward -171.464, steps 1600
123452: done 939 episodes, mean reward -124.425, speed 38.47 f/s
Test done in 11.80 sec, reward -179.652, steps 1600
Test done in 11.39 sec, reward -174.819, steps 1600
125052: done 940 episodes, mean reward -124.846, speed 37.16 f/s
Test done in 11.23 sec, reward -177.832, steps 1600
126652: done 941 episodes, mean reward -125.256, speed 51.37 f/s
Test done in 10.94 sec, reward -176.362, steps 1600
Test done in 10.62 sec, reward -168.378, steps 1600
128252: done 942 episodes, mean reward -125.623, speed 39.03 f/s
Test done in 10.98 sec, reward -175.421, steps 1600
129852: done 943 episodes, mean reward -126.044, speed 51.98 f/s
Test done in 10.91 sec, reward -176.628, steps 1600
Test done in 10.78 sec, reward -178.546, steps 1600
131452: done 944 episodes, mean reward -126.451, speed 38.91 f/s
Test done in 11.07 sec, reward -165.291, steps 1600
Test done in 10.67 sec, reward -154.638, steps 1600
133052: done 945 episodes, mean reward -126.796, speed 38.63 f/s
Test done in 10.97 sec, reward -147.495, steps 1600
134652: done 946 episodes, mean reward -127.035, speed 51.78 f/s
Test done in 10.90 sec, reward -142.559, steps 1600
Test done in 10.68 sec, reward -141.777, steps 1600
136252: done 947 episodes, mean reward -127.234, speed 38.93 f/s
136381: done 948 episodes, mean reward -127.118, speed 85.60 f/s
Test done in 2.73 sec, reward -116.687, steps 381
137981: done 949 episodes, mean reward -127.276, speed 70.35 f/s
Test done in 0.44 sec, reward -102.295, steps 64
138145: done 950 episodes, mean reward -127.210, speed 66.33 f/s
Test done in 10.54 sec, reward -172.076, steps 1600
139745: done 951 episodes, mean reward -127.551, speed 52.54 f/s
Test done in 10.34 sec, reward -151.599, steps 1457
Test done in 0.50 sec, reward -100.771, steps 70
141345: done 952 episodes, mean reward -127.889, speed 51.80 f/s
Test done in 0.39 sec, reward -102.145, steps 51
142945: done 953 episodes, mean reward -128.282, speed 80.69 f/s
Test done in 10.00 sec, reward -156.508, steps 1449
Test done in 3.82 sec, reward -123.360, steps 559
144585: done 955 episodes, mean reward -128.662, speed 48.32 f/s
Test done in 10.10 sec, reward -169.349, steps 1448
Test done in 10.84 sec, reward -179.052, steps 1600
146254: done 957 episodes, mean reward -129.011, speed 40.08 f/s
Test done in 10.92 sec, reward -165.561, steps 1600
147854: done 958 episodes, mean reward -129.383, speed 52.20 f/s
147955: done 960 episodes, mean reward -129.289, speed 82.27 f/s
Test done in 0.44 sec, reward -100.724, steps 58
148001: done 961 episodes, mean reward -129.248, speed 44.79 f/s
148110: done 963 episodes, mean reward -129.126, speed 80.27 f/s
Test done in 10.97 sec, reward -153.976, steps 1600
149768: done 965 episodes, mean reward -129.341, speed 53.04 f/s
149864: done 967 episodes, mean reward -129.284, speed 78.60 f/s
149978: done 968 episodes, mean reward -129.208, speed 82.97 f/s
Test done in 10.91 sec, reward -161.014, steps 1600
Test done in 11.02 sec, reward -156.228, steps 1600
151578: done 969 episodes, mean reward -129.560, speed 38.33 f/s
Test done in 9.87 sec, reward -138.051, steps 1456
Test done in 11.06 sec, reward -178.772, steps 1600
153178: done 970 episodes, mean reward -129.894, speed 39.47 f/s
Test done in 10.77 sec, reward -161.069, steps 1600
154778: done 971 episodes, mean reward -130.215, speed 52.53 f/s
Test done in 10.87 sec, reward -141.626, steps 1600
Test done in 10.87 sec, reward -158.600, steps 1600
156378: done 972 episodes, mean reward -130.553, speed 39.07 f/s
Test done in 10.97 sec, reward -169.871, steps 1600
157978: done 973 episodes, mean reward -130.938, speed 52.10 f/s
Test done in 10.98 sec, reward -183.561, steps 1597
Test done in 10.86 sec, reward -158.214, steps 1600
159578: done 974 episodes, mean reward -131.178, speed 38.62 f/s
Test done in 10.77 sec, reward -147.511, steps 1600
Test done in 10.05 sec, reward -171.575, steps 1484
161178: done 975 episodes, mean reward -131.531, speed 39.66 f/s
Test done in 11.02 sec, reward -155.592, steps 1600
162819: done 977 episodes, mean reward -131.825, speed 52.73 f/s
Test done in 10.80 sec, reward -164.402, steps 1600
Test done in 10.67 sec, reward -152.790, steps 1600
164419: done 978 episodes, mean reward -132.088, speed 38.82 f/s
Test done in 10.68 sec, reward -161.606, steps 1600
Test done in 10.52 sec, reward -163.863, steps 1600
166019: done 979 episodes, mean reward -132.400, speed 39.33 f/s
Test done in 10.84 sec, reward -163.321, steps 1600
167619: done 980 episodes, mean reward -132.712, speed 52.73 f/s
Test done in 10.85 sec, reward -163.664, steps 1600
Test done in 10.99 sec, reward -178.527, steps 1600
169219: done 981 episodes, mean reward -133.043, speed 38.50 f/s
Test done in 10.78 sec, reward -168.348, steps 1600
170819: done 982 episodes, mean reward -133.315, speed 52.89 f/s
Test done in 10.81 sec, reward -178.456, steps 1600
Test done in 11.13 sec, reward -166.205, steps 1600
172419: done 983 episodes, mean reward -133.609, speed 38.32 f/s
Test done in 11.13 sec, reward -161.341, steps 1600
Test done in 11.09 sec, reward -169.001, steps 1600
174019: done 984 episodes, mean reward -133.867, speed 37.85 f/s
Test done in 10.68 sec, reward -167.035, steps 1600
175619: done 985 episodes, mean reward -134.143, speed 52.92 f/s
Test done in 11.23 sec, reward -162.046, steps 1600
Test done in 10.96 sec, reward -164.058, steps 1600
177219: done 986 episodes, mean reward -134.358, speed 38.18 f/s
Test done in 11.11 sec, reward -168.546, steps 1600
178819: done 987 episodes, mean reward -134.719, speed 51.43 f/s
Test done in 11.14 sec, reward -168.810, steps 1600
Test done in 10.94 sec, reward -175.485, steps 1600
180419: done 988 episodes, mean reward -134.997, speed 37.78 f/s
Test done in 10.76 sec, reward -169.825, steps 1600
Test done in 12.21 sec, reward -168.085, steps 1600
182019: done 989 episodes, mean reward -134.799, speed 36.24 f/s
Test done in 11.36 sec, reward -152.591, steps 1600
183619: done 990 episodes, mean reward -135.072, speed 48.37 f/s
Test done in 9.91 sec, reward -164.475, steps 1342
Test done in 10.96 sec, reward -154.347, steps 1600
185219: done 991 episodes, mean reward -135.374, speed 38.31 f/s
185893: done 992 episodes, mean reward -135.804, speed 75.87 f/s
Test done in 10.95 sec, reward -169.751, steps 1600
Test done in 8.75 sec, reward -158.482, steps 1150
187079: done 993 episodes, mean reward -136.590, speed 34.22 f/s
Test done in 11.35 sec, reward -174.819, steps 1600
188518: done 994 episodes, mean reward -137.861, speed 46.67 f/s
188607: done 995 episodes, mean reward -137.994, speed 78.43 f/s
188704: done 997 episodes, mean reward -137.682, speed 77.43 f/s
188886: done 999 episodes, mean reward -137.585, speed 73.02 f/s
Test done in 7.20 sec, reward -162.121, steps 1008
189027: done 1001 episodes, mean reward -137.667, speed 15.71 f/s
189230: done 1002 episodes, mean reward -137.874, speed 79.66 f/s
189366: done 1005 episodes, mean reward -137.816, speed 82.53 f/s
Test done in 9.15 sec, reward -198.271, steps 1316
Test done in 11.20 sec, reward -170.393, steps 1600
191009: done 1007 episodes, mean reward -137.504, speed 40.79 f/s
Test done in 11.22 sec, reward -171.204, steps 1600
192609: done 1008 episodes, mean reward -137.373, speed 52.23 f/s
Test done in 11.40 sec, reward -153.012, steps 1600
Test done in 11.45 sec, reward -133.152, steps 1600
194209: done 1009 episodes, mean reward -137.459, speed 37.68 f/s
Test done in 11.43 sec, reward -147.076, steps 1600
195809: done 1010 episodes, mean reward -137.075, speed 51.12 f/s
Test done in 11.15 sec, reward -155.726, steps 1600
Test done in 11.40 sec, reward -150.944, steps 1600
197409: done 1011 episodes, mean reward -136.942, speed 38.25 f/s
Test done in 11.26 sec, reward -147.301, steps 1600
Test done in 11.41 sec, reward -150.238, steps 1600
199009: done 1012 episodes, mean reward -136.928, speed 37.78 f/s
Test done in 11.90 sec, reward -143.231, steps 1600
200609: done 1013 episodes, mean reward -136.906, speed 49.43 f/s
Test done in 11.98 sec, reward -148.887, steps 1600
Test done in 12.53 sec, reward -148.768, steps 1600
202209: done 1014 episodes, mean reward -136.906, speed 35.50 f/s
Test done in 11.30 sec, reward -149.544, steps 1600
203809: done 1015 episodes, mean reward -136.857, speed 51.77 f/s
Test done in 11.63 sec, reward -149.608, steps 1600
Test done in 11.25 sec, reward -148.102, steps 1600
205409: done 1016 episodes, mean reward -136.760, speed 37.61 f/s
Test done in 11.85 sec, reward -155.999, steps 1600
Test done in 11.48 sec, reward -165.481, steps 1600
207009: done 1017 episodes, mean reward -136.611, speed 36.29 f/s
Test done in 11.98 sec, reward -152.601, steps 1600
208609: done 1018 episodes, mean reward -136.484, speed 49.08 f/s
Test done in 11.73 sec, reward -144.057, steps 1600
Test done in 1.65 sec, reward -118.592, steps 204
210180: done 1019 episodes, mean reward -137.383, speed 46.31 f/s
210653: done 1020 episodes, mean reward -137.230, speed 78.10 f/s
210850: done 1022 episodes, mean reward -137.070, speed 79.42 f/s
Test done in 12.07 sec, reward -90.758, steps 1600
Best reward updated: -92.664 -> -90.758
Test done in 11.26 sec, reward -175.794, steps 1600
212450: done 1023 episodes, mean reward -136.922, speed 36.61 f/s
Test done in 11.43 sec, reward -148.503, steps 1600
213924: done 1024 episodes, mean reward -137.376, speed 49.91 f/s
Test done in 1.03 sec, reward -122.435, steps 149
214061: done 1025 episodes, mean reward -137.040, speed 50.79 f/s
214198: done 1026 episodes, mean reward -136.637, speed 82.58 f/s
214371: done 1027 episodes, mean reward -136.311, speed 82.08 f/s
214542: done 1028 episodes, mean reward -135.952, speed 83.37 f/s
214725: done 1029 episodes, mean reward -135.667, speed 79.88 f/s
214864: done 1030 episodes, mean reward -135.776, speed 86.13 f/s
Test done in 2.57 sec, reward -118.351, steps 377
215010: done 1031 episodes, mean reward -135.425, speed 33.34 f/s
215199: done 1032 episodes, mean reward -135.106, speed 81.87 f/s
215389: done 1033 episodes, mean reward -134.751, speed 83.62 f/s
215557: done 1034 episodes, mean reward -134.341, speed 80.80 f/s
Test done in 10.98 sec, reward -145.414, steps 1600
216190: done 1035 episodes, mean reward -134.387, speed 33.58 f/s
216335: done 1036 episodes, mean reward -134.075, speed 72.67 f/s
216517: done 1037 episodes, mean reward -133.804, speed 82.48 f/s
216664: done 1038 episodes, mean reward -133.436, speed 80.02 f/s
Test done in 1.78 sec, reward -129.632, steps 250
217073: done 1039 episodes, mean reward -133.303, speed 57.23 f/s
217382: done 1040 episodes, mean reward -133.246, speed 76.58 f/s
217534: done 1041 episodes, mean reward -132.979, speed 74.85 f/s
217916: done 1042 episodes, mean reward -133.114, speed 77.72 f/s
Test done in 1.36 sec, reward -130.061, steps 165
218037: done 1043 episodes, mean reward -132.820, speed 40.41 f/s
218159: done 1044 episodes, mean reward -132.634, speed 73.55 f/s
218317: done 1046 episodes, mean reward -132.478, speed 77.62 f/s
218461: done 1047 episodes, mean reward -132.451, speed 75.63 f/s
218647: done 1048 episodes, mean reward -132.802, speed 85.97 f/s
218791: done 1049 episodes, mean reward -132.855, speed 82.99 f/s
218957: done 1050 episodes, mean reward -133.161, speed 81.08 f/s
Test done in 2.04 sec, reward -138.481, steps 300
219128: done 1051 episodes, mean reward -133.124, speed 40.97 f/s
219230: done 1052 episodes, mean reward -132.829, speed 77.82 f/s
219344: done 1053 episodes, mean reward -132.708, speed 80.05 f/s
Test done in 10.04 sec, reward -172.037, steps 1458
220944: done 1054 episodes, mean reward -133.158, speed 54.66 f/s
Test done in 3.51 sec, reward -142.641, steps 462
221145: done 1055 episodes, mean reward -132.996, speed 33.62 f/s
221315: done 1056 episodes, mean reward -133.304, speed 80.80 f/s
221460: done 1057 episodes, mean reward -133.065, speed 83.03 f/s
Test done in 9.32 sec, reward -157.407, steps 1296
Test done in 3.78 sec, reward -133.388, steps 550
223060: done 1058 episodes, mean reward -133.166, speed 49.44 f/s
223201: done 1059 episodes, mean reward -133.394, speed 82.25 f/s
Test done in 10.21 sec, reward -179.096, steps 1460
224801: done 1060 episodes, mean reward -133.851, speed 54.58 f/s
224894: done 1061 episodes, mean reward -133.966, speed 82.75 f/s
Test done in 0.54 sec, reward -120.313, steps 77
Test done in 11.04 sec, reward -162.270, steps 1600
226494: done 1062 episodes, mean reward -134.411, speed 51.07 f/s
226586: done 1063 episodes, mean reward -134.525, speed 83.73 f/s
Test done in 5.78 sec, reward -140.257, steps 840
227509: done 1064 episodes, mean reward -135.434, speed 54.43 f/s
Test done in 11.25 sec, reward -161.900, steps 1600
Test done in 10.12 sec, reward -157.051, steps 1456
229109: done 1065 episodes, mean reward -135.513, speed 39.10 f/s
229516: done 1066 episodes, mean reward -135.837, speed 82.94 f/s
229709: done 1067 episodes, mean reward -136.003, speed 83.45 f/s
229952: done 1068 episodes, mean reward -136.379, speed 81.64 f/s
Test done in 6.19 sec, reward -139.090, steps 871
230005: done 1069 episodes, mean reward -136.039, speed 7.81 f/s
230668: done 1071 episodes, mean reward -135.848, speed 83.45 f/s
230836: done 1073 episodes, mean reward -135.219, speed 84.01 f/s
Test done in 2.42 sec, reward -123.650, steps 352
231119: done 1075 episodes, mean reward -134.846, speed 49.39 f/s
231858: done 1076 episodes, mean reward -135.342, speed 81.44 f/s
231944: done 1077 episodes, mean reward -135.064, speed 84.90 f/s
Test done in 7.29 sec, reward -142.159, steps 1073
232018: done 1078 episodes, mean reward -134.872, speed 9.00 f/s
232346: done 1080 episodes, mean reward -134.456, speed 81.26 f/s
232980: done 1081 episodes, mean reward -134.538, speed 78.15 f/s
Test done in 4.33 sec, reward -117.724, steps 613
233143: done 1082 episodes, mean reward -134.276, speed 25.61 f/s
233394: done 1083 episodes, mean reward -134.292, speed 79.27 f/s
233624: done 1084 episodes, mean reward -134.323, speed 82.24 f/s
233708: done 1085 episodes, mean reward -134.106, speed 82.75 f/s
Test done in 2.62 sec, reward -124.313, steps 362
234217: done 1086 episodes, mean reward -134.276, speed 58.01 f/s
234303: done 1087 episodes, mean reward -133.993, speed 81.82 f/s
234508: done 1088 episodes, mean reward -133.975, speed 79.35 f/s
234591: done 1089 episodes, mean reward -133.788, speed 80.70 f/s
234732: done 1091 episodes, mean reward -133.223, speed 80.82 f/s
Test done in 5.57 sec, reward -145.054, steps 796
235065: done 1092 episodes, mean reward -133.094, speed 34.68 f/s
235815: done 1094 episodes, mean reward -131.457, speed 79.79 f/s
235954: done 1096 episodes, mean reward -131.278, speed 81.24 f/s
Test done in 4.96 sec, reward -137.946, steps 734
236075: done 1097 episodes, mean reward -131.383, speed 18.77 f/s
236904: done 1098 episodes, mean reward -131.855, speed 82.80 f/s
236991: done 1099 episodes, mean reward -131.547, speed 84.59 f/s
Test done in 5.37 sec, reward -134.163, steps 774
237290: done 1100 episodes, mean reward -131.751, speed 33.32 f/s
237418: done 1102 episodes, mean reward -131.278, speed 81.25 f/s
Test done in 6.51 sec, reward -122.084, steps 910
238217: done 1103 episodes, mean reward -131.883, speed 49.79 f/s
238302: done 1104 episodes, mean reward -131.799, speed 80.85 f/s
Test done in 11.06 sec, reward -131.240, steps 1600
239440: done 1105 episodes, mean reward -132.520, speed 45.00 f/s
239538: done 1106 episodes, mean reward -132.412, speed 83.37 f/s
Test done in 0.72 sec, reward -138.475, steps 100
240447: done 1107 episodes, mean reward -132.925, speed 78.25 f/s
240794: done 1108 episodes, mean reward -133.043, speed 74.13 f/s
240974: done 1109 episodes, mean reward -133.212, speed 75.53 f/s
Test done in 1.10 sec, reward -134.928, steps 145
241185: done 1110 episodes, mean reward -133.440, speed 56.01 f/s
Test done in 11.62 sec, reward -118.533, steps 1600
242785: done 1111 episodes, mean reward -133.472, speed 50.10 f/s
Test done in 5.59 sec, reward -134.818, steps 820
243427: done 1112 episodes, mean reward -133.678, speed 46.69 f/s
Test done in 10.86 sec, reward -124.798, steps 1460

Test done in 11.73 sec, reward -82.179, steps 1600
Best reward updated: -90.758 -> -82.179
245027: done 1113 episodes, mean reward -133.312, speed 36.30 f/s
Test done in 11.23 sec, reward -41.998, steps 1362
Best reward updated: -82.179 -> -41.998
246627: done 1114 episodes, mean reward -132.604, speed 49.49 f/s
Test done in 10.64 sec, reward -97.757, steps 1473
247220: done 1115 episodes, mean reward -132.607, speed 32.53 f/s
Test done in 11.46 sec, reward -107.006, steps 1600
248820: done 1116 episodes, mean reward -132.097, speed 50.54 f/s
Test done in 6.65 sec, reward -141.868, steps 920
249193: done 1117 episodes, mean reward -132.110, speed 32.10 f/s
249608: done 1118 episodes, mean reward -132.225, speed 81.06 f/s
Test done in 11.66 sec, reward -175.756, steps 1600
Test done in 11.48 sec, reward -169.696, steps 1600
251208: done 1119 episodes, mean reward -131.240, speed 37.15 f/s
Test done in 11.14 sec, reward -163.103, steps 1600
252808: done 1120 episodes, mean reward -131.334, speed 50.17 f/s
252986: done 1121 episodes, mean reward -131.475, speed 80.32 f/s
Test done in 11.50 sec, reward -170.758, steps 1600
Test done in 11.29 sec, reward -173.647, steps 1600
254523: done 1122 episodes, mean reward -132.748, speed 36.51 f/s
Test done in 11.63 sec, reward -139.095, steps 1600
Test done in 11.89 sec, reward -125.471, steps 1600
256123: done 1123 episodes, mean reward -132.532, speed 36.31 f/s
Test done in 11.09 sec, reward -121.439, steps 1600
257723: done 1124 episodes, mean reward -131.758, speed 51.63 f/s
Test done in 10.73 sec, reward -139.364, steps 1600
Test done in 11.23 sec, reward -120.223, steps 1600
259323: done 1125 episodes, mean reward -131.796, speed 38.29 f/s
Test done in 11.26 sec, reward -128.389, steps 1600
260923: done 1126 episodes, mean reward -131.930, speed 51.42 f/s
Test done in 11.33 sec, reward -142.117, steps 1600
Test done in 11.13 sec, reward -121.170, steps 1600
262523: done 1127 episodes, mean reward -131.925, speed 37.88 f/s
Test done in 10.77 sec, reward -150.379, steps 1600
Test done in 11.09 sec, reward -140.672, steps 1600
264123: done 1128 episodes, mean reward -131.971, speed 38.74 f/s
Test done in 11.05 sec, reward -116.814, steps 1600
265723: done 1129 episodes, mean reward -131.935, speed 51.92 f/s
Test done in 10.99 sec, reward -131.774, steps 1600
Test done in 11.55 sec, reward -144.251, steps 1600
267323: done 1130 episodes, mean reward -132.013, speed 37.96 f/s
Test done in 11.10 sec, reward -127.849, steps 1600
268923: done 1131 episodes, mean reward -131.980, speed 52.29 f/s
Test done in 11.12 sec, reward -118.993, steps 1600
Test done in 10.92 sec, reward -126.104, steps 1600
270523: done 1132 episodes, mean reward -131.802, speed 38.38 f/s
Test done in 11.53 sec, reward -123.955, steps 1600
Test done in 11.05 sec, reward -128.251, steps 1600
272123: done 1133 episodes, mean reward -131.690, speed 37.82 f/s
Test done in 10.95 sec, reward -124.616, steps 1600
273723: done 1134 episodes, mean reward -131.791, speed 52.45 f/s
Test done in 11.05 sec, reward -124.563, steps 1600
Test done in 11.21 sec, reward -119.283, steps 1600
275323: done 1135 episodes, mean reward -131.234, speed 38.19 f/s
Test done in 11.12 sec, reward -124.590, steps 1600
276972: done 1137 episodes, mean reward -131.031, speed 52.78 f/s
Test done in 11.36 sec, reward -115.615, steps 1600
Test done in 11.56 sec, reward -127.265, steps 1600
278572: done 1138 episodes, mean reward -131.019, speed 37.52 f/s
Test done in 11.22 sec, reward -146.920, steps 1600
Test done in 11.13 sec, reward -150.166, steps 1600
280074: done 1139 episodes, mean reward -131.750, speed 36.83 f/s
Test done in 11.26 sec, reward -159.972, steps 1600
281674: done 1140 episodes, mean reward -131.592, speed 51.27 f/s
Test done in 12.36 sec, reward -148.859, steps 1600
Test done in 11.89 sec, reward -149.326, steps 1600
283274: done 1141 episodes, mean reward -131.605, speed 35.93 f/s
Test done in 11.67 sec, reward -164.483, steps 1600
284874: done 1142 episodes, mean reward -131.372, speed 50.24 f/s
Test done in 11.30 sec, reward -170.869, steps 1600
Test done in 11.20 sec, reward -153.532, steps 1600
286474: done 1143 episodes, mean reward -131.506, speed 37.54 f/s
Test done in 12.08 sec, reward -180.230, steps 1600
Test done in 11.36 sec, reward -182.593, steps 1600
288074: done 1144 episodes, mean reward -131.692, speed 36.45 f/s
Test done in 9.45 sec, reward -172.495, steps 1341
289674: done 1145 episodes, mean reward -132.062, speed 53.86 f/s
Test done in 5.00 sec, reward -154.411, steps 716
Test done in 4.88 sec, reward -162.711, steps 722
291274: done 1146 episodes, mean reward -132.260, speed 53.62 f/s
291518: done 1147 episodes, mean reward -132.299, speed 80.53 f/s
291653: done 1148 episodes, mean reward -132.192, speed 78.47 f/s
Test done in 5.67 sec, reward -184.629, steps 819
Test done in 2.05 sec, reward -150.463, steps 273
293071: done 1149 episodes, mean reward -133.292, speed 56.53 f/s
Test done in 1.87 sec, reward -152.412, steps 272
294594: done 1150 episodes, mean reward -134.471, speed 72.84 f/s
Test done in 1.40 sec, reward -146.397, steps 200
295147: done 1151 episodes, mean reward -134.651, speed 64.50 f/s
295364: done 1152 episodes, mean reward -134.799, speed 80.71 f/s
295576: done 1153 episodes, mean reward -134.830, speed 77.94 f/s
295755: done 1154 episodes, mean reward -134.481, speed 79.14 f/s
Test done in 2.79 sec, reward -163.630, steps 402
296053: done 1155 episodes, mean reward -134.600, speed 45.97 f/s
296142: done 1156 episodes, mean reward -134.474, speed 79.95 f/s
296953: done 1157 episodes, mean reward -135.063, speed 79.74 f/s
Test done in 2.57 sec, reward -152.381, steps 364
Test done in 5.50 sec, reward -195.998, steps 766
298026: done 1158 episodes, mean reward -135.758, speed 50.16 f/s
298487: done 1159 episodes, mean reward -136.158, speed 81.99 f/s
298637: done 1160 episodes, mean reward -135.959, speed 81.73 f/s
Test done in 6.46 sec, reward -220.037, steps 887
299907: done 1161 episodes, mean reward -137.235, speed 56.95 f/s
Test done in 6.81 sec, reward -202.033, steps 933
300708: done 1162 episodes, mean reward -137.787, speed 47.22 f/s
300844: done 1163 episodes, mean reward -138.004, speed 76.37 f/s
Test done in 3.27 sec, reward -171.737, steps 453
Test done in 5.53 sec, reward -209.534, steps 803
302444: done 1164 episodes, mean reward -137.660, speed 55.65 f/s
Test done in 6.59 sec, reward -180.874, steps 930
Test done in 10.86 sec, reward -185.674, steps 1470
304044: done 1165 episodes, mean reward -137.695, speed 42.44 f/s
Test done in 11.11 sec, reward -185.691, steps 1600
305644: done 1166 episodes, mean reward -137.885, speed 51.27 f/s
Test done in 4.31 sec, reward -151.687, steps 600
Test done in 11.53 sec, reward -182.975, steps 1600
307244: done 1167 episodes, mean reward -138.253, speed 44.29 f/s
Test done in 11.19 sec, reward -184.401, steps 1600
308844: done 1168 episodes, mean reward -138.413, speed 51.38 f/s
Test done in 11.43 sec, reward -166.884, steps 1600
Test done in 11.58 sec, reward -153.482, steps 1600
310444: done 1169 episodes, mean reward -138.777, speed 36.85 f/s
Test done in 11.31 sec, reward -156.936, steps 1600
Test done in 11.59 sec, reward -169.321, steps 1600
312044: done 1170 episodes, mean reward -139.170, speed 37.42 f/s
312425: done 1171 episodes, mean reward -139.159, speed 78.54 f/s
Test done in 11.51 sec, reward -174.925, steps 1600
Test done in 11.49 sec, reward -177.723, steps 1600
314025: done 1172 episodes, mean reward -139.557, speed 37.35 f/s
Test done in 3.81 sec, reward -153.757, steps 508
315358: done 1173 episodes, mean reward -140.816, speed 63.96 f/s
Test done in 11.31 sec, reward -180.023, steps 1600
316958: done 1174 episodes, mean reward -141.310, speed 50.68 f/s
Test done in 3.02 sec, reward -149.134, steps 389
317206: done 1175 episodes, mean reward -141.429, speed 40.59 f/s
317413: done 1176 episodes, mean reward -141.269, speed 80.61 f/s
317768: done 1177 episodes, mean reward -141.803, speed 78.58 f/s
Test done in 3.42 sec, reward -140.666, steps 467
318017: done 1178 episodes, mean reward -142.089, speed 38.43 f/s
318130: done 1179 episodes, mean reward -142.242, speed 76.10 f/s
318466: done 1180 episodes, mean reward -142.485, speed 79.02 f/s
318634: done 1181 episodes, mean reward -142.316, speed 78.92 f/s
318859: done 1182 episodes, mean reward -142.575, speed 82.89 f/s
Test done in 2.29 sec, reward -140.127, steps 317
319050: done 1183 episodes, mean reward -142.574, speed 40.41 f/s
319289: done 1184 episodes, mean reward -142.631, speed 80.98 f/s
319550: done 1185 episodes, mean reward -142.901, speed 80.13 f/s
319725: done 1186 episodes, mean reward -142.735, speed 78.28 f/s
319922: done 1187 episodes, mean reward -142.940, speed 77.98 f/s
Test done in 1.48 sec, reward -139.680, steps 196
320155: done 1188 episodes, mean reward -142.974, speed 53.85 f/s
Test done in 8.32 sec, reward -170.339, steps 1192
321755: done 1189 episodes, mean reward -143.291, speed 56.29 f/s
Test done in 9.65 sec, reward -172.525, steps 1323
Test done in 8.47 sec, reward -166.249, steps 1203
323355: done 1190 episodes, mean reward -143.707, speed 41.21 f/s
Test done in 6.34 sec, reward -159.660, steps 910
324733: done 1191 episodes, mean reward -145.058, speed 58.54 f/s
Test done in 3.35 sec, reward -134.889, steps 458
Test done in 5.93 sec, reward -128.961, steps 848
326333: done 1192 episodes, mean reward -145.159, speed 53.68 f/s
Test done in 11.30 sec, reward -170.945, steps 1600
327136: done 1193 episodes, mean reward -145.958, speed 36.93 f/s
Test done in 0.52 sec, reward -116.548, steps 72
328813: done 1195 episodes, mean reward -145.731, speed 78.46 f/s
328941: done 1196 episodes, mean reward -145.726, speed 80.55 f/s
Test done in 4.05 sec, reward -118.611, steps 578
Test done in 11.32 sec, reward -116.578, steps 1600
330541: done 1197 episodes, mean reward -145.665, speed 45.04 f/s
330628: done 1198 episodes, mean reward -145.136, speed 78.84 f/s
Test done in 5.97 sec, reward -125.170, steps 848
Test done in 10.74 sec, reward -128.842, steps 1600
332228: done 1199 episodes, mean reward -145.173, speed 43.26 f/s
Test done in 11.26 sec, reward -114.852, steps 1600
333828: done 1200 episodes, mean reward -144.854, speed 50.47 f/s
Test done in 11.24 sec, reward -129.446, steps 1600
Test done in 10.89 sec, reward -117.662, steps 1600
335428: done 1201 episodes, mean reward -144.756, speed 37.61 f/s
Test done in 10.87 sec, reward -119.194, steps 1600
Test done in 11.16 sec, reward -115.436, steps 1600
337028: done 1202 episodes, mean reward -144.492, speed 37.90 f/s
Test done in 10.25 sec, reward -84.182, steps 1419
338628: done 1203 episodes, mean reward -143.390, speed 53.02 f/s
Test done in 11.39 sec, reward -118.506, steps 1600
Test done in 11.02 sec, reward -106.606, steps 1600
340228: done 1204 episodes, mean reward -142.882, speed 37.81 f/s
Test done in 10.91 sec, reward -103.071, steps 1600
341500: done 1205 episodes, mean reward -142.813, speed 46.72 f/s
Test done in 11.22 sec, reward -116.348, steps 1600
Test done in 11.41 sec, reward -132.445, steps 1600
343100: done 1206 episodes, mean reward -142.843, speed 37.58 f/s
Test done in 11.92 sec, reward -127.339, steps 1600
344700: done 1207 episodes, mean reward -141.914, speed 48.81 f/s
Test done in 11.43 sec, reward -95.175, steps 1600
Test done in 11.51 sec, reward -87.814, steps 1600
346300: done 1208 episodes, mean reward -141.150, speed 36.39 f/s
346893: done 1209 episodes, mean reward -141.062, speed 71.83 f/s
Test done in 8.38 sec, reward -117.790, steps 1166
Test done in 12.43 sec, reward -72.353, steps 1600
348493: done 1210 episodes, mean reward -140.317, speed 37.33 f/s
Test done in 10.91 sec, reward -70.371, steps 1562
Test done in 8.49 sec, reward -109.530, steps 1234
350093: done 1211 episodes, mean reward -139.240, speed 40.05 f/s
Test done in 10.53 sec, reward -74.141, steps 1510
351693: done 1212 episodes, mean reward -138.178, speed 51.86 f/s
Test done in 10.71 sec, reward -70.536, steps 1600
Test done in 8.91 sec, reward -107.946, steps 1274
353293: done 1213 episodes, mean reward -137.474, speed 40.21 f/s
Test done in 12.64 sec, reward -103.167, steps 1600
354893: done 1214 episodes, mean reward -137.398, speed 45.20 f/s
Test done in 12.65 sec, reward -126.985, steps 1600
Test done in 12.38 sec, reward -109.961, steps 1600
356493: done 1215 episodes, mean reward -136.780, speed 34.36 f/s
Test done in 12.48 sec, reward -97.443, steps 1600
Test done in 11.35 sec, reward -108.709, steps 1600
358093: done 1216 episodes, mean reward -136.776, speed 36.13 f/s
Test done in 11.02 sec, reward -103.561, steps 1600
359693: done 1217 episodes, mean reward -136.383, speed 51.61 f/s
Test done in 11.23 sec, reward -117.648, steps 1600
Test done in 12.15 sec, reward -108.649, steps 1600
361293: done 1218 episodes, mean reward -135.924, speed 36.88 f/s
Test done in 11.21 sec, reward -95.338, steps 1600
362893: done 1219 episodes, mean reward -135.312, speed 51.07 f/s
Test done in 11.23 sec, reward -97.296, steps 1600
Test done in 10.46 sec, reward -111.302, steps 1504
364493: done 1220 episodes, mean reward -134.542, speed 38.08 f/s
Test done in 10.20 sec, reward -88.477, steps 1463
Test done in 10.02 sec, reward -103.972, steps 1459
366093: done 1221 episodes, mean reward -133.909, speed 40.08 f/s
Test done in 9.57 sec, reward -94.218, steps 1427
367693: done 1222 episodes, mean reward -131.913, speed 53.74 f/s
Test done in 9.05 sec, reward -143.303, steps 1315
Test done in 9.04 sec, reward -128.913, steps 1317
369293: done 1223 episodes, mean reward -131.525, speed 42.36 f/s
369743: done 1224 episodes, mean reward -131.639, speed 80.65 f/s
Test done in 9.65 sec, reward -111.845, steps 1424
Test done in 9.78 sec, reward -135.807, steps 1432
371166: done 1225 episodes, mean reward -132.287, speed 38.36 f/s
Test done in 8.55 sec, reward -165.002, steps 1273
372766: done 1226 episodes, mean reward -132.268, speed 56.12 f/s
372940: done 1227 episodes, mean reward -132.372, speed 78.64 f/s
Test done in 9.23 sec, reward -144.841, steps 1325
373810: done 1228 episodes, mean reward -132.878, speed 42.86 f/s
Test done in 9.97 sec, reward -153.585, steps 1455
Test done in 9.24 sec, reward -176.502, steps 1333
375410: done 1229 episodes, mean reward -132.762, speed 40.89 f/s
Test done in 3.24 sec, reward -168.093, steps 469
Test done in 10.29 sec, reward -138.504, steps 1495
377010: done 1230 episodes, mean reward -132.592, speed 47.03 f/s
377130: done 1231 episodes, mean reward -132.686, speed 79.90 f/s
Test done in 9.51 sec, reward -151.616, steps 1451
378730: done 1232 episodes, mean reward -132.712, speed 55.05 f/s
Test done in 9.53 sec, reward -145.607, steps 1428
Test done in 10.53 sec, reward -132.462, steps 1529
380173: done 1233 episodes, mean reward -134.102, speed 37.76 f/s
380878: done 1234 episodes, mean reward -134.556, speed 81.64 f/s
Test done in 9.49 sec, reward -168.721, steps 1432
Test done in 11.07 sec, reward -150.323, steps 1572
382182: done 1235 episodes, mean reward -135.427, speed 35.49 f/s
Test done in 8.56 sec, reward -154.485, steps 1265
383782: done 1236 episodes, mean reward -135.559, speed 56.76 f/s
Test done in 7.70 sec, reward -160.834, steps 1124
Test done in 9.97 sec, reward -166.692, steps 1432
385207: done 1237 episodes, mean reward -136.840, speed 40.29 f/s
385370: done 1238 episodes, mean reward -137.023, speed 81.22 f/s
385696: done 1239 episodes, mean reward -136.280, speed 82.06 f/s
Test done in 7.02 sec, reward -184.199, steps 1026
386459: done 1240 episodes, mean reward -136.841, speed 46.59 f/s
Test done in 7.62 sec, reward -166.885, steps 1108
387889: done 1241 episodes, mean reward -137.858, speed 56.69 f/s
Test done in 7.79 sec, reward -152.364, steps 1158
388242: done 1242 episodes, mean reward -137.796, speed 28.52 f/s
388512: done 1243 episodes, mean reward -137.805, speed 80.79 f/s
388854: done 1244 episodes, mean reward -137.641, speed 82.12 f/s
Test done in 6.68 sec, reward -181.337, steps 988
389072: done 1245 episodes, mean reward -137.329, speed 23.07 f/s
389432: done 1246 episodes, mean reward -137.437, speed 82.83 f/s
389992: done 1247 episodes, mean reward -137.877, speed 80.96 f/s
Test done in 3.49 sec, reward -162.128, steps 516
390536: done 1248 episodes, mean reward -138.389, speed 53.10 f/s
390900: done 1249 episodes, mean reward -137.269, speed 80.49 f/s
Test done in 3.17 sec, reward -165.217, steps 484
391895: done 1250 episodes, mean reward -136.889, speed 64.66 f/s
Test done in 3.63 sec, reward -173.708, steps 517
392052: done 1251 episodes, mean reward -136.546, speed 27.44 f/s
392286: done 1252 episodes, mean reward -136.596, speed 77.62 f/s
392794: done 1253 episodes, mean reward -136.874, speed 80.28 f/s
Test done in 6.30 sec, reward -173.109, steps 943
393321: done 1254 episodes, mean reward -137.394, speed 41.56 f/s
393892: done 1255 episodes, mean reward -137.701, speed 80.36 f/s
Test done in 6.52 sec, reward -206.760, steps 995
394099: done 1256 episodes, mean reward -137.940, speed 22.72 f/s
394701: done 1257 episodes, mean reward -137.846, speed 81.87 f/s
Test done in 2.75 sec, reward -164.988, steps 390
395143: done 1258 episodes, mean reward -137.233, speed 53.76 f/s
395657: done 1259 episodes, mean reward -137.292, speed 76.30 f/s
Test done in 2.56 sec, reward -163.242, steps 387
396240: done 1260 episodes, mean reward -137.778, speed 59.61 f/s
396556: done 1261 episodes, mean reward -136.872, speed 82.47 f/s
396732: done 1262 episodes, mean reward -136.275, speed 81.58 f/s
Test done in 2.43 sec, reward -156.529, steps 347
397107: done 1263 episodes, mean reward -136.455, speed 53.79 f/s
397207: done 1264 episodes, mean reward -136.169, speed 80.19 f/s
397570: done 1265 episodes, mean reward -136.261, speed 79.45 f/s
Test done in 3.64 sec, reward -179.726, steps 543
398313: done 1266 episodes, mean reward -136.593, speed 58.10 f/s
398786: done 1267 episodes, mean reward -136.676, speed 82.67 f/s
Test done in 3.58 sec, reward -179.019, steps 534
399869: done 1268 episodes, mean reward -137.360, speed 62.91 f/s
Test done in 6.60 sec, reward -214.484, steps 975
Test done in 9.99 sec, reward -156.868, steps 1470
401469: done 1269 episodes, mean reward -137.284, speed 43.93 f/s
Test done in 10.99 sec, reward -124.960, steps 1600
Test done in 10.97 sec, reward -138.641, steps 1592
403069: done 1270 episodes, mean reward -136.677, speed 37.77 f/s
Test done in 11.14 sec, reward -129.534, steps 1600
404669: done 1271 episodes, mean reward -135.935, speed 51.64 f/s
Test done in 10.66 sec, reward -122.687, steps 1513
405265: done 1272 episodes, mean reward -135.843, speed 33.17 f/s
Test done in 10.00 sec, reward -112.442, steps 1454
406696: done 1273 episodes, mean reward -135.222, speed 51.82 f/s
Test done in 10.02 sec, reward -124.377, steps 1466
407726: done 1274 episodes, mean reward -135.328, speed 45.88 f/s
Test done in 9.12 sec, reward -148.575, steps 1349
408614: done 1275 episodes, mean reward -135.368, speed 43.51 f/s
Test done in 9.31 sec, reward -153.785, steps 1382
Test done in 9.94 sec, reward -112.089, steps 1423
410214: done 1276 episodes, mean reward -134.741, speed 41.34 f/s
410366: done 1277 episodes, mean reward -134.401, speed 80.08 f/s
Test done in 5.43 sec, reward -170.162, steps 825
411737: done 1278 episodes, mean reward -134.858, speed 61.00 f/s
411938: done 1279 episodes, mean reward -134.988, speed 81.70 f/s
Test done in 9.18 sec, reward -148.950, steps 1335
412934: done 1280 episodes, mean reward -135.060, speed 45.85 f/s
Test done in 7.82 sec, reward -162.480, steps 1159
413692: done 1281 episodes, mean reward -135.297, speed 43.91 f/s
Test done in 10.60 sec, reward -99.108, steps 1556
414372: done 1282 episodes, mean reward -135.571, speed 35.94 f/s
414804: done 1283 episodes, mean reward -135.612, speed 80.00 f/s
Test done in 9.25 sec, reward -119.160, steps 1318
Test done in 7.89 sec, reward -140.021, steps 1194
416404: done 1284 episodes, mean reward -134.759, speed 42.89 f/s
Test done in 10.29 sec, reward -59.453, steps 1544
Test done in 10.94 sec, reward -82.068, steps 1592
418004: done 1285 episodes, mean reward -133.573, speed 38.88 f/s
Test done in 10.78 sec, reward -40.892, steps 1600
Best reward updated: -41.998 -> -40.892
419604: done 1286 episodes, mean reward -132.190, speed 52.55 f/s
Test done in 10.73 sec, reward -31.098, steps 1600
Best reward updated: -40.892 -> -31.098
Test done in 10.68 sec, reward -5.407, steps 1600
Best reward updated: -31.098 -> -5.407
421204: done 1287 episodes, mean reward -130.891, speed 38.55 f/s
Test done in 10.66 sec, reward 11.478, steps 1600
Best reward updated: -5.407 -> 11.478
422804: done 1288 episodes, mean reward -129.169, speed 52.48 f/s
Test done in 11.20 sec, reward 17.612, steps 1600
Best reward updated: 11.478 -> 17.612
423505: done 1289 episodes, mean reward -128.879, speed 35.13 f/s
Test done in 11.19 sec, reward 6.688, steps 1600
Test done in 10.53 sec, reward -9.031, steps 1569
425105: done 1290 episodes, mean reward -127.205, speed 38.45 f/s
Test done in 10.36 sec, reward -15.388, steps 1451
426705: done 1291 episodes, mean reward -124.344, speed 53.05 f/s
Test done in 10.80 sec, reward -1.130, steps 1600
Test done in 10.90 sec, reward 7.901, steps 1600
428305: done 1292 episodes, mean reward -122.957, speed 38.28 f/s
Test done in 10.84 sec, reward 37.222, steps 1600
Best reward updated: 17.612 -> 37.222
429905: done 1293 episodes, mean reward -120.965, speed 51.73 f/s
Test done in 9.43 sec, reward -44.744, steps 1413
Test done in 10.80 sec, reward 35.422, steps 1600
431505: done 1294 episodes, mean reward -119.493, speed 40.09 f/s
Test done in 10.93 sec, reward 21.930, steps 1594
432937: done 1295 episodes, mean reward -118.938, speed 49.86 f/s
Test done in 10.56 sec, reward 64.385, steps 1600
Best reward updated: 37.222 -> 64.385
433414: done 1296 episodes, mean reward -118.784, speed 28.97 f/s
Test done in 6.29 sec, reward -93.608, steps 919
434312: done 1297 episodes, mean reward -118.618, speed 51.65 f/s
Test done in 9.88 sec, reward 31.304, steps 1488
435912: done 1298 episodes, mean reward -117.025, speed 53.41 f/s
Test done in 10.11 sec, reward 48.989, steps 1511
Test done in 10.27 sec, reward 70.255, steps 1506
Best reward updated: 64.385 -> 70.255
437512: done 1299 episodes, mean reward -115.164, speed 39.69 f/s
Test done in 10.78 sec, reward 89.900, steps 1600
Best reward updated: 70.255 -> 89.900
Test done in 8.93 sec, reward 12.681, steps 1304
439112: done 1300 episodes, mean reward -113.637, speed 40.34 f/s
Test done in 10.72 sec, reward 84.460, steps 1600
440712: done 1301 episodes, mean reward -111.812, speed 51.85 f/s
440972: done 1302 episodes, mean reward -112.302, speed 81.14 f/s
Test done in 8.06 sec, reward -44.843, steps 1200
Test done in 10.90 sec, reward 84.435, steps 1600
442572: done 1303 episodes, mean reward -111.404, speed 40.93 f/s
Test done in 10.60 sec, reward 50.503, steps 1556
443082: done 1304 episodes, mean reward -111.835, speed 29.98 f/s
Test done in 6.71 sec, reward -42.424, steps 966
444682: done 1305 episodes, mean reward -109.351, speed 60.13 f/s
Test done in 8.83 sec, reward -15.020, steps 1280
Test done in 3.66 sec, reward -92.202, steps 522
446282: done 1306 episodes, mean reward -107.835, speed 48.85 f/s
Test done in 3.06 sec, reward -102.868, steps 455
447882: done 1307 episodes, mean reward -106.232, speed 70.14 f/s
Test done in 2.59 sec, reward -108.215, steps 392
448148: done 1308 episodes, mean reward -106.678, speed 45.47 f/s
448356: done 1309 episodes, mean reward -106.666, speed 79.56 f/s
448544: done 1310 episodes, mean reward -107.330, speed 80.64 f/s
Test done in 2.42 sec, reward -116.372, steps 374
449502: done 1311 episodes, mean reward -108.140, speed 65.18 f/s
Test done in 1.21 sec, reward -121.695, steps 174
450880: done 1312 episodes, mean reward -108.819, speed 75.75 f/s
Test done in 4.86 sec, reward -135.781, steps 728
Test done in 1.85 sec, reward -127.098, steps 266
452480: done 1313 episodes, mean reward -107.921, speed 60.43 f/s
Test done in 8.64 sec, reward -65.954, steps 1165
Test done in 3.70 sec, reward -115.567, steps 536
454080: done 1314 episodes, mean reward -107.905, speed 49.38 f/s
454674: done 1315 episodes, mean reward -108.600, speed 81.18 f/s
Test done in 10.06 sec, reward -95.853, steps 1462
Test done in 7.32 sec, reward -112.682, steps 1108
456274: done 1316 episodes, mean reward -107.912, speed 42.28 f/s
456664: done 1317 episodes, mean reward -108.269, speed 79.90 f/s
456813: done 1318 episodes, mean reward -108.494, speed 79.64 f/s
Test done in 6.40 sec, reward -100.940, steps 902
457035: done 1319 episodes, mean reward -108.952, speed 24.26 f/s
457265: done 1320 episodes, mean reward -109.440, speed 81.59 f/s
Test done in 6.49 sec, reward -138.521, steps 963
458865: done 1321 episodes, mean reward -109.166, speed 58.73 f/s
Test done in 6.03 sec, reward -137.817, steps 892
459123: done 1322 episodes, mean reward -110.075, speed 27.60 f/s
459394: done 1323 episodes, mean reward -110.514, speed 75.33 f/s
459814: done 1324 episodes, mean reward -110.434, speed 78.86 f/s
Test done in 4.41 sec, reward -84.577, steps 656
460156: done 1325 episodes, mean reward -109.945, speed 39.19 f/s
460339: done 1326 episodes, mean reward -110.050, speed 80.90 f/s
460440: done 1327 episodes, mean reward -109.916, speed 80.30 f/s
460613: done 1328 episodes, mean reward -109.419, speed 80.58 f/s
460735: done 1329 episodes, mean reward -109.671, speed 81.73 f/s
Test done in 2.29 sec, reward -141.590, steps 330
461076: done 1330 episodes, mean reward -109.964, speed 51.64 f/s
461213: done 1331 episodes, mean reward -110.065, speed 79.81 f/s
461323: done 1332 episodes, mean reward -110.218, speed 78.68 f/s
461566: done 1333 episodes, mean reward -109.105, speed 74.10 f/s
461941: done 1334 episodes, mean reward -108.854, speed 77.59 f/s
Test done in 3.90 sec, reward -85.063, steps 560
462071: done 1335 episodes, mean reward -108.216, speed 23.57 f/s
462341: done 1336 episodes, mean reward -108.317, speed 78.98 f/s
462535: done 1337 episodes, mean reward -107.291, speed 79.67 f/s
Test done in 2.07 sec, reward -115.530, steps 292
463691: done 1338 episodes, mean reward -107.224, speed 68.91 f/s
Test done in 4.56 sec, reward -118.201, steps 686
464874: done 1339 episodes, mean reward -107.658, speed 61.60 f/s
Test done in 5.76 sec, reward -144.705, steps 812
465492: done 1340 episodes, mean reward -107.306, speed 45.22 f/s
Test done in 6.97 sec, reward -51.402, steps 1043
Test done in 6.66 sec, reward -90.599, steps 1003
467092: done 1341 episodes, mean reward -105.422, speed 47.43 f/s
Test done in 4.52 sec, reward -141.154, steps 664
468237: done 1342 episodes, mean reward -105.343, speed 60.67 f/s
468321: done 1343 episodes, mean reward -105.154, speed 80.55 f/s
468765: done 1344 episodes, mean reward -105.184, speed 75.40 f/s
Test done in 10.24 sec, reward -105.529, steps 1552
Test done in 7.81 sec, reward -141.337, steps 1137
470365: done 1345 episodes, mean reward -104.628, speed 42.17 f/s
Test done in 6.09 sec, reward -169.388, steps 885
471965: done 1346 episodes, mean reward -103.886, speed 60.38 f/s
Test done in 8.16 sec, reward -130.373, steps 1225
Test done in 8.07 sec, reward -169.977, steps 1220
473106: done 1347 episodes, mean reward -103.997, speed 37.24 f/s
473954: done 1348 episodes, mean reward -103.935, speed 79.72 f/s
Test done in 8.94 sec, reward -118.286, steps 1315
Test done in 6.74 sec, reward -161.136, steps 978
475554: done 1349 episodes, mean reward -103.730, speed 44.51 f/s
Test done in 11.05 sec, reward -94.522, steps 1600
Test done in 9.86 sec, reward -137.506, steps 1468
477154: done 1350 episodes, mean reward -102.389, speed 39.08 f/s
477548: done 1351 episodes, mean reward -102.376, speed 74.88 f/s
Test done in 10.73 sec, reward -124.018, steps 1561
Test done in 10.15 sec, reward -139.465, steps 1527
479148: done 1352 episodes, mean reward -101.934, speed 39.38 f/s
Test done in 11.02 sec, reward -105.111, steps 1600
480635: done 1353 episodes, mean reward -102.284, speed 50.09 f/s
Test done in 11.07 sec, reward -142.791, steps 1600
481576: done 1354 episodes, mean reward -102.071, speed 41.08 f/s
481856: done 1355 episodes, mean reward -101.790, speed 79.34 f/s
Test done in 10.54 sec, reward -122.172, steps 1580
Test done in 10.93 sec, reward -132.820, steps 1600
483456: done 1356 episodes, mean reward -101.324, speed 38.41 f/s
Test done in 10.62 sec, reward -138.867, steps 1600
Test done in 10.79 sec, reward -124.135, steps 1600
485056: done 1357 episodes, mean reward -100.314, speed 38.45 f/s
485302: done 1358 episodes, mean reward -99.745, speed 74.43 f/s
Test done in 10.94 sec, reward -114.673, steps 1600
486902: done 1359 episodes, mean reward -98.771, speed 51.31 f/s
Test done in 9.45 sec, reward -134.688, steps 1374
Test done in 11.04 sec, reward -137.067, steps 1538
488502: done 1360 episodes, mean reward -98.024, speed 39.42 f/s
Test done in 9.15 sec, reward -98.764, steps 1295
489072: done 1361 episodes, mean reward -97.963, speed 34.47 f/s
489637: done 1362 episodes, mean reward -97.846, speed 76.45 f/s
Test done in 6.82 sec, reward -143.932, steps 928
490293: done 1363 episodes, mean reward -97.554, speed 43.19 f/s
Test done in 4.36 sec, reward -153.526, steps 594
491893: done 1364 episodes, mean reward -96.900, speed 65.35 f/s
Test done in 5.33 sec, reward -146.755, steps 782
492528: done 1365 episodes, mean reward -96.365, speed 47.73 f/s
Test done in 6.79 sec, reward -165.781, steps 991
493035: done 1366 episodes, mean reward -95.566, speed 38.23 f/s
Test done in 5.70 sec, reward -112.289, steps 852
494635: done 1367 episodes, mean reward -94.314, speed 61.92 f/s
Test done in 2.17 sec, reward -109.943, steps 317
495076: done 1368 episodes, mean reward -93.257, speed 56.80 f/s
495670: done 1369 episodes, mean reward -92.994, speed 81.00 f/s
495811: done 1370 episodes, mean reward -93.264, speed 81.03 f/s
Test done in 11.55 sec, reward -3.588, steps 1600
Test done in 9.26 sec, reward -66.805, steps 1375
497045: done 1371 episodes, mean reward -93.751, speed 34.05 f/s
497155: done 1372 episodes, mean reward -93.367, speed 81.40 f/s
497975: done 1373 episodes, mean reward -92.451, speed 79.52 f/s
Test done in 10.40 sec, reward -24.257, steps 1489
498843: done 1374 episodes, mean reward -92.025, speed 40.70 f/s
Test done in 6.92 sec, reward -64.780, steps 1022
499237: done 1375 episodes, mean reward -91.642, speed 33.33 f/s
Test done in 4.16 sec, reward -95.234, steps 621
500837: done 1376 episodes, mean reward -90.604, speed 64.84 f/s
Test done in 9.92 sec, reward 50.723, steps 1440
501198: done 1377 episodes, mean reward -90.501, speed 25.01 f/s
Test done in 3.31 sec, reward -72.164, steps 473
502096: done 1378 episodes, mean reward -89.297, speed 61.48 f/s
502197: done 1379 episodes, mean reward -88.974, speed 78.32 f/s
502544: done 1380 episodes, mean reward -88.433, speed 81.42 f/s
Test done in 10.19 sec, reward 23.346, steps 1502
503858: done 1381 episodes, mean reward -87.090, speed 48.94 f/s
503987: done 1382 episodes, mean reward -86.560, speed 80.85 f/s
Test done in 5.39 sec, reward -54.595, steps 784
Test done in 8.75 sec, reward 39.358, steps 1272
505005: done 1383 episodes, mean reward -85.901, speed 37.99 f/s
Test done in 2.33 sec, reward -93.272, steps 339
506605: done 1384 episodes, mean reward -84.539, speed 70.31 f/s
506703: done 1385 episodes, mean reward -85.423, speed 78.78 f/s
Test done in 6.90 sec, reward -36.848, steps 1006
507639: done 1386 episodes, mean reward -86.436, speed 50.11 f/s
507858: done 1387 episodes, mean reward -87.510, speed 81.12 f/s
507958: done 1388 episodes, mean reward -88.968, speed 80.59 f/s
Test done in 1.94 sec, reward -105.540, steps 291
508439: done 1389 episodes, mean reward -88.728, speed 59.97 f/s
Test done in 4.60 sec, reward -82.075, steps 631
509916: done 1390 episodes, mean reward -89.083, speed 63.34 f/s
Test done in 3.26 sec, reward -85.736, steps 491
510031: done 1391 episodes, mean reward -90.565, speed 24.56 f/s
510294: done 1392 episodes, mean reward -91.512, speed 79.06 f/s
510542: done 1393 episodes, mean reward -92.605, speed 79.03 f/s
510812: done 1394 episodes, mean reward -93.978, speed 79.73 f/s
510924: done 1395 episodes, mean reward -94.245, speed 78.55 f/s
Test done in 5.87 sec, reward -58.106, steps 858
511126: done 1396 episodes, mean reward -94.396, speed 24.00 f/s
511811: done 1397 episodes, mean reward -94.331, speed 81.00 f/s
511914: done 1398 episodes, mean reward -95.839, speed 79.61 f/s
Test done in 7.60 sec, reward -21.189, steps 1143
512136: done 1399 episodes, mean reward -97.612, speed 21.45 f/s
512412: done 1400 episodes, mean reward -99.246, speed 73.24 f/s
512529: done 1401 episodes, mean reward -101.174, speed 76.18 f/s
Test done in 2.56 sec, reward -99.735, steps 377
513194: done 1402 episodes, mean reward -100.611, speed 61.44 f/s
513373: done 1403 episodes, mean reward -101.855, speed 80.35 f/s
Test done in 5.60 sec, reward -57.338, steps 789
514397: done 1404 episodes, mean reward -101.434, speed 55.08 f/s
514770: done 1405 episodes, mean reward -103.033, speed 79.90 f/s
Test done in 10.26 sec, reward 50.429, steps 1473
Test done in 7.44 sec, reward -24.109, steps 1065
516004: done 1406 episodes, mean reward -104.417, speed 37.07 f/s
516115: done 1407 episodes, mean reward -106.043, speed 79.23 f/s
516856: done 1408 episodes, mean reward -105.722, speed 80.28 f/s
Test done in 6.70 sec, reward -61.076, steps 965
517136: done 1409 episodes, mean reward -105.600, speed 27.64 f/s
517650: done 1410 episodes, mean reward -105.447, speed 79.22 f/s
Test done in 6.56 sec, reward -83.208, steps 969
518106: done 1411 episodes, mean reward -105.269, speed 37.27 f/s
518562: done 1412 episodes, mean reward -104.982, speed 81.04 f/s
Test done in 7.40 sec, reward -40.485, steps 1040
Test done in 7.24 sec, reward -47.704, steps 1080
520162: done 1413 episodes, mean reward -104.965, speed 45.94 f/s
520815: done 1414 episodes, mean reward -105.148, speed 79.53 f/s
Test done in 9.44 sec, reward -15.551, steps 1377
521962: done 1415 episodes, mean reward -104.443, speed 48.87 f/s
Test done in 8.82 sec, reward -50.464, steps 1270
Test done in 10.19 sec, reward -18.636, steps 1501
523363: done 1416 episodes, mean reward -105.147, speed 38.25 f/s
Test done in 10.55 sec, reward -30.982, steps 1551
524732: done 1417 episodes, mean reward -104.829, speed 49.58 f/s
Test done in 10.00 sec, reward 33.729, steps 1495
Test done in 10.56 sec, reward 22.850, steps 1562
526245: done 1418 episodes, mean reward -104.430, speed 38.16 f/s
Test done in 10.78 sec, reward 29.353, steps 1580
527014: done 1419 episodes, mean reward -104.139, speed 37.63 f/s
Test done in 10.83 sec, reward 25.127, steps 1600
528614: done 1420 episodes, mean reward -102.819, speed 51.23 f/s
Test done in 11.07 sec, reward 15.139, steps 1600
Test done in 11.16 sec, reward 45.109, steps 1600
530214: done 1421 episodes, mean reward -102.071, speed 37.89 f/s
Test done in 10.99 sec, reward 15.606, steps 1600
531814: done 1422 episodes, mean reward -100.613, speed 51.86 f/s
Test done in 10.85 sec, reward 34.258, steps 1600
Test done in 10.26 sec, reward -16.880, steps 1498
533414: done 1423 episodes, mean reward -99.283, speed 38.57 f/s
Test done in 10.83 sec, reward 28.315, steps 1600
534308: done 1424 episodes, mean reward -98.794, speed 40.43 f/s
Test done in 10.71 sec, reward 42.831, steps 1600
535908: done 1425 episodes, mean reward -97.073, speed 50.93 f/s
Test done in 10.96 sec, reward -55.595, steps 1600
Test done in 9.72 sec, reward 13.380, steps 1458
537508: done 1426 episodes, mean reward -95.922, speed 39.33 f/s
Test done in 11.52 sec, reward 35.830, steps 1590
Test done in 10.91 sec, reward 7.076, steps 1600
539108: done 1427 episodes, mean reward -94.473, speed 37.97 f/s
Test done in 10.59 sec, reward -18.654, steps 1600
540585: done 1428 episodes, mean reward -94.549, speed 50.99 f/s
Test done in 10.69 sec, reward -51.675, steps 1600
541903: done 1429 episodes, mean reward -94.692, speed 48.48 f/s
Test done in 10.46 sec, reward -14.829, steps 1536
542389: done 1430 episodes, mean reward -94.467, speed 29.18 f/s
Test done in 10.53 sec, reward -21.078, steps 1523
543989: done 1431 episodes, mean reward -93.206, speed 51.88 f/s
Test done in 10.71 sec, reward -0.995, steps 1600
544411: done 1432 episodes, mean reward -93.138, speed 26.30 f/s
Test done in 10.89 sec, reward 9.286, steps 1600
Test done in 11.29 sec, reward 46.985, steps 1600
546011: done 1433 episodes, mean reward -91.534, speed 37.92 f/s
Test done in 10.78 sec, reward 24.233, steps 1600
547611: done 1434 episodes, mean reward -89.771, speed 52.35 f/s
Test done in 9.67 sec, reward 33.028, steps 1455
Test done in 9.89 sec, reward 102.568, steps 1463
Best reward updated: 89.900 -> 102.568
549211: done 1435 episodes, mean reward -87.745, speed 40.45 f/s
Test done in 10.92 sec, reward 79.187, steps 1600
550811: done 1436 episodes, mean reward -85.572, speed 51.20 f/s
Test done in 10.16 sec, reward 45.222, steps 1485
Test done in 9.39 sec, reward 69.427, steps 1365
552411: done 1437 episodes, mean reward -83.485, speed 40.49 f/s
Test done in 4.02 sec, reward -42.981, steps 580
Test done in 8.86 sec, reward 112.518, steps 1254
Best reward updated: 102.568 -> 112.518
554011: done 1438 episodes, mean reward -80.851, speed 48.61 f/s
Test done in 9.28 sec, reward 122.381, steps 1389
Best reward updated: 112.518 -> 122.381
555611: done 1439 episodes, mean reward -77.838, speed 54.71 f/s
Test done in 8.28 sec, reward 91.177, steps 1222
Test done in 7.67 sec, reward 92.456, steps 1116
557146: done 1440 episodes, mean reward -76.266, speed 43.41 f/s
Test done in 10.16 sec, reward 138.999, steps 1518
Best reward updated: 122.381 -> 138.999
558746: done 1441 episodes, mean reward -74.515, speed 52.93 f/s
558866: done 1442 episodes, mean reward -74.638, speed 70.39 f/s
Test done in 10.78 sec, reward 196.941, steps 1532
Best reward updated: 138.999 -> 196.941
Test done in 9.38 sec, reward 167.867, steps 1383
560466: done 1443 episodes, mean reward -72.095, speed 39.68 f/s
Test done in 9.54 sec, reward 167.274, steps 1375
Test done in 10.08 sec, reward 166.821, steps 1477
562066: done 1444 episodes, mean reward -69.343, speed 39.87 f/s
Test done in 9.66 sec, reward 164.710, steps 1434
563532: done 1445 episodes, mean reward -68.386, speed 52.43 f/s
Test done in 9.03 sec, reward 143.498, steps 1305
Test done in 9.32 sec, reward 185.435, steps 1346
565132: done 1446 episodes, mean reward -66.071, speed 41.59 f/s
Test done in 8.72 sec, reward 131.655, steps 1294
566732: done 1447 episodes, mean reward -62.871, speed 55.96 f/s
Test done in 10.63 sec, reward 214.675, steps 1489
Best reward updated: 196.941 -> 214.675
Test done in 10.04 sec, reward 224.056, steps 1472
Best reward updated: 214.675 -> 224.056
568332: done 1448 episodes, mean reward -59.802, speed 39.10 f/s
568883: done 1449 episodes, mean reward -59.197, speed 80.10 f/s
Test done in 10.21 sec, reward 236.977, steps 1466
Best reward updated: 224.056 -> 236.977
569024: done 1450 episodes, mean reward -59.518, speed 11.79 f/s
Test done in 10.44 sec, reward 234.995, steps 1473
570624: done 1451 episodes, mean reward -56.713, speed 52.03 f/s
Test done in 10.63 sec, reward 213.058, steps 1597
Test done in 10.53 sec, reward 172.567, steps 1536
572134: done 1452 episodes, mean reward -55.772, speed 38.11 f/s
Test done in 9.98 sec, reward 237.793, steps 1450
Best reward updated: 236.977 -> 237.793
573734: done 1453 episodes, mean reward -52.534, speed 53.60 f/s
Test done in 10.41 sec, reward 137.350, steps 1566
Test done in 7.62 sec, reward 13.598, steps 1142
575334: done 1454 episodes, mean reward -50.142, speed 41.81 f/s
Test done in 6.85 sec, reward -12.609, steps 1036
576693: done 1455 episodes, mean reward -49.688, speed 56.64 f/s
Test done in 5.63 sec, reward -74.652, steps 856
577857: done 1456 episodes, mean reward -49.593, speed 58.07 f/s
Test done in 8.09 sec, reward -93.653, steps 1202
Test done in 7.74 sec, reward -74.007, steps 1160
579457: done 1457 episodes, mean reward -48.616, speed 44.48 f/s
Test done in 7.69 sec, reward -99.272, steps 1147
580171: done 1458 episodes, mean reward -48.756, speed 43.38 f/s
580463: done 1460 episodes, mean reward -49.562, speed 80.31 f/s
Test done in 2.71 sec, reward -130.455, steps 384
581479: done 1461 episodes, mean reward -49.697, speed 65.52 f/s
Test done in 3.38 sec, reward -126.747, steps 474
582226: done 1462 episodes, mean reward -49.563, speed 58.02 f/s
Test done in 3.83 sec, reward -129.248, steps 558
583283: done 1463 episodes, mean reward -49.923, speed 63.24 f/s
583648: done 1464 episodes, mean reward -50.513, speed 81.06 f/s
Test done in 2.45 sec, reward -125.524, steps 324
584218: done 1465 episodes, mean reward -50.682, speed 58.73 f/s
584490: done 1466 episodes, mean reward -50.897, speed 76.65 f/s
584801: done 1467 episodes, mean reward -51.662, speed 76.34 f/s
584893: done 1468 episodes, mean reward -51.627, speed 81.43 f/s
584976: done 1469 episodes, mean reward -51.651, speed 78.97 f/s
Test done in 2.20 sec, reward -125.953, steps 311
585078: done 1470 episodes, mean reward -51.645, speed 29.30 f/s
585722: done 1471 episodes, mean reward -51.710, speed 80.83 f/s
Test done in 3.49 sec, reward -130.278, steps 481
586664: done 1472 episodes, mean reward -51.753, speed 61.19 f/s
586904: done 1473 episodes, mean reward -52.051, speed 81.63 f/s
Test done in 4.12 sec, reward -114.528, steps 617
587124: done 1474 episodes, mean reward -52.042, speed 32.16 f/s
587311: done 1475 episodes, mean reward -52.017, speed 81.04 f/s
Test done in 5.27 sec, reward -112.665, steps 770
588574: done 1476 episodes, mean reward -53.600, speed 60.50 f/s
Test done in 8.03 sec, reward -88.305, steps 1153
589115: done 1477 episodes, mean reward -53.806, speed 36.78 f/s
589262: done 1478 episodes, mean reward -54.239, speed 76.27 f/s
589434: done 1479 episodes, mean reward -54.309, speed 72.74 f/s
589813: done 1480 episodes, mean reward -54.500, speed 79.90 f/s
Test done in 3.48 sec, reward -120.670, steps 496
590129: done 1481 episodes, mean reward -55.336, speed 42.83 f/s
590390: done 1482 episodes, mean reward -55.543, speed 79.95 f/s
Test done in 5.57 sec, reward -117.821, steps 846
591990: done 1483 episodes, mean reward -55.150, speed 63.31 f/s
Test done in 6.55 sec, reward -126.858, steps 1007
592649: done 1484 episodes, mean reward -57.185, speed 44.57 f/s
Test done in 7.65 sec, reward -148.626, steps 1144
593831: done 1485 episodes, mean reward -57.811, speed 52.71 f/s
Test done in 9.57 sec, reward -111.612, steps 1434
Test done in 10.53 sec, reward -97.466, steps 1600
595431: done 1486 episodes, mean reward -57.196, speed 39.69 f/s
Test done in 9.66 sec, reward -91.625, steps 1457
Test done in 10.25 sec, reward -61.699, steps 1508
597031: done 1487 episodes, mean reward -56.467, speed 40.46 f/s
Test done in 8.89 sec, reward -111.416, steps 1338
598631: done 1488 episodes, mean reward -55.651, speed 55.39 f/s
598878: done 1489 episodes, mean reward -55.754, speed 81.98 f/s
Test done in 7.80 sec, reward -85.606, steps 1211
599988: done 1490 episodes, mean reward -57.112, speed 52.08 f/s
Test done in 6.25 sec, reward -86.182, steps 898
600169: done 1491 episodes, mean reward -57.161, speed 21.06 f/s
Test done in 1.91 sec, reward -119.629, steps 287
601450: done 1492 episodes, mean reward -56.838, speed 71.76 f/s
Test done in 7.05 sec, reward -73.204, steps 1044
602105: done 1493 episodes, mean reward -56.752, speed 42.96 f/s
Test done in 3.19 sec, reward -105.600, steps 472
603476: done 1494 episodes, mean reward -56.598, speed 68.27 f/s
Test done in 0.88 sec, reward -121.041, steps 122
604041: done 1495 episodes, mean reward -56.519, speed 69.05 f/s
604455: done 1496 episodes, mean reward -56.476, speed 80.13 f/s
604684: done 1497 episodes, mean reward -56.703, speed 80.28 f/s
604845: done 1498 episodes, mean reward -56.837, speed 81.24 f/s
604940: done 1499 episodes, mean reward -57.001, speed 80.35 f/s
Test done in 4.10 sec, reward -87.933, steps 592
605146: done 1500 episodes, mean reward -57.037, speed 30.36 f/s
605927: done 1501 episodes, mean reward -56.895, speed 81.44 f/s
Test done in 2.49 sec, reward -106.987, steps 387
606028: done 1502 episodes, mean reward -57.360, speed 26.78 f/s
606162: done 1503 episodes, mean reward -57.489, speed 82.93 f/s
606664: done 1504 episodes, mean reward -57.822, speed 80.55 f/s
Test done in 2.99 sec, reward -102.871, steps 444
607029: done 1505 episodes, mean reward -57.907, speed 49.15 f/s
607432: done 1506 episodes, mean reward -58.116, speed 80.75 f/s
607864: done 1507 episodes, mean reward -57.958, speed 81.40 f/s
607958: done 1508 episodes, mean reward -58.268, speed 70.43 f/s
Test done in 3.42 sec, reward -100.679, steps 511
608593: done 1509 episodes, mean reward -57.997, speed 55.68 f/s
Test done in 3.44 sec, reward -96.541, steps 490
609033: done 1510 episodes, mean reward -57.758, speed 49.75 f/s
609378: done 1511 episodes, mean reward -57.915, speed 81.16 f/s
609570: done 1512 episodes, mean reward -58.028, speed 82.47 f/s
Test done in 5.65 sec, reward -79.177, steps 871
610612: done 1513 episodes, mean reward -59.422, speed 56.94 f/s
Test done in 3.85 sec, reward -73.188, steps 570
611100: done 1514 episodes, mean reward -59.251, speed 49.59 f/s
611719: done 1515 episodes, mean reward -59.206, speed 77.73 f/s
Test done in 7.93 sec, reward -55.097, steps 1208
612438: done 1516 episodes, mean reward -58.931, speed 42.41 f/s
612639: done 1517 episodes, mean reward -58.817, speed 78.40 f/s
612844: done 1518 episodes, mean reward -59.029, speed 81.87 f/s
Test done in 5.23 sec, reward -41.155, steps 789
613035: done 1519 episodes, mean reward -59.073, speed 25.15 f/s
613525: done 1520 episodes, mean reward -59.940, speed 82.03 f/s
Test done in 6.68 sec, reward -17.903, steps 1009
614145: done 1521 episodes, mean reward -61.016, speed 43.26 f/s
Test done in 6.98 sec, reward -24.283, steps 1023
615232: done 1522 episodes, mean reward -61.714, speed 52.64 f/s
615816: done 1523 episodes, mean reward -62.614, speed 78.29 f/s
615977: done 1524 episodes, mean reward -62.846, speed 79.44 f/s
Test done in 5.26 sec, reward -27.883, steps 808
616264: done 1525 episodes, mean reward -64.049, speed 32.27 f/s
616522: done 1526 episodes, mean reward -64.908, speed 79.65 f/s
Test done in 5.29 sec, reward -14.875, steps 814
617657: done 1527 episodes, mean reward -65.577, speed 58.86 f/s
617898: done 1528 episodes, mean reward -65.216, speed 79.84 f/s
Test done in 5.12 sec, reward -28.243, steps 753
618055: done 1529 episodes, mean reward -64.908, speed 22.27 f/s
618695: done 1530 episodes, mean reward -64.382, speed 77.79 f/s
Test done in 4.68 sec, reward -47.472, steps 691
619362: done 1531 episodes, mean reward -64.900, speed 51.41 f/s
619501: done 1532 episodes, mean reward -64.762, speed 79.42 f/s
619969: done 1533 episodes, mean reward -65.792, speed 79.58 f/s
Test done in 5.32 sec, reward -30.344, steps 798
620138: done 1534 episodes, mean reward -67.195, speed 22.87 f/s
620598: done 1535 episodes, mean reward -69.003, speed 79.91 f/s
620880: done 1536 episodes, mean reward -70.854, speed 79.50 f/s
Test done in 4.97 sec, reward -48.460, steps 725
Test done in 8.03 sec, reward 36.049, steps 1186
622480: done 1537 episodes, mean reward -70.517, speed 48.21 f/s
Test done in 8.81 sec, reward 9.878, steps 1284
623449: done 1538 episodes, mean reward -72.356, speed 46.86 f/s
Test done in 9.15 sec, reward 84.464, steps 1341
624049: done 1539 episodes, mean reward -74.478, speed 36.11 f/s
Test done in 6.05 sec, reward -40.825, steps 901
625321: done 1540 episodes, mean reward -74.727, speed 57.14 f/s
Test done in 3.35 sec, reward -91.166, steps 499
626325: done 1541 episodes, mean reward -76.437, speed 63.91 f/s
626441: done 1542 episodes, mean reward -76.195, speed 79.73 f/s
626612: done 1543 episodes, mean reward -78.922, speed 81.31 f/s
Test done in 7.62 sec, reward -2.861, steps 1129
627015: done 1544 episodes, mean reward -81.305, speed 31.91 f/s
627470: done 1545 episodes, mean reward -82.288, speed 83.61 f/s
627566: done 1546 episodes, mean reward -84.897, speed 79.52 f/s
Test done in 6.99 sec, reward 17.882, steps 966
Test done in 8.17 sec, reward -74.216, steps 1188
629166: done 1547 episodes, mean reward -87.837, speed 45.70 f/s
629517: done 1548 episodes, mean reward -90.298, speed 81.82 f/s
629707: done 1549 episodes, mean reward -91.259, speed 81.30 f/s
Test done in 10.24 sec, reward -94.119, steps 1439
Test done in 8.85 sec, reward 57.207, steps 1238
631307: done 1550 episodes, mean reward -90.886, speed 41.13 f/s
Test done in 11.70 sec, reward -115.769, steps 1600
632907: done 1551 episodes, mean reward -91.498, speed 51.20 f/s
Test done in 6.05 sec, reward -44.508, steps 904
633073: done 1552 episodes, mean reward -92.837, speed 20.40 f/s
633377: done 1553 episodes, mean reward -95.191, speed 82.57 f/s
Test done in 9.47 sec, reward 73.921, steps 1358
634778: done 1554 episodes, mean reward -96.129, speed 52.65 f/s
634864: done 1555 episodes, mean reward -96.282, speed 78.45 f/s
Test done in 10.86 sec, reward 99.926, steps 1597
635285: done 1556 episodes, mean reward -96.214, speed 26.07 f/s
635519: done 1557 episodes, mean reward -97.431, speed 81.22 f/s
Test done in 10.61 sec, reward 98.273, steps 1468
636004: done 1558 episodes, mean reward -96.924, speed 29.21 f/s
636453: done 1559 episodes, mean reward -96.396, speed 79.79 f/s
636592: done 1561 episodes, mean reward -95.616, speed 78.17 f/s
636679: done 1562 episodes, mean reward -95.563, speed 75.89 f/s
Test done in 9.57 sec, reward 63.969, steps 1406
637344: done 1563 episodes, mean reward -94.542, speed 36.39 f/s
637815: done 1564 episodes, mean reward -94.117, speed 82.17 f/s
637973: done 1565 episodes, mean reward -93.861, speed 79.51 f/s
Test done in 7.75 sec, reward 28.975, steps 1157
Test done in 8.41 sec, reward 26.095, steps 1243
639573: done 1566 episodes, mean reward -91.396, speed 44.87 f/s
639856: done 1567 episodes, mean reward -91.130, speed 79.11 f/s
Test done in 7.24 sec, reward -19.364, steps 1055
640232: done 1568 episodes, mean reward -90.835, speed 31.66 f/s
640445: done 1569 episodes, mean reward -90.835, speed 81.09 f/s
640920: done 1570 episodes, mean reward -90.450, speed 81.95 f/s
Test done in 8.95 sec, reward 58.248, steps 1333
641603: done 1571 episodes, mean reward -89.644, speed 39.36 f/s
641998: done 1572 episodes, mean reward -89.372, speed 81.86 f/s
Test done in 4.44 sec, reward -46.949, steps 638
642081: done 1573 episodes, mean reward -89.243, speed 15.19 f/s
642196: done 1574 episodes, mean reward -88.930, speed 82.53 f/s
Test done in 8.36 sec, reward 18.948, steps 1227
643032: done 1575 episodes, mean reward -88.229, speed 44.25 f/s
643331: done 1576 episodes, mean reward -87.671, speed 81.12 f/s
643552: done 1578 episodes, mean reward -87.350, speed 81.70 f/s
Test done in 10.05 sec, reward 67.791, steps 1541
644541: done 1579 episodes, mean reward -86.391, speed 44.13 f/s
644781: done 1580 episodes, mean reward -85.995, speed 81.47 f/s
644879: done 1581 episodes, mean reward -86.032, speed 80.31 f/s
Test done in 9.77 sec, reward 88.979, steps 1478
645943: done 1582 episodes, mean reward -85.012, speed 46.89 f/s
Test done in 9.60 sec, reward 68.921, steps 1467
646274: done 1583 episodes, mean reward -85.612, speed 24.17 f/s
Test done in 10.97 sec, reward 61.364, steps 1600
647874: done 1584 episodes, mean reward -83.290, speed 51.96 f/s
Test done in 10.78 sec, reward 79.260, steps 1600
648247: done 1585 episodes, mean reward -82.508, speed 24.18 f/s
648393: done 1586 episodes, mean reward -83.086, speed 79.86 f/s
Test done in 9.56 sec, reward 48.598, steps 1451
649470: done 1587 episodes, mean reward -83.068, speed 47.21 f/s
649667: done 1588 episodes, mean reward -83.680, speed 81.21 f/s
Test done in 10.14 sec, reward 8.164, steps 1449
650015: done 1589 episodes, mean reward -83.620, speed 23.71 f/s
650453: done 1590 episodes, mean reward -82.980, speed 79.98 f/s
650705: done 1591 episodes, mean reward -82.803, speed 81.14 f/s
Test done in 10.51 sec, reward 65.418, steps 1572
Test done in 10.61 sec, reward 65.050, steps 1570
652305: done 1592 episodes, mean reward -81.113, speed 39.20 f/s
Test done in 8.30 sec, reward 8.330, steps 1237
653154: done 1593 episodes, mean reward -80.668, speed 44.15 f/s
Test done in 2.47 sec, reward -85.064, steps 362
654112: done 1594 episodes, mean reward -80.385, speed 66.89 f/s
654544: done 1595 episodes, mean reward -80.310, speed 82.20 f/s
Test done in 10.98 sec, reward 69.932, steps 1600
Test done in 9.62 sec, reward 91.370, steps 1453
656144: done 1596 episodes, mean reward -78.022, speed 39.52 f/s
Test done in 10.26 sec, reward 132.791, steps 1515
657371: done 1597 episodes, mean reward -76.894, speed 48.45 f/s
Test done in 9.65 sec, reward 105.168, steps 1455
658217: done 1598 episodes, mean reward -76.179, speed 41.83 f/s
Test done in 10.59 sec, reward 125.604, steps 1600
659465: done 1599 episodes, mean reward -74.787, speed 47.67 f/s
Test done in 9.53 sec, reward 134.765, steps 1453
660748: done 1600 episodes, mean reward -73.565, speed 50.36 f/s
Test done in 11.12 sec, reward 153.016, steps 1600
Test done in 7.29 sec, reward 40.662, steps 1070
662038: done 1601 episodes, mean reward -72.463, speed 37.39 f/s
Test done in 9.70 sec, reward 95.733, steps 1462
663634: done 1603 episodes, mean reward -71.100, speed 54.15 f/s
Test done in 9.35 sec, reward 83.041, steps 1376
664783: done 1604 episodes, mean reward -70.358, speed 49.78 f/s
Test done in 8.64 sec, reward 70.269, steps 1280
Test done in 9.55 sec, reward 140.820, steps 1397
666298: done 1605 episodes, mean reward -69.104, speed 41.12 f/s
666865: done 1606 episodes, mean reward -68.250, speed 80.89 f/s
Test done in 7.11 sec, reward 44.306, steps 971
Test done in 8.84 sec, reward 112.531, steps 1323
668080: done 1607 episodes, mean reward -67.481, speed 39.43 f/s
Test done in 8.20 sec, reward 75.109, steps 1189
669321: done 1608 episodes, mean reward -66.132, speed 52.70 f/s
Test done in 11.11 sec, reward 154.260, steps 1600
670626: done 1609 episodes, mean reward -65.136, speed 47.61 f/s
Test done in 11.03 sec, reward 151.627, steps 1600
671756: done 1610 episodes, mean reward -64.180, speed 45.13 f/s
Test done in 10.64 sec, reward 159.222, steps 1592
Test done in 9.47 sec, reward 134.467, steps 1410
673356: done 1611 episodes, mean reward -61.550, speed 39.92 f/s
Test done in 10.05 sec, reward 171.524, steps 1484
674956: done 1612 episodes, mean reward -58.867, speed 53.62 f/s
Test done in 10.41 sec, reward 212.467, steps 1554
Test done in 8.88 sec, reward 142.639, steps 1325
676556: done 1613 episodes, mean reward -56.487, speed 40.77 f/s
Test done in 10.03 sec, reward 143.836, steps 1486
Test done in 8.85 sec, reward 99.804, steps 1257
678156: done 1614 episodes, mean reward -54.272, speed 41.31 f/s
Test done in 10.91 sec, reward 158.647, steps 1600
679756: done 1615 episodes, mean reward -52.249, speed 52.56 f/s
Test done in 8.33 sec, reward 31.542, steps 1220
Test done in 10.96 sec, reward 116.273, steps 1589
681356: done 1616 episodes, mean reward -51.509, speed 41.16 f/s
681547: done 1617 episodes, mean reward -51.545, speed 81.80 f/s
Test done in 9.97 sec, reward 34.960, steps 1444
682366: done 1618 episodes, mean reward -51.389, speed 40.53 f/s
Test done in 9.25 sec, reward 51.866, steps 1363
683966: done 1619 episodes, mean reward -49.372, speed 55.03 f/s
Test done in 8.54 sec, reward 27.130, steps 1264
684157: done 1620 episodes, mean reward -49.863, speed 17.64 f/s
684581: done 1621 episodes, mean reward -50.284, speed 81.45 f/s
Test done in 10.11 sec, reward 91.511, steps 1488
Test done in 6.49 sec, reward -21.893, steps 961
686181: done 1622 episodes, mean reward -48.519, speed 43.46 f/s
Test done in 10.57 sec, reward 137.974, steps 1554
687781: done 1623 episodes, mean reward -46.123, speed 52.61 f/s
687995: done 1624 episodes, mean reward -46.040, speed 81.42 f/s
Test done in 10.07 sec, reward 95.024, steps 1459
Test done in 9.73 sec, reward 103.091, steps 1426
689444: done 1625 episodes, mean reward -45.257, speed 38.20 f/s
Test done in 8.30 sec, reward 73.029, steps 1266
Test done in 11.06 sec, reward 134.131, steps 1574
691044: done 1626 episodes, mean reward -42.718, speed 41.25 f/s
Test done in 10.82 sec, reward 137.405, steps 1580
692644: done 1627 episodes, mean reward -41.073, speed 52.48 f/s
692874: done 1628 episodes, mean reward -41.091, speed 79.29 f/s
Test done in 9.50 sec, reward 104.237, steps 1422
693559: done 1629 episodes, mean reward -40.468, speed 37.86 f/s
693781: done 1630 episodes, mean reward -40.711, speed 79.91 f/s
Test done in 10.87 sec, reward 115.375, steps 1600
694792: done 1631 episodes, mean reward -40.484, speed 42.88 f/s
Test done in 9.23 sec, reward 95.534, steps 1367
695208: done 1632 episodes, mean reward -40.273, speed 28.70 f/s
Test done in 8.28 sec, reward -1.875, steps 1198
696808: done 1633 episodes, mean reward -38.148, speed 56.52 f/s
Test done in 6.21 sec, reward -31.142, steps 935
697610: done 1634 episodes, mean reward -37.680, speed 49.24 f/s
Test done in 9.31 sec, reward 58.581, steps 1383
698609: done 1635 episodes, mean reward -36.911, speed 46.37 f/s
Test done in 7.56 sec, reward -0.044, steps 1144
Test done in 9.51 sec, reward 122.922, steps 1417
700209: done 1636 episodes, mean reward -34.452, speed 43.22 f/s
Test done in 9.75 sec, reward 109.610, steps 1447
701155: done 1637 episodes, mean reward -35.946, speed 44.51 f/s
701347: done 1638 episodes, mean reward -36.547, speed 78.82 f/s
701992: done 1639 episodes, mean reward -36.402, speed 81.36 f/s
Test done in 9.52 sec, reward 101.368, steps 1427
702433: done 1640 episodes, mean reward -36.947, speed 29.02 f/s
Test done in 10.00 sec, reward 77.730, steps 1465
703444: done 1641 episodes, mean reward -37.019, speed 45.06 f/s
703585: done 1642 episodes, mean reward -36.926, speed 81.41 f/s
703861: done 1643 episodes, mean reward -36.639, speed 80.79 f/s
Test done in 9.19 sec, reward 72.832, steps 1370
704342: done 1644 episodes, mean reward -36.495, speed 31.72 f/s
704758: done 1645 episodes, mean reward -36.563, speed 80.97 f/s
704954: done 1646 episodes, mean reward -36.430, speed 79.93 f/s
Test done in 10.72 sec, reward 99.760, steps 1600
705037: done 1647 episodes, mean reward -35.936, speed 7.07 f/s
705463: done 1648 episodes, mean reward -35.700, speed 80.76 f/s
Test done in 10.08 sec, reward 116.408, steps 1506
706974: done 1649 episodes, mean reward -34.153, speed 52.73 f/s
Test done in 11.20 sec, reward 165.689, steps 1600
707684: done 1650 episodes, mean reward -33.869, speed 35.57 f/s
Test done in 10.29 sec, reward 111.463, steps 1452
708043: done 1651 episodes, mean reward -35.735, speed 24.59 f/s
708835: done 1652 episodes, mean reward -34.922, speed 78.36 f/s
Test done in 10.61 sec, reward 106.173, steps 1600
709155: done 1653 episodes, mean reward -34.701, speed 21.81 f/s
709332: done 1654 episodes, mean reward -35.676, speed 81.23 f/s
709477: done 1655 episodes, mean reward -35.606, speed 79.95 f/s
709828: done 1656 episodes, mean reward -35.655, speed 78.37 f/s
709977: done 1657 episodes, mean reward -35.628, speed 80.27 f/s
Test done in 5.49 sec, reward -64.340, steps 832
710356: done 1658 episodes, mean reward -35.879, speed 37.15 f/s
710812: done 1659 episodes, mean reward -35.951, speed 78.90 f/s
Test done in 10.27 sec, reward 91.408, steps 1518
711529: done 1660 episodes, mean reward -35.375, speed 37.29 f/s
Test done in 9.78 sec, reward 95.319, steps 1453
712243: done 1661 episodes, mean reward -34.978, speed 38.22 f/s
Test done in 10.63 sec, reward 117.024, steps 1600
713213: done 1662 episodes, mean reward -34.342, speed 42.97 f/s
713401: done 1663 episodes, mean reward -34.747, speed 82.22 f/s
Test done in 7.97 sec, reward 63.620, steps 1187
714919: done 1664 episodes, mean reward -33.613, speed 55.98 f/s
Test done in 7.79 sec, reward 41.907, steps 1158
715313: done 1665 episodes, mean reward -33.542, speed 30.89 f/s
715916: done 1666 episodes, mean reward -35.545, speed 81.90 f/s
Test done in 5.22 sec, reward -15.519, steps 776
716366: done 1667 episodes, mean reward -35.624, speed 41.83 f/s
716792: done 1668 episodes, mean reward -35.702, speed 80.67 f/s
Test done in 1.06 sec, reward -123.321, steps 148
717380: done 1669 episodes, mean reward -35.232, speed 67.42 f/s
717996: done 1670 episodes, mean reward -35.022, speed 80.98 f/s
Test done in 9.28 sec, reward 79.683, steps 1390
718502: done 1671 episodes, mean reward -35.146, speed 32.54 f/s
718661: done 1672 episodes, mean reward -35.497, speed 85.10 f/s
Test done in 7.51 sec, reward 49.389, steps 1115
719545: done 1673 episodes, mean reward -35.082, speed 47.73 f/s
719668: done 1674 episodes, mean reward -35.329, speed 77.16 f/s
Test done in 10.34 sec, reward 122.006, steps 1466
720580: done 1675 episodes, mean reward -35.203, speed 41.83 f/s
720753: done 1676 episodes, mean reward -35.574, speed 80.39 f/s
Test done in 7.62 sec, reward 20.350, steps 1139
721173: done 1677 episodes, mean reward -35.450, speed 32.72 f/s
721471: done 1678 episodes, mean reward -35.411, speed 79.50 f/s
721653: done 1679 episodes, mean reward -36.305, speed 81.36 f/s
721854: done 1680 episodes, mean reward -36.507, speed 81.03 f/s
Test done in 9.73 sec, reward 94.124, steps 1452
722158: done 1681 episodes, mean reward -36.444, speed 22.50 f/s
Test done in 11.18 sec, reward 98.614, steps 1600
723226: done 1682 episodes, mean reward -36.576, speed 43.80 f/s
723417: done 1683 episodes, mean reward -36.709, speed 81.47 f/s
Test done in 10.92 sec, reward 153.314, steps 1600
724123: done 1684 episodes, mean reward -38.631, speed 35.79 f/s
724613: done 1685 episodes, mean reward -38.637, speed 81.45 f/s
724841: done 1686 episodes, mean reward -38.831, speed 79.54 f/s
724956: done 1687 episodes, mean reward -39.538, speed 78.49 f/s
Test done in 7.56 sec, reward 64.890, steps 1111
725289: done 1688 episodes, mean reward -39.540, speed 28.48 f/s
725390: done 1689 episodes, mean reward -39.611, speed 81.73 f/s
725669: done 1690 episodes, mean reward -39.721, speed 80.98 f/s
725887: done 1691 episodes, mean reward -39.714, speed 75.06 f/s
Test done in 4.87 sec, reward -53.448, steps 708
726587: done 1692 episodes, mean reward -41.174, speed 51.44 f/s
Test done in 10.00 sec, reward 139.381, steps 1487
727020: done 1693 episodes, mean reward -41.461, speed 28.22 f/s
727389: done 1694 episodes, mean reward -41.821, speed 80.03 f/s
Test done in 9.06 sec, reward 35.753, steps 1360
728009: done 1695 episodes, mean reward -41.518, speed 37.21 f/s
728352: done 1696 episodes, mean reward -43.408, speed 78.24 f/s
728525: done 1697 episodes, mean reward -44.361, speed 81.67 f/s
728726: done 1698 episodes, mean reward -44.925, speed 79.18 f/s
Test done in 10.51 sec, reward 75.450, steps 1542
729753: done 1699 episodes, mean reward -45.308, speed 44.36 f/s
729886: done 1700 episodes, mean reward -46.355, speed 80.84 f/s
729972: done 1701 episodes, mean reward -47.614, speed 84.59 f/s
Test done in 5.90 sec, reward -36.486, steps 902
730641: done 1702 episodes, mean reward -46.869, speed 47.27 f/s
Test done in 10.54 sec, reward 99.233, steps 1600
731047: done 1703 episodes, mean reward -47.992, speed 26.25 f/s
731201: done 1704 episodes, mean reward -48.742, speed 81.89 f/s
731482: done 1705 episodes, mean reward -50.143, speed 80.21 f/s
Test done in 10.95 sec, reward 128.111, steps 1600
732276: done 1706 episodes, mean reward -50.329, speed 38.14 f/s
732414: done 1707 episodes, mean reward -51.291, speed 80.17 f/s
Test done in 11.24 sec, reward 133.835, steps 1597
733125: done 1708 episodes, mean reward -51.816, speed 35.43 f/s
733410: done 1709 episodes, mean reward -52.734, speed 82.54 f/s
Test done in 10.58 sec, reward 170.467, steps 1600
734826: done 1710 episodes, mean reward -52.620, speed 50.20 f/s
Test done in 10.74 sec, reward 130.330, steps 1600
Test done in 10.90 sec, reward 163.098, steps 1600
736426: done 1711 episodes, mean reward -52.707, speed 38.20 f/s
736567: done 1712 episodes, mean reward -55.275, speed 74.53 f/s
736819: done 1713 episodes, mean reward -57.647, speed 81.55 f/s
Test done in 9.51 sec, reward 140.018, steps 1377
737012: done 1714 episodes, mean reward -60.068, speed 16.15 f/s
737570: done 1715 episodes, mean reward -61.744, speed 80.59 f/s
Test done in 10.68 sec, reward 142.828, steps 1600
738496: done 1716 episodes, mean reward -62.182, speed 41.65 f/s
738779: done 1717 episodes, mean reward -62.262, speed 77.93 f/s
Test done in 10.71 sec, reward 158.440, steps 1600
739142: done 1718 episodes, mean reward -62.331, speed 23.75 f/s
739516: done 1719 episodes, mean reward -64.002, speed 81.03 f/s
Test done in 10.82 sec, reward 160.303, steps 1600
Test done in 11.07 sec, reward 171.356, steps 1600
741116: done 1720 episodes, mean reward -61.162, speed 38.33 f/s
741577: done 1721 episodes, mean reward -60.562, speed 80.50 f/s
741997: done 1723 episodes, mean reward -65.123, speed 80.66 f/s
Test done in 10.18 sec, reward 152.529, steps 1481
742294: done 1724 episodes, mean reward -65.017, speed 21.41 f/s
742819: done 1725 episodes, mean reward -65.469, speed 80.60 f/s
Test done in 10.85 sec, reward 199.551, steps 1600
743101: done 1726 episodes, mean reward -67.919, speed 19.67 f/s
743204: done 1727 episodes, mean reward -70.371, speed 83.32 f/s
Test done in 10.22 sec, reward 164.442, steps 1488
744475: done 1728 episodes, mean reward -69.101, speed 48.05 f/s
Test done in 10.74 sec, reward 200.858, steps 1600
745856: done 1729 episodes, mean reward -68.278, speed 49.97 f/s
Test done in 10.66 sec, reward 183.018, steps 1600
746091: done 1730 episodes, mean reward -68.239, speed 17.28 f/s
746721: done 1731 episodes, mean reward -68.308, speed 76.71 f/s
746929: done 1732 episodes, mean reward -68.449, speed 81.32 f/s
Test done in 9.99 sec, reward 136.231, steps 1424
Test done in 9.45 sec, reward 150.367, steps 1408
748529: done 1733 episodes, mean reward -67.936, speed 40.83 f/s
Test done in 9.34 sec, reward 141.482, steps 1368
749257: done 1734 episodes, mean reward -67.466, speed 39.50 f/s
Test done in 9.07 sec, reward 120.570, steps 1358
750230: done 1735 episodes, mean reward -67.014, speed 45.94 f/s
750372: done 1736 episodes, mean reward -69.387, speed 80.40 f/s
750707: done 1737 episodes, mean reward -69.758, speed 82.53 f/s
750968: done 1738 episodes, mean reward -69.424, speed 83.30 f/s
Test done in 9.74 sec, reward 150.626, steps 1475
751427: done 1739 episodes, mean reward -69.328, speed 29.43 f/s
751852: done 1740 episodes, mean reward -69.126, speed 81.88 f/s
Test done in 8.52 sec, reward 104.552, steps 1246
752125: done 1741 episodes, mean reward -69.468, speed 23.01 f/s
752338: done 1742 episodes, mean reward -69.307, speed 81.59 f/s
752907: done 1743 episodes, mean reward -68.698, speed 81.11 f/s
Test done in 9.91 sec, reward 173.504, steps 1454
753583: done 1744 episodes, mean reward -68.228, speed 36.78 f/s
753870: done 1745 episodes, mean reward -68.327, speed 81.70 f/s
Test done in 10.13 sec, reward 162.825, steps 1510
754400: done 1746 episodes, mean reward -67.782, speed 31.53 f/s
754659: done 1747 episodes, mean reward -67.644, speed 80.49 f/s
754743: done 1748 episodes, mean reward -67.913, speed 76.50 f/s
Test done in 4.21 sec, reward -28.178, steps 608
755090: done 1749 episodes, mean reward -68.678, speed 40.14 f/s
755364: done 1750 episodes, mean reward -69.052, speed 80.88 f/s
755750: done 1751 episodes, mean reward -68.968, speed 80.73 f/s
755901: done 1752 episodes, mean reward -69.408, speed 78.59 f/s
Test done in 9.44 sec, reward 131.328, steps 1408
756772: done 1753 episodes, mean reward -68.407, speed 42.93 f/s
Test done in 9.57 sec, reward 123.291, steps 1365
757207: done 1754 episodes, mean reward -68.053, speed 28.89 f/s
757416: done 1755 episodes, mean reward -67.951, speed 82.90 f/s
757907: done 1756 episodes, mean reward -67.512, speed 80.64 f/s
Test done in 8.81 sec, reward 81.978, steps 1231
758441: done 1757 episodes, mean reward -67.093, speed 34.60 f/s
758985: done 1758 episodes, mean reward -66.648, speed 79.67 f/s
Test done in 9.27 sec, reward 167.224, steps 1397
759421: done 1759 episodes, mean reward -66.429, speed 29.77 f/s
Test done in 8.31 sec, reward 112.656, steps 1245
Test done in 10.47 sec, reward 179.919, steps 1529
761021: done 1760 episodes, mean reward -63.683, speed 41.20 f/s
761478: done 1761 episodes, mean reward -63.623, speed 81.06 f/s
Test done in 10.69 sec, reward 179.145, steps 1600
762084: done 1762 episodes, mean reward -63.489, speed 33.80 f/s
762194: done 1763 episodes, mean reward -63.746, speed 81.74 f/s
762855: done 1764 episodes, mean reward -64.431, speed 81.73 f/s
Test done in 9.52 sec, reward 141.160, steps 1423
763363: done 1765 episodes, mean reward -64.135, speed 32.15 f/s
763657: done 1766 episodes, mean reward -64.100, speed 74.92 f/s
763848: done 1767 episodes, mean reward -64.049, speed 78.21 f/s
Test done in 10.29 sec, reward 148.124, steps 1526
764339: done 1768 episodes, mean reward -63.792, speed 30.00 f/s
764582: done 1769 episodes, mean reward -64.149, speed 82.58 f/s
764920: done 1770 episodes, mean reward -64.392, speed 82.97 f/s
Test done in 10.55 sec, reward 170.336, steps 1600
765428: done 1771 episodes, mean reward -64.101, speed 29.88 f/s
765633: done 1772 episodes, mean reward -63.869, speed 79.63 f/s
765818: done 1773 episodes, mean reward -64.088, speed 79.65 f/s
765937: done 1774 episodes, mean reward -63.902, speed 80.35 f/s
Test done in 10.45 sec, reward 137.304, steps 1534
766003: done 1775 episodes, mean reward -64.702, speed 5.86 f/s
766720: done 1776 episodes, mean reward -64.001, speed 79.07 f/s
Test done in 10.11 sec, reward 139.383, steps 1494
767330: done 1777 episodes, mean reward -63.559, speed 34.42 f/s
767618: done 1778 episodes, mean reward -63.357, speed 79.23 f/s
Test done in 10.81 sec, reward 165.634, steps 1600
768686: done 1779 episodes, mean reward -62.295, speed 44.24 f/s
Test done in 10.49 sec, reward 125.247, steps 1538
769017: done 1780 episodes, mean reward -62.221, speed 22.69 f/s
769934: done 1781 episodes, mean reward -61.272, speed 82.24 f/s
Test done in 9.58 sec, reward 104.030, steps 1462
770144: done 1782 episodes, mean reward -61.855, speed 17.14 f/s
Test done in 10.71 sec, reward 106.780, steps 1600
771329: done 1783 episodes, mean reward -60.740, speed 46.30 f/s
771506: done 1784 episodes, mean reward -61.156, speed 81.58 f/s
771939: done 1785 episodes, mean reward -60.804, speed 79.18 f/s
Test done in 10.29 sec, reward 124.686, steps 1540
772327: done 1786 episodes, mean reward -60.494, speed 25.62 f/s
Test done in 9.60 sec, reward 94.990, steps 1414
773134: done 1787 episodes, mean reward -59.696, speed 41.31 f/s
773415: done 1788 episodes, mean reward -59.655, speed 80.18 f/s
773950: done 1789 episodes, mean reward -59.172, speed 81.37 f/s
Test done in 9.55 sec, reward 104.662, steps 1429
774431: done 1790 episodes, mean reward -58.767, speed 31.10 f/s
Test done in 9.30 sec, reward 72.109, steps 1396
775272: done 1791 episodes, mean reward -58.524, speed 42.69 f/s
775413: done 1792 episodes, mean reward -58.990, speed 78.47 f/s
775936: done 1793 episodes, mean reward -58.844, speed 80.68 f/s
Test done in 10.55 sec, reward 131.232, steps 1598
776337: done 1794 episodes, mean reward -58.443, speed 25.87 f/s
776563: done 1795 episodes, mean reward -58.834, speed 80.00 f/s
776906: done 1796 episodes, mean reward -58.837, speed 81.92 f/s
Test done in 10.60 sec, reward 137.591, steps 1600
777416: done 1797 episodes, mean reward -58.537, speed 30.11 f/s
777717: done 1798 episodes, mean reward -58.141, speed 81.66 f/s
777824: done 1799 episodes, mean reward -59.004, speed 82.01 f/s
Test done in 9.87 sec, reward 140.344, steps 1468
778034: done 1800 episodes, mean reward -59.166, speed 16.70 f/s
778304: done 1801 episodes, mean reward -58.875, speed 81.42 f/s
778426: done 1802 episodes, mean reward -59.486, speed 80.95 f/s
778557: done 1803 episodes, mean reward -59.634, speed 79.80 f/s
778960: done 1804 episodes, mean reward -59.573, speed 76.57 f/s
Test done in 9.44 sec, reward 121.068, steps 1385
779570: done 1805 episodes, mean reward -58.852, speed 36.10 f/s
779717: done 1806 episodes, mean reward -59.262, speed 78.75 f/s
779939: done 1807 episodes, mean reward -59.578, speed 79.41 f/s
Test done in 7.04 sec, reward 60.406, steps 1061
780692: done 1808 episodes, mean reward -59.994, speed 45.64 f/s
780898: done 1809 episodes, mean reward -60.145, speed 80.46 f/s
Test done in 10.15 sec, reward 144.352, steps 1516
781674: done 1810 episodes, mean reward -60.329, speed 38.84 f/s
781992: done 1811 episodes, mean reward -62.460, speed 81.18 f/s
Test done in 6.94 sec, reward 72.376, steps 1043
782098: done 1812 episodes, mean reward -62.575, speed 12.84 f/s
Test done in 10.87 sec, reward 174.903, steps 1600
783573: done 1813 episodes, mean reward -63.513, speed 50.22 f/s
783940: done 1814 episodes, mean reward -63.082, speed 79.56 f/s
Test done in 8.04 sec, reward 65.065, steps 1183
784070: done 1815 episodes, mean reward -63.724, speed 13.43 f/s
784424: done 1816 episodes, mean reward -64.011, speed 78.80 f/s
784622: done 1817 episodes, mean reward -63.885, speed 81.40 f/s
784873: done 1818 episodes, mean reward -63.687, speed 80.09 f/s
784979: done 1819 episodes, mean reward -64.011, speed 82.51 f/s
Test done in 10.19 sec, reward 63.264, steps 1477
785097: done 1820 episodes, mean reward -66.715, speed 10.09 f/s
Test done in 9.99 sec, reward -3.985, steps 1464
786051: done 1821 episodes, mean reward -66.311, speed 43.77 f/s
786935: done 1822 episodes, mean reward -65.496, speed 80.86 f/s
Test done in 10.39 sec, reward -15.094, steps 1476
787506: done 1823 episodes, mean reward -65.743, speed 32.43 f/s
787781: done 1824 episodes, mean reward -65.784, speed 80.57 f/s
787983: done 1825 episodes, mean reward -66.291, speed 78.39 f/s
Test done in 10.90 sec, reward 149.376, steps 1600
Test done in 10.58 sec, reward 104.334, steps 1567
789231: done 1826 episodes, mean reward -66.827, speed 33.80 f/s
789773: done 1827 episodes, mean reward -66.670, speed 79.86 f/s
Test done in 10.59 sec, reward 111.327, steps 1474
790774: done 1828 episodes, mean reward -68.216, speed 43.26 f/s
Test done in 9.73 sec, reward 73.657, steps 1391
791177: done 1829 episodes, mean reward -69.094, speed 27.36 f/s
791764: done 1830 episodes, mean reward -69.398, speed 80.55 f/s
791992: done 1831 episodes, mean reward -69.892, speed 80.37 f/s
Test done in 10.13 sec, reward -34.385, steps 1478
792235: done 1832 episodes, mean reward -69.682, speed 18.46 f/s
792607: done 1833 episodes, mean reward -72.530, speed 76.85 f/s
792873: done 1834 episodes, mean reward -73.368, speed 75.70 f/s
Test done in 9.67 sec, reward 139.344, steps 1445
793205: done 1835 episodes, mean reward -74.224, speed 24.04 f/s
793541: done 1836 episodes, mean reward -74.315, speed 80.11 f/s
Test done in 8.65 sec, reward 68.661, steps 1266
794256: done 1837 episodes, mean reward -75.193, speed 40.66 f/s
794429: done 1838 episodes, mean reward -75.537, speed 84.87 f/s
Test done in 4.55 sec, reward -27.941, steps 674
795261: done 1839 episodes, mean reward -76.820, speed 56.23 f/s
795477: done 1841 episodes, mean reward -77.741, speed 83.17 f/s
795802: done 1842 episodes, mean reward -77.743, speed 83.11 f/s
795955: done 1843 episodes, mean reward -78.293, speed 80.01 f/s
Test done in 8.22 sec, reward 113.186, steps 1172
796079: done 1844 episodes, mean reward -79.075, speed 12.81 f/s
796282: done 1845 episodes, mean reward -79.197, speed 83.00 f/s
796996: done 1846 episodes, mean reward -78.947, speed 82.06 f/s
Test done in 8.82 sec, reward 147.711, steps 1314
797259: done 1847 episodes, mean reward -78.868, speed 21.78 f/s
797612: done 1848 episodes, mean reward -78.593, speed 83.27 f/s
Test done in 9.63 sec, reward 165.144, steps 1400
798004: done 1849 episodes, mean reward -78.682, speed 27.34 f/s
798289: done 1850 episodes, mean reward -78.686, speed 81.42 f/s
798669: done 1851 episodes, mean reward -78.485, speed 79.36 f/s
798881: done 1852 episodes, mean reward -78.490, speed 77.07 f/s
Test done in 7.17 sec, reward 87.914, steps 1045
799153: done 1853 episodes, mean reward -79.416, speed 25.97 f/s
799453: done 1854 episodes, mean reward -79.514, speed 80.86 f/s
799850: done 1855 episodes, mean reward -79.107, speed 79.85 f/s
Test done in 0.75 sec, reward -127.784, steps 100
800128: done 1856 episodes, mean reward -79.600, speed 66.32 f/s
800425: done 1857 episodes, mean reward -79.830, speed 77.93 f/s
800587: done 1858 episodes, mean reward -80.255, speed 78.82 f/s
800849: done 1859 episodes, mean reward -80.688, speed 84.01 f/s
Test done in 2.19 sec, reward -73.633, steps 306
801149: done 1860 episodes, mean reward -83.556, speed 51.20 f/s
801545: done 1861 episodes, mean reward -83.725, speed 79.70 f/s
801863: done 1862 episodes, mean reward -84.049, speed 77.06 f/s
801960: done 1863 episodes, mean reward -83.913, speed 75.29 f/s
Test done in 0.55 sec, reward -114.773, steps 64
802028: done 1864 episodes, mean reward -84.692, speed 47.72 f/s
802145: done 1866 episodes, mean reward -85.693, speed 76.43 f/s
802299: done 1868 episodes, mean reward -86.373, speed 73.72 f/s
802464: done 1870 episodes, mean reward -86.733, speed 78.22 f/s
802942: done 1872 episodes, mean reward -87.170, speed 76.03 f/s
Test done in 0.46 sec, reward -115.966, steps 60
803020: done 1873 episodes, mean reward -87.403, speed 55.02 f/s
803118: done 1874 episodes, mean reward -87.516, speed 71.22 f/s
803203: done 1875 episodes, mean reward -87.511, speed 76.95 f/s
803434: done 1877 episodes, mean reward -88.509, speed 80.46 f/s
803708: done 1878 episodes, mean reward -88.532, speed 78.89 f/s
803829: done 1880 episodes, mean reward -89.842, speed 78.77 f/s
Test done in 0.49 sec, reward -113.607, steps 71
804387: done 1882 episodes, mean reward -90.331, speed 75.52 f/s
804686: done 1883 episodes, mean reward -91.124, speed 80.89 f/s
804776: done 1884 episodes, mean reward -90.889, speed 80.32 f/s
804909: done 1886 episodes, mean reward -91.760, speed 83.44 f/s
Test done in 0.63 sec, reward -109.227, steps 90
805032: done 1888 episodes, mean reward -92.837, speed 56.69 f/s
805163: done 1890 episodes, mean reward -93.903, speed 76.08 f/s
805439: done 1892 episodes, mean reward -94.378, speed 81.17 f/s
805574: done 1894 episodes, mean reward -95.542, speed 83.00 f/s
805708: done 1896 episodes, mean reward -96.168, speed 81.97 f/s
805896: done 1898 episodes, mean reward -97.048, speed 81.76 f/s
Test done in 0.68 sec, reward -110.506, steps 92
806017: done 1900 episodes, mean reward -97.092, speed 57.27 f/s
806456: done 1902 episodes, mean reward -96.946, speed 80.23 f/s
806633: done 1903 episodes, mean reward -96.800, speed 80.42 f/s
806831: done 1904 episodes, mean reward -96.829, speed 78.78 f/s
Test done in 5.30 sec, reward 35.122, steps 776
807096: done 1905 episodes, mean reward -97.237, speed 30.45 f/s
807237: done 1906 episodes, mean reward -97.295, speed 76.75 f/s
807368: done 1907 episodes, mean reward -96.859, speed 75.69 f/s
807780: done 1908 episodes, mean reward -96.597, speed 81.43 f/s
807946: done 1909 episodes, mean reward -96.460, speed 82.91 f/s
Test done in 8.12 sec, reward 126.701, steps 1184
808451: done 1910 episodes, mean reward -96.918, speed 35.32 f/s
808683: done 1911 episodes, mean reward -96.972, speed 83.12 f/s
808805: done 1912 episodes, mean reward -96.842, speed 80.68 f/s
808959: done 1913 episodes, mean reward -95.912, speed 82.09 f/s
Test done in 6.98 sec, reward 94.916, steps 1029
809069: done 1914 episodes, mean reward -96.424, speed 13.15 f/s
809237: done 1915 episodes, mean reward -96.317, speed 82.29 f/s
809444: done 1916 episodes, mean reward -96.457, speed 80.74 f/s
809771: done 1917 episodes, mean reward -96.113, speed 83.99 f/s
809980: done 1918 episodes, mean reward -96.241, speed 81.59 f/s
Test done in 4.11 sec, reward -14.888, steps 598
810282: done 1919 episodes, mean reward -95.883, speed 39.38 f/s
810431: done 1920 episodes, mean reward -95.701, speed 82.95 f/s
810706: done 1921 episodes, mean reward -96.221, speed 76.41 f/s
810894: done 1922 episodes, mean reward -96.816, speed 78.51 f/s
Test done in 6.24 sec, reward 87.131, steps 944
811066: done 1923 episodes, mean reward -96.621, speed 20.56 f/s
811202: done 1924 episodes, mean reward -96.755, speed 81.34 f/s
811624: done 1925 episodes, mean reward -96.182, speed 82.60 f/s
811797: done 1926 episodes, mean reward -95.426, speed 84.06 f/s
Test done in 10.39 sec, reward 232.788, steps 1572
812034: done 1927 episodes, mean reward -95.088, speed 17.90 f/s
812222: done 1928 episodes, mean reward -94.701, speed 81.57 f/s
812514: done 1929 episodes, mean reward -94.934, speed 80.59 f/s
812650: done 1930 episodes, mean reward -94.666, speed 82.92 f/s
812858: done 1931 episodes, mean reward -94.516, speed 82.00 f/s
Test done in 2.20 sec, reward -57.664, steps 320
813005: done 1932 episodes, mean reward -94.671, speed 36.64 f/s
813305: done 1933 episodes, mean reward -94.414, speed 81.06 f/s
813564: done 1934 episodes, mean reward -94.243, speed 81.53 f/s
813957: done 1935 episodes, mean reward -93.987, speed 80.24 f/s
Test done in 4.22 sec, reward 6.119, steps 569
814095: done 1936 episodes, mean reward -93.856, speed 23.07 f/s
814270: done 1937 episodes, mean reward -93.009, speed 80.20 f/s
814400: done 1938 episodes, mean reward -92.840, speed 75.41 f/s
814678: done 1939 episodes, mean reward -91.503, speed 79.90 f/s
814882: done 1940 episodes, mean reward -91.056, speed 81.71 f/s
Test done in 7.06 sec, reward 121.488, steps 1082
815135: done 1941 episodes, mean reward -90.693, speed 25.04 f/s
815487: done 1942 episodes, mean reward -90.373, speed 78.97 f/s
815896: done 1943 episodes, mean reward -89.873, speed 81.87 f/s
Test done in 4.09 sec, reward 1.248, steps 599
816045: done 1944 episodes, mean reward -89.672, speed 25.37 f/s
816152: done 1945 episodes, mean reward -89.593, speed 81.37 f/s
816392: done 1946 episodes, mean reward -90.101, speed 81.01 f/s
816558: done 1947 episodes, mean reward -90.165, speed 80.91 f/s
816645: done 1948 episodes, mean reward -90.316, speed 78.75 f/s
816783: done 1949 episodes, mean reward -90.518, speed 81.45 f/s
816984: done 1950 episodes, mean reward -90.447, speed 82.59 f/s
Test done in 4.97 sec, reward 39.630, steps 758
817137: done 1951 episodes, mean reward -90.755, speed 22.25 f/s
817364: done 1952 episodes, mean reward -90.661, speed 77.09 f/s
817504: done 1953 episodes, mean reward -90.809, speed 80.72 f/s
817667: done 1954 episodes, mean reward -90.955, speed 82.49 f/s
Test done in 4.72 sec, reward 28.190, steps 709
818041: done 1955 episodes, mean reward -90.818, speed 39.80 f/s
818330: done 1956 episodes, mean reward -90.625, speed 81.43 f/s
818432: done 1957 episodes, mean reward -90.913, speed 80.87 f/s
818596: done 1958 episodes, mean reward -90.766, speed 83.59 f/s
818777: done 1959 episodes, mean reward -90.603, speed 83.01 f/s
Test done in 3.73 sec, reward -5.212, steps 549
819117: done 1960 episodes, mean reward -90.486, speed 42.73 f/s
819276: done 1961 episodes, mean reward -90.669, speed 79.90 f/s
819459: done 1962 episodes, mean reward -90.775, speed 79.45 f/s
819780: done 1963 episodes, mean reward -90.341, speed 77.81 f/s
819917: done 1964 episodes, mean reward -90.034, speed 78.58 f/s
Test done in 5.35 sec, reward 54.692, steps 787
820008: done 1965 episodes, mean reward -89.931, speed 14.01 f/s
820130: done 1966 episodes, mean reward -89.612, speed 80.78 f/s
820269: done 1967 episodes, mean reward -89.310, speed 83.80 f/s
820556: done 1968 episodes, mean reward -88.944, speed 81.44 f/s
820744: done 1969 episodes, mean reward -88.634, speed 83.53 f/s
820911: done 1970 episodes, mean reward -88.575, speed 83.44 f/s
Test done in 7.47 sec, reward 125.907, steps 1114
821189: done 1971 episodes, mean reward -88.266, speed 25.38 f/s
821472: done 1972 episodes, mean reward -88.323, speed 80.96 f/s
821665: done 1973 episodes, mean reward -88.128, speed 84.00 f/s
821926: done 1974 episodes, mean reward -87.605, speed 80.52 f/s
Test done in 7.03 sec, reward 133.323, steps 1041
822084: done 1975 episodes, mean reward -87.443, speed 17.64 f/s
822378: done 1976 episodes, mean reward -86.954, speed 81.98 f/s
822504: done 1977 episodes, mean reward -87.026, speed 82.12 f/s
822661: done 1978 episodes, mean reward -87.094, speed 79.04 f/s
822784: done 1979 episodes, mean reward -86.946, speed 75.57 f/s
822943: done 1980 episodes, mean reward -86.798, speed 74.92 f/s
Test done in 1.92 sec, reward -67.681, steps 276
823407: done 1981 episodes, mean reward -86.039, speed 58.37 f/s
823591: done 1982 episodes, mean reward -86.471, speed 80.52 f/s
823806: done 1983 episodes, mean reward -86.586, speed 80.79 f/s
Test done in 3.90 sec, reward -2.044, steps 581
824062: done 1984 episodes, mean reward -86.186, speed 36.10 f/s
824232: done 1985 episodes, mean reward -85.878, speed 78.97 f/s
824621: done 1986 episodes, mean reward -85.218, speed 78.92 f/s
824848: done 1987 episodes, mean reward -84.858, speed 83.05 f/s
Test done in 8.67 sec, reward 201.016, steps 1323
825177: done 1988 episodes, mean reward -84.320, speed 25.90 f/s
825283: done 1989 episodes, mean reward -84.200, speed 85.78 f/s
825517: done 1990 episodes, mean reward -83.813, speed 83.34 f/s
825815: done 1991 episodes, mean reward -83.351, speed 83.35 f/s
Test done in 6.50 sec, reward 117.609, steps 956
826057: done 1992 episodes, mean reward -83.241, speed 25.37 f/s
826224: done 1993 episodes, mean reward -82.892, speed 81.57 f/s
826464: done 1994 episodes, mean reward -82.433, speed 80.40 f/s
826557: done 1995 episodes, mean reward -82.538, speed 80.49 f/s
826975: done 1996 episodes, mean reward -81.808, speed 79.74 f/s
Test done in 6.78 sec, reward 118.384, steps 1008
827178: done 1997 episodes, mean reward -81.639, speed 21.92 f/s
827336: done 1998 episodes, mean reward -81.581, speed 76.32 f/s
827851: done 1999 episodes, mean reward -80.575, speed 83.38 f/s
Test done in 7.24 sec, reward 137.313, steps 1095
828054: done 2000 episodes, mean reward -80.811, speed 20.66 f/s
828183: done 2001 episodes, mean reward -80.550, speed 80.69 f/s
828348: done 2002 episodes, mean reward -80.943, speed 81.36 f/s
828467: done 2003 episodes, mean reward -81.058, speed 83.25 f/s
828721: done 2004 episodes, mean reward -80.923, speed 79.95 f/s
828839: done 2005 episodes, mean reward -81.130, speed 80.67 f/s
828952: done 2006 episodes, mean reward -81.183, speed 80.96 f/s
Test done in 5.86 sec, reward 71.305, steps 880
829128: done 2007 episodes, mean reward -81.125, speed 21.91 f/s
829411: done 2008 episodes, mean reward -81.329, speed 75.39 f/s
829739: done 2009 episodes, mean reward -81.198, speed 77.77 f/s
829864: done 2010 episodes, mean reward -81.772, speed 78.82 f/s
Test done in 2.95 sec, reward -38.758, steps 431
830082: done 2011 episodes, mean reward -82.443, speed 38.17 f/s
830403: done 2012 episodes, mean reward -82.153, speed 80.54 f/s
830653: done 2013 episodes, mean reward -82.004, speed 83.67 f/s
830767: done 2014 episodes, mean reward -82.037, speed 76.80 f/s
830870: done 2015 episodes, mean reward -82.072, speed 80.93 f/s
Test done in 2.36 sec, reward -54.868, steps 351
831080: done 2016 episodes, mean reward -82.709, speed 42.60 f/s
831306: done 2017 episodes, mean reward -83.011, speed 81.41 f/s
831590: done 2018 episodes, mean reward -83.134, speed 81.66 f/s
831924: done 2019 episodes, mean reward -83.752, speed 80.54 f/s
Test done in 0.56 sec, reward -120.770, steps 75
832039: done 2020 episodes, mean reward -83.833, speed 59.69 f/s
832386: done 2022 episodes, mean reward -84.554, speed 80.80 f/s
832494: done 2023 episodes, mean reward -84.769, speed 78.38 f/s
832578: done 2024 episodes, mean reward -84.920, speed 79.08 f/s
832949: done 2025 episodes, mean reward -85.019, speed 80.55 f/s
Test done in 4.20 sec, reward -3.523, steps 591
833199: done 2026 episodes, mean reward -85.346, speed 34.31 f/s
833281: done 2027 episodes, mean reward -85.809, speed 75.69 f/s
833461: done 2028 episodes, mean reward -85.919, speed 78.39 f/s
833869: done 2029 episodes, mean reward -85.988, speed 76.98 f/s
Test done in 9.41 sec, reward 194.571, steps 1385
834075: done 2030 episodes, mean reward -86.329, speed 17.20 f/s
834210: done 2031 episodes, mean reward -86.729, speed 79.27 f/s
834605: done 2032 episodes, mean reward -86.540, speed 78.59 f/s
Test done in 3.64 sec, reward -19.367, steps 549
835052: done 2033 episodes, mean reward -86.589, speed 49.32 f/s
835216: done 2034 episodes, mean reward -86.826, speed 82.02 f/s
835353: done 2036 episodes, mean reward -87.762, speed 83.96 f/s
835741: done 2037 episodes, mean reward -87.794, speed 80.60 f/s
Test done in 6.37 sec, reward 57.233, steps 944
836673: done 2039 episodes, mean reward -88.225, speed 52.46 f/s
836949: done 2040 episodes, mean reward -88.308, speed 76.94 f/s
Test done in 1.90 sec, reward -91.882, steps 251
837036: done 2041 episodes, mean reward -88.697, speed 28.55 f/s
837163: done 2043 episodes, mean reward -89.810, speed 83.13 f/s
Test done in 7.69 sec, reward 50.322, steps 1115
838763: done 2044 episodes, mean reward -89.106, speed 58.44 f/s
Test done in 5.32 sec, reward 54.093, steps 802
Test done in 5.79 sec, reward -11.819, steps 841
840363: done 2045 episodes, mean reward -89.012, speed 51.28 f/s
Test done in 5.31 sec, reward -72.572, steps 751
841963: done 2046 episodes, mean reward -89.193, speed 63.09 f/s
Test done in 11.34 sec, reward -49.087, steps 1600
Test done in 11.38 sec, reward -81.973, steps 1600
843563: done 2047 episodes, mean reward -88.856, speed 37.68 f/s
Test done in 10.45 sec, reward 103.820, steps 1540
Test done in 9.75 sec, reward -134.411, steps 1358
845163: done 2048 episodes, mean reward -88.862, speed 40.12 f/s
Test done in 8.86 sec, reward -40.276, steps 1225
846763: done 2049 episodes, mean reward -89.132, speed 56.21 f/s
Test done in 11.41 sec, reward -174.525, steps 1600
Test done in 11.40 sec, reward -191.396, steps 1569
848363: done 2050 episodes, mean reward -89.823, speed 37.63 f/s
Test done in 0.86 sec, reward -129.705, steps 114
849963: done 2051 episodes, mean reward -90.131, speed 75.44 f/s
Test done in 10.21 sec, reward 67.833, steps 1473
850030: done 2052 episodes, mean reward -90.418, speed 6.07 f/s
850180: done 2053 episodes, mean reward -90.526, speed 81.71 f/s
850448: done 2054 episodes, mean reward -90.364, speed 81.72 f/s
Test done in 9.10 sec, reward 223.366, steps 1329
Test done in 11.92 sec, reward -152.464, steps 1600
852048: done 2055 episodes, mean reward -91.099, speed 39.28 f/s
Test done in 11.35 sec, reward -154.963, steps 1600
853648: done 2056 episodes, mean reward -91.647, speed 51.66 f/s
Test done in 9.17 sec, reward 62.353, steps 1316
Test done in 11.32 sec, reward -146.053, steps 1600
855248: done 2057 episodes, mean reward -91.997, speed 39.39 f/s
Test done in 11.28 sec, reward -84.984, steps 1600
856848: done 2058 episodes, mean reward -92.178, speed 52.19 f/s
Test done in 11.38 sec, reward -61.961, steps 1600
Test done in 9.49 sec, reward 141.710, steps 1418
858448: done 2059 episodes, mean reward -92.370, speed 39.28 f/s
Test done in 9.83 sec, reward 78.080, steps 1459
859131: done 2060 episodes, mean reward -92.691, speed 37.62 f/s
859578: done 2061 episodes, mean reward -92.808, speed 82.18 f/s
Test done in 11.05 sec, reward 120.695, steps 1586
860400: done 2062 episodes, mean reward -92.342, speed 39.16 f/s
860840: done 2063 episodes, mean reward -92.243, speed 80.73 f/s
Test done in 10.34 sec, reward 87.489, steps 1557
861334: done 2064 episodes, mean reward -92.783, speed 30.00 f/s
861593: done 2065 episodes, mean reward -92.744, speed 81.54 f/s
Test done in 9.59 sec, reward 175.442, steps 1442
862724: done 2066 episodes, mean reward -92.183, speed 48.35 f/s
862905: done 2067 episodes, mean reward -92.414, speed 83.47 f/s
Test done in 9.61 sec, reward 200.545, steps 1453
863290: done 2068 episodes, mean reward -92.377, speed 26.74 f/s
863515: done 2069 episodes, mean reward -92.583, speed 83.11 f/s
863889: done 2070 episodes, mean reward -92.448, speed 81.51 f/s
Test done in 9.52 sec, reward 187.610, steps 1395
864351: done 2071 episodes, mean reward -92.178, speed 30.21 f/s
864893: done 2072 episodes, mean reward -92.016, speed 80.48 f/s
Test done in 8.67 sec, reward 184.445, steps 1295
865209: done 2073 episodes, mean reward -92.159, speed 25.24 f/s
865615: done 2074 episodes, mean reward -92.504, speed 81.92 f/s
865888: done 2075 episodes, mean reward -92.510, speed 81.68 f/s
Test done in 7.81 sec, reward 105.227, steps 1154
866002: done 2076 episodes, mean reward -92.998, speed 12.35 f/s
866229: done 2077 episodes, mean reward -93.112, speed 81.61 f/s
866526: done 2078 episodes, mean reward -93.052, speed 81.20 f/s
866833: done 2079 episodes, mean reward -92.858, speed 82.81 f/s
Test done in 7.54 sec, reward 85.211, steps 1100
867182: done 2080 episodes, mean reward -92.596, speed 29.46 f/s
867482: done 2081 episodes, mean reward -93.104, speed 78.20 f/s
867913: done 2082 episodes, mean reward -92.835, speed 80.54 f/s
Test done in 7.31 sec, reward 131.175, steps 1108
868511: done 2083 episodes, mean reward -92.284, speed 40.91 f/s
868748: done 2084 episodes, mean reward -92.754, speed 82.15 f/s
Test done in 9.56 sec, reward 137.151, steps 1416
869085: done 2085 episodes, mean reward -92.672, speed 24.84 f/s
869381: done 2086 episodes, mean reward -93.037, speed 83.95 f/s
869597: done 2087 episodes, mean reward -93.218, speed 82.28 f/s
869917: done 2088 episodes, mean reward -93.436, speed 83.15 f/s
Test done in 9.18 sec, reward 189.952, steps 1363
870233: done 2089 episodes, mean reward -93.583, speed 24.08 f/s
870664: done 2090 episodes, mean reward -93.573, speed 81.48 f/s
870978: done 2091 episodes, mean reward -93.728, speed 83.46 f/s
Test done in 8.05 sec, reward 157.054, steps 1213
871286: done 2092 episodes, mean reward -93.745, speed 25.82 f/s
871384: done 2093 episodes, mean reward -94.107, speed 84.38 f/s
871671: done 2094 episodes, mean reward -94.358, speed 83.62 f/s
Test done in 9.70 sec, reward 199.616, steps 1447
872006: done 2095 episodes, mean reward -93.782, speed 24.14 f/s
872512: done 2096 episodes, mean reward -93.688, speed 80.60 f/s
872599: done 2097 episodes, mean reward -93.861, speed 84.45 f/s
Test done in 7.08 sec, reward 72.700, steps 1015
873048: done 2098 episodes, mean reward -93.465, speed 36.11 f/s
873425: done 2099 episodes, mean reward -94.032, speed 81.02 f/s
873731: done 2100 episodes, mean reward -93.407, speed 84.07 f/s
Test done in 9.05 sec, reward 190.767, steps 1319
874148: done 2101 episodes, mean reward -93.087, speed 29.73 f/s
874462: done 2103 episodes, mean reward -92.904, speed 81.03 f/s
874783: done 2104 episodes, mean reward -92.799, speed 81.77 f/s
874873: done 2105 episodes, mean reward -92.931, speed 79.15 f/s
Test done in 8.91 sec, reward 241.700, steps 1357
Best reward updated: 237.793 -> 241.700
875200: done 2106 episodes, mean reward -92.555, speed 25.33 f/s
875366: done 2107 episodes, mean reward -92.565, speed 82.09 f/s
875649: done 2108 episodes, mean reward -92.552, speed 83.20 f/s
875747: done 2109 episodes, mean reward -93.029, speed 82.27 f/s
875955: done 2110 episodes, mean reward -93.061, speed 80.23 f/s
Test done in 9.09 sec, reward 164.471, steps 1398
876166: done 2111 episodes, mean reward -92.431, speed 18.11 f/s
876268: done 2112 episodes, mean reward -92.805, speed 82.20 f/s
876573: done 2113 episodes, mean reward -92.864, speed 82.02 f/s
Test done in 10.17 sec, reward 258.705, steps 1551
Best reward updated: 241.700 -> 258.705
877031: done 2114 episodes, mean reward -92.276, speed 28.96 f/s
877234: done 2116 episodes, mean reward -92.076, speed 83.41 f/s
877332: done 2117 episodes, mean reward -92.272, speed 79.85 f/s
877470: done 2118 episodes, mean reward -92.148, speed 82.51 f/s
877911: done 2119 episodes, mean reward -91.395, speed 82.73 f/s
Test done in 10.35 sec, reward 235.277, steps 1599
878176: done 2120 episodes, mean reward -90.993, speed 19.54 f/s
878468: done 2121 episodes, mean reward -90.469, speed 82.15 f/s
878667: done 2122 episodes, mean reward -90.458, speed 81.52 f/s
878792: done 2123 episodes, mean reward -90.448, speed 83.27 f/s
Test done in 11.01 sec, reward 236.252, steps 1599
879138: done 2124 episodes, mean reward -89.866, speed 22.62 f/s
879424: done 2125 episodes, mean reward -90.264, speed 81.50 f/s
879551: done 2126 episodes, mean reward -90.246, speed 84.34 f/s
Test done in 9.81 sec, reward 162.834, steps 1423
880212: done 2127 episodes, mean reward -89.121, speed 36.77 f/s
880601: done 2128 episodes, mean reward -88.844, speed 82.95 f/s
Test done in 8.66 sec, reward 190.057, steps 1325
881095: done 2130 episodes, mean reward -88.349, speed 33.54 f/s
881486: done 2131 episodes, mean reward -87.794, speed 79.88 f/s
881891: done 2133 episodes, mean reward -88.426, speed 77.67 f/s
Test done in 7.94 sec, reward 166.743, steps 1149
882123: done 2134 episodes, mean reward -88.163, speed 21.20 f/s
882263: done 2135 episodes, mean reward -87.930, speed 80.43 f/s
882348: done 2136 episodes, mean reward -87.979, speed 82.35 f/s
882595: done 2137 episodes, mean reward -87.916, speed 79.17 f/s
Test done in 9.06 sec, reward 226.313, steps 1321
883163: done 2138 episodes, mean reward -87.129, speed 35.16 f/s
883522: done 2139 episodes, mean reward -86.853, speed 81.08 f/s
883736: done 2140 episodes, mean reward -86.870, speed 79.93 f/s
883946: done 2141 episodes, mean reward -86.843, speed 83.58 f/s
Test done in 9.00 sec, reward 225.624, steps 1363
884202: done 2142 episodes, mean reward -86.591, speed 21.05 f/s
884615: done 2143 episodes, mean reward -85.995, speed 80.12 f/s
Test done in 10.83 sec, reward 242.850, steps 1593
885313: done 2144 episodes, mean reward -85.912, speed 35.65 f/s
885467: done 2145 episodes, mean reward -86.018, speed 82.76 f/s
Test done in 8.12 sec, reward 118.916, steps 1190
886150: done 2146 episodes, mean reward -86.457, speed 41.34 f/s
886394: done 2147 episodes, mean reward -87.098, speed 83.33 f/s
886588: done 2148 episodes, mean reward -87.100, speed 83.47 f/s
Test done in 9.17 sec, reward 217.655, steps 1399
887173: done 2149 episodes, mean reward -86.221, speed 35.78 f/s
Test done in 9.27 sec, reward 231.473, steps 1398
888232: done 2150 episodes, mean reward -85.593, speed 47.32 f/s
888858: done 2151 episodes, mean reward -85.093, speed 84.01 f/s
Test done in 9.06 sec, reward 215.735, steps 1340
Test done in 10.07 sec, reward 209.074, steps 1507
890458: done 2152 episodes, mean reward -84.689, speed 41.26 f/s
890637: done 2153 episodes, mean reward -84.551, speed 81.45 f/s
Test done in 10.74 sec, reward 53.918, steps 1566
891077: done 2154 episodes, mean reward -84.246, speed 27.35 f/s
891364: done 2155 episodes, mean reward -84.094, speed 80.66 f/s
891842: done 2156 episodes, mean reward -84.003, speed 81.53 f/s
Test done in 10.13 sec, reward -57.735, steps 1495
Test done in 10.05 sec, reward 122.553, steps 1401
893442: done 2157 episodes, mean reward -83.294, speed 40.10 f/s
Test done in 9.36 sec, reward 201.635, steps 1383
894028: done 2158 episodes, mean reward -83.224, speed 35.47 f/s
894333: done 2159 episodes, mean reward -83.311, speed 81.91 f/s
Test done in 11.10 sec, reward 21.865, steps 1600
895933: done 2160 episodes, mean reward -82.654, speed 52.42 f/s
Test done in 11.57 sec, reward -34.626, steps 1600
896397: done 2161 episodes, mean reward -82.495, speed 26.72 f/s
Test done in 11.04 sec, reward 97.569, steps 1599
897146: done 2162 episodes, mean reward -83.051, speed 36.85 f/s
897988: done 2163 episodes, mean reward -82.653, speed 82.43 f/s
Test done in 10.42 sec, reward 62.221, steps 1504
898908: done 2164 episodes, mean reward -82.116, speed 41.60 f/s
Test done in 7.95 sec, reward 52.234, steps 1160
899449: done 2165 episodes, mean reward -81.675, speed 37.14 f/s
Test done in 11.20 sec, reward 23.071, steps 1600
900336: done 2166 episodes, mean reward -81.552, speed 40.49 f/s
Test done in 10.82 sec, reward 12.679, steps 1595
901936: done 2167 episodes, mean reward -80.079, speed 52.36 f/s
Test done in 11.28 sec, reward 39.490, steps 1600
Test done in 7.78 sec, reward -11.707, steps 1160
903323: done 2168 episodes, mean reward -79.873, speed 38.65 f/s
Test done in 11.05 sec, reward 72.662, steps 1600
904923: done 2169 episodes, mean reward -78.078, speed 52.24 f/s
Test done in 10.87 sec, reward 54.049, steps 1529
905135: done 2170 episodes, mean reward -78.136, speed 15.66 f/s
Test done in 10.74 sec, reward 90.030, steps 1600
906735: done 2171 episodes, mean reward -76.904, speed 51.75 f/s
Test done in 11.47 sec, reward 90.810, steps 1600
Test done in 11.09 sec, reward 97.559, steps 1600
908335: done 2172 episodes, mean reward -75.773, speed 37.94 f/s
Test done in 11.08 sec, reward 95.864, steps 1600
909935: done 2173 episodes, mean reward -74.070, speed 52.69 f/s
Test done in 10.73 sec, reward 106.910, steps 1600
Test done in 11.15 sec, reward 93.930, steps 1600
911535: done 2174 episodes, mean reward -72.451, speed 38.81 f/s
Test done in 10.90 sec, reward 114.091, steps 1600
Test done in 10.71 sec, reward 120.863, steps 1600
913135: done 2175 episodes, mean reward -70.682, speed 39.09 f/s
Test done in 11.08 sec, reward 96.412, steps 1600
914735: done 2176 episodes, mean reward -68.746, speed 51.92 f/s
Test done in 10.56 sec, reward 90.263, steps 1600
Test done in 11.26 sec, reward 86.582, steps 1600
916335: done 2177 episodes, mean reward -66.986, speed 39.07 f/s
Test done in 10.92 sec, reward 90.908, steps 1600
917935: done 2178 episodes, mean reward -65.293, speed 52.94 f/s
Test done in 11.00 sec, reward 82.987, steps 1600
Test done in 11.40 sec, reward 96.224, steps 1600
919535: done 2179 episodes, mean reward -63.529, speed 38.33 f/s
Test done in 10.81 sec, reward 95.332, steps 1600
Test done in 11.06 sec, reward 99.381, steps 1600
921135: done 2180 episodes, mean reward -62.152, speed 38.64 f/s
Test done in 10.98 sec, reward 91.811, steps 1600
922735: done 2181 episodes, mean reward -60.364, speed 52.44 f/s
Test done in 11.04 sec, reward 88.935, steps 1600
Test done in 11.29 sec, reward 114.845, steps 1600
924335: done 2182 episodes, mean reward -58.799, speed 38.50 f/s
Test done in 10.77 sec, reward 127.100, steps 1600
925935: done 2183 episodes, mean reward -57.499, speed 53.05 f/s
Test done in 10.92 sec, reward 120.633, steps 1600
Test done in 11.16 sec, reward 108.598, steps 1600
927535: done 2184 episodes, mean reward -55.637, speed 38.79 f/s
Test done in 10.72 sec, reward 117.885, steps 1600
Test done in 11.13 sec, reward 101.604, steps 1600
929135: done 2185 episodes, mean reward -54.097, speed 38.65 f/s
Test done in 7.82 sec, reward 58.607, steps 1133
930735: done 2186 episodes, mean reward -52.330, speed 59.60 f/s
Test done in 1.91 sec, reward -99.807, steps 285
931044: done 2187 episodes, mean reward -52.232, speed 56.19 f/s
931275: done 2188 episodes, mean reward -52.542, speed 82.73 f/s
931439: done 2189 episodes, mean reward -52.266, speed 84.59 f/s
931685: done 2190 episodes, mean reward -52.342, speed 84.10 f/s
931878: done 2191 episodes, mean reward -52.581, speed 81.73 f/s
Test done in 4.64 sec, reward -24.387, steps 699
932170: done 2192 episodes, mean reward -52.503, speed 35.94 f/s
932418: done 2193 episodes, mean reward -52.468, speed 82.87 f/s
932678: done 2194 episodes, mean reward -52.530, speed 84.71 f/s
932984: done 2195 episodes, mean reward -52.566, speed 82.18 f/s
Test done in 6.74 sec, reward 38.852, steps 980
933312: done 2196 episodes, mean reward -53.048, speed 30.34 f/s
933534: done 2197 episodes, mean reward -52.723, speed 82.07 f/s
933673: done 2198 episodes, mean reward -53.120, speed 78.54 f/s
Test done in 5.03 sec, reward 14.339, steps 750
934073: done 2199 episodes, mean reward -53.401, speed 40.68 f/s
934239: done 2200 episodes, mean reward -53.527, speed 83.18 f/s
934484: done 2201 episodes, mean reward -53.745, speed 83.88 f/s
934806: done 2202 episodes, mean reward -53.318, speed 83.63 f/s
Test done in 9.46 sec, reward 217.421, steps 1429
935789: done 2203 episodes, mean reward -53.137, speed 46.53 f/s
Test done in 10.87 sec, reward 130.120, steps 1568
936005: done 2204 episodes, mean reward -53.241, speed 16.02 f/s
Test done in 10.35 sec, reward 122.841, steps 1505
937345: done 2205 episodes, mean reward -52.676, speed 49.94 f/s
937520: done 2206 episodes, mean reward -52.880, speed 75.35 f/s
Test done in 10.87 sec, reward 144.474, steps 1500
Test done in 6.96 sec, reward 31.382, steps 958
939120: done 2207 episodes, mean reward -52.324, speed 42.27 f/s
939323: done 2208 episodes, mean reward -52.718, speed 83.50 f/s
939625: done 2209 episodes, mean reward -52.478, speed 81.58 f/s
939813: done 2210 episodes, mean reward -52.277, speed 81.31 f/s
Test done in 7.85 sec, reward 135.048, steps 1188
940228: done 2211 episodes, mean reward -52.190, speed 32.54 f/s
940514: done 2212 episodes, mean reward -51.893, speed 86.27 f/s
940759: done 2213 episodes, mean reward -51.889, speed 83.06 f/s
Test done in 9.92 sec, reward 220.235, steps 1528
941131: done 2214 episodes, mean reward -52.423, speed 25.87 f/s
Test done in 7.86 sec, reward 113.486, steps 1116
942252: done 2215 episodes, mean reward -51.744, speed 52.41 f/s
Test done in 3.87 sec, reward -83.337, steps 585
943039: done 2216 episodes, mean reward -51.710, speed 58.84 f/s
943511: done 2217 episodes, mean reward -51.594, speed 83.44 f/s
943816: done 2218 episodes, mean reward -51.583, speed 81.71 f/s
Test done in 10.07 sec, reward 174.108, steps 1453
944320: done 2219 episodes, mean reward -52.104, speed 31.16 f/s
944509: done 2220 episodes, mean reward -52.232, speed 84.84 f/s
944730: done 2221 episodes, mean reward -52.345, speed 83.44 f/s
Test done in 10.08 sec, reward 227.191, steps 1483
945957: done 2222 episodes, mean reward -51.717, speed 47.75 f/s
Test done in 9.27 sec, reward 202.976, steps 1339
946120: done 2223 episodes, mean reward -51.686, speed 14.40 f/s
946393: done 2224 episodes, mean reward -51.974, speed 79.17 f/s
946579: done 2225 episodes, mean reward -51.937, speed 80.13 f/s
Test done in 9.63 sec, reward 259.243, steps 1405
Best reward updated: 258.705 -> 259.243
947052: done 2226 episodes, mean reward -51.668, speed 30.16 f/s
947421: done 2227 episodes, mean reward -52.347, speed 49.88 f/s
947695: done 2228 episodes, mean reward -52.589, speed 62.18 f/s
Test done in 11.73 sec, reward 243.719, steps 1551
948070: done 2229 episodes, mean reward -52.153, speed 22.10 f/s
948439: done 2230 episodes, mean reward -52.701, speed 73.01 f/s
948698: done 2231 episodes, mean reward -53.051, speed 72.21 f/s
948953: done 2232 episodes, mean reward -52.885, speed 67.78 f/s
Test done in 10.68 sec, reward 183.305, steps 1324
949071: done 2233 episodes, mean reward -52.883, speed 9.51 f/s
949328: done 2234 episodes, mean reward -52.953, speed 70.09 f/s
949883: done 2235 episodes, mean reward -52.575, speed 69.43 f/s
Test done in 10.97 sec, reward 256.848, steps 1390
950246: done 2236 episodes, mean reward -52.406, speed 22.53 f/s
950507: done 2237 episodes, mean reward -52.647, speed 73.09 f/s
950845: done 2238 episodes, mean reward -53.199, speed 71.70 f/s
Test done in 11.84 sec, reward 251.787, steps 1558
951176: done 2239 episodes, mean reward -53.240, speed 20.22 f/s
951491: done 2240 episodes, mean reward -53.291, speed 75.94 f/s
951637: done 2241 episodes, mean reward -53.285, speed 75.76 f/s
951995: done 2242 episodes, mean reward -52.967, speed 80.60 f/s
Test done in 9.71 sec, reward 242.278, steps 1439
952255: done 2243 episodes, mean reward -53.384, speed 20.07 f/s
952430: done 2244 episodes, mean reward -54.128, speed 79.10 f/s
952626: done 2245 episodes, mean reward -54.111, speed 74.88 f/s
952706: done 2246 episodes, mean reward -53.886, speed 75.45 f/s
952958: done 2247 episodes, mean reward -53.557, speed 82.32 f/s
Test done in 9.94 sec, reward 253.984, steps 1481
953104: done 2248 episodes, mean reward -53.603, speed 12.41 f/s
953286: done 2249 episodes, mean reward -54.175, speed 79.92 f/s
953687: done 2250 episodes, mean reward -53.963, speed 80.76 f/s
953882: done 2251 episodes, mean reward -54.202, speed 77.46 f/s
Test done in 9.51 sec, reward 218.728, steps 1309
954363: done 2252 episodes, mean reward -54.002, speed 29.89 f/s
954656: done 2253 episodes, mean reward -53.803, speed 76.45 f/s
954786: done 2254 episodes, mean reward -54.474, speed 79.31 f/s
954903: done 2255 episodes, mean reward -54.578, speed 79.85 f/s
Test done in 7.77 sec, reward 167.724, steps 1095
955051: done 2256 episodes, mean reward -54.459, speed 15.30 f/s
955229: done 2257 episodes, mean reward -54.761, speed 80.49 f/s
955473: done 2258 episodes, mean reward -54.689, speed 79.27 f/s
955653: done 2259 episodes, mean reward -54.552, speed 79.04 f/s
955862: done 2260 episodes, mean reward -55.073, speed 77.48 f/s
Test done in 7.38 sec, reward 144.633, steps 1025
956054: done 2261 episodes, mean reward -55.224, speed 19.67 f/s
956184: done 2262 episodes, mean reward -55.453, speed 75.17 f/s
956286: done 2263 episodes, mean reward -56.320, speed 78.99 f/s
956440: done 2264 episodes, mean reward -56.503, speed 79.60 f/s
956645: done 2265 episodes, mean reward -56.772, speed 77.60 f/s
956803: done 2266 episodes, mean reward -57.602, speed 76.10 f/s
956913: done 2267 episodes, mean reward -59.072, speed 54.83 f/s
Test done in 9.22 sec, reward 187.121, steps 1201
957018: done 2268 episodes, mean reward -59.624, speed 9.93 f/s
957158: done 2269 episodes, mean reward -61.324, speed 70.49 f/s
957238: done 2270 episodes, mean reward -61.650, speed 79.40 f/s
957570: done 2271 episodes, mean reward -63.337, speed 72.03 f/s
957737: done 2272 episodes, mean reward -65.122, speed 76.81 f/s
Test done in 10.36 sec, reward 197.301, steps 1400
958012: done 2273 episodes, mean reward -66.600, speed 19.99 f/s
958277: done 2274 episodes, mean reward -68.068, speed 81.77 f/s
958470: done 2275 episodes, mean reward -69.893, speed 80.61 f/s
958593: done 2276 episodes, mean reward -71.859, speed 80.15 f/s
958717: done 2277 episodes, mean reward -73.685, speed 81.05 f/s
958800: done 2278 episodes, mean reward -75.732, speed 77.29 f/s
958938: done 2279 episodes, mean reward -77.659, speed 81.38 f/s
Test done in 7.99 sec, reward 157.125, steps 1138
959147: done 2280 episodes, mean reward -79.102, speed 19.82 f/s
959356: done 2281 episodes, mean reward -80.844, speed 84.37 f/s
959588: done 2282 episodes, mean reward -82.600, speed 82.83 f/s
959724: done 2283 episodes, mean reward -84.572, speed 81.98 f/s
959860: done 2284 episodes, mean reward -86.406, speed 83.28 f/s
Test done in 7.51 sec, reward 130.026, steps 1077
960043: done 2285 episodes, mean reward -88.070, speed 18.71 f/s
960222: done 2286 episodes, mean reward -90.017, speed 83.55 f/s
960390: done 2287 episodes, mean reward -90.193, speed 77.85 f/s
960530: done 2288 episodes, mean reward -90.121, speed 81.73 f/s
960723: done 2289 episodes, mean reward -90.256, speed 78.56 f/s
960890: done 2290 episodes, mean reward -90.451, speed 77.22 f/s
Test done in 8.60 sec, reward 191.479, steps 1260
961396: done 2291 episodes, mean reward -89.628, speed 34.61 f/s
Test done in 9.93 sec, reward 248.015, steps 1449
962083: done 2292 episodes, mean reward -88.825, speed 37.65 f/s
962368: done 2293 episodes, mean reward -88.357, speed 81.86 f/s
962725: done 2294 episodes, mean reward -88.119, speed 83.63 f/s
962930: done 2295 episodes, mean reward -88.485, speed 85.06 f/s
Test done in 9.74 sec, reward 234.450, steps 1347
963089: done 2296 episodes, mean reward -88.701, speed 13.24 f/s
963274: done 2297 episodes, mean reward -88.909, speed 70.51 f/s
963514: done 2298 episodes, mean reward -88.793, speed 71.94 f/s
963652: done 2299 episodes, mean reward -88.889, speed 79.77 f/s
963799: done 2300 episodes, mean reward -89.118, speed 74.59 f/s
Test done in 7.76 sec, reward 188.370, steps 1112
964046: done 2301 episodes, mean reward -89.206, speed 22.26 f/s
964184: done 2302 episodes, mean reward -89.635, speed 72.74 f/s
964385: done 2303 episodes, mean reward -89.987, speed 76.39 f/s
964608: done 2304 episodes, mean reward -90.081, speed 81.11 f/s
964781: done 2305 episodes, mean reward -90.479, speed 83.11 f/s
Test done in 8.70 sec, reward 205.022, steps 1273
965196: done 2306 episodes, mean reward -90.069, speed 29.35 f/s
965360: done 2307 episodes, mean reward -90.811, speed 74.04 f/s
965742: done 2308 episodes, mean reward -90.187, speed 77.43 f/s
965885: done 2309 episodes, mean reward -90.460, speed 80.63 f/s
Test done in 7.73 sec, reward 156.323, steps 1123
966041: done 2310 episodes, mean reward -90.662, speed 16.26 f/s
966222: done 2311 episodes, mean reward -91.011, speed 85.15 f/s
966470: done 2312 episodes, mean reward -91.082, speed 84.05 f/s
966617: done 2313 episodes, mean reward -91.307, speed 77.26 f/s
966735: done 2314 episodes, mean reward -91.473, speed 83.19 f/s
966847: done 2315 episodes, mean reward -92.142, speed 83.18 f/s
Test done in 9.55 sec, reward 216.155, steps 1327
967041: done 2316 episodes, mean reward -91.983, speed 16.15 f/s
967365: done 2317 episodes, mean reward -91.750, speed 80.32 f/s
967596: done 2318 episodes, mean reward -91.707, speed 78.87 f/s
967830: done 2319 episodes, mean reward -91.522, speed 80.95 f/s
967972: done 2320 episodes, mean reward -91.888, speed 79.18 f/s
Test done in 6.63 sec, reward 95.907, steps 955
968445: done 2321 episodes, mean reward -91.578, speed 36.86 f/s
968653: done 2322 episodes, mean reward -92.092, speed 78.96 f/s
968849: done 2323 episodes, mean reward -91.971, speed 76.46 f/s
968986: done 2324 episodes, mean reward -92.244, speed 69.72 f/s
Test done in 9.55 sec, reward 217.403, steps 1392
969245: done 2325 episodes, mean reward -92.150, speed 20.17 f/s
969503: done 2326 episodes, mean reward -92.215, speed 73.25 f/s
969595: done 2327 episodes, mean reward -92.641, speed 77.85 f/s
969759: done 2328 episodes, mean reward -92.701, speed 80.03 f/s
Test done in 9.95 sec, reward 239.315, steps 1458
970005: done 2329 episodes, mean reward -93.024, speed 18.84 f/s
970196: done 2330 episodes, mean reward -92.956, speed 85.84 f/s
970519: done 2331 episodes, mean reward -92.692, speed 83.95 f/s
970709: done 2332 episodes, mean reward -92.922, speed 84.85 f/s
Test done in 10.64 sec, reward 261.364, steps 1506
Best reward updated: 259.243 -> 261.364
971419: done 2333 episodes, mean reward -91.800, speed 36.42 f/s
971699: done 2334 episodes, mean reward -91.967, speed 70.37 f/s
971836: done 2335 episodes, mean reward -92.553, speed 77.06 f/s
Test done in 5.93 sec, reward 97.635, steps 866
972041: done 2336 episodes, mean reward -92.502, speed 23.78 f/s
972300: done 2337 episodes, mean reward -92.461, speed 84.31 f/s
972704: done 2338 episodes, mean reward -92.500, speed 85.08 f/s
972839: done 2339 episodes, mean reward -92.974, speed 78.32 f/s
972952: done 2340 episodes, mean reward -93.299, speed 82.85 f/s
Test done in 9.15 sec, reward 236.686, steps 1367
973378: done 2341 episodes, mean reward -93.011, speed 30.01 f/s
973524: done 2342 episodes, mean reward -93.459, speed 83.01 f/s
973631: done 2343 episodes, mean reward -93.640, speed 85.83 f/s
973916: done 2344 episodes, mean reward -93.579, speed 84.51 f/s
Test done in 8.54 sec, reward 204.674, steps 1274
974018: done 2345 episodes, mean reward -93.697, speed 10.44 f/s
974196: done 2346 episodes, mean reward -93.498, speed 79.80 f/s
974410: done 2347 episodes, mean reward -93.657, speed 79.35 f/s
974793: done 2348 episodes, mean reward -93.275, speed 75.75 f/s
Test done in 8.17 sec, reward 159.934, steps 1157
975001: done 2349 episodes, mean reward -93.291, speed 19.43 f/s
975166: done 2350 episodes, mean reward -93.723, speed 78.73 f/s
975319: done 2351 episodes, mean reward -93.855, speed 85.35 f/s
975457: done 2352 episodes, mean reward -94.448, speed 79.93 f/s
975810: done 2353 episodes, mean reward -94.357, speed 84.18 f/s
975946: done 2354 episodes, mean reward -94.327, speed 82.27 f/s
Test done in 8.74 sec, reward 224.487, steps 1302
976135: done 2355 episodes, mean reward -94.371, speed 17.12 f/s
976271: done 2356 episodes, mean reward -94.406, speed 81.21 f/s
976499: done 2357 episodes, mean reward -94.359, speed 84.69 f/s
976589: done 2358 episodes, mean reward -94.630, speed 87.24 f/s
976785: done 2359 episodes, mean reward -94.482, speed 86.10 f/s
976899: done 2360 episodes, mean reward -94.738, speed 84.77 f/s
Test done in 9.13 sec, reward 242.468, steps 1358
977142: done 2361 episodes, mean reward -94.493, speed 20.17 f/s
977301: done 2362 episodes, mean reward -94.501, speed 78.60 f/s
977506: done 2363 episodes, mean reward -94.436, speed 68.54 f/s
977819: done 2364 episodes, mean reward -94.080, speed 58.53 f/s
Test done in 9.22 sec, reward 232.324, steps 1384
978077: done 2365 episodes, mean reward -93.972, speed 20.21 f/s
978282: done 2366 episodes, mean reward -93.855, speed 81.38 f/s
978507: done 2367 episodes, mean reward -93.633, speed 81.38 f/s
Test done in 9.69 sec, reward 259.690, steps 1408
979078: done 2368 episodes, mean reward -92.797, speed 34.55 f/s
979292: done 2369 episodes, mean reward -92.758, speed 80.64 f/s
979423: done 2371 episodes, mean reward -92.832, speed 84.72 f/s
979707: done 2372 episodes, mean reward -92.367, speed 81.65 f/s
979982: done 2373 episodes, mean reward -92.316, speed 78.47 f/s
Test done in 9.84 sec, reward 266.288, steps 1419
Best reward updated: 261.364 -> 266.288
980286: done 2374 episodes, mean reward -92.118, speed 21.69 f/s
980424: done 2375 episodes, mean reward -92.109, speed 75.90 f/s
980710: done 2376 episodes, mean reward -91.717, speed 78.36 f/s
980830: done 2377 episodes, mean reward -91.701, speed 78.77 f/s
980991: done 2378 episodes, mean reward -91.566, speed 77.49 f/s
Test done in 10.11 sec, reward 247.226, steps 1431
981061: done 2379 episodes, mean reward -91.719, speed 6.34 f/s
981341: done 2380 episodes, mean reward -91.635, speed 77.96 f/s
981449: done 2381 episodes, mean reward -91.734, speed 77.34 f/s
981650: done 2382 episodes, mean reward -91.856, speed 76.14 f/s
981923: done 2383 episodes, mean reward -91.585, speed 77.96 f/s
Test done in 10.31 sec, reward 261.736, steps 1346
982065: done 2384 episodes, mean reward -91.608, speed 11.72 f/s
982336: done 2385 episodes, mean reward -91.575, speed 79.09 f/s
982587: done 2386 episodes, mean reward -91.430, speed 82.33 f/s
982825: done 2387 episodes, mean reward -91.318, speed 78.49 f/s
Test done in 10.65 sec, reward 239.588, steps 1429
983077: done 2388 episodes, mean reward -91.101, speed 17.83 f/s
983344: done 2389 episodes, mean reward -90.911, speed 78.30 f/s
983502: done 2390 episodes, mean reward -91.002, speed 79.11 f/s
Test done in 9.56 sec, reward 241.095, steps 1368
984174: done 2391 episodes, mean reward -90.810, speed 37.66 f/s
984381: done 2392 episodes, mean reward -91.579, speed 79.37 f/s
984559: done 2393 episodes, mean reward -91.800, speed 80.22 f/s
984752: done 2394 episodes, mean reward -91.941, speed 78.88 f/s
984865: done 2395 episodes, mean reward -91.978, speed 79.17 f/s
Test done in 9.64 sec, reward 239.118, steps 1357
985058: done 2396 episodes, mean reward -91.766, speed 16.01 f/s
985601: done 2397 episodes, mean reward -91.003, speed 78.24 f/s
985689: done 2398 episodes, mean reward -91.223, speed 74.32 f/s
985882: done 2399 episodes, mean reward -90.910, speed 73.41 f/s
Test done in 11.17 sec, reward 256.506, steps 1544
986137: done 2400 episodes, mean reward -90.636, speed 17.37 f/s
986298: done 2401 episodes, mean reward -90.673, speed 86.93 f/s
986435: done 2402 episodes, mean reward -90.674, speed 78.84 f/s
986556: done 2403 episodes, mean reward -90.722, speed 84.66 f/s
986915: done 2405 episodes, mean reward -90.621, speed 83.85 f/s
Test done in 7.81 sec, reward 179.601, steps 1134
987112: done 2406 episodes, mean reward -90.954, speed 19.28 f/s
987310: done 2407 episodes, mean reward -90.813, speed 85.87 f/s
987524: done 2408 episodes, mean reward -91.211, speed 78.43 f/s
987772: done 2409 episodes, mean reward -90.783, speed 75.80 f/s
987917: done 2410 episodes, mean reward -90.738, speed 76.80 f/s
987998: done 2411 episodes, mean reward -90.711, speed 79.96 f/s
Test done in 9.74 sec, reward 245.465, steps 1413
988280: done 2412 episodes, mean reward -90.650, speed 21.52 f/s
988645: done 2413 episodes, mean reward -90.189, speed 76.21 f/s
988879: done 2414 episodes, mean reward -89.857, speed 71.98 f/s
Test done in 11.61 sec, reward 262.028, steps 1491
989189: done 2415 episodes, mean reward -89.401, speed 19.64 f/s
989424: done 2416 episodes, mean reward -89.434, speed 73.24 f/s
989696: done 2417 episodes, mean reward -89.333, speed 74.65 f/s
989845: done 2418 episodes, mean reward -89.331, speed 72.72 f/s
Test done in 10.06 sec, reward 248.166, steps 1419
990062: done 2419 episodes, mean reward -89.239, speed 16.83 f/s
990150: done 2420 episodes, mean reward -89.265, speed 74.24 f/s
990404: done 2421 episodes, mean reward -89.544, speed 81.72 f/s
990927: done 2422 episodes, mean reward -88.793, speed 79.76 f/s
Test done in 9.74 sec, reward 262.440, steps 1407
991081: done 2423 episodes, mean reward -88.775, speed 13.12 f/s
991353: done 2424 episodes, mean reward -88.387, speed 83.55 f/s
991722: done 2426 episodes, mean reward -88.438, speed 79.94 f/s
991866: done 2427 episodes, mean reward -88.243, speed 81.16 f/s
991984: done 2428 episodes, mean reward -88.190, speed 82.05 f/s
Test done in 9.39 sec, reward 262.128, steps 1366
992299: done 2429 episodes, mean reward -87.891, speed 23.82 f/s
992524: done 2430 episodes, mean reward -87.782, speed 78.78 f/s
992755: done 2431 episodes, mean reward -87.979, speed 79.22 f/s
992999: done 2432 episodes, mean reward -87.732, speed 73.40 f/s
Test done in 10.16 sec, reward 264.216, steps 1391
993189: done 2433 episodes, mean reward -88.718, speed 14.80 f/s
993673: done 2434 episodes, mean reward -88.095, speed 70.68 f/s
993830: done 2435 episodes, mean reward -87.883, speed 77.27 f/s
993978: done 2436 episodes, mean reward -87.851, speed 78.90 f/s
Test done in 8.25 sec, reward 202.827, steps 1198
994282: done 2437 episodes, mean reward -87.477, speed 25.17 f/s
994437: done 2438 episodes, mean reward -87.491, speed 79.27 f/s
994517: done 2439 episodes, mean reward -87.519, speed 75.02 f/s
994852: done 2440 episodes, mean reward -86.812, speed 72.21 f/s
Test done in 9.08 sec, reward 179.615, steps 1201
995188: done 2441 episodes, mean reward -86.690, speed 24.18 f/s
995316: done 2442 episodes, mean reward -86.698, speed 68.40 f/s
995464: done 2443 episodes, mean reward -86.481, speed 78.29 f/s
995635: done 2444 episodes, mean reward -86.572, speed 75.12 f/s
995849: done 2446 episodes, mean reward -86.394, speed 74.19 f/s
Test done in 6.22 sec, reward 67.642, steps 807
996092: done 2447 episodes, mean reward -86.168, speed 25.75 f/s
996250: done 2448 episodes, mean reward -86.424, speed 74.34 f/s
996355: done 2449 episodes, mean reward -86.467, speed 78.94 f/s
996578: done 2450 episodes, mean reward -86.138, speed 76.08 f/s
996807: done 2451 episodes, mean reward -85.919, speed 73.75 f/s
996967: done 2452 episodes, mean reward -85.704, speed 75.58 f/s
Test done in 8.32 sec, reward 175.945, steps 1130
997151: done 2453 episodes, mean reward -85.992, speed 17.05 f/s
997333: done 2454 episodes, mean reward -85.798, speed 68.98 f/s
997557: done 2455 episodes, mean reward -85.347, speed 70.58 f/s
997631: done 2456 episodes, mean reward -85.413, speed 66.61 f/s
997740: done 2457 episodes, mean reward -85.506, speed 68.61 f/s
997929: done 2458 episodes, mean reward -85.313, speed 68.80 f/s
Test done in 9.31 sec, reward 237.941, steps 1263
998061: done 2459 episodes, mean reward -85.373, speed 11.91 f/s
998214: done 2461 episodes, mean reward -85.607, speed 77.58 f/s
998391: done 2462 episodes, mean reward -85.507, speed 79.66 f/s
998544: done 2463 episodes, mean reward -85.517, speed 79.12 f/s
998652: done 2464 episodes, mean reward -85.801, speed 80.56 f/s
998913: done 2465 episodes, mean reward -85.687, speed 74.06 f/s
Test done in 10.05 sec, reward 262.415, steps 1361
999105: done 2466 episodes, mean reward -85.633, speed 15.35 f/s
999273: done 2467 episodes, mean reward -85.569, speed 77.76 f/s
999437: done 2468 episodes, mean reward -86.209, speed 75.04 f/s
999579: done 2469 episodes, mean reward -86.097, speed 75.79 f/s
999882: done 2470 episodes, mean reward -85.690, speed 81.27 f/s
Test done in 9.84 sec, reward 259.535, steps 1421
1000063: done 2471 episodes, mean reward -85.391, speed 14.97 f/s
1000222: done 2472 episodes, mean reward -85.753, speed 81.35 f/s
1000535: done 2473 episodes, mean reward -85.597, speed 74.68 f/s
1000664: done 2474 episodes, mean reward -85.985, speed 78.06 f/s
Test done in 10.06 sec, reward 259.546, steps 1422
1001035: done 2475 episodes, mean reward -85.526, speed 24.97 f/s
1001195: done 2477 episodes, mean reward -85.775, speed 77.48 f/s
1001433: done 2478 episodes, mean reward -85.569, speed 78.01 f/s
1001599: done 2479 episodes, mean reward -85.349, speed 78.08 f/s
1001759: done 2480 episodes, mean reward -85.567, speed 77.95 f/s
Test done in 9.46 sec, reward 230.906, steps 1310
1002092: done 2482 episodes, mean reward -85.427, speed 23.91 f/s
1002295: done 2483 episodes, mean reward -85.523, speed 78.95 f/s
1002482: done 2484 episodes, mean reward -85.340, speed 81.30 f/s
1002718: done 2485 episodes, mean reward -85.311, speed 78.76 f/s
Test done in 9.85 sec, reward 251.467, steps 1343
1003007: done 2486 episodes, mean reward -85.017, speed 21.35 f/s
1003262: done 2487 episodes, mean reward -84.789, speed 78.12 f/s
1003519: done 2488 episodes, mean reward -84.696, speed 77.27 f/s
1003888: done 2489 episodes, mean reward -84.393, speed 75.02 f/s
1003990: done 2490 episodes, mean reward -84.407, speed 74.04 f/s
Test done in 9.70 sec, reward 263.519, steps 1335
1004108: done 2491 episodes, mean reward -85.227, speed 10.57 f/s
1004378: done 2492 episodes, mean reward -85.158, speed 76.36 f/s
1004595: done 2493 episodes, mean reward -85.056, speed 77.19 f/s
1004728: done 2494 episodes, mean reward -85.197, speed 78.77 f/s
1004945: done 2495 episodes, mean reward -84.912, speed 76.48 f/s
Test done in 11.25 sec, reward 264.984, steps 1417
1005210: done 2496 episodes, mean reward -84.810, speed 17.78 f/s
1005317: done 2497 episodes, mean reward -85.544, speed 61.33 f/s
1005418: done 2498 episodes, mean reward -85.494, speed 71.39 f/s
1005726: done 2499 episodes, mean reward -85.371, speed 76.47 f/s
1005837: done 2500 episodes, mean reward -85.570, speed 62.98 f/s
Test done in 9.68 sec, reward 235.503, steps 1282
1006314: done 2501 episodes, mean reward -84.972, speed 28.72 f/s
1006701: done 2502 episodes, mean reward -84.423, speed 78.90 f/s
1006980: done 2503 episodes, mean reward -84.087, speed 77.45 f/s
Test done in 9.66 sec, reward 247.445, steps 1350
1007463: done 2504 episodes, mean reward -83.239, speed 30.48 f/s
1007639: done 2505 episodes, mean reward -83.345, speed 77.66 f/s
1007779: done 2506 episodes, mean reward -83.575, speed 71.30 f/s
Test done in 9.47 sec, reward 259.155, steps 1372
1008231: done 2507 episodes, mean reward -82.872, speed 29.44 f/s
1008405: done 2508 episodes, mean reward -82.912, speed 81.44 f/s
1008509: done 2509 episodes, mean reward -83.289, speed 80.80 f/s
1008611: done 2510 episodes, mean reward -83.380, speed 81.74 f/s
1008717: done 2511 episodes, mean reward -83.294, speed 78.58 f/s
1008940: done 2512 episodes, mean reward -83.265, speed 80.40 f/s
Test done in 9.56 sec, reward 237.615, steps 1353
1009022: done 2513 episodes, mean reward -83.741, speed 7.74 f/s
1009206: done 2514 episodes, mean reward -83.702, speed 72.62 f/s
1009400: done 2515 episodes, mean reward -83.958, speed 68.93 f/s
1009553: done 2516 episodes, mean reward -83.960, speed 64.49 f/s
1009705: done 2517 episodes, mean reward -84.341, speed 69.03 f/s
1009968: done 2518 episodes, mean reward -84.238, speed 77.92 f/s
Test done in 10.17 sec, reward 256.967, steps 1429
1010228: done 2519 episodes, mean reward -84.090, speed 19.28 f/s
1010582: done 2520 episodes, mean reward -83.487, speed 78.44 f/s
1010977: done 2521 episodes, mean reward -83.257, speed 81.92 f/s
Test done in 7.95 sec, reward 169.599, steps 1091
1011351: done 2522 episodes, mean reward -83.593, speed 28.81 f/s
1011768: done 2523 episodes, mean reward -83.184, speed 79.44 f/s
1011873: done 2524 episodes, mean reward -83.446, speed 82.11 f/s
1011981: done 2525 episodes, mean reward -83.393, speed 78.43 f/s
Test done in 11.63 sec, reward 250.379, steps 1485
1012151: done 2526 episodes, mean reward -83.710, speed 12.20 f/s
1012490: done 2527 episodes, mean reward -83.246, speed 70.24 f/s
1012628: done 2528 episodes, mean reward -83.284, speed 71.66 f/s
1012861: done 2529 episodes, mean reward -83.451, speed 73.86 f/s
Test done in 14.22 sec, reward 249.945, steps 1519
1013049: done 2530 episodes, mean reward -83.485, speed 11.13 f/s
1013212: done 2531 episodes, mean reward -83.535, speed 73.98 f/s
1013488: done 2532 episodes, mean reward -83.355, speed 78.07 f/s
1013648: done 2533 episodes, mean reward -83.380, speed 77.19 f/s
1013878: done 2534 episodes, mean reward -83.922, speed 78.23 f/s
Test done in 9.98 sec, reward 218.988, steps 1339
1014013: done 2535 episodes, mean reward -83.993, speed 11.34 f/s
1014159: done 2536 episodes, mean reward -84.089, speed 78.86 f/s
1014244: done 2537 episodes, mean reward -84.635, speed 76.51 f/s
1014713: done 2538 episodes, mean reward -84.187, speed 75.37 f/s
1014841: done 2539 episodes, mean reward -84.089, speed 82.00 f/s
Test done in 10.64 sec, reward 247.523, steps 1480
1015067: done 2540 episodes, mean reward -84.370, speed 16.70 f/s
1015303: done 2541 episodes, mean reward -84.588, speed 83.30 f/s
1015738: done 2542 episodes, mean reward -84.025, speed 80.52 f/s
Test done in 10.21 sec, reward 238.506, steps 1397
1016064: done 2543 episodes, mean reward -83.850, speed 22.69 f/s
1016207: done 2544 episodes, mean reward -83.911, speed 81.96 f/s
1016391: done 2545 episodes, mean reward -83.911, speed 82.61 f/s
1016682: done 2546 episodes, mean reward -83.820, speed 82.06 f/s
1016950: done 2547 episodes, mean reward -83.755, speed 79.22 f/s
Test done in 10.72 sec, reward 219.814, steps 1528
1017509: done 2548 episodes, mean reward -83.110, speed 31.87 f/s
1017650: done 2549 episodes, mean reward -83.196, speed 80.93 f/s
1017885: done 2550 episodes, mean reward -83.330, speed 80.42 f/s
1017999: done 2551 episodes, mean reward -83.449, speed 77.42 f/s
Test done in 10.13 sec, reward 251.057, steps 1445
1018110: done 2552 episodes, mean reward -83.510, speed 9.68 f/s
1018406: done 2553 episodes, mean reward -83.217, speed 81.45 f/s
Test done in 11.84 sec, reward 244.903, steps 1548
1019014: done 2554 episodes, mean reward -82.632, speed 31.35 f/s
1019114: done 2555 episodes, mean reward -82.946, speed 75.54 f/s
1019407: done 2556 episodes, mean reward -82.382, speed 78.83 f/s
1019605: done 2557 episodes, mean reward -82.220, speed 74.57 f/s
1019798: done 2558 episodes, mean reward -82.165, speed 69.75 f/s
Test done in 11.44 sec, reward 244.469, steps 1511
1020093: done 2559 episodes, mean reward -82.033, speed 19.23 f/s
1020347: done 2560 episodes, mean reward -81.702, speed 75.95 f/s
1020542: done 2561 episodes, mean reward -81.584, speed 68.34 f/s
1020999: done 2562 episodes, mean reward -81.039, speed 74.31 f/s
Test done in 10.79 sec, reward 251.252, steps 1461
1021119: done 2563 episodes, mean reward -81.110, speed 9.65 f/s
1021393: done 2564 episodes, mean reward -80.956, speed 70.91 f/s
1021544: done 2565 episodes, mean reward -81.312, speed 74.37 f/s
1021671: done 2566 episodes, mean reward -81.379, speed 74.05 f/s
Test done in 11.83 sec, reward 247.968, steps 1548
1022071: done 2567 episodes, mean reward -81.155, speed 23.45 f/s
1022217: done 2568 episodes, mean reward -81.272, speed 72.48 f/s
1022350: done 2569 episodes, mean reward -81.471, speed 71.57 f/s
1022511: done 2570 episodes, mean reward -81.709, speed 69.68 f/s
1022643: done 2571 episodes, mean reward -81.922, speed 73.00 f/s
1022889: done 2572 episodes, mean reward -81.545, speed 73.64 f/s
Test done in 10.51 sec, reward 250.621, steps 1386
1023298: done 2573 episodes, mean reward -81.493, speed 25.73 f/s
1023435: done 2574 episodes, mean reward -81.561, speed 74.79 f/s
1023612: done 2575 episodes, mean reward -82.068, speed 74.22 f/s
1023711: done 2576 episodes, mean reward -81.988, speed 74.92 f/s
1023897: done 2577 episodes, mean reward -81.899, speed 71.86 f/s
Test done in 11.73 sec, reward 249.206, steps 1552
1024112: done 2578 episodes, mean reward -81.921, speed 14.81 f/s
1024424: done 2579 episodes, mean reward -81.544, speed 75.80 f/s
1024497: done 2580 episodes, mean reward -81.653, speed 67.72 f/s
1024803: done 2581 episodes, mean reward -81.300, speed 68.03 f/s
Test done in 11.08 sec, reward 251.866, steps 1478
1025050: done 2582 episodes, mean reward -81.507, speed 17.23 f/s
1025184: done 2583 episodes, mean reward -81.605, speed 78.10 f/s
1025518: done 2584 episodes, mean reward -81.413, speed 74.49 f/s
1025674: done 2585 episodes, mean reward -81.647, speed 83.02 f/s
1025886: done 2587 episodes, mean reward -82.375, speed 81.42 f/s
Test done in 10.90 sec, reward 253.253, steps 1465
1026542: done 2588 episodes, mean reward -81.670, speed 34.10 f/s
1026709: done 2589 episodes, mean reward -82.081, speed 75.14 f/s
1026799: done 2590 episodes, mean reward -82.074, speed 72.90 f/s
1026991: done 2591 episodes, mean reward -81.907, speed 73.51 f/s
Test done in 12.19 sec, reward 249.914, steps 1513
1027105: done 2592 episodes, mean reward -82.150, speed 8.25 f/s
1027211: done 2593 episodes, mean reward -82.313, speed 72.20 f/s
1027320: done 2594 episodes, mean reward -82.288, speed 75.56 f/s
1027627: done 2595 episodes, mean reward -82.180, speed 77.45 f/s
1027742: done 2596 episodes, mean reward -82.308, speed 73.44 f/s
1027836: done 2597 episodes, mean reward -82.367, speed 81.58 f/s
1027983: done 2598 episodes, mean reward -82.316, speed 75.48 f/s
Test done in 10.82 sec, reward 250.357, steps 1438
1028248: done 2599 episodes, mean reward -82.349, speed 18.68 f/s
1028339: done 2600 episodes, mean reward -82.395, speed 76.58 f/s
1028709: done 2601 episodes, mean reward -82.421, speed 75.35 f/s
1028883: done 2602 episodes, mean reward -82.704, speed 73.09 f/s
Test done in 11.20 sec, reward 249.849, steps 1440
1029101: done 2603 episodes, mean reward -82.803, speed 15.43 f/s
1029351: done 2604 episodes, mean reward -83.300, speed 73.99 f/s
1029531: done 2605 episodes, mean reward -83.287, speed 74.91 f/s
1029879: done 2606 episodes, mean reward -82.740, speed 78.62 f/s
Test done in 9.94 sec, reward 248.935, steps 1461
1030042: done 2607 episodes, mean reward -83.402, speed 13.51 f/s
1030230: done 2608 episodes, mean reward -83.278, speed 83.01 f/s
1030602: done 2610 episodes, mean reward -82.741, speed 75.76 f/s
1030747: done 2611 episodes, mean reward -82.778, speed 70.68 f/s
1030833: done 2612 episodes, mean reward -83.136, speed 75.13 f/s
1030944: done 2613 episodes, mean reward -83.088, speed 78.19 f/s
Test done in 11.58 sec, reward 243.686, steps 1528
1031083: done 2614 episodes, mean reward -83.302, speed 10.40 f/s
1031329: done 2615 episodes, mean reward -83.016, speed 75.06 f/s
1031781: done 2616 episodes, mean reward -82.464, speed 73.04 f/s
1031910: done 2617 episodes, mean reward -82.518, speed 77.22 f/s
Test done in 10.63 sec, reward 255.512, steps 1452
1032050: done 2618 episodes, mean reward -82.756, speed 11.29 f/s
1032267: done 2619 episodes, mean reward -82.943, speed 79.00 f/s
1032409: done 2620 episodes, mean reward -83.366, speed 78.34 f/s
1032531: done 2621 episodes, mean reward -83.762, speed 77.95 f/s
1032632: done 2622 episodes, mean reward -84.250, speed 80.20 f/s
1032797: done 2623 episodes, mean reward -84.668, speed 78.81 f/s
1032909: done 2624 episodes, mean reward -84.718, speed 76.36 f/s
Test done in 11.18 sec, reward 251.880, steps 1447
1033128: done 2625 episodes, mean reward -84.471, speed 15.12 f/s
1033384: done 2626 episodes, mean reward -84.325, speed 80.27 f/s
1033480: done 2627 episodes, mean reward -84.921, speed 80.60 f/s
1033657: done 2628 episodes, mean reward -84.892, speed 77.48 f/s
1033932: done 2629 episodes, mean reward -84.635, speed 78.35 f/s
Test done in 10.37 sec, reward 255.316, steps 1477
1034005: done 2630 episodes, mean reward -84.820, speed 6.50 f/s
1034243: done 2631 episodes, mean reward -84.701, speed 78.84 f/s
1034547: done 2632 episodes, mean reward -84.699, speed 82.58 f/s
1034646: done 2633 episodes, mean reward -84.730, speed 84.60 f/s
1034772: done 2634 episodes, mean reward -84.866, speed 82.19 f/s
1034898: done 2635 episodes, mean reward -84.817, speed 78.20 f/s
Test done in 8.87 sec, reward 190.193, steps 1270
1035269: done 2637 episodes, mean reward -84.347, speed 27.63 f/s
1035709: done 2638 episodes, mean reward -84.351, speed 81.37 f/s
Test done in 9.99 sec, reward 248.501, steps 1439
1036015: done 2639 episodes, mean reward -83.997, speed 22.22 f/s
1036514: done 2640 episodes, mean reward -83.488, speed 82.85 f/s
1036614: done 2641 episodes, mean reward -83.786, speed 88.27 f/s
1036817: done 2642 episodes, mean reward -84.259, speed 81.45 f/s
Test done in 10.44 sec, reward 247.449, steps 1447
1037015: done 2643 episodes, mean reward -84.398, speed 15.38 f/s
1037136: done 2644 episodes, mean reward -84.302, speed 70.43 f/s
1037250: done 2645 episodes, mean reward -84.254, speed 76.96 f/s
1037477: done 2646 episodes, mean reward -84.288, speed 78.02 f/s
1037743: done 2647 episodes, mean reward -84.248, speed 75.95 f/s
1037954: done 2648 episodes, mean reward -84.858, speed 78.97 f/s
Test done in 10.13 sec, reward 249.322, steps 1455
1038025: done 2649 episodes, mean reward -84.970, speed 6.46 f/s
1038224: done 2651 episodes, mean reward -85.068, speed 79.44 f/s
1038422: done 2652 episodes, mean reward -85.049, speed 77.59 f/s
1038620: done 2653 episodes, mean reward -85.260, speed 80.36 f/s
1038861: done 2654 episodes, mean reward -85.762, speed 79.09 f/s
1038967: done 2655 episodes, mean reward -85.826, speed 76.19 f/s
Test done in 10.05 sec, reward 251.514, steps 1398
1039105: done 2656 episodes, mean reward -86.264, speed 11.66 f/s
1039203: done 2657 episodes, mean reward -86.443, speed 79.56 f/s
1039353: done 2658 episodes, mean reward -86.617, speed 79.54 f/s
1039534: done 2659 episodes, mean reward -86.613, speed 78.50 f/s
1039722: done 2660 episodes, mean reward -86.656, speed 79.53 f/s
1039829: done 2661 episodes, mean reward -86.856, speed 81.86 f/s
1039992: done 2662 episodes, mean reward -87.368, speed 78.19 f/s
Test done in 10.28 sec, reward 230.698, steps 1348
1040107: done 2663 episodes, mean reward -87.340, speed 9.37 f/s
1040252: done 2664 episodes, mean reward -87.561, speed 63.63 f/s
1040355: done 2665 episodes, mean reward -87.575, speed 59.30 f/s
1040930: done 2666 episodes, mean reward -86.774, speed 71.44 f/s
Test done in 10.14 sec, reward 252.874, steps 1433
1041273: done 2667 episodes, mean reward -86.722, speed 23.71 f/s
1041410: done 2668 episodes, mean reward -86.656, speed 78.71 f/s
Test done in 10.25 sec, reward 253.851, steps 1461
1042328: done 2670 episodes, mean reward -85.169, speed 41.57 f/s
1042474: done 2671 episodes, mean reward -85.097, speed 76.65 f/s
1042932: done 2672 episodes, mean reward -84.822, speed 79.95 f/s
Test done in 11.15 sec, reward 247.705, steps 1533
1043113: done 2673 episodes, mean reward -85.157, speed 13.46 f/s
1043275: done 2674 episodes, mean reward -84.950, speed 77.95 f/s
1043425: done 2675 episodes, mean reward -84.995, speed 81.21 f/s
1043526: done 2676 episodes, mean reward -85.062, speed 80.25 f/s
1043757: done 2677 episodes, mean reward -84.992, speed 79.24 f/s
Test done in 10.74 sec, reward 255.269, steps 1494
1044130: done 2678 episodes, mean reward -84.664, speed 24.12 f/s
1044247: done 2679 episodes, mean reward -85.150, speed 75.99 f/s
1044383: done 2680 episodes, mean reward -85.077, speed 80.26 f/s
1044477: done 2681 episodes, mean reward -85.441, speed 84.57 f/s
1044630: done 2682 episodes, mean reward -85.425, speed 75.66 f/s
1044817: done 2683 episodes, mean reward -85.349, speed 77.48 f/s
Test done in 10.98 sec, reward 252.792, steps 1494
1045050: done 2684 episodes, mean reward -85.338, speed 16.79 f/s
1045163: done 2685 episodes, mean reward -85.368, speed 76.63 f/s
1045349: done 2686 episodes, mean reward -85.111, speed 80.64 f/s
1045459: done 2687 episodes, mean reward -85.237, speed 80.39 f/s
1045609: done 2688 episodes, mean reward -86.053, speed 76.44 f/s
1045713: done 2689 episodes, mean reward -86.031, speed 77.68 f/s
1045920: done 2690 episodes, mean reward -85.730, speed 82.81 f/s
Test done in 10.55 sec, reward 256.555, steps 1495
1046195: done 2691 episodes, mean reward -85.679, speed 19.74 f/s
1046293: done 2692 episodes, mean reward -85.757, speed 81.06 f/s
1046465: done 2693 episodes, mean reward -85.589, speed 79.63 f/s
1046664: done 2694 episodes, mean reward -85.343, speed 80.16 f/s
1046982: done 2695 episodes, mean reward -85.249, speed 78.90 f/s
Test done in 10.97 sec, reward 244.288, steps 1522
1047123: done 2696 episodes, mean reward -85.424, speed 11.08 f/s
1047289: done 2697 episodes, mean reward -85.176, speed 81.08 f/s
1047438: done 2698 episodes, mean reward -85.062, speed 78.73 f/s
1047546: done 2699 episodes, mean reward -85.480, speed 75.82 f/s
1047713: done 2700 episodes, mean reward -85.203, speed 76.78 f/s
1047835: done 2701 episodes, mean reward -85.889, speed 82.01 f/s
Test done in 9.74 sec, reward 251.673, steps 1355
1048106: done 2702 episodes, mean reward -85.894, speed 20.67 f/s
Test done in 10.10 sec, reward 253.808, steps 1443
1049130: done 2703 episodes, mean reward -84.402, speed 44.70 f/s
1049328: done 2704 episodes, mean reward -84.605, speed 80.38 f/s
1049446: done 2705 episodes, mean reward -84.703, speed 79.19 f/s
1049569: done 2706 episodes, mean reward -85.138, speed 77.81 f/s
1049687: done 2707 episodes, mean reward -85.167, speed 76.02 f/s
Test done in 10.04 sec, reward 254.012, steps 1488
1050125: done 2708 episodes, mean reward -84.745, speed 28.14 f/s
1050239: done 2709 episodes, mean reward -84.804, speed 80.19 f/s
1050644: done 2710 episodes, mean reward -84.608, speed 80.01 f/s
Test done in 10.40 sec, reward 254.998, steps 1504
1051138: done 2711 episodes, mean reward -83.958, speed 29.86 f/s
1051300: done 2712 episodes, mean reward -83.705, speed 80.93 f/s
1051579: done 2713 episodes, mean reward -83.409, speed 80.34 f/s
1051699: done 2714 episodes, mean reward -83.505, speed 84.27 f/s
Test done in 10.37 sec, reward 254.353, steps 1472
1052116: done 2715 episodes, mean reward -83.234, speed 26.89 f/s
1052241: done 2716 episodes, mean reward -83.876, speed 80.49 f/s
1052385: done 2717 episodes, mean reward -84.050, speed 73.67 f/s
1052500: done 2718 episodes, mean reward -83.997, speed 75.70 f/s
1052694: done 2719 episodes, mean reward -83.986, speed 78.71 f/s
1052850: done 2720 episodes, mean reward -83.979, speed 79.05 f/s
Test done in 10.69 sec, reward 167.662, steps 1518
1053129: done 2721 episodes, mean reward -84.070, speed 19.60 f/s
1053273: done 2722 episodes, mean reward -84.089, speed 82.60 f/s
1053681: done 2723 episodes, mean reward -83.480, speed 80.37 f/s
1053804: done 2724 episodes, mean reward -83.315, speed 77.62 f/s
Test done in 10.03 sec, reward 240.398, steps 1429
1054187: done 2725 episodes, mean reward -83.039, speed 25.98 f/s
1054458: done 2726 episodes, mean reward -83.001, speed 83.64 f/s
1054565: done 2727 episodes, mean reward -82.936, speed 80.91 f/s
Test done in 10.31 sec, reward 258.246, steps 1434
1055091: done 2728 episodes, mean reward -82.651, speed 31.29 f/s
1055244: done 2729 episodes, mean reward -83.016, speed 81.04 f/s
1055414: done 2730 episodes, mean reward -82.692, speed 81.50 f/s
1055501: done 2731 episodes, mean reward -82.962, speed 74.11 f/s
1055644: done 2732 episodes, mean reward -83.411, speed 80.71 f/s
1055825: done 2733 episodes, mean reward -83.149, speed 78.32 f/s
1055924: done 2734 episodes, mean reward -83.071, speed 77.45 f/s
Test done in 9.89 sec, reward 255.665, steps 1422
1056063: done 2735 episodes, mean reward -83.014, speed 11.96 f/s
1056174: done 2736 episodes, mean reward -82.999, speed 75.97 f/s
1056593: done 2737 episodes, mean reward -82.820, speed 73.81 f/s
1056701: done 2738 episodes, mean reward -83.351, speed 75.14 f/s
1056802: done 2739 episodes, mean reward -83.754, speed 83.17 f/s
Test done in 9.46 sec, reward 222.014, steps 1305
1057034: done 2740 episodes, mean reward -84.281, speed 18.50 f/s
1057186: done 2741 episodes, mean reward -84.073, speed 71.59 f/s
1057302: done 2742 episodes, mean reward -84.222, speed 81.86 f/s
1057390: done 2743 episodes, mean reward -84.526, speed 75.15 f/s
1057760: done 2744 episodes, mean reward -83.971, speed 79.81 f/s
Test done in 11.11 sec, reward 251.500, steps 1517
1058004: done 2745 episodes, mean reward -83.784, speed 17.10 f/s
1058097: done 2746 episodes, mean reward -84.101, speed 81.37 f/s
1058202: done 2747 episodes, mean reward -84.527, speed 82.40 f/s
1058424: done 2748 episodes, mean reward -84.387, speed 83.93 f/s
1058517: done 2749 episodes, mean reward -84.202, speed 78.01 f/s
1058770: done 2751 episodes, mean reward -84.371, speed 77.70 f/s
1058919: done 2752 episodes, mean reward -84.418, speed 80.87 f/s
Test done in 9.87 sec, reward 258.203, steps 1451
1059073: done 2753 episodes, mean reward -84.514, speed 13.00 f/s
1059274: done 2754 episodes, mean reward -84.451, speed 77.55 f/s
1059635: done 2755 episodes, mean reward -83.869, speed 78.58 f/s
Test done in 12.01 sec, reward 254.570, steps 1493
1060203: done 2756 episodes, mean reward -83.105, speed 28.72 f/s
1060408: done 2757 episodes, mean reward -82.945, speed 75.00 f/s
1060531: done 2758 episodes, mean reward -82.990, speed 69.62 f/s
1060685: done 2759 episodes, mean reward -83.263, speed 71.56 f/s
1060956: done 2760 episodes, mean reward -83.290, speed 73.61 f/s
Test done in 10.41 sec, reward 254.323, steps 1441
Test done in 10.16 sec, reward 250.035, steps 1452
1062080: done 2761 episodes, mean reward -83.672, speed 32.28 f/s
1062336: done 2762 episodes, mean reward -83.477, speed 76.02 f/s
1062452: done 2763 episodes, mean reward -83.493, speed 79.98 f/s
1062598: done 2764 episodes, mean reward -83.535, speed 76.80 f/s
1062729: done 2765 episodes, mean reward -83.419, speed 77.79 f/s
Test done in 9.66 sec, reward 261.796, steps 1373
1063068: done 2766 episodes, mean reward -84.011, speed 24.22 f/s
1063320: done 2767 episodes, mean reward -84.269, speed 76.62 f/s
1063448: done 2768 episodes, mean reward -84.480, speed 76.78 f/s
1063676: done 2769 episodes, mean reward -84.350, speed 79.51 f/s
1063858: done 2770 episodes, mean reward -85.702, speed 76.76 f/s
Test done in 9.22 sec, reward 223.355, steps 1275
1064007: done 2771 episodes, mean reward -85.696, speed 13.23 f/s
1064267: done 2772 episodes, mean reward -86.120, speed 72.30 f/s
1064407: done 2773 episodes, mean reward -86.283, speed 75.76 f/s
1064555: done 2774 episodes, mean reward -86.375, speed 73.06 f/s
Test done in 10.38 sec, reward 259.472, steps 1379
1065015: done 2775 episodes, mean reward -85.719, speed 26.87 f/s
1065323: done 2776 episodes, mean reward -85.398, speed 75.99 f/s
1065731: done 2777 episodes, mean reward -84.872, speed 76.78 f/s
1065917: done 2778 episodes, mean reward -85.290, speed 72.31 f/s
Test done in 10.80 sec, reward 260.408, steps 1370
1066210: done 2779 episodes, mean reward -84.779, speed 19.36 f/s
1066376: done 2780 episodes, mean reward -84.659, speed 78.92 f/s
1066671: done 2781 episodes, mean reward -84.277, speed 81.31 f/s
1066849: done 2782 episodes, mean reward -84.419, speed 78.18 f/s
Test done in 10.62 sec, reward 254.765, steps 1422
1067113: done 2783 episodes, mean reward -84.383, speed 18.95 f/s
1067331: done 2784 episodes, mean reward -84.569, speed 77.10 f/s
1067519: done 2785 episodes, mean reward -84.345, speed 72.31 f/s
1067746: done 2786 episodes, mean reward -84.382, speed 75.41 f/s
1067970: done 2787 episodes, mean reward -84.196, speed 79.81 f/s
Test done in 8.58 sec, reward 190.147, steps 1201
1068028: done 2788 episodes, mean reward -84.522, speed 6.22 f/s
1068112: done 2789 episodes, mean reward -84.730, speed 83.61 f/s
1068290: done 2790 episodes, mean reward -84.914, speed 78.54 f/s
1068452: done 2791 episodes, mean reward -85.092, speed 75.36 f/s
1068600: done 2792 episodes, mean reward -85.080, speed 77.56 f/s
1068771: done 2793 episodes, mean reward -85.229, speed 80.49 f/s
1068899: done 2794 episodes, mean reward -85.516, speed 79.92 f/s
Test done in 9.57 sec, reward 258.323, steps 1364
1069268: done 2795 episodes, mean reward -85.443, speed 25.83 f/s
1069385: done 2796 episodes, mean reward -85.384, speed 74.10 f/s
1069576: done 2797 episodes, mean reward -85.511, speed 66.18 f/s
1069880: done 2798 episodes, mean reward -85.287, speed 68.93 f/s
Test done in 10.07 sec, reward 263.746, steps 1395
1070026: done 2799 episodes, mean reward -85.158, speed 11.87 f/s
1070332: done 2800 episodes, mean reward -84.915, speed 77.87 f/s
1070702: done 2801 episodes, mean reward -84.374, speed 77.81 f/s
1070844: done 2802 episodes, mean reward -84.614, speed 80.24 f/s
1070959: done 2803 episodes, mean reward -86.426, speed 82.69 f/s
Test done in 9.89 sec, reward 260.083, steps 1414
1071125: done 2804 episodes, mean reward -86.390, speed 13.91 f/s
1071359: done 2805 episodes, mean reward -86.157, speed 80.03 f/s
1071732: done 2806 episodes, mean reward -85.685, speed 81.88 f/s
1071895: done 2807 episodes, mean reward -85.542, speed 76.82 f/s
Test done in 10.92 sec, reward 263.006, steps 1345
1072291: done 2808 episodes, mean reward -85.634, speed 24.32 f/s
1072438: done 2809 episodes, mean reward -85.456, speed 76.24 f/s
1072619: done 2810 episodes, mean reward -85.860, speed 77.76 f/s
1072813: done 2811 episodes, mean reward -86.503, speed 74.56 f/s
Test done in 10.11 sec, reward 262.635, steps 1367
1073048: done 2812 episodes, mean reward -86.294, speed 17.74 f/s
1073436: done 2813 episodes, mean reward -86.088, speed 75.67 f/s
1073686: done 2814 episodes, mean reward -85.646, speed 73.18 f/s
1073839: done 2815 episodes, mean reward -86.111, speed 79.74 f/s
Test done in 9.87 sec, reward 256.306, steps 1412
1074078: done 2816 episodes, mean reward -85.914, speed 18.56 f/s
1074185: done 2817 episodes, mean reward -85.782, speed 75.45 f/s
1074289: done 2818 episodes, mean reward -85.918, speed 76.38 f/s
1074390: done 2819 episodes, mean reward -86.197, speed 75.81 f/s
1074773: done 2820 episodes, mean reward -85.528, speed 79.47 f/s
Test done in 9.41 sec, reward 235.831, steps 1296
1075101: done 2821 episodes, mean reward -85.071, speed 24.16 f/s
1075448: done 2822 episodes, mean reward -84.729, speed 77.92 f/s
1075642: done 2823 episodes, mean reward -85.136, speed 76.54 f/s
1075742: done 2824 episodes, mean reward -85.387, speed 80.90 f/s
1075835: done 2825 episodes, mean reward -85.934, speed 80.90 f/s
Test done in 10.00 sec, reward 262.930, steps 1330
1076205: done 2826 episodes, mean reward -85.661, speed 25.13 f/s
1076333: done 2827 episodes, mean reward -85.696, speed 70.34 f/s
1076532: done 2828 episodes, mean reward -85.896, speed 79.07 f/s
1076678: done 2829 episodes, mean reward -85.974, speed 79.70 f/s
1076831: done 2830 episodes, mean reward -86.064, speed 77.14 f/s
Test done in 10.13 sec, reward 263.256, steps 1364
1077073: done 2831 episodes, mean reward -85.616, speed 18.18 f/s
1077241: done 2832 episodes, mean reward -85.322, speed 79.30 f/s
1077324: done 2833 episodes, mean reward -85.668, speed 81.20 f/s
1077546: done 2834 episodes, mean reward -85.375, speed 79.38 f/s
1077689: done 2835 episodes, mean reward -85.364, speed 82.90 f/s
1077852: done 2836 episodes, mean reward -85.191, speed 80.56 f/s
Test done in 9.41 sec, reward 262.757, steps 1335
1078144: done 2837 episodes, mean reward -85.303, speed 22.31 f/s
1078278: done 2838 episodes, mean reward -85.412, speed 73.82 f/s
1078470: done 2839 episodes, mean reward -85.210, speed 77.23 f/s
1078587: done 2840 episodes, mean reward -85.277, speed 78.16 f/s
1078674: done 2841 episodes, mean reward -85.466, speed 78.79 f/s
1078822: done 2842 episodes, mean reward -85.566, speed 74.82 f/s
1078964: done 2843 episodes, mean reward -85.439, speed 75.21 f/s
Test done in 8.46 sec, reward 200.855, steps 1155
1079215: done 2844 episodes, mean reward -85.870, speed 21.51 f/s
1079345: done 2845 episodes, mean reward -86.133, speed 74.06 f/s
1079694: done 2846 episodes, mean reward -85.665, speed 76.40 f/s
1079903: done 2847 episodes, mean reward -85.285, speed 76.26 f/s
Test done in 9.77 sec, reward 249.498, steps 1370
1080106: done 2848 episodes, mean reward -85.349, speed 16.42 f/s
1080247: done 2849 episodes, mean reward -85.387, speed 76.90 f/s
1080354: done 2850 episodes, mean reward -85.322, speed 79.47 f/s
1080669: done 2851 episodes, mean reward -84.830, speed 77.79 f/s
1080894: done 2852 episodes, mean reward -84.754, speed 72.04 f/s
Test done in 10.07 sec, reward 260.691, steps 1406
1081028: done 2853 episodes, mean reward -84.910, speed 11.29 f/s
1081140: done 2854 episodes, mean reward -85.277, speed 76.55 f/s
1081452: done 2855 episodes, mean reward -85.309, speed 74.28 f/s
1081899: done 2856 episodes, mean reward -85.378, speed 75.54 f/s
Test done in 9.38 sec, reward 263.426, steps 1299
1082347: done 2857 episodes, mean reward -84.782, speed 29.47 f/s
1082569: done 2858 episodes, mean reward -84.501, speed 75.44 f/s
1082752: done 2859 episodes, mean reward -84.365, speed 76.18 f/s
Test done in 9.65 sec, reward 263.571, steps 1313
1083034: done 2860 episodes, mean reward -84.260, speed 20.99 f/s
1083163: done 2861 episodes, mean reward -83.742, speed 74.34 f/s
1083478: done 2862 episodes, mean reward -83.685, speed 74.35 f/s
1083608: done 2863 episodes, mean reward -83.623, speed 76.39 f/s
1083788: done 2864 episodes, mean reward -83.625, speed 77.12 f/s
1083893: done 2865 episodes, mean reward -83.756, speed 77.20 f/s
Test done in 9.79 sec, reward 260.008, steps 1347
1084069: done 2866 episodes, mean reward -83.962, speed 14.65 f/s
1084253: done 2867 episodes, mean reward -84.100, speed 77.41 f/s
1084413: done 2868 episodes, mean reward -83.975, speed 69.30 f/s
1084733: done 2869 episodes, mean reward -83.624, speed 68.12 f/s
1084888: done 2870 episodes, mean reward -83.644, speed 60.42 f/s
Test done in 10.66 sec, reward 265.820, steps 1324
1085174: done 2871 episodes, mean reward -83.205, speed 19.10 f/s
1085331: done 2872 episodes, mean reward -83.302, speed 67.25 f/s
1085405: done 2873 episodes, mean reward -83.259, speed 70.12 f/s
1085605: done 2874 episodes, mean reward -83.109, speed 75.86 f/s
1085721: done 2875 episodes, mean reward -83.775, speed 74.12 f/s
1085796: done 2876 episodes, mean reward -84.110, speed 74.17 f/s
1085944: done 2877 episodes, mean reward -84.664, speed 80.19 f/s
Test done in 9.72 sec, reward 252.573, steps 1288
1086079: done 2878 episodes, mean reward -84.638, speed 11.75 f/s
1086249: done 2879 episodes, mean reward -84.933, speed 75.30 f/s
1086351: done 2880 episodes, mean reward -85.151, speed 77.71 f/s
1086472: done 2881 episodes, mean reward -85.446, speed 73.15 f/s
1086727: done 2882 episodes, mean reward -85.189, speed 75.51 f/s
1086840: done 2883 episodes, mean reward -85.401, speed 76.12 f/s
Test done in 9.42 sec, reward 264.442, steps 1274
1087184: done 2885 episodes, mean reward -85.337, speed 24.88 f/s
1087300: done 2886 episodes, mean reward -85.591, speed 77.77 f/s
1087514: done 2887 episodes, mean reward -85.573, speed 76.33 f/s
1087675: done 2888 episodes, mean reward -85.187, speed 73.12 f/s
1087874: done 2890 episodes, mean reward -85.021, speed 77.54 f/s
Test done in 9.78 sec, reward 264.774, steps 1313
1088021: done 2891 episodes, mean reward -85.123, speed 12.56 f/s
1088275: done 2893 episodes, mean reward -85.167, speed 74.32 f/s
1088507: done 2894 episodes, mean reward -84.838, speed 79.51 f/s
1088667: done 2895 episodes, mean reward -85.190, speed 80.90 f/s
Test done in 10.29 sec, reward 267.104, steps 1319
Best reward updated: 266.288 -> 267.104
1089028: done 2896 episodes, mean reward -84.650, speed 23.75 f/s
1089177: done 2897 episodes, mean reward -84.689, speed 72.38 f/s
1089332: done 2898 episodes, mean reward -84.883, speed 68.20 f/s
1089494: done 2899 episodes, mean reward -84.708, speed 69.77 f/s
1089622: done 2900 episodes, mean reward -85.166, speed 67.92 f/s
1089789: done 2901 episodes, mean reward -85.547, speed 77.79 f/s
1089920: done 2902 episodes, mean reward -85.463, speed 75.41 f/s
Test done in 8.67 sec, reward 235.543, steps 1276
1090089: done 2903 episodes, mean reward -85.237, speed 15.41 f/s
1090307: done 2904 episodes, mean reward -85.124, speed 77.79 f/s
1090462: done 2905 episodes, mean reward -85.397, speed 81.51 f/s
1090623: done 2906 episodes, mean reward -85.776, speed 76.47 f/s
1090801: done 2908 episodes, mean reward -86.722, speed 76.51 f/s
1090972: done 2909 episodes, mean reward -86.778, speed 76.64 f/s
Test done in 8.82 sec, reward 235.099, steps 1242
1091293: done 2910 episodes, mean reward -86.426, speed 23.72 f/s
1091481: done 2911 episodes, mean reward -86.343, speed 67.83 f/s
1091657: done 2912 episodes, mean reward -86.462, speed 68.86 f/s
1091915: done 2913 episodes, mean reward -86.570, speed 63.36 f/s
Test done in 10.43 sec, reward 268.292, steps 1365
Best reward updated: 267.104 -> 268.292
1092079: done 2914 episodes, mean reward -86.754, speed 12.64 f/s
1092169: done 2915 episodes, mean reward -86.872, speed 73.83 f/s
1092297: done 2916 episodes, mean reward -87.066, speed 73.73 f/s
1092616: done 2917 episodes, mean reward -86.456, speed 77.05 f/s
1092877: done 2918 episodes, mean reward -86.015, speed 77.38 f/s
Test done in 10.00 sec, reward 248.393, steps 1326
1093006: done 2919 episodes, mean reward -85.978, speed 11.09 f/s
1093123: done 2920 episodes, mean reward -86.610, speed 79.86 f/s
1093327: done 2921 episodes, mean reward -86.856, speed 78.05 f/s
1093458: done 2922 episodes, mean reward -87.114, speed 71.45 f/s
1093641: done 2923 episodes, mean reward -87.183, speed 70.49 f/s
1093804: done 2924 episodes, mean reward -86.901, speed 72.14 f/s
1093904: done 2925 episodes, mean reward -86.810, speed 74.12 f/s
Test done in 10.19 sec, reward 240.420, steps 1355
1094298: done 2926 episodes, mean reward -86.658, speed 25.15 f/s
1094413: done 2927 episodes, mean reward -86.681, speed 76.02 f/s
1094536: done 2928 episodes, mean reward -86.860, speed 74.26 f/s
1094623: done 2929 episodes, mean reward -86.955, speed 74.34 f/s
1094730: done 2930 episodes, mean reward -87.151, speed 73.98 f/s
Test done in 3.47 sec, reward 14.219, steps 484
1095174: done 2931 episodes, mean reward -86.691, speed 48.24 f/s
1095450: done 2932 episodes, mean reward -86.504, speed 75.26 f/s
1095686: done 2934 episodes, mean reward -86.758, speed 77.67 f/s
1095766: done 2935 episodes, mean reward -86.968, speed 74.42 f/s
1095890: done 2936 episodes, mean reward -87.089, speed 81.27 f/s
Test done in 6.19 sec, reward 122.571, steps 883
1096068: done 2937 episodes, mean reward -87.377, speed 21.05 f/s
1096297: done 2938 episodes, mean reward -87.028, speed 77.12 f/s
1096459: done 2939 episodes, mean reward -87.094, speed 73.74 f/s
1096610: done 2940 episodes, mean reward -86.965, speed 77.08 f/s
1096725: done 2941 episodes, mean reward -86.761, speed 77.91 f/s
1096812: done 2942 episodes, mean reward -86.715, speed 75.87 f/s
Test done in 5.92 sec, reward 119.551, steps 831
1097148: done 2943 episodes, mean reward -86.136, speed 32.07 f/s
1097233: done 2944 episodes, mean reward -86.470, speed 70.50 f/s
1097386: done 2945 episodes, mean reward -86.215, speed 77.77 f/s
1097586: done 2946 episodes, mean reward -86.335, speed 75.70 f/s
Test done in 9.41 sec, reward 243.318, steps 1328
1098008: done 2948 episodes, mean reward -86.373, speed 28.03 f/s
1098208: done 2949 episodes, mean reward -86.266, speed 64.91 f/s
1098439: done 2950 episodes, mean reward -85.943, speed 72.41 f/s
1098687: done 2951 episodes, mean reward -86.136, speed 69.40 f/s
1098792: done 2952 episodes, mean reward -86.365, speed 69.17 f/s
Test done in 10.13 sec, reward 264.216, steps 1382
1099085: done 2953 episodes, mean reward -86.049, speed 20.45 f/s
1099249: done 2954 episodes, mean reward -85.925, speed 74.48 f/s
1099546: done 2955 episodes, mean reward -85.927, speed 76.78 f/s
1099661: done 2956 episodes, mean reward -86.521, speed 79.71 f/s
1099785: done 2957 episodes, mean reward -87.271, speed 77.49 f/s
1099943: done 2958 episodes, mean reward -87.443, speed 74.76 f/s
Test done in 9.85 sec, reward 267.883, steps 1333
1100083: done 2959 episodes, mean reward -87.424, speed 11.98 f/s
1100348: done 2960 episodes, mean reward -87.495, speed 75.82 f/s
1100425: done 2961 episodes, mean reward -87.614, speed 73.64 f/s
1100527: done 2962 episodes, mean reward -87.843, speed 70.10 f/s
1100693: done 2963 episodes, mean reward -87.776, speed 71.36 f/s
1100883: done 2964 episodes, mean reward -87.429, speed 69.74 f/s
Test done in 7.05 sec, reward 145.411, steps 985
1101042: done 2965 episodes, mean reward -87.162, speed 17.23 f/s
1101173: done 2966 episodes, mean reward -87.201, speed 78.43 f/s
1101466: done 2967 episodes, mean reward -86.796, speed 75.29 f/s
1101546: done 2968 episodes, mean reward -86.934, speed 72.86 f/s
1101789: done 2969 episodes, mean reward -86.984, speed 77.98 f/s
1101912: done 2970 episodes, mean reward -87.006, speed 79.84 f/s
Test done in 9.19 sec, reward 266.329, steps 1282
1102111: done 2971 episodes, mean reward -87.382, speed 16.76 f/s
1102204: done 2972 episodes, mean reward -87.481, speed 68.65 f/s
1102365: done 2973 episodes, mean reward -87.435, speed 58.97 f/s
1102827: done 2974 episodes, mean reward -86.891, speed 77.71 f/s
Test done in 9.99 sec, reward 265.116, steps 1286
1103017: done 2975 episodes, mean reward -86.655, speed 15.03 f/s
1103225: done 2976 episodes, mean reward -86.333, speed 67.74 f/s
1103451: done 2977 episodes, mean reward -86.068, speed 62.68 f/s
1103672: done 2978 episodes, mean reward -85.836, speed 72.34 f/s
1103800: done 2979 episodes, mean reward -86.070, speed 63.98 f/s
1103909: done 2980 episodes, mean reward -86.095, speed 72.74 f/s
Test done in 9.19 sec, reward 240.493, steps 1227
1104052: done 2981 episodes, mean reward -85.992, speed 12.74 f/s
1104268: done 2982 episodes, mean reward -86.143, speed 74.94 f/s
1104483: done 2983 episodes, mean reward -85.777, speed 71.19 f/s
1104602: done 2984 episodes, mean reward -85.512, speed 67.05 f/s
1105000: done 2985 episodes, mean reward -85.261, speed 69.26 f/s
Test done in 9.55 sec, reward 252.948, steps 1266
1105142: done 2986 episodes, mean reward -85.269, speed 12.05 f/s
1105301: done 2987 episodes, mean reward -85.288, speed 60.25 f/s
1105490: done 2988 episodes, mean reward -85.273, speed 72.16 f/s
1105638: done 2989 episodes, mean reward -85.078, speed 70.03 f/s
1105804: done 2990 episodes, mean reward -85.017, speed 70.78 f/s
1105981: done 2991 episodes, mean reward -84.821, speed 73.45 f/s
Test done in 9.92 sec, reward 259.926, steps 1268
1106196: done 2992 episodes, mean reward -84.476, speed 16.55 f/s
1106330: done 2993 episodes, mean reward -84.596, speed 74.42 f/s
1106423: done 2994 episodes, mean reward -84.872, speed 70.84 f/s
1106675: done 2995 episodes, mean reward -84.791, speed 70.84 f/s
1106764: done 2996 episodes, mean reward -85.360, speed 66.71 f/s
1106918: done 2997 episodes, mean reward -85.211, speed 66.08 f/s
Test done in 10.26 sec, reward 262.568, steps 1291
1107068: done 2998 episodes, mean reward -85.229, speed 12.17 f/s
1107209: done 2999 episodes, mean reward -85.335, speed 72.56 f/s
1107472: done 3000 episodes, mean reward -84.893, speed 77.72 f/s
1107693: done 3001 episodes, mean reward -84.678, speed 76.23 f/s
Test done in 7.81 sec, reward 187.642, steps 1080
1108052: done 3002 episodes, mean reward -84.079, speed 28.68 f/s
1108360: done 3003 episodes, mean reward -83.915, speed 66.76 f/s
1108713: done 3004 episodes, mean reward -83.553, speed 70.24 f/s
Test done in 9.33 sec, reward 240.252, steps 1180
1109017: done 3005 episodes, mean reward -83.096, speed 22.31 f/s
1109165: done 3006 episodes, mean reward -83.185, speed 72.12 f/s
1109386: done 3007 episodes, mean reward -82.734, speed 71.96 f/s
1109538: done 3008 episodes, mean reward -82.620, speed 67.80 f/s
1109676: done 3009 episodes, mean reward -82.679, speed 71.89 f/s
1109974: done 3010 episodes, mean reward -82.887, speed 69.58 f/s
Test done in 9.75 sec, reward 261.565, steps 1262
1110275: done 3011 episodes, mean reward -82.635, speed 21.49 f/s
1110610: done 3012 episodes, mean reward -82.254, speed 69.75 f/s
1110910: done 3013 episodes, mean reward -82.360, speed 74.48 f/s
Test done in 8.95 sec, reward 235.889, steps 1255
1111285: done 3014 episodes, mean reward -81.865, speed 26.69 f/s
1111454: done 3015 episodes, mean reward -81.760, speed 75.57 f/s
1111626: done 3016 episodes, mean reward -81.728, speed 74.69 f/s
1111954: done 3017 episodes, mean reward -81.718, speed 76.57 f/s
Test done in 9.33 sec, reward 249.288, steps 1257
1112064: done 3018 episodes, mean reward -82.018, speed 10.14 f/s
1112322: done 3019 episodes, mean reward -81.668, speed 74.67 f/s
1112555: done 3020 episodes, mean reward -81.594, speed 66.88 f/s
1112825: done 3021 episodes, mean reward -81.336, speed 77.65 f/s
1112971: done 3022 episodes, mean reward -81.477, speed 77.21 f/s
Test done in 8.86 sec, reward 222.819, steps 1169
1113148: done 3023 episodes, mean reward -81.639, speed 15.47 f/s
1113292: done 3024 episodes, mean reward -81.661, speed 73.00 f/s
1113436: done 3025 episodes, mean reward -81.633, speed 77.76 f/s
1113561: done 3026 episodes, mean reward -82.380, speed 74.46 f/s
1113789: done 3027 episodes, mean reward -82.069, speed 70.68 f/s
1113951: done 3028 episodes, mean reward -82.445, speed 65.58 f/s
Test done in 9.03 sec, reward 263.252, steps 1240
1114275: done 3029 episodes, mean reward -81.942, speed 24.29 f/s
1114419: done 3030 episodes, mean reward -81.890, speed 74.94 f/s
1114571: done 3031 episodes, mean reward -82.537, speed 79.18 f/s
1114798: done 3032 episodes, mean reward -82.562, speed 69.94 f/s
1114960: done 3033 episodes, mean reward -82.525, speed 71.80 f/s
Test done in 8.57 sec, reward 245.806, steps 1223
1115148: done 3034 episodes, mean reward -82.621, speed 17.11 f/s
1115307: done 3035 episodes, mean reward -82.446, speed 78.25 f/s
1115784: done 3036 episodes, mean reward -81.706, speed 66.77 f/s
1115930: done 3037 episodes, mean reward -81.796, speed 68.81 f/s
Test done in 9.13 sec, reward 185.711, steps 1107
1116073: done 3038 episodes, mean reward -82.130, speed 12.89 f/s
1116170: done 3039 episodes, mean reward -82.194, speed 75.19 f/s
1116463: done 3040 episodes, mean reward -81.928, speed 78.53 f/s
1116630: done 3041 episodes, mean reward -81.956, speed 76.78 f/s
1116886: done 3042 episodes, mean reward -81.708, speed 77.76 f/s
Test done in 6.96 sec, reward 178.650, steps 996
1117042: done 3043 episodes, mean reward -82.157, speed 17.29 f/s
1117218: done 3044 episodes, mean reward -81.806, speed 81.10 f/s
1117382: done 3045 episodes, mean reward -81.869, speed 73.50 f/s
1117650: done 3046 episodes, mean reward -81.678, speed 79.20 f/s
1117801: done 3047 episodes, mean reward -81.456, speed 78.68 f/s
1117993: done 3048 episodes, mean reward -81.722, speed 81.67 f/s
Test done in 8.00 sec, reward 207.726, steps 1120
1118173: done 3049 episodes, mean reward -81.704, speed 16.98 f/s
1118312: done 3050 episodes, mean reward -82.012, speed 74.94 f/s
1118507: done 3051 episodes, mean reward -82.183, speed 76.82 f/s
1118651: done 3052 episodes, mean reward -81.961, speed 83.79 f/s
1118762: done 3053 episodes, mean reward -82.126, speed 80.38 f/s
1118859: done 3054 episodes, mean reward -82.296, speed 79.82 f/s
1118993: done 3055 episodes, mean reward -82.786, speed 75.65 f/s
Test done in 8.97 sec, reward 258.442, steps 1248
1119163: done 3056 episodes, mean reward -82.805, speed 15.35 f/s
1119268: done 3057 episodes, mean reward -82.815, speed 80.10 f/s
1119456: done 3058 episodes, mean reward -82.637, speed 76.51 f/s
1119715: done 3059 episodes, mean reward -82.575, speed 77.79 f/s
1119932: done 3060 episodes, mean reward -82.650, speed 81.71 f/s
Test done in 8.32 sec, reward 230.376, steps 1182
1120186: done 3061 episodes, mean reward -82.416, speed 22.03 f/s
1120560: done 3062 episodes, mean reward -82.023, speed 72.13 f/s
1120719: done 3063 episodes, mean reward -82.010, speed 78.66 f/s
1120900: done 3064 episodes, mean reward -82.228, speed 78.62 f/s
Test done in 7.44 sec, reward 181.628, steps 1099
1121207: done 3065 episodes, mean reward -81.986, speed 26.96 f/s
1121409: done 3066 episodes, mean reward -81.945, speed 78.99 f/s
1121708: done 3067 episodes, mean reward -82.001, speed 77.53 f/s
1121956: done 3068 episodes, mean reward -81.516, speed 78.39 f/s
Test done in 8.49 sec, reward 238.714, steps 1161
1122175: done 3069 episodes, mean reward -81.695, speed 19.55 f/s
1122288: done 3070 episodes, mean reward -81.720, speed 84.73 f/s
1122508: done 3071 episodes, mean reward -81.477, speed 81.73 f/s
1122626: done 3072 episodes, mean reward -81.530, speed 79.51 f/s
1122749: done 3073 episodes, mean reward -81.625, speed 77.94 f/s
1122918: done 3074 episodes, mean reward -82.198, speed 82.05 f/s
Test done in 8.35 sec, reward 221.829, steps 1168
1123149: done 3075 episodes, mean reward -82.010, speed 20.61 f/s
1123267: done 3076 episodes, mean reward -82.248, speed 78.28 f/s
1123375: done 3077 episodes, mean reward -82.653, speed 65.82 f/s
1123555: done 3078 episodes, mean reward -82.717, speed 71.76 f/s
1123680: done 3079 episodes, mean reward -82.755, speed 75.32 f/s
1123760: done 3080 episodes, mean reward -82.756, speed 77.48 f/s
1123923: done 3081 episodes, mean reward -82.797, speed 77.49 f/s
Test done in 8.53 sec, reward 237.070, steps 1165
1124038: done 3082 episodes, mean reward -82.998, speed 11.29 f/s
1124301: done 3083 episodes, mean reward -82.903, speed 64.13 f/s
1124380: done 3084 episodes, mean reward -83.122, speed 71.35 f/s
1124534: done 3085 episodes, mean reward -83.820, speed 62.37 f/s
1124642: done 3086 episodes, mean reward -83.608, speed 65.34 f/s
1124851: done 3087 episodes, mean reward -83.419, speed 62.86 f/s
1124968: done 3088 episodes, mean reward -83.548, speed 71.71 f/s
Test done in 9.67 sec, reward 237.135, steps 1209
1125163: done 3089 episodes, mean reward -83.431, speed 15.39 f/s
1125389: done 3090 episodes, mean reward -83.421, speed 76.19 f/s
1125600: done 3091 episodes, mean reward -83.354, speed 66.60 f/s
1125932: done 3092 episodes, mean reward -83.172, speed 70.21 f/s
Test done in 10.18 sec, reward 231.397, steps 1199
1126351: done 3093 episodes, mean reward -82.542, speed 26.26 f/s
1126558: done 3094 episodes, mean reward -82.386, speed 72.62 f/s
1126730: done 3095 episodes, mean reward -82.364, speed 63.87 f/s
1126860: done 3096 episodes, mean reward -82.421, speed 57.37 f/s
Test done in 9.85 sec, reward 217.455, steps 1202
1127142: done 3097 episodes, mean reward -82.223, speed 19.93 f/s
1127435: done 3098 episodes, mean reward -82.046, speed 74.00 f/s
1127545: done 3099 episodes, mean reward -82.124, speed 74.24 f/s
1127663: done 3100 episodes, mean reward -82.626, speed 78.16 f/s
1127798: done 3101 episodes, mean reward -82.977, speed 65.36 f/s
Test done in 10.00 sec, reward 260.315, steps 1260
1128009: done 3102 episodes, mean reward -83.378, speed 16.09 f/s
1128310: done 3103 episodes, mean reward -83.209, speed 68.36 f/s
1128507: done 3104 episodes, mean reward -83.689, speed 61.55 f/s
1128672: done 3105 episodes, mean reward -84.157, speed 62.37 f/s
1128892: done 3106 episodes, mean reward -83.882, speed 68.01 f/s
Test done in 9.97 sec, reward 246.796, steps 1210
1129091: done 3107 episodes, mean reward -84.042, speed 15.64 f/s
1129190: done 3108 episodes, mean reward -84.054, speed 60.24 f/s
1129350: done 3109 episodes, mean reward -83.876, speed 65.58 f/s
1129593: done 3110 episodes, mean reward -83.923, speed 65.87 f/s
1129920: done 3111 episodes, mean reward -83.674, speed 67.23 f/s
Test done in 10.47 sec, reward 250.397, steps 1231
1130118: done 3113 episodes, mean reward -84.686, speed 14.49 f/s
1130326: done 3114 episodes, mean reward -85.033, speed 75.96 f/s
1130458: done 3115 episodes, mean reward -85.101, speed 59.25 f/s
1130715: done 3116 episodes, mean reward -84.708, speed 69.50 f/s
1130861: done 3117 episodes, mean reward -85.050, speed 70.53 f/s
Test done in 11.39 sec, reward 262.669, steps 1268
1131083: done 3118 episodes, mean reward -84.748, speed 15.20 f/s
1131261: done 3119 episodes, mean reward -85.375, speed 72.62 f/s
1131454: done 3120 episodes, mean reward -85.283, speed 70.01 f/s
1131702: done 3121 episodes, mean reward -85.380, speed 72.49 f/s
1131830: done 3122 episodes, mean reward -85.277, speed 77.56 f/s
Test done in 8.94 sec, reward 210.717, steps 1093
1132029: done 3123 episodes, mean reward -85.193, speed 17.33 f/s
1132168: done 3124 episodes, mean reward -85.328, speed 72.04 f/s
1132338: done 3125 episodes, mean reward -85.157, speed 69.81 f/s
1132467: done 3126 episodes, mean reward -85.088, speed 76.62 f/s
1132555: done 3127 episodes, mean reward -85.434, speed 76.29 f/s
Test done in 10.78 sec, reward 246.242, steps 1317
1133046: done 3128 episodes, mean reward -84.134, speed 28.68 f/s
1133210: done 3129 episodes, mean reward -84.347, speed 63.83 f/s
1133344: done 3130 episodes, mean reward -84.231, speed 71.57 f/s
1133441: done 3131 episodes, mean reward -84.398, speed 61.37 f/s
1133594: done 3132 episodes, mean reward -84.572, speed 56.80 f/s
1133744: done 3133 episodes, mean reward -84.513, speed 70.27 f/s
1133833: done 3134 episodes, mean reward -84.448, speed 65.82 f/s
Test done in 8.40 sec, reward 189.447, steps 1184
1134008: done 3135 episodes, mean reward -84.356, speed 15.62 f/s
1134363: done 3136 episodes, mean reward -84.555, speed 73.10 f/s
1134459: done 3137 episodes, mean reward -84.694, speed 79.31 f/s
1134572: done 3138 episodes, mean reward -84.672, speed 79.08 f/s
1134761: done 3139 episodes, mean reward -84.526, speed 77.74 f/s
1134868: done 3140 episodes, mean reward -85.083, speed 77.70 f/s
Test done in 5.47 sec, reward 108.700, steps 793
1135074: done 3141 episodes, mean reward -84.878, speed 24.62 f/s
1135248: done 3142 episodes, mean reward -84.938, speed 71.13 f/s
1135405: done 3143 episodes, mean reward -84.858, speed 68.53 f/s
1135549: done 3144 episodes, mean reward -85.119, speed 68.56 f/s
1135675: done 3145 episodes, mean reward -85.279, speed 72.97 f/s
1135793: done 3146 episodes, mean reward -85.667, speed 67.19 f/s
Test done in 11.01 sec, reward 259.773, steps 1384
1136017: done 3147 episodes, mean reward -85.584, speed 15.64 f/s
1136113: done 3148 episodes, mean reward -85.709, speed 64.91 f/s
1136257: done 3149 episodes, mean reward -85.888, speed 60.32 f/s
1136427: done 3150 episodes, mean reward -85.634, speed 59.44 f/s
1136795: done 3151 episodes, mean reward -85.209, speed 68.29 f/s
1136965: done 3152 episodes, mean reward -85.086, speed 74.18 f/s
Test done in 10.54 sec, reward 243.647, steps 1281
1137109: done 3153 episodes, mean reward -85.174, speed 11.50 f/s
1137194: done 3154 episodes, mean reward -85.162, speed 74.95 f/s
1137379: done 3155 episodes, mean reward -84.899, speed 69.29 f/s
1137496: done 3156 episodes, mean reward -85.001, speed 65.14 f/s
1137569: done 3157 episodes, mean reward -85.040, speed 70.42 f/s
1137837: done 3158 episodes, mean reward -84.973, speed 71.66 f/s
Test done in 10.20 sec, reward 266.328, steps 1285
1138002: done 3159 episodes, mean reward -85.102, speed 13.04 f/s
1138093: done 3160 episodes, mean reward -85.249, speed 56.30 f/s
1138307: done 3161 episodes, mean reward -85.430, speed 71.99 f/s
1138615: done 3162 episodes, mean reward -85.426, speed 67.43 f/s
1138899: done 3163 episodes, mean reward -85.205, speed 71.21 f/s
Test done in 9.76 sec, reward 264.607, steps 1244
1139104: done 3164 episodes, mean reward -85.066, speed 15.78 f/s
1139353: done 3165 episodes, mean reward -85.268, speed 69.07 f/s
1139534: done 3166 episodes, mean reward -85.109, speed 76.11 f/s
1139685: done 3167 episodes, mean reward -85.337, speed 71.39 f/s
1139752: done 3168 episodes, mean reward -85.818, speed 62.88 f/s
1139937: done 3169 episodes, mean reward -85.972, speed 74.05 f/s
Test done in 9.99 sec, reward 261.058, steps 1344
1140121: done 3170 episodes, mean reward -85.972, speed 14.86 f/s
1140261: done 3171 episodes, mean reward -86.392, speed 72.02 f/s
1140360: done 3172 episodes, mean reward -86.395, speed 75.76 f/s
1140459: done 3173 episodes, mean reward -86.290, speed 71.03 f/s
1140563: done 3174 episodes, mean reward -86.430, speed 76.87 f/s
1140674: done 3175 episodes, mean reward -86.781, speed 79.28 f/s
1140943: done 3176 episodes, mean reward -86.429, speed 73.34 f/s
Test done in 9.27 sec, reward 231.089, steps 1236
1141097: done 3177 episodes, mean reward -86.171, speed 13.56 f/s
1141277: done 3178 episodes, mean reward -86.279, speed 66.56 f/s
1141408: done 3179 episodes, mean reward -86.220, speed 62.57 f/s
1141536: done 3180 episodes, mean reward -86.128, speed 64.55 f/s
1141932: done 3181 episodes, mean reward -85.688, speed 72.00 f/s
Test done in 9.75 sec, reward 246.772, steps 1247
1142016: done 3182 episodes, mean reward -85.671, speed 7.74 f/s
1142171: done 3183 episodes, mean reward -86.070, speed 65.68 f/s
1142262: done 3184 episodes, mean reward -86.009, speed 72.72 f/s
1142588: done 3185 episodes, mean reward -85.641, speed 70.60 f/s
1142694: done 3186 episodes, mean reward -85.723, speed 58.44 f/s
Test done in 8.36 sec, reward 207.105, steps 1118
1143040: done 3187 episodes, mean reward -85.528, speed 26.70 f/s
1143199: done 3188 episodes, mean reward -85.484, speed 66.98 f/s
1143291: done 3189 episodes, mean reward -85.688, speed 67.10 f/s
1143565: done 3190 episodes, mean reward -85.451, speed 74.38 f/s
1143731: done 3191 episodes, mean reward -85.724, speed 68.78 f/s
1143930: done 3192 episodes, mean reward -85.845, speed 74.69 f/s
Test done in 8.95 sec, reward 264.464, steps 1270
1144147: done 3193 episodes, mean reward -86.144, speed 18.46 f/s
1144254: done 3194 episodes, mean reward -86.334, speed 76.26 f/s
1144492: done 3195 episodes, mean reward -86.298, speed 79.99 f/s
1144681: done 3196 episodes, mean reward -85.990, speed 80.67 f/s
1144856: done 3197 episodes, mean reward -86.094, speed 78.74 f/s
Test done in 8.56 sec, reward 237.485, steps 1270
1145239: done 3198 episodes, mean reward -85.851, speed 28.62 f/s
1145417: done 3199 episodes, mean reward -85.627, speed 80.58 f/s
1145745: done 3200 episodes, mean reward -84.984, speed 79.69 f/s
1145861: done 3201 episodes, mean reward -84.975, speed 80.57 f/s
Test done in 8.46 sec, reward 250.479, steps 1216
1146041: done 3202 episodes, mean reward -85.041, speed 16.87 f/s
1146204: done 3203 episodes, mean reward -85.332, speed 79.21 f/s
1146424: done 3205 episodes, mean reward -85.377, speed 78.71 f/s
1146650: done 3206 episodes, mean reward -85.456, speed 81.61 f/s
1146800: done 3207 episodes, mean reward -85.547, speed 78.88 f/s
1146977: done 3208 episodes, mean reward -85.437, speed 77.62 f/s
Test done in 10.54 sec, reward 262.120, steps 1293
1147170: done 3209 episodes, mean reward -85.379, speed 14.28 f/s
1147253: done 3210 episodes, mean reward -85.695, speed 67.42 f/s
1147511: done 3211 episodes, mean reward -86.070, speed 78.53 f/s
1147667: done 3212 episodes, mean reward -85.920, speed 73.35 f/s
1147946: done 3213 episodes, mean reward -85.613, speed 72.00 f/s
Test done in 7.76 sec, reward 189.363, steps 1095
1148078: done 3214 episodes, mean reward -85.944, speed 14.02 f/s
1148180: done 3215 episodes, mean reward -85.980, speed 78.12 f/s
1148516: done 3216 episodes, mean reward -85.813, speed 79.19 f/s
1148679: done 3217 episodes, mean reward -85.814, speed 78.47 f/s
1148777: done 3218 episodes, mean reward -86.148, speed 76.43 f/s
1148916: done 3219 episodes, mean reward -85.698, speed 84.80 f/s
Test done in 9.24 sec, reward 259.606, steps 1319
1149203: done 3220 episodes, mean reward -85.453, speed 21.81 f/s
1149443: done 3221 episodes, mean reward -85.377, speed 68.87 f/s
1149569: done 3222 episodes, mean reward -85.501, speed 72.11 f/s
1149767: done 3223 episodes, mean reward -85.316, speed 61.71 f/s
1149947: done 3224 episodes, mean reward -85.025, speed 63.14 f/s
Test done in 10.21 sec, reward 226.465, steps 1303
1150054: done 3225 episodes, mean reward -85.240, speed 9.12 f/s
1150169: done 3226 episodes, mean reward -85.314, speed 68.30 f/s
1150293: done 3227 episodes, mean reward -85.152, speed 71.25 f/s
1150429: done 3228 episodes, mean reward -86.061, speed 81.13 f/s
1150500: done 3229 episodes, mean reward -86.389, speed 61.86 f/s
1150686: done 3230 episodes, mean reward -86.240, speed 65.84 f/s
1150791: done 3231 episodes, mean reward -86.172, speed 65.25 f/s
1150924: done 3232 episodes, mean reward -86.358, speed 79.44 f/s
Test done in 9.20 sec, reward 237.398, steps 1217
1151035: done 3233 episodes, mean reward -86.460, speed 10.41 f/s
1151262: done 3234 episodes, mean reward -86.252, speed 70.19 f/s
1151413: done 3235 episodes, mean reward -86.355, speed 68.38 f/s
1151571: done 3236 episodes, mean reward -86.827, speed 71.21 f/s
1151684: done 3237 episodes, mean reward -86.711, speed 75.80 f/s
1151816: done 3238 episodes, mean reward -86.477, speed 63.77 f/s
1151916: done 3239 episodes, mean reward -86.584, speed 68.23 f/s
Test done in 11.39 sec, reward 238.120, steps 1356
1152358: done 3240 episodes, mean reward -85.731, speed 25.60 f/s
1152639: done 3241 episodes, mean reward -85.535, speed 80.09 f/s
1152961: done 3242 episodes, mean reward -85.246, speed 81.47 f/s
Test done in 8.88 sec, reward 263.294, steps 1264
1153059: done 3243 episodes, mean reward -85.456, speed 9.69 f/s
1153392: done 3244 episodes, mean reward -84.911, speed 80.66 f/s
1153935: done 3246 episodes, mean reward -84.193, speed 77.54 f/s
Test done in 8.96 sec, reward 260.472, steps 1322
1154027: done 3247 episodes, mean reward -84.458, speed 9.06 f/s
1154467: done 3248 episodes, mean reward -83.849, speed 81.40 f/s
1154645: done 3250 episodes, mean reward -84.029, speed 79.78 f/s
1154812: done 3251 episodes, mean reward -84.526, speed 71.37 f/s
Test done in 9.34 sec, reward 263.178, steps 1268
1155042: done 3252 episodes, mean reward -84.488, speed 18.69 f/s
1155130: done 3253 episodes, mean reward -84.555, speed 72.63 f/s
1155357: done 3254 episodes, mean reward -84.214, speed 74.47 f/s
1155461: done 3255 episodes, mean reward -84.435, speed 78.06 f/s
1155628: done 3256 episodes, mean reward -84.319, speed 75.86 f/s
1155914: done 3257 episodes, mean reward -83.797, speed 78.95 f/s
Test done in 8.41 sec, reward 261.412, steps 1285
1156086: done 3259 episodes, mean reward -84.096, speed 16.36 f/s
1156339: done 3260 episodes, mean reward -83.753, speed 81.66 f/s
1156483: done 3261 episodes, mean reward -83.601, speed 81.33 f/s
1156604: done 3262 episodes, mean reward -83.970, speed 73.46 f/s
1156692: done 3263 episodes, mean reward -84.270, speed 75.28 f/s
1156817: done 3264 episodes, mean reward -84.357, speed 81.25 f/s
Test done in 9.13 sec, reward 259.805, steps 1361
1157075: done 3265 episodes, mean reward -84.219, speed 21.00 f/s
1157234: done 3266 episodes, mean reward -84.421, speed 81.73 f/s
1157493: done 3267 episodes, mean reward -84.267, speed 81.41 f/s
1157645: done 3268 episodes, mean reward -84.059, speed 83.04 f/s
1157766: done 3269 episodes, mean reward -83.969, speed 80.35 f/s
Test done in 8.94 sec, reward 257.916, steps 1340
1158091: done 3270 episodes, mean reward -83.459, speed 25.30 f/s
1158273: done 3272 episodes, mean reward -83.293, speed 81.91 f/s
1158402: done 3273 episodes, mean reward -83.274, speed 83.50 f/s
1158715: done 3274 episodes, mean reward -82.787, speed 82.21 f/s
1158876: done 3275 episodes, mean reward -82.762, speed 81.57 f/s
1158964: done 3276 episodes, mean reward -83.221, speed 82.55 f/s
Test done in 8.46 sec, reward 258.757, steps 1282
1159064: done 3277 episodes, mean reward -83.404, speed 10.27 f/s
1159500: done 3279 episodes, mean reward -83.003, speed 79.29 f/s
1159579: done 3280 episodes, mean reward -83.118, speed 77.57 f/s
1159739: done 3281 episodes, mean reward -83.476, speed 81.55 f/s
1159995: done 3283 episodes, mean reward -83.255, speed 84.30 f/s
Test done in 8.63 sec, reward 260.965, steps 1249
1160985: done 3284 episodes, mean reward -81.338, speed 47.56 f/s
Test done in 8.81 sec, reward 260.752, steps 1286
1161277: done 3285 episodes, mean reward -81.260, speed 23.63 f/s
1161532: done 3286 episodes, mean reward -80.892, speed 83.21 f/s
1161662: done 3287 episodes, mean reward -81.303, speed 85.33 f/s
1161810: done 3288 episodes, mean reward -81.300, speed 81.47 f/s
Test done in 8.48 sec, reward 259.588, steps 1283
1162236: done 3289 episodes, mean reward -80.646, speed 31.34 f/s
1162439: done 3290 episodes, mean reward -80.813, speed 78.28 f/s
1162569: done 3291 episodes, mean reward -80.700, speed 76.71 f/s
1162658: done 3292 episodes, mean reward -81.018, speed 82.18 f/s
Test done in 9.71 sec, reward 246.083, steps 1470
1163217: done 3293 episodes, mean reward -80.487, speed 33.67 f/s
1163343: done 3294 episodes, mean reward -80.354, speed 79.92 f/s
1163542: done 3295 episodes, mean reward -80.355, speed 85.54 f/s
1163656: done 3296 episodes, mean reward -80.530, speed 81.17 f/s
1163873: done 3297 episodes, mean reward -80.450, speed 81.07 f/s
1163982: done 3298 episodes, mean reward -81.080, speed 80.44 f/s
Test done in 9.77 sec, reward 247.953, steps 1449
1164108: done 3299 episodes, mean reward -81.253, speed 11.16 f/s
1164243: done 3300 episodes, mean reward -81.818, speed 81.89 f/s
1164340: done 3301 episodes, mean reward -81.836, speed 80.11 f/s
1164450: done 3302 episodes, mean reward -81.965, speed 78.17 f/s
1164718: done 3303 episodes, mean reward -81.844, speed 80.71 f/s
Test done in 9.35 sec, reward 252.817, steps 1392
1165193: done 3304 episodes, mean reward -81.042, speed 30.93 f/s
1165484: done 3305 episodes, mean reward -80.805, speed 83.04 f/s
1165614: done 3306 episodes, mean reward -81.076, speed 83.13 f/s
1165935: done 3307 episodes, mean reward -80.644, speed 83.18 f/s
Test done in 8.77 sec, reward 260.209, steps 1290
1166109: done 3308 episodes, mean reward -80.606, speed 16.04 f/s
1166371: done 3309 episodes, mean reward -80.414, speed 82.63 f/s
1166489: done 3310 episodes, mean reward -80.278, speed 81.21 f/s
1166623: done 3311 episodes, mean reward -80.391, speed 82.26 f/s
1166781: done 3312 episodes, mean reward -80.508, speed 79.24 f/s
1166905: done 3313 episodes, mean reward -80.795, speed 75.06 f/s
Test done in 8.66 sec, reward 258.353, steps 1264
1167103: done 3314 episodes, mean reward -80.586, speed 17.98 f/s
1167296: done 3315 episodes, mean reward -80.731, speed 82.37 f/s
1167516: done 3316 episodes, mean reward -80.992, speed 83.96 f/s
1167692: done 3317 episodes, mean reward -80.940, speed 85.35 f/s
1167892: done 3318 episodes, mean reward -80.797, speed 80.96 f/s
1167985: done 3319 episodes, mean reward -80.931, speed 80.36 f/s
Test done in 9.03 sec, reward 250.479, steps 1336
1168238: done 3320 episodes, mean reward -81.165, speed 20.95 f/s
1168339: done 3321 episodes, mean reward -81.574, speed 82.38 f/s
1168426: done 3322 episodes, mean reward -81.651, speed 83.52 f/s
1168613: done 3324 episodes, mean reward -82.227, speed 83.68 f/s
1168709: done 3325 episodes, mean reward -82.288, speed 79.70 f/s
Test done in 8.32 sec, reward 97.365, steps 1208
1169052: done 3326 episodes, mean reward -81.729, speed 27.48 f/s
1169167: done 3327 episodes, mean reward -81.739, speed 83.29 f/s
1169323: done 3328 episodes, mean reward -81.512, speed 79.29 f/s
1169636: done 3329 episodes, mean reward -80.816, speed 81.04 f/s
1169746: done 3330 episodes, mean reward -81.160, speed 83.61 f/s
1169955: done 3331 episodes, mean reward -81.164, speed 84.77 f/s
Test done in 9.20 sec, reward 249.550, steps 1371
1170081: done 3332 episodes, mean reward -81.059, speed 11.79 f/s
1170210: done 3333 episodes, mean reward -81.051, speed 83.39 f/s
1170443: done 3334 episodes, mean reward -81.193, speed 85.22 f/s
1170628: done 3335 episodes, mean reward -81.155, speed 84.39 f/s
1170830: done 3337 episodes, mean reward -81.304, speed 83.82 f/s
Test done in 9.05 sec, reward 241.582, steps 1298
1171050: done 3338 episodes, mean reward -81.229, speed 18.66 f/s
1171535: done 3339 episodes, mean reward -80.359, speed 81.15 f/s
1171655: done 3340 episodes, mean reward -81.076, speed 79.82 f/s
1171968: done 3341 episodes, mean reward -81.086, speed 82.13 f/s
Test done in 9.76 sec, reward 247.366, steps 1419
1172202: done 3342 episodes, mean reward -81.068, speed 18.57 f/s
1172312: done 3343 episodes, mean reward -81.174, speed 79.23 f/s
1172633: done 3344 episodes, mean reward -81.329, speed 82.20 f/s
1172737: done 3345 episodes, mean reward -81.478, speed 83.29 f/s
1172844: done 3346 episodes, mean reward -82.293, speed 81.38 f/s
Test done in 7.96 sec, reward 207.631, steps 1202
1173235: done 3348 episodes, mean reward -82.545, speed 30.98 f/s
1173656: done 3349 episodes, mean reward -81.819, speed 83.71 f/s
1173799: done 3350 episodes, mean reward -81.874, speed 84.83 f/s
Test done in 10.20 sec, reward 240.458, steps 1479
1174015: done 3351 episodes, mean reward -81.664, speed 16.79 f/s
1174473: done 3352 episodes, mean reward -81.326, speed 82.13 f/s
1174823: done 3353 episodes, mean reward -80.675, speed 80.69 f/s
Test done in 8.84 sec, reward 255.817, steps 1301
1175189: done 3354 episodes, mean reward -80.367, speed 27.47 f/s
1175287: done 3355 episodes, mean reward -80.508, speed 79.78 f/s
1175391: done 3356 episodes, mean reward -80.680, speed 83.37 f/s
1175500: done 3357 episodes, mean reward -81.104, speed 81.94 f/s
1175944: done 3358 episodes, mean reward -80.509, speed 83.08 f/s
Test done in 8.48 sec, reward 257.849, steps 1277
1176042: done 3359 episodes, mean reward -80.644, speed 10.16 f/s
1176409: done 3360 episodes, mean reward -80.507, speed 82.99 f/s
1176937: done 3361 episodes, mean reward -79.830, speed 82.86 f/s
Test done in 8.84 sec, reward 255.327, steps 1306
1177119: done 3362 episodes, mean reward -79.761, speed 16.44 f/s
1177245: done 3363 episodes, mean reward -79.698, speed 81.02 f/s
1177453: done 3364 episodes, mean reward -79.608, speed 80.33 f/s
1177653: done 3365 episodes, mean reward -79.844, speed 78.75 f/s
1177807: done 3366 episodes, mean reward -79.830, speed 85.20 f/s
1177991: done 3367 episodes, mean reward -80.083, speed 80.23 f/s
Test done in 8.92 sec, reward 255.639, steps 1307
1178050: done 3368 episodes, mean reward -80.311, speed 6.10 f/s
1178240: done 3369 episodes, mean reward -80.325, speed 79.53 f/s
1178483: done 3371 episodes, mean reward -80.893, speed 83.58 f/s
1178566: done 3372 episodes, mean reward -80.999, speed 81.01 f/s
1178698: done 3373 episodes, mean reward -81.082, speed 86.25 f/s
1178907: done 3374 episodes, mean reward -81.328, speed 80.45 f/s
Test done in 8.77 sec, reward 250.578, steps 1337
1179155: done 3375 episodes, mean reward -81.091, speed 21.18 f/s
1179282: done 3376 episodes, mean reward -80.984, speed 79.53 f/s
1179418: done 3377 episodes, mean reward -80.923, speed 78.91 f/s
1179588: done 3378 episodes, mean reward -80.737, speed 82.79 f/s
1179793: done 3380 episodes, mean reward -81.172, speed 83.99 f/s
1179943: done 3381 episodes, mean reward -81.148, speed 83.05 f/s
Test done in 9.30 sec, reward 253.779, steps 1318
1180222: done 3382 episodes, mean reward -80.780, speed 21.85 f/s
1180330: done 3383 episodes, mean reward -81.023, speed 83.14 f/s
1180483: done 3384 episodes, mean reward -82.888, speed 83.04 f/s
1180659: done 3385 episodes, mean reward -83.245, speed 81.26 f/s
1180756: done 3386 episodes, mean reward -83.715, speed 82.33 f/s
1180947: done 3387 episodes, mean reward -83.615, speed 81.27 f/s
Test done in 8.52 sec, reward 258.850, steps 1269
1181134: done 3388 episodes, mean reward -83.525, speed 17.43 f/s
1181266: done 3389 episodes, mean reward -84.231, speed 79.75 f/s
1181353: done 3390 episodes, mean reward -84.524, speed 83.87 f/s
1181511: done 3391 episodes, mean reward -84.626, speed 81.92 f/s
1181623: done 3392 episodes, mean reward -84.541, speed 81.87 f/s
1181949: done 3394 episodes, mean reward -85.086, speed 81.78 f/s
Test done in 8.54 sec, reward 261.407, steps 1269
1182091: done 3395 episodes, mean reward -85.357, speed 13.86 f/s
1182222: done 3396 episodes, mean reward -85.446, speed 81.96 f/s
1182366: done 3397 episodes, mean reward -85.623, speed 83.01 f/s
1182501: done 3398 episodes, mean reward -85.552, speed 82.97 f/s
1182614: done 3399 episodes, mean reward -85.587, speed 81.77 f/s
1182903: done 3400 episodes, mean reward -85.173, speed 81.11 f/s
Test done in 8.85 sec, reward 258.742, steps 1302
1183026: done 3401 episodes, mean reward -85.088, speed 11.90 f/s
1183172: done 3402 episodes, mean reward -85.104, speed 79.89 f/s
1183286: done 3403 episodes, mean reward -85.484, speed 74.73 f/s
1183589: done 3404 episodes, mean reward -85.743, speed 81.96 f/s
1183749: done 3405 episodes, mean reward -86.043, speed 81.46 f/s
1183914: done 3406 episodes, mean reward -85.823, speed 85.45 f/s
Test done in 8.97 sec, reward 253.529, steps 1343
1184158: done 3407 episodes, mean reward -86.009, speed 20.39 f/s
1184242: done 3408 episodes, mean reward -86.291, speed 77.74 f/s
1184350: done 3409 episodes, mean reward -86.803, speed 83.81 f/s
1184465: done 3410 episodes, mean reward -86.850, speed 84.29 f/s
1184699: done 3412 episodes, mean reward -86.898, speed 82.02 f/s
1184843: done 3413 episodes, mean reward -86.783, speed 82.84 f/s
Test done in 8.75 sec, reward 256.089, steps 1327
1185006: done 3415 episodes, mean reward -86.865, speed 15.16 f/s
1185341: done 3416 episodes, mean reward -86.501, speed 83.30 f/s
1185458: done 3417 episodes, mean reward -86.758, speed 83.26 f/s
1185619: done 3418 episodes, mean reward -86.888, speed 81.96 f/s
1185903: done 3419 episodes, mean reward -86.414, speed 83.84 f/s
Test done in 8.91 sec, reward 259.829, steps 1270
1186163: done 3420 episodes, mean reward -86.312, speed 21.60 f/s
1186275: done 3421 episodes, mean reward -86.389, speed 84.99 f/s
1186463: done 3422 episodes, mean reward -86.015, speed 81.09 f/s
1186615: done 3424 episodes, mean reward -86.208, speed 82.89 f/s
1186770: done 3425 episodes, mean reward -85.955, speed 79.68 f/s
1186874: done 3426 episodes, mean reward -86.444, speed 81.41 f/s
Test done in 8.43 sec, reward 260.990, steps 1250
1187185: done 3427 episodes, mean reward -85.933, speed 25.39 f/s
1187373: done 3428 episodes, mean reward -85.889, speed 79.89 f/s
1187545: done 3429 episodes, mean reward -86.226, speed 83.59 f/s
1187697: done 3430 episodes, mean reward -85.937, speed 79.64 f/s
1187836: done 3431 episodes, mean reward -85.966, speed 82.15 f/s
Test done in 8.89 sec, reward 257.560, steps 1292
1188027: done 3432 episodes, mean reward -85.895, speed 16.95 f/s
1188131: done 3433 episodes, mean reward -85.826, speed 84.83 f/s
1188225: done 3434 episodes, mean reward -85.896, speed 81.80 f/s
1188356: done 3435 episodes, mean reward -85.851, speed 80.05 f/s
1188658: done 3436 episodes, mean reward -85.400, speed 83.85 f/s
1188794: done 3437 episodes, mean reward -85.356, speed 81.15 f/s
1188955: done 3439 episodes, mean reward -86.610, speed 83.39 f/s
Test done in 8.53 sec, reward 261.711, steps 1243
1189087: done 3440 episodes, mean reward -86.584, speed 12.91 f/s
1189213: done 3441 episodes, mean reward -87.040, speed 86.67 f/s
1189581: done 3442 episodes, mean reward -86.742, speed 80.99 f/s
1189758: done 3443 episodes, mean reward -86.373, speed 79.65 f/s
1189903: done 3444 episodes, mean reward -86.588, speed 80.81 f/s
Test done in 8.41 sec, reward 261.511, steps 1268
1190205: done 3445 episodes, mean reward -85.848, speed 24.70 f/s
1190332: done 3446 episodes, mean reward -85.832, speed 78.52 f/s
1190489: done 3447 episodes, mean reward -85.554, speed 81.94 f/s
1190710: done 3448 episodes, mean reward -85.713, speed 83.38 f/s
1190797: done 3449 episodes, mean reward -86.474, speed 81.07 f/s
1190900: done 3450 episodes, mean reward -86.562, speed 82.21 f/s
Test done in 8.48 sec, reward 263.390, steps 1245
1191002: done 3451 episodes, mean reward -86.947, speed 10.53 f/s
1191141: done 3452 episodes, mean reward -87.449, speed 82.24 f/s
1191390: done 3453 episodes, mean reward -87.696, speed 82.55 f/s
1191548: done 3454 episodes, mean reward -88.025, speed 81.07 f/s
1191644: done 3455 episodes, mean reward -87.902, speed 80.00 f/s
1191810: done 3457 episodes, mean reward -88.010, speed 80.17 f/s
Test done in 8.72 sec, reward 259.981, steps 1272
1192103: done 3459 episodes, mean reward -88.306, speed 23.22 f/s
1192226: done 3460 episodes, mean reward -88.668, speed 83.37 f/s
1192345: done 3461 episodes, mean reward -89.563, speed 79.87 f/s
1192534: done 3462 episodes, mean reward -89.596, speed 82.89 f/s
1192717: done 3463 episodes, mean reward -89.423, speed 81.06 f/s
1192915: done 3465 episodes, mean reward -89.866, speed 84.07 f/s
Test done in 8.44 sec, reward 262.043, steps 1260
1193162: done 3467 episodes, mean reward -90.089, speed 21.47 f/s
1193271: done 3468 episodes, mean reward -89.951, speed 81.55 f/s
1193380: done 3469 episodes, mean reward -90.038, speed 81.38 f/s
1193478: done 3470 episodes, mean reward -89.945, speed 77.58 f/s
1193674: done 3471 episodes, mean reward -89.775, speed 82.72 f/s
1193809: done 3472 episodes, mean reward -89.697, speed 82.37 f/s
1193988: done 3474 episodes, mean reward -89.973, speed 83.85 f/s
Test done in 8.47 sec, reward 259.006, steps 1269
1194299: done 3475 episodes, mean reward -89.746, speed 25.48 f/s
1194578: done 3476 episodes, mean reward -89.391, speed 81.18 f/s
1194833: done 3477 episodes, mean reward -89.148, speed 83.62 f/s
Test done in 8.75 sec, reward 258.322, steps 1293
1195002: done 3478 episodes, mean reward -89.095, speed 15.70 f/s
1195182: done 3480 episodes, mean reward -89.199, speed 78.39 f/s
1195279: done 3481 episodes, mean reward -89.338, speed 86.74 f/s
1195452: done 3482 episodes, mean reward -89.562, speed 81.13 f/s
1195670: done 3484 episodes, mean reward -89.592, speed 84.11 f/s
1195876: done 3485 episodes, mean reward -89.409, speed 83.46 f/s
1195979: done 3486 episodes, mean reward -89.315, speed 84.07 f/s
Test done in 8.83 sec, reward 256.320, steps 1307
1196251: done 3487 episodes, mean reward -89.163, speed 21.96 f/s
1196377: done 3488 episodes, mean reward -89.314, speed 82.09 f/s
1196462: done 3489 episodes, mean reward -89.333, speed 78.72 f/s
1196581: done 3490 episodes, mean reward -89.352, speed 80.41 f/s
1196689: done 3491 episodes, mean reward -89.284, speed 87.06 f/s
Test done in 8.29 sec, reward 260.406, steps 1232
1197070: done 3492 episodes, mean reward -88.782, speed 29.33 f/s
1197232: done 3493 episodes, mean reward -88.675, speed 81.28 f/s
1197390: done 3494 episodes, mean reward -88.918, speed 85.61 f/s
1197589: done 3495 episodes, mean reward -88.609, speed 78.05 f/s
1197879: done 3496 episodes, mean reward -88.073, speed 82.80 f/s
Test done in 8.21 sec, reward 260.300, steps 1229
1198046: done 3497 episodes, mean reward -87.999, speed 16.34 f/s
1198337: done 3498 episodes, mean reward -87.465, speed 81.54 f/s
1198461: done 3499 episodes, mean reward -87.495, speed 83.40 f/s
1198689: done 3500 episodes, mean reward -87.493, speed 83.96 f/s
1198862: done 3501 episodes, mean reward -87.298, speed 78.15 f/s
Test done in 7.95 sec, reward 263.637, steps 1192
1199006: done 3503 episodes, mean reward -87.356, speed 14.71 f/s
1199154: done 3504 episodes, mean reward -87.643, speed 76.19 f/s
1199331: done 3506 episodes, mean reward -88.095, speed 75.61 f/s
1199452: done 3507 episodes, mean reward -88.443, speed 80.86 f/s
1199576: done 3508 episodes, mean reward -88.291, speed 82.00 f/s
1199828: done 3509 episodes, mean reward -87.793, speed 81.59 f/s
1199981: done 3511 episodes, mean reward -87.920, speed 82.17 f/s
Test done in 7.63 sec, reward 266.944, steps 1159
1200073: done 3512 episodes, mean reward -87.999, speed 10.49 f/s
1200226: done 3513 episodes, mean reward -87.942, speed 87.44 f/s
1200375: done 3514 episodes, mean reward -87.780, speed 82.73 f/s
1200496: done 3515 episodes, mean reward -87.886, speed 83.66 f/s
1200988: done 3516 episodes, mean reward -87.502, speed 82.32 f/s
Test done in 7.94 sec, reward 263.512, steps 1188
1201133: done 3517 episodes, mean reward -87.313, speed 14.95 f/s
1201221: done 3518 episodes, mean reward -87.362, speed 82.96 f/s
1201402: done 3519 episodes, mean reward -87.552, speed 82.99 f/s
1201936: done 3520 episodes, mean reward -86.881, speed 82.31 f/s
Test done in 7.73 sec, reward 265.583, steps 1174
1202334: done 3521 episodes, mean reward -86.020, speed 31.26 f/s
1202448: done 3522 episodes, mean reward -86.192, speed 76.63 f/s
1202850: done 3523 episodes, mean reward -85.497, speed 83.93 f/s
Test done in 7.63 sec, reward 266.006, steps 1169
1203030: done 3525 episodes, mean reward -85.651, speed 18.32 f/s
1203424: done 3526 episodes, mean reward -84.882, speed 84.20 f/s
1203519: done 3527 episodes, mean reward -85.458, speed 83.59 f/s
1203729: done 3528 episodes, mean reward -85.512, speed 77.35 f/s
1203905: done 3529 episodes, mean reward -85.611, speed 83.71 f/s
1203989: done 3530 episodes, mean reward -85.851, speed 81.43 f/s
Test done in 7.39 sec, reward 232.756, steps 1095
1204112: done 3531 episodes, mean reward -85.765, speed 13.73 f/s
1204275: done 3532 episodes, mean reward -85.700, speed 80.87 f/s
1204409: done 3533 episodes, mean reward -85.640, speed 82.26 f/s
1204601: done 3534 episodes, mean reward -85.512, speed 79.62 f/s
1204710: done 3535 episodes, mean reward -85.535, speed 81.59 f/s
1204825: done 3536 episodes, mean reward -85.963, speed 81.40 f/s
1204983: done 3537 episodes, mean reward -85.911, speed 83.96 f/s
Test done in 7.77 sec, reward 264.713, steps 1194
1205076: done 3538 episodes, mean reward -85.872, speed 10.40 f/s
1205244: done 3540 episodes, mean reward -86.097, speed 83.02 f/s
1205760: done 3541 episodes, mean reward -85.170, speed 78.21 f/s
1205938: done 3542 episodes, mean reward -85.630, speed 83.86 f/s
Test done in 6.76 sec, reward 210.628, steps 1007
1206094: done 3543 episodes, mean reward -85.711, speed 17.90 f/s
1206337: done 3544 episodes, mean reward -85.427, speed 81.65 f/s
1206429: done 3545 episodes, mean reward -86.098, speed 80.12 f/s
1206556: done 3546 episodes, mean reward -85.956, speed 81.45 f/s
1206792: done 3547 episodes, mean reward -85.843, speed 81.77 f/s
1206980: done 3548 episodes, mean reward -86.021, speed 85.05 f/s
Test done in 7.74 sec, reward 265.684, steps 1140
1207137: done 3549 episodes, mean reward -85.822, speed 16.10 f/s
1207289: done 3551 episodes, mean reward -85.850, speed 81.49 f/s
1207456: done 3552 episodes, mean reward -85.779, speed 82.71 f/s
1207649: done 3553 episodes, mean reward -85.782, speed 80.62 f/s
1207812: done 3555 episodes, mean reward -86.075, speed 81.30 f/s
Test done in 7.65 sec, reward 267.279, steps 1136
1208032: done 3556 episodes, mean reward -85.559, speed 21.43 f/s
1208434: done 3557 episodes, mean reward -84.650, speed 83.17 f/s
1208621: done 3559 episodes, mean reward -84.996, speed 81.31 f/s
1208813: done 3560 episodes, mean reward -84.795, speed 84.44 f/s
Test done in 7.56 sec, reward 267.051, steps 1154
1209160: done 3561 episodes, mean reward -84.038, speed 29.44 f/s
1209270: done 3562 episodes, mean reward -84.154, speed 85.07 f/s
1209519: done 3564 episodes, mean reward -84.346, speed 83.08 f/s
1209612: done 3565 episodes, mean reward -84.383, speed 85.55 f/s
1209783: done 3566 episodes, mean reward -84.147, speed 78.52 f/s
1209978: done 3567 episodes, mean reward -84.066, speed 82.71 f/s
Test done in 7.79 sec, reward 267.688, steps 1152
1210050: done 3568 episodes, mean reward -84.131, speed 8.25 f/s
1210256: done 3569 episodes, mean reward -83.913, speed 83.31 f/s
1210379: done 3570 episodes, mean reward -83.896, speed 80.92 f/s
1210512: done 3571 episodes, mean reward -84.048, speed 86.44 f/s
1210697: done 3572 episodes, mean reward -83.934, speed 84.37 f/s
1210807: done 3573 episodes, mean reward -83.860, speed 82.21 f/s
1210910: done 3574 episodes, mean reward -83.937, speed 82.94 f/s
Test done in 7.48 sec, reward 269.033, steps 1123
Best reward updated: 268.292 -> 269.033
1211004: done 3575 episodes, mean reward -84.557, speed 10.82 f/s
1211272: done 3576 episodes, mean reward -84.557, speed 79.93 f/s
1211517: done 3577 episodes, mean reward -84.429, speed 82.02 f/s
1211619: done 3578 episodes, mean reward -84.616, speed 77.20 f/s
1211949: done 3580 episodes, mean reward -84.158, speed 80.49 f/s
Test done in 7.93 sec, reward 267.345, steps 1178
1212085: done 3581 episodes, mean reward -84.177, speed 14.11 f/s
1212182: done 3582 episodes, mean reward -84.288, speed 82.37 f/s
1212356: done 3583 episodes, mean reward -84.075, speed 83.51 f/s
1212508: done 3585 episodes, mean reward -84.574, speed 83.58 f/s
1212633: done 3586 episodes, mean reward -84.477, speed 81.86 f/s
1212724: done 3587 episodes, mean reward -84.831, speed 79.48 f/s
Test done in 7.74 sec, reward 269.671, steps 1141
Best reward updated: 269.033 -> 269.671
1213094: done 3588 episodes, mean reward -84.169, speed 30.21 f/s
1213209: done 3589 episodes, mean reward -84.091, speed 82.24 f/s
1213311: done 3590 episodes, mean reward -84.091, speed 80.99 f/s
1213420: done 3591 episodes, mean reward -84.203, speed 82.57 f/s
1213590: done 3592 episodes, mean reward -84.545, speed 83.69 f/s
1213737: done 3593 episodes, mean reward -84.594, speed 81.03 f/s
1213929: done 3594 episodes, mean reward -84.499, speed 81.66 f/s
Test done in 6.99 sec, reward 237.611, steps 1070
1214029: done 3595 episodes, mean reward -84.714, speed 12.11 f/s
1214109: done 3596 episodes, mean reward -85.275, speed 78.67 f/s
1214591: done 3597 episodes, mean reward -84.477, speed 80.75 f/s
1214746: done 3598 episodes, mean reward -84.768, speed 80.63 f/s
1214848: done 3599 episodes, mean reward -84.864, speed 81.07 f/s
1214963: done 3600 episodes, mean reward -85.188, speed 81.36 f/s
Test done in 8.18 sec, reward 269.394, steps 1119
1215026: done 3601 episodes, mean reward -85.517, speed 7.04 f/s
1215124: done 3602 episodes, mean reward -85.528, speed 78.75 f/s
1215237: done 3603 episodes, mean reward -85.414, speed 81.24 f/s
1215363: done 3604 episodes, mean reward -85.595, speed 80.30 f/s
1215465: done 3605 episodes, mean reward -85.584, speed 82.13 f/s
1215724: done 3607 episodes, mean reward -85.254, speed 78.44 f/s
1215812: done 3608 episodes, mean reward -85.397, speed 78.78 f/s
1215904: done 3609 episodes, mean reward -85.788, speed 83.10 f/s
Test done in 7.68 sec, reward 269.117, steps 1134
1216168: done 3610 episodes, mean reward -85.159, speed 24.21 f/s
1216278: done 3611 episodes, mean reward -85.203, speed 81.91 f/s
1216397: done 3612 episodes, mean reward -85.183, speed 82.42 f/s
1216485: done 3613 episodes, mean reward -85.435, speed 80.00 f/s
1216573: done 3614 episodes, mean reward -85.485, speed 82.19 f/s
1216659: done 3615 episodes, mean reward -85.559, speed 84.19 f/s
Test done in 7.90 sec, reward 269.015, steps 1144
1217070: done 3616 episodes, mean reward -85.812, speed 31.74 f/s
1217226: done 3617 episodes, mean reward -85.762, speed 81.20 f/s
1217326: done 3618 episodes, mean reward -85.676, speed 83.07 f/s
1217512: done 3619 episodes, mean reward -85.751, speed 82.72 f/s
1217670: done 3620 episodes, mean reward -86.683, speed 79.72 f/s
1217798: done 3621 episodes, mean reward -87.380, speed 80.88 f/s
1217896: done 3622 episodes, mean reward -87.392, speed 81.79 f/s
Test done in 7.93 sec, reward 249.927, steps 1108
1218061: done 3623 episodes, mean reward -87.830, speed 16.56 f/s
1218365: done 3624 episodes, mean reward -87.149, speed 82.02 f/s
1218493: done 3625 episodes, mean reward -86.956, speed 78.58 f/s
1218597: done 3626 episodes, mean reward -87.724, speed 85.87 f/s
1218707: done 3627 episodes, mean reward -87.719, speed 75.28 f/s
1218915: done 3628 episodes, mean reward -87.698, speed 81.45 f/s
Test done in 8.00 sec, reward 267.409, steps 1170
1219049: done 3629 episodes, mean reward -87.725, speed 13.87 f/s
1219284: done 3630 episodes, mean reward -87.364, speed 82.10 f/s
1219391: done 3631 episodes, mean reward -87.523, speed 84.03 f/s
1219525: done 3632 episodes, mean reward -87.568, speed 78.28 f/s
1219608: done 3633 episodes, mean reward -87.749, speed 79.07 f/s
1219794: done 3634 episodes, mean reward -87.648, speed 80.82 f/s
1219917: done 3635 episodes, mean reward -87.781, speed 80.18 f/s
Test done in 7.69 sec, reward 267.633, steps 1138
1220125: done 3636 episodes, mean reward -87.504, speed 20.25 f/s
1220267: done 3637 episodes, mean reward -87.613, speed 80.60 f/s
1220485: done 3638 episodes, mean reward -87.201, speed 81.95 f/s
1220672: done 3639 episodes, mean reward -86.857, speed 81.57 f/s
1220779: done 3640 episodes, mean reward -86.898, speed 79.70 f/s
1220996: done 3641 episodes, mean reward -87.517, speed 82.89 f/s
Test done in 7.59 sec, reward 254.414, steps 1076
1221120: done 3642 episodes, mean reward -87.756, speed 13.66 f/s
1221233: done 3643 episodes, mean reward -87.956, speed 80.28 f/s
1221455: done 3644 episodes, mean reward -88.088, speed 78.94 f/s
1221818: done 3645 episodes, mean reward -87.265, speed 80.08 f/s
Test done in 7.61 sec, reward 268.042, steps 1154
1222091: done 3646 episodes, mean reward -86.995, speed 24.76 f/s
1222201: done 3647 episodes, mean reward -87.290, speed 82.76 f/s
1222408: done 3648 episodes, mean reward -87.092, speed 80.88 f/s
1222557: done 3649 episodes, mean reward -87.038, speed 82.44 f/s
1222734: done 3650 episodes, mean reward -86.616, speed 84.08 f/s
1222870: done 3651 episodes, mean reward -86.362, speed 82.84 f/s
1222982: done 3652 episodes, mean reward -86.581, speed 82.47 f/s
Test done in 6.78 sec, reward 224.880, steps 994
1223103: done 3653 episodes, mean reward -86.740, speed 14.63 f/s
1223398: done 3654 episodes, mean reward -86.083, speed 82.61 f/s
1223637: done 3655 episodes, mean reward -85.791, speed 82.59 f/s
1223778: done 3656 episodes, mean reward -86.061, speed 85.03 f/s
1223879: done 3657 episodes, mean reward -86.814, speed 82.17 f/s
Test done in 7.73 sec, reward 268.768, steps 1125
1224049: done 3659 episodes, mean reward -86.849, speed 17.23 f/s
1224170: done 3660 episodes, mean reward -87.072, speed 80.13 f/s
1224268: done 3661 episodes, mean reward -87.874, speed 82.65 f/s
1224385: done 3662 episodes, mean reward -87.945, speed 81.36 f/s
1224498: done 3663 episodes, mean reward -87.925, speed 79.67 f/s
1224635: done 3664 episodes, mean reward -88.028, speed 79.74 f/s
1224825: done 3665 episodes, mean reward -87.731, speed 76.81 f/s
Test done in 7.40 sec, reward 270.114, steps 1107
Best reward updated: 269.671 -> 270.114
1225030: done 3666 episodes, mean reward -87.543, speed 20.66 f/s
1225121: done 3667 episodes, mean reward -87.717, speed 78.17 f/s
1225266: done 3668 episodes, mean reward -87.483, speed 84.63 f/s
1225438: done 3669 episodes, mean reward -87.541, speed 83.64 f/s
1225843: done 3670 episodes, mean reward -86.785, speed 82.52 f/s
Test done in 7.04 sec, reward 249.832, steps 1063
1226050: done 3671 episodes, mean reward -86.651, speed 21.77 f/s
1226231: done 3672 episodes, mean reward -86.459, speed 83.44 f/s
1226323: done 3673 episodes, mean reward -86.480, speed 79.71 f/s
1226443: done 3674 episodes, mean reward -86.267, speed 83.84 f/s
1226608: done 3675 episodes, mean reward -85.893, speed 80.70 f/s
1226846: done 3676 episodes, mean reward -85.890, speed 83.18 f/s
1226987: done 3677 episodes, mean reward -86.120, speed 78.87 f/s
Test done in 7.45 sec, reward 269.220, steps 1130
1227082: done 3678 episodes, mean reward -86.143, speed 11.08 f/s
1227223: done 3679 episodes, mean reward -85.906, speed 84.86 f/s
1227505: done 3680 episodes, mean reward -85.909, speed 81.25 f/s
1227658: done 3681 episodes, mean reward -85.885, speed 80.24 f/s
1227818: done 3682 episodes, mean reward -85.811, speed 84.42 f/s
Test done in 7.45 sec, reward 270.569, steps 1104
Best reward updated: 270.114 -> 270.569
1228014: done 3683 episodes, mean reward -85.697, speed 19.83 f/s
1228151: done 3684 episodes, mean reward -85.559, speed 79.86 f/s
1228285: done 3685 episodes, mean reward -85.514, speed 81.71 f/s
1228420: done 3686 episodes, mean reward -85.588, speed 81.33 f/s
1228501: done 3687 episodes, mean reward -85.702, speed 80.84 f/s
1228682: done 3689 episodes, mean reward -86.598, speed 82.87 f/s
1228769: done 3690 episodes, mean reward -86.685, speed 86.18 f/s
1228858: done 3691 episodes, mean reward -86.654, speed 79.91 f/s
1228976: done 3692 episodes, mean reward -86.887, speed 80.15 f/s
Test done in 7.69 sec, reward 269.912, steps 1122
1229112: done 3693 episodes, mean reward -86.858, speed 14.64 f/s
1229236: done 3694 episodes, mean reward -87.162, speed 82.73 f/s
1229416: done 3695 episodes, mean reward -87.081, speed 81.23 f/s
1229585: done 3696 episodes, mean reward -86.829, speed 84.17 f/s
1229694: done 3697 episodes, mean reward -87.783, speed 85.00 f/s
1229798: done 3698 episodes, mean reward -88.007, speed 80.81 f/s
1229918: done 3699 episodes, mean reward -87.965, speed 82.84 f/s
Test done in 7.48 sec, reward 272.736, steps 1121
Best reward updated: 270.569 -> 272.736
1230233: done 3700 episodes, mean reward -87.545, speed 27.79 f/s
1230365: done 3701 episodes, mean reward -87.478, speed 81.08 f/s
1230532: done 3702 episodes, mean reward -87.178, speed 76.43 f/s
1230615: done 3703 episodes, mean reward -87.376, speed 76.55 f/s
1230702: done 3704 episodes, mean reward -87.402, speed 75.59 f/s
1230808: done 3705 episodes, mean reward -87.439, speed 76.20 f/s
1230918: done 3706 episodes, mean reward -87.393, speed 80.59 f/s
Test done in 7.55 sec, reward 272.080, steps 1110
1231048: done 3707 episodes, mean reward -87.710, speed 14.17 f/s
1231173: done 3708 episodes, mean reward -87.637, speed 83.04 f/s
1231347: done 3709 episodes, mean reward -87.418, speed 78.20 f/s
1231484: done 3710 episodes, mean reward -87.762, speed 80.61 f/s
1231587: done 3711 episodes, mean reward -87.661, speed 83.22 f/s
1231818: done 3712 episodes, mean reward -87.232, speed 83.12 f/s
1231951: done 3713 episodes, mean reward -87.017, speed 82.48 f/s
Test done in 6.99 sec, reward 234.058, steps 1038
1232160: done 3714 episodes, mean reward -86.694, speed 21.91 f/s
1232288: done 3715 episodes, mean reward -86.590, speed 82.58 f/s
1232369: done 3716 episodes, mean reward -87.441, speed 79.32 f/s
1232539: done 3717 episodes, mean reward -87.540, speed 78.84 f/s
1232769: done 3718 episodes, mean reward -87.649, speed 81.41 f/s
Test done in 7.60 sec, reward 271.174, steps 1130
1233053: done 3719 episodes, mean reward -87.834, speed 25.47 f/s
1233273: done 3720 episodes, mean reward -87.843, speed 84.26 f/s
1233384: done 3721 episodes, mean reward -87.993, speed 80.71 f/s
1233547: done 3722 episodes, mean reward -87.911, speed 82.25 f/s
Test done in 8.08 sec, reward 259.760, steps 1199
1234031: done 3723 episodes, mean reward -88.604, speed 33.79 f/s
1234166: done 3724 episodes, mean reward -89.197, speed 82.10 f/s
1234495: done 3725 episodes, mean reward -89.389, speed 79.77 f/s
1234715: done 3726 episodes, mean reward -89.285, speed 83.05 f/s
Test done in 8.33 sec, reward 259.735, steps 1236
1235010: done 3727 episodes, mean reward -89.333, speed 24.94 f/s
1235305: done 3728 episodes, mean reward -89.707, speed 83.35 f/s
1235499: done 3729 episodes, mean reward -89.875, speed 80.20 f/s
Test done in 8.27 sec, reward 225.217, steps 1233
1236013: done 3730 episodes, mean reward -90.378, speed 35.37 f/s
1236248: done 3731 episodes, mean reward -90.218, speed 83.20 f/s
1236777: done 3732 episodes, mean reward -89.587, speed 80.46 f/s
1236951: done 3733 episodes, mean reward -89.317, speed 73.41 f/s
Test done in 5.18 sec, reward 104.878, steps 753
1237162: done 3734 episodes, mean reward -89.561, speed 27.29 f/s
Test done in 7.80 sec, reward 145.677, steps 1168
1238762: done 3735 episodes, mean reward -89.832, speed 59.28 f/s
1238965: done 3736 episodes, mean reward -90.005, speed 82.83 f/s
Test done in 9.91 sec, reward 17.411, steps 1442
Test done in 3.00 sec, reward -141.951, steps 418
1240565: done 3737 episodes, mean reward -90.521, speed 49.30 f/s
1240652: done 3738 episodes, mean reward -91.039, speed 78.29 f/s
1240942: done 3739 episodes, mean reward -90.825, speed 83.47 f/s
Test done in 10.36 sec, reward -60.898, steps 1515
1241055: done 3740 episodes, mean reward -90.904, speed 9.65 f/s
Test done in 7.57 sec, reward 270.004, steps 1128
1242655: done 3741 episodes, mean reward -91.564, speed 59.49 f/s
Test done in 11.03 sec, reward -123.508, steps 1558
Test done in 10.77 sec, reward -108.239, steps 1562
1244255: done 3742 episodes, mean reward -92.091, speed 38.60 f/s
1244350: done 3743 episodes, mean reward -92.110, speed 80.26 f/s
1244496: done 3744 episodes, mean reward -92.284, speed 83.08 f/s
1244671: done 3745 episodes, mean reward -92.832, speed 81.34 f/s
1244795: done 3746 episodes, mean reward -93.309, speed 85.01 f/s
1244899: done 3747 episodes, mean reward -93.561, speed 81.02 f/s
Test done in 8.09 sec, reward 225.522, steps 1189
1245107: done 3748 episodes, mean reward -94.011, speed 19.66 f/s
1245260: done 3749 episodes, mean reward -94.001, speed 81.84 f/s
Test done in 10.32 sec, reward -24.656, steps 1498
1246860: done 3750 episodes, mean reward -94.771, speed 54.03 f/s
Test done in 8.16 sec, reward 228.842, steps 1206
1247121: done 3751 episodes, mean reward -94.487, speed 23.08 f/s
1247339: done 3752 episodes, mean reward -94.084, speed 83.03 f/s
1247451: done 3753 episodes, mean reward -94.135, speed 83.55 f/s
1247608: done 3754 episodes, mean reward -94.787, speed 83.82 f/s
1247719: done 3755 episodes, mean reward -95.164, speed 80.94 f/s
1247903: done 3756 episodes, mean reward -94.975, speed 84.78 f/s
Test done in 7.50 sec, reward 269.638, steps 1128
1248010: done 3757 episodes, mean reward -95.207, speed 12.14 f/s
1248133: done 3758 episodes, mean reward -95.050, speed 81.66 f/s
1248289: done 3759 episodes, mean reward -94.875, speed 84.49 f/s
1248437: done 3760 episodes, mean reward -94.799, speed 81.35 f/s
1248688: done 3761 episodes, mean reward -94.308, speed 81.46 f/s
1248832: done 3762 episodes, mean reward -94.091, speed 81.02 f/s
1248950: done 3763 episodes, mean reward -93.914, speed 85.25 f/s
Test done in 7.15 sec, reward 237.209, steps 1071
1249054: done 3764 episodes, mean reward -94.056, speed 12.32 f/s
1249173: done 3765 episodes, mean reward -94.054, speed 83.77 f/s
1249333: done 3766 episodes, mean reward -94.222, speed 83.63 f/s
1249458: done 3767 episodes, mean reward -94.258, speed 85.68 f/s
1249799: done 3768 episodes, mean reward -93.750, speed 82.95 f/s
Test done in 7.58 sec, reward 251.449, steps 1140
1250016: done 3769 episodes, mean reward -93.636, speed 21.21 f/s
1250316: done 3770 episodes, mean reward -93.918, speed 81.32 f/s
1250558: done 3771 episodes, mean reward -93.878, speed 81.10 f/s
1250682: done 3772 episodes, mean reward -94.027, speed 84.97 f/s
1250878: done 3773 episodes, mean reward -93.870, speed 85.53 f/s
Test done in 7.65 sec, reward 270.293, steps 1138
1251144: done 3774 episodes, mean reward -93.466, speed 24.42 f/s
1251227: done 3775 episodes, mean reward -93.832, speed 79.86 f/s
1251458: done 3776 episodes, mean reward -93.796, speed 82.55 f/s
1251694: done 3777 episodes, mean reward -93.731, speed 78.33 f/s
1251832: done 3778 episodes, mean reward -93.544, speed 83.48 f/s
1251996: done 3779 episodes, mean reward -93.494, speed 82.32 f/s
Test done in 7.40 sec, reward 271.717, steps 1116
1252110: done 3780 episodes, mean reward -93.968, speed 12.98 f/s
1252194: done 3781 episodes, mean reward -94.108, speed 81.49 f/s
1252328: done 3782 episodes, mean reward -94.071, speed 83.69 f/s
1252446: done 3783 episodes, mean reward -94.282, speed 86.75 f/s
1252583: done 3784 episodes, mean reward -94.270, speed 83.54 f/s
1252695: done 3785 episodes, mean reward -94.325, speed 85.62 f/s
1252834: done 3786 episodes, mean reward -94.211, speed 84.19 f/s
1252939: done 3787 episodes, mean reward -94.147, speed 82.35 f/s
Test done in 7.53 sec, reward 259.724, steps 1121
1253072: done 3788 episodes, mean reward -94.085, speed 14.60 f/s
1253218: done 3790 episodes, mean reward -94.040, speed 83.96 f/s
1253449: done 3792 episodes, mean reward -93.969, speed 82.56 f/s
1253532: done 3793 episodes, mean reward -94.127, speed 78.43 f/s
1253697: done 3794 episodes, mean reward -93.886, speed 75.72 f/s
1253860: done 3795 episodes, mean reward -93.818, speed 80.86 f/s
1253973: done 3796 episodes, mean reward -94.079, speed 80.88 f/s
Test done in 6.96 sec, reward 256.645, steps 1089
1254059: done 3797 episodes, mean reward -94.118, speed 10.66 f/s
1254171: done 3798 episodes, mean reward -94.186, speed 84.13 f/s
1254269: done 3799 episodes, mean reward -94.167, speed 81.16 f/s
1254479: done 3800 episodes, mean reward -94.443, speed 83.28 f/s
1254600: done 3801 episodes, mean reward -94.435, speed 81.36 f/s
1254833: done 3802 episodes, mean reward -94.409, speed 85.93 f/s
1254972: done 3803 episodes, mean reward -94.085, speed 84.97 f/s
Test done in 7.46 sec, reward 247.774, steps 1098
1255090: done 3804 episodes, mean reward -94.136, speed 13.21 f/s
1255388: done 3805 episodes, mean reward -93.516, speed 85.13 f/s
1255608: done 3806 episodes, mean reward -93.037, speed 83.10 f/s
1255741: done 3807 episodes, mean reward -92.869, speed 83.60 f/s
1255886: done 3808 episodes, mean reward -92.788, speed 84.94 f/s
Test done in 7.20 sec, reward 249.576, steps 1102
1256152: done 3809 episodes, mean reward -92.533, speed 25.38 f/s
1256282: done 3810 episodes, mean reward -92.668, speed 84.62 f/s
1256409: done 3811 episodes, mean reward -92.636, speed 81.17 f/s
1256620: done 3812 episodes, mean reward -92.823, speed 80.89 f/s
1256744: done 3813 episodes, mean reward -92.957, speed 83.87 f/s
1256864: done 3814 episodes, mean reward -93.311, speed 76.13 f/s
1256990: done 3815 episodes, mean reward -93.281, speed 80.03 f/s
Test done in 7.49 sec, reward 271.074, steps 1108
1257138: done 3816 episodes, mean reward -93.178, speed 16.10 f/s
1257264: done 3817 episodes, mean reward -93.199, speed 82.42 f/s
1257423: done 3818 episodes, mean reward -92.965, speed 84.63 f/s
1257560: done 3819 episodes, mean reward -92.880, speed 83.22 f/s
1257925: done 3820 episodes, mean reward -92.230, speed 81.98 f/s
Test done in 7.60 sec, reward 269.156, steps 1125
1258053: done 3821 episodes, mean reward -92.245, speed 13.99 f/s
1258237: done 3822 episodes, mean reward -92.338, speed 78.99 f/s
1258349: done 3823 episodes, mean reward -91.864, speed 81.62 f/s
1258516: done 3824 episodes, mean reward -91.659, speed 83.06 f/s
1258660: done 3825 episodes, mean reward -91.448, speed 79.47 f/s
1258796: done 3826 episodes, mean reward -91.524, speed 86.56 f/s
1258969: done 3828 episodes, mean reward -91.439, speed 85.32 f/s
Test done in 7.50 sec, reward 247.462, steps 1122
1259186: done 3829 episodes, mean reward -91.378, speed 21.42 f/s
1259329: done 3830 episodes, mean reward -91.049, speed 82.01 f/s
1259419: done 3831 episodes, mean reward -91.265, speed 82.77 f/s
1259893: done 3832 episodes, mean reward -91.156, speed 84.63 f/s
1259997: done 3833 episodes, mean reward -91.443, speed 79.81 f/s
Test done in 7.89 sec, reward 269.292, steps 1149
1260296: done 3834 episodes, mean reward -91.180, speed 26.13 f/s
1260497: done 3835 episodes, mean reward -90.582, speed 86.07 f/s
1260668: done 3836 episodes, mean reward -90.503, speed 81.00 f/s
1260928: done 3838 episodes, mean reward -89.794, speed 79.72 f/s
Test done in 8.01 sec, reward 266.633, steps 1175
1261187: done 3839 episodes, mean reward -89.876, speed 23.25 f/s
1261278: done 3840 episodes, mean reward -90.080, speed 81.67 f/s
1261418: done 3842 episodes, mean reward -89.587, speed 81.00 f/s
1261559: done 3843 episodes, mean reward -89.573, speed 82.30 f/s
1261713: done 3844 episodes, mean reward -89.707, speed 81.63 f/s
1261843: done 3845 episodes, mean reward -89.883, speed 84.78 f/s
1261969: done 3846 episodes, mean reward -89.646, speed 82.22 f/s
Test done in 6.35 sec, reward 187.523, steps 963
1262269: done 3847 episodes, mean reward -88.737, speed 30.21 f/s
1262425: done 3848 episodes, mean reward -88.462, speed 84.35 f/s
1262700: done 3849 episodes, mean reward -88.091, speed 83.21 f/s
1262864: done 3850 episodes, mean reward -87.422, speed 84.55 f/s
1262953: done 3851 episodes, mean reward -87.820, speed 82.55 f/s
Test done in 7.89 sec, reward 270.233, steps 1147
1263085: done 3852 episodes, mean reward -88.167, speed 13.96 f/s
1263173: done 3853 episodes, mean reward -88.313, speed 84.29 f/s
1263408: done 3854 episodes, mean reward -87.726, speed 81.21 f/s
1263573: done 3856 episodes, mean reward -88.164, speed 82.72 f/s
1263710: done 3858 episodes, mean reward -88.336, speed 83.47 f/s
1263922: done 3859 episodes, mean reward -88.558, speed 80.80 f/s
Test done in 6.41 sec, reward 202.012, steps 968
1264087: done 3860 episodes, mean reward -88.646, speed 19.29 f/s
1264231: done 3861 episodes, mean reward -88.930, speed 82.29 f/s
1264426: done 3862 episodes, mean reward -89.600, speed 83.90 f/s
1264676: done 3863 episodes, mean reward -89.381, speed 80.52 f/s
1264838: done 3865 episodes, mean reward -89.810, speed 82.22 f/s
Test done in 0.98 sec, reward -142.390, steps 148
1265031: done 3867 episodes, mean reward -90.118, speed 59.74 f/s
1265205: done 3868 episodes, mean reward -91.252, speed 80.08 f/s
1265310: done 3869 episodes, mean reward -91.534, speed 80.11 f/s
1265457: done 3870 episodes, mean reward -91.824, speed 82.12 f/s
1265591: done 3871 episodes, mean reward -91.966, speed 84.58 f/s
1265772: done 3872 episodes, mean reward -92.232, speed 84.99 f/s
1265889: done 3873 episodes, mean reward -92.391, speed 86.58 f/s
1265987: done 3874 episodes, mean reward -92.966, speed 82.26 f/s
Test done in 6.37 sec, reward 186.599, steps 963
1266135: done 3875 episodes, mean reward -93.239, speed 18.34 f/s
1266258: done 3876 episodes, mean reward -94.010, speed 80.18 f/s
1266337: done 3877 episodes, mean reward -94.604, speed 78.59 f/s
1266454: done 3878 episodes, mean reward -95.189, speed 85.75 f/s
1266610: done 3879 episodes, mean reward -95.167, speed 81.29 f/s
1266921: done 3881 episodes, mean reward -94.784, speed 79.52 f/s
Test done in 7.95 sec, reward 269.473, steps 1141
1267112: done 3883 episodes, mean reward -95.054, speed 18.62 f/s
1267247: done 3884 episodes, mean reward -95.109, speed 80.05 f/s
1267374: done 3885 episodes, mean reward -94.916, speed 84.10 f/s
1267606: done 3887 episodes, mean reward -94.732, speed 79.95 f/s
1267733: done 3888 episodes, mean reward -94.608, speed 81.84 f/s
1267953: done 3889 episodes, mean reward -94.085, speed 83.27 f/s
Test done in 7.42 sec, reward 268.840, steps 1136
1268061: done 3890 episodes, mean reward -94.044, speed 12.44 f/s
1268186: done 3891 episodes, mean reward -94.032, speed 84.30 f/s
1268270: done 3892 episodes, mean reward -94.264, speed 83.77 f/s
1268404: done 3893 episodes, mean reward -94.146, speed 81.60 f/s
1268583: done 3894 episodes, mean reward -94.139, speed 82.72 f/s
1268678: done 3895 episodes, mean reward -94.466, speed 81.83 f/s
1268884: done 3896 episodes, mean reward -94.028, speed 82.99 f/s
1268983: done 3897 episodes, mean reward -94.165, speed 83.31 f/s
Test done in 4.38 sec, reward 32.773, steps 643
1269085: done 3898 episodes, mean reward -93.987, speed 18.34 f/s
1269191: done 3899 episodes, mean reward -94.279, speed 82.30 f/s
1269300: done 3900 episodes, mean reward -94.405, speed 81.28 f/s
1269500: done 3901 episodes, mean reward -94.297, speed 84.39 f/s
1269688: done 3902 episodes, mean reward -94.417, speed 83.71 f/s
1269825: done 3903 episodes, mean reward -94.747, speed 86.15 f/s
Test done in 5.98 sec, reward 149.227, steps 873
1270017: done 3904 episodes, mean reward -94.317, speed 23.19 f/s
1270251: done 3906 episodes, mean reward -95.322, speed 83.08 f/s
1270383: done 3907 episodes, mean reward -95.503, speed 83.12 f/s
1270536: done 3909 episodes, mean reward -96.282, speed 82.79 f/s
1270652: done 3910 episodes, mean reward -96.349, speed 76.82 f/s
1270737: done 3911 episodes, mean reward -96.365, speed 83.34 f/s
1270931: done 3913 episodes, mean reward -96.620, speed 84.31 f/s
Test done in 7.81 sec, reward 269.175, steps 1145
1271075: done 3914 episodes, mean reward -96.603, speed 15.09 f/s
1271223: done 3915 episodes, mean reward -96.447, speed 80.63 f/s
1271352: done 3917 episodes, mean reward -96.801, speed 82.97 f/s
1271466: done 3918 episodes, mean reward -97.052, speed 87.28 f/s
1271551: done 3919 episodes, mean reward -97.120, speed 82.86 f/s
1271679: done 3920 episodes, mean reward -97.704, speed 85.94 f/s
1271937: done 3921 episodes, mean reward -97.190, speed 82.01 f/s
Test done in 7.45 sec, reward 265.796, steps 1161
1272003: done 3922 episodes, mean reward -97.434, speed 8.03 f/s
1272266: done 3924 episodes, mean reward -97.705, speed 84.26 f/s
1272433: done 3925 episodes, mean reward -97.749, speed 82.72 f/s
1272564: done 3927 episodes, mean reward -97.938, speed 86.24 f/s
1272650: done 3928 episodes, mean reward -97.991, speed 81.68 f/s
1272765: done 3930 episodes, mean reward -98.348, speed 79.14 f/s
Test done in 7.93 sec, reward 264.935, steps 1174
1273055: done 3931 episodes, mean reward -97.831, speed 25.14 f/s
1273165: done 3932 episodes, mean reward -98.786, speed 79.43 f/s
1273353: done 3934 episodes, mean reward -98.892, speed 84.22 f/s
1273455: done 3935 episodes, mean reward -99.146, speed 81.94 f/s
1273613: done 3936 episodes, mean reward -99.095, speed 84.91 f/s
1273896: done 3937 episodes, mean reward -98.820, speed 77.35 f/s
1273999: done 3938 episodes, mean reward -99.060, speed 83.83 f/s
Test done in 6.64 sec, reward 214.676, steps 984
1274093: done 3939 episodes, mean reward -99.484, speed 11.80 f/s
1274249: done 3940 episodes, mean reward -99.179, speed 83.15 f/s
1274340: done 3941 episodes, mean reward -98.990, speed 81.80 f/s
1274507: done 3942 episodes, mean reward -98.946, speed 87.18 f/s
1274709: done 3943 episodes, mean reward -98.805, speed 84.97 f/s
1274822: done 3944 episodes, mean reward -98.751, speed 84.21 f/s
Test done in 8.08 sec, reward 265.971, steps 1219
1275011: done 3945 episodes, mean reward -98.582, speed 18.20 f/s
1275222: done 3946 episodes, mean reward -98.391, speed 82.50 f/s
1275310: done 3947 episodes, mean reward -99.147, speed 85.16 f/s
1275511: done 3948 episodes, mean reward -99.005, speed 83.89 f/s
1275636: done 3949 episodes, mean reward -99.461, speed 84.56 f/s
1275930: done 3950 episodes, mean reward -99.090, speed 85.04 f/s
Test done in 7.61 sec, reward 266.285, steps 1141
1276039: done 3951 episodes, mean reward -99.214, speed 12.26 f/s
1276235: done 3952 episodes, mean reward -99.019, speed 77.94 f/s
1276340: done 3953 episodes, mean reward -99.027, speed 78.29 f/s
1276432: done 3954 episodes, mean reward -99.562, speed 76.42 f/s
1276555: done 3955 episodes, mean reward -99.345, speed 83.00 f/s
1276794: done 3956 episodes, mean reward -99.050, speed 84.43 f/s
1276957: done 3957 episodes, mean reward -98.866, speed 86.13 f/s
Test done in 7.64 sec, reward 265.224, steps 1128
1277159: done 3958 episodes, mean reward -98.334, speed 20.21 f/s
1277344: done 3959 episodes, mean reward -98.052, speed 83.71 f/s
1277440: done 3960 episodes, mean reward -98.266, speed 81.03 f/s
1277654: done 3961 episodes, mean reward -98.008, speed 84.75 f/s
1277804: done 3962 episodes, mean reward -97.383, speed 84.52 f/s
Test done in 6.90 sec, reward 243.510, steps 1036
1278173: done 3963 episodes, mean reward -97.015, speed 32.56 f/s
1278309: done 3964 episodes, mean reward -96.611, speed 83.73 f/s
1278433: done 3965 episodes, mean reward -96.541, speed 80.44 f/s
1278565: done 3966 episodes, mean reward -96.119, speed 84.46 f/s
1278717: done 3967 episodes, mean reward -95.954, speed 84.98 f/s
1278822: done 3968 episodes, mean reward -95.453, speed 83.77 f/s
1278978: done 3969 episodes, mean reward -95.250, speed 82.56 f/s
Test done in 7.65 sec, reward 266.503, steps 1159
1279125: done 3970 episodes, mean reward -95.211, speed 15.69 f/s
1279223: done 3971 episodes, mean reward -95.412, speed 83.18 f/s
1279346: done 3972 episodes, mean reward -95.329, speed 83.36 f/s
1279651: done 3974 episodes, mean reward -94.848, speed 79.79 f/s
1279747: done 3975 episodes, mean reward -94.436, speed 78.59 f/s
1279975: done 3976 episodes, mean reward -93.806, speed 80.61 f/s
Test done in 7.96 sec, reward 264.301, steps 1174
1280111: done 3977 episodes, mean reward -93.486, speed 14.19 f/s
1280308: done 3978 episodes, mean reward -92.795, speed 81.23 f/s
1280482: done 3979 episodes, mean reward -92.762, speed 84.55 f/s
1280588: done 3980 episodes, mean reward -92.574, speed 85.11 f/s
1280709: done 3981 episodes, mean reward -92.973, speed 84.65 f/s
1280877: done 3982 episodes, mean reward -92.772, speed 83.09 f/s
Test done in 7.85 sec, reward 264.028, steps 1170
1281007: done 3983 episodes, mean reward -92.515, speed 13.81 f/s
1281154: done 3984 episodes, mean reward -92.369, speed 83.82 f/s
1281254: done 3985 episodes, mean reward -92.616, speed 82.89 f/s
1281364: done 3986 episodes, mean reward -92.537, speed 81.59 f/s
1281462: done 3987 episodes, mean reward -92.728, speed 85.17 f/s
1281588: done 3988 episodes, mean reward -92.887, speed 84.91 f/s
1281680: done 3989 episodes, mean reward -93.371, speed 79.81 f/s
1281944: done 3990 episodes, mean reward -92.843, speed 85.74 f/s
Test done in 7.90 sec, reward 264.257, steps 1165
1282111: done 3991 episodes, mean reward -92.586, speed 16.88 f/s
1282228: done 3992 episodes, mean reward -92.368, speed 83.20 f/s
1282361: done 3993 episodes, mean reward -92.252, speed 83.95 f/s
1282506: done 3994 episodes, mean reward -92.440, speed 86.06 f/s
1282670: done 3995 episodes, mean reward -92.252, speed 81.66 f/s
1282833: done 3996 episodes, mean reward -92.411, speed 74.04 f/s
1282961: done 3997 episodes, mean reward -92.249, speed 75.69 f/s
Test done in 7.89 sec, reward 261.395, steps 1194
1283078: done 3998 episodes, mean reward -92.468, speed 12.62 f/s
1283208: done 3999 episodes, mean reward -92.081, speed 81.56 f/s
1283319: done 4000 episodes, mean reward -92.220, speed 85.84 f/s
1283482: done 4002 episodes, mean reward -92.390, speed 83.85 f/s
1283585: done 4003 episodes, mean reward -92.162, speed 84.73 f/s
1283708: done 4004 episodes, mean reward -92.545, speed 88.88 f/s
1283804: done 4005 episodes, mean reward -92.375, speed 87.96 f/s
1283906: done 4006 episodes, mean reward -92.470, speed 83.79 f/s
Test done in 7.80 sec, reward 262.801, steps 1209
1284006: done 4007 episodes, mean reward -92.585, speed 11.13 f/s
1284153: done 4008 episodes, mean reward -92.517, speed 85.92 f/s
1284272: done 4009 episodes, mean reward -92.333, speed 84.15 f/s
1284543: done 4010 episodes, mean reward -91.728, speed 84.37 f/s
1284879: done 4011 episodes, mean reward -90.996, speed 84.22 f/s
1284973: done 4012 episodes, mean reward -90.918, speed 86.91 f/s
Test done in 8.25 sec, reward 246.196, steps 1227
1285178: done 4013 episodes, mean reward -90.663, speed 19.11 f/s
1285305: done 4014 episodes, mean reward -90.489, speed 86.04 f/s
1285459: done 4015 episodes, mean reward -90.458, speed 82.46 f/s
1285782: done 4016 episodes, mean reward -89.814, speed 82.57 f/s
1285940: done 4017 episodes, mean reward -89.582, speed 79.04 f/s
Test done in 8.17 sec, reward 218.484, steps 1191
1286074: done 4018 episodes, mean reward -89.498, speed 13.62 f/s
1286254: done 4019 episodes, mean reward -89.404, speed 81.41 f/s
1286605: done 4020 episodes, mean reward -88.880, speed 82.84 f/s
Test done in 7.98 sec, reward 257.652, steps 1228
1287460: done 4021 episodes, mean reward -87.406, speed 47.61 f/s
1287594: done 4022 episodes, mean reward -87.093, speed 83.11 f/s
1287759: done 4023 episodes, mean reward -86.831, speed 81.85 f/s
1287892: done 4024 episodes, mean reward -86.928, speed 85.12 f/s
Test done in 7.85 sec, reward 260.401, steps 1206
1288105: done 4025 episodes, mean reward -86.701, speed 20.48 f/s
1288320: done 4026 episodes, mean reward -86.221, speed 82.48 f/s
1288779: done 4027 episodes, mean reward -85.215, speed 84.93 f/s
Test done in 7.04 sec, reward 197.122, steps 1055
1289098: done 4028 episodes, mean reward -84.665, speed 29.17 f/s
1289240: done 4029 episodes, mean reward -84.414, speed 82.52 f/s
1289440: done 4030 episodes, mean reward -84.037, speed 81.02 f/s
1289527: done 4031 episodes, mean reward -84.497, speed 81.18 f/s
1289687: done 4032 episodes, mean reward -84.282, speed 85.07 f/s
Test done in 8.13 sec, reward 232.769, steps 1227
1290057: done 4034 episodes, mean reward -83.759, speed 29.60 f/s
1290441: done 4035 episodes, mean reward -82.950, speed 83.01 f/s
1290552: done 4036 episodes, mean reward -83.047, speed 81.65 f/s
1290654: done 4037 episodes, mean reward -83.189, speed 86.41 f/s
Test done in 7.94 sec, reward 216.404, steps 1160
1291186: done 4038 episodes, mean reward -81.883, speed 37.42 f/s
1291377: done 4039 episodes, mean reward -81.522, speed 83.17 f/s
1291509: done 4040 episodes, mean reward -81.438, speed 86.04 f/s
1291920: done 4041 episodes, mean reward -80.960, speed 84.11 f/s
Test done in 7.53 sec, reward 227.660, steps 1092
1292131: done 4042 episodes, mean reward -80.640, speed 21.16 f/s
1292258: done 4043 episodes, mean reward -80.810, speed 83.20 f/s
1292369: done 4044 episodes, mean reward -80.776, speed 84.62 f/s
1292563: done 4045 episodes, mean reward -80.738, speed 82.39 f/s
1292688: done 4046 episodes, mean reward -80.907, speed 83.90 f/s
1292832: done 4047 episodes, mean reward -80.811, speed 82.15 f/s
Test done in 7.32 sec, reward 203.801, steps 1088
1293477: done 4048 episodes, mean reward -79.721, speed 42.98 f/s
1293739: done 4049 episodes, mean reward -79.344, speed 77.49 f/s
Test done in 8.12 sec, reward 244.625, steps 1216
1294052: done 4050 episodes, mean reward -79.389, speed 26.36 f/s
1294191: done 4051 episodes, mean reward -79.159, speed 83.27 f/s
1294408: done 4053 episodes, mean reward -79.308, speed 84.29 f/s
1294512: done 4054 episodes, mean reward -79.133, speed 88.29 f/s
1294651: done 4055 episodes, mean reward -79.255, speed 85.02 f/s
1294776: done 4056 episodes, mean reward -79.475, speed 85.30 f/s
1294924: done 4057 episodes, mean reward -79.432, speed 83.02 f/s
Test done in 8.09 sec, reward 241.969, steps 1181
1295050: done 4058 episodes, mean reward -79.604, speed 13.19 f/s
1295349: done 4059 episodes, mean reward -79.444, speed 84.80 f/s
1295516: done 4060 episodes, mean reward -79.125, speed 80.70 f/s
1295648: done 4061 episodes, mean reward -79.424, speed 85.43 f/s
1295783: done 4062 episodes, mean reward -79.573, speed 84.36 f/s
1295975: done 4064 episodes, mean reward -80.481, speed 82.66 f/s
Test done in 8.01 sec, reward 249.607, steps 1200
1296137: done 4065 episodes, mean reward -80.420, speed 16.28 f/s
1296419: done 4067 episodes, mean reward -80.706, speed 85.06 f/s
1296657: done 4068 episodes, mean reward -80.563, speed 82.74 f/s
Test done in 6.98 sec, reward 175.001, steps 1036
1297113: done 4069 episodes, mean reward -80.007, speed 36.02 f/s
1297285: done 4070 episodes, mean reward -79.931, speed 83.95 f/s
1297516: done 4071 episodes, mean reward -79.622, speed 82.47 f/s
1297680: done 4072 episodes, mean reward -79.370, speed 81.53 f/s
1297766: done 4073 episodes, mean reward -79.615, speed 85.95 f/s
1297928: done 4074 episodes, mean reward -79.963, speed 85.83 f/s
Test done in 7.29 sec, reward 238.497, steps 1107
1298103: done 4075 episodes, mean reward -79.738, speed 18.54 f/s
1298353: done 4076 episodes, mean reward -79.669, speed 81.42 f/s
Test done in 8.10 sec, reward 262.824, steps 1202
1299953: done 4077 episodes, mean reward -80.032, speed 58.04 f/s
Test done in 7.49 sec, reward 231.025, steps 1119
1300036: done 4078 episodes, mean reward -80.345, speed 9.80 f/s
1300184: done 4079 episodes, mean reward -80.534, speed 86.45 f/s
1300330: done 4080 episodes, mean reward -80.344, speed 77.53 f/s
1300573: done 4081 episodes, mean reward -79.969, speed 82.44 f/s
Test done in 7.77 sec, reward 253.911, steps 1160
Test done in 7.41 sec, reward 239.162, steps 1098
1302092: done 4082 episodes, mean reward -80.376, speed 45.16 f/s
1302236: done 4083 episodes, mean reward -80.395, speed 83.42 f/s
1302488: done 4084 episodes, mean reward -80.118, speed 85.33 f/s
1302603: done 4085 episodes, mean reward -79.902, speed 83.07 f/s
1302845: done 4086 episodes, mean reward -79.725, speed 86.06 f/s
1302979: done 4087 episodes, mean reward -79.658, speed 86.12 f/s
Test done in 7.70 sec, reward 226.457, steps 1078
1303063: done 4088 episodes, mean reward -79.613, speed 9.68 f/s
1303312: done 4089 episodes, mean reward -79.139, speed 84.03 f/s
1303474: done 4090 episodes, mean reward -79.518, speed 78.75 f/s
1303590: done 4091 episodes, mean reward -79.669, speed 86.00 f/s
1303736: done 4092 episodes, mean reward -79.702, speed 84.20 f/s
1303892: done 4093 episodes, mean reward -79.609, speed 83.67 f/s
Test done in 8.24 sec, reward 267.366, steps 1192
1304009: done 4094 episodes, mean reward -79.575, speed 12.19 f/s
1304263: done 4096 episodes, mean reward -79.553, speed 84.63 f/s
1304380: done 4097 episodes, mean reward -79.467, speed 82.33 f/s
1304616: done 4099 episodes, mean reward -79.444, speed 80.30 f/s
1304785: done 4101 episodes, mean reward -79.547, speed 80.03 f/s
Test done in 6.18 sec, reward 195.066, steps 931
1305003: done 4103 episodes, mean reward -79.568, speed 24.80 f/s
1305166: done 4104 episodes, mean reward -79.409, speed 82.55 f/s
1305319: done 4105 episodes, mean reward -79.296, speed 81.87 f/s
1305445: done 4106 episodes, mean reward -79.208, speed 83.75 f/s
1305578: done 4107 episodes, mean reward -79.041, speed 82.32 f/s
1305711: done 4108 episodes, mean reward -78.917, speed 83.68 f/s
1305879: done 4109 episodes, mean reward -78.908, speed 83.57 f/s
Test done in 7.65 sec, reward 255.121, steps 1094
1306077: done 4111 episodes, mean reward -80.219, speed 19.51 f/s
1306407: done 4112 episodes, mean reward -79.541, speed 79.68 f/s
1306573: done 4113 episodes, mean reward -79.640, speed 79.31 f/s
1306704: done 4114 episodes, mean reward -79.765, speed 79.14 f/s
1306842: done 4115 episodes, mean reward -79.913, speed 80.21 f/s
1306952: done 4116 episodes, mean reward -80.392, speed 83.13 f/s
Test done in 7.31 sec, reward 237.520, steps 1064
1307106: done 4117 episodes, mean reward -80.278, speed 16.77 f/s
1307250: done 4118 episodes, mean reward -80.333, speed 83.71 f/s
1307486: done 4119 episodes, mean reward -80.036, speed 84.30 f/s
1307623: done 4120 episodes, mean reward -80.669, speed 80.06 f/s
1307838: done 4121 episodes, mean reward -82.194, speed 84.68 f/s
Test done in 6.42 sec, reward 185.416, steps 964
1308147: done 4122 episodes, mean reward -81.971, speed 30.00 f/s
1308334: done 4124 episodes, mean reward -82.210, speed 82.99 f/s
1308434: done 4125 episodes, mean reward -82.452, speed 83.57 f/s
1308726: done 4126 episodes, mean reward -82.647, speed 84.60 f/s
1308871: done 4127 episodes, mean reward -83.310, speed 85.06 f/s
Test done in 7.75 sec, reward 270.683, steps 1155
1309009: done 4128 episodes, mean reward -83.699, speed 14.64 f/s
1309132: done 4129 episodes, mean reward -83.894, speed 86.45 f/s
1309255: done 4130 episodes, mean reward -84.145, speed 82.97 f/s
1309476: done 4131 episodes, mean reward -83.804, speed 75.82 f/s
1309658: done 4132 episodes, mean reward -83.637, speed 82.04 f/s
1309842: done 4133 episodes, mean reward -83.276, speed 82.56 f/s
1309966: done 4134 episodes, mean reward -83.887, speed 84.34 f/s
Test done in 7.66 sec, reward 271.103, steps 1129
1310098: done 4135 episodes, mean reward -84.516, speed 14.35 f/s
1310221: done 4136 episodes, mean reward -84.609, speed 84.05 f/s
1310516: done 4138 episodes, mean reward -85.776, speed 85.13 f/s
1310623: done 4139 episodes, mean reward -86.048, speed 83.61 f/s
1310720: done 4140 episodes, mean reward -86.219, speed 84.43 f/s
1310830: done 4142 episodes, mean reward -86.941, speed 82.11 f/s
Test done in 7.22 sec, reward 256.963, steps 1092
1311011: done 4143 episodes, mean reward -86.656, speed 19.10 f/s
1311218: done 4144 episodes, mean reward -86.327, speed 81.38 f/s
1311419: done 4145 episodes, mean reward -86.207, speed 84.02 f/s
1311621: done 4146 episodes, mean reward -86.024, speed 83.57 f/s
1311905: done 4148 episodes, mean reward -87.635, speed 84.34 f/s
Test done in 5.70 sec, reward 160.721, steps 826
1312020: done 4149 episodes, mean reward -87.955, speed 16.28 f/s
1312145: done 4150 episodes, mean reward -88.459, speed 84.97 f/s
1312388: done 4152 episodes, mean reward -88.389, speed 82.23 f/s
1312681: done 4153 episodes, mean reward -87.869, speed 85.23 f/s
1312833: done 4154 episodes, mean reward -87.673, speed 80.24 f/s
1312946: done 4155 episodes, mean reward -87.677, speed 87.25 f/s
Test done in 7.45 sec, reward 246.377, steps 1118
1313107: done 4156 episodes, mean reward -87.635, speed 16.96 f/s
1313246: done 4157 episodes, mean reward -87.651, speed 77.91 f/s
1313443: done 4159 episodes, mean reward -88.150, speed 82.39 f/s
1313573: done 4160 episodes, mean reward -88.351, speed 87.19 f/s
1313730: done 4161 episodes, mean reward -88.285, speed 81.05 f/s
1313869: done 4162 episodes, mean reward -88.139, speed 82.80 f/s
Test done in 7.48 sec, reward 270.686, steps 1140
1314002: done 4163 episodes, mean reward -87.941, speed 14.61 f/s
1314153: done 4165 episodes, mean reward -88.062, speed 84.16 f/s
1314256: done 4166 episodes, mean reward -87.975, speed 81.89 f/s
1314347: done 4167 episodes, mean reward -88.142, speed 80.52 f/s
1314590: done 4168 episodes, mean reward -87.951, speed 84.08 f/s
1314675: done 4169 episodes, mean reward -88.706, speed 82.70 f/s
1314810: done 4170 episodes, mean reward -88.843, speed 85.30 f/s
Test done in 7.88 sec, reward 273.159, steps 1142
Best reward updated: 272.736 -> 273.159
1315005: done 4171 episodes, mean reward -88.912, speed 19.20 f/s
1315222: done 4172 episodes, mean reward -88.797, speed 81.84 f/s
1315423: done 4174 episodes, mean reward -88.527, speed 82.27 f/s
1315523: done 4175 episodes, mean reward -88.836, speed 84.52 f/s
1315622: done 4176 episodes, mean reward -89.220, speed 86.83 f/s
1315721: done 4177 episodes, mean reward -88.955, speed 80.58 f/s
1315910: done 4178 episodes, mean reward -88.586, speed 82.99 f/s
Test done in 6.73 sec, reward 214.221, steps 972
1316058: done 4179 episodes, mean reward -88.468, speed 17.50 f/s
1316314: done 4180 episodes, mean reward -88.304, speed 83.82 f/s
1316482: done 4181 episodes, mean reward -88.738, speed 82.58 f/s
1316590: done 4182 episodes, mean reward -88.385, speed 82.01 f/s
1316698: done 4183 episodes, mean reward -88.509, speed 82.26 f/s
1316919: done 4184 episodes, mean reward -88.900, speed 84.27 f/s
Test done in 5.54 sec, reward 147.966, steps 786
1317019: done 4185 episodes, mean reward -89.107, speed 14.84 f/s
1317112: done 4186 episodes, mean reward -89.355, speed 80.42 f/s
1317388: done 4187 episodes, mean reward -89.202, speed 83.20 f/s
1317538: done 4188 episodes, mean reward -89.241, speed 84.19 f/s
1317681: done 4189 episodes, mean reward -89.629, speed 83.54 f/s
1317793: done 4190 episodes, mean reward -89.912, speed 86.46 f/s
Test done in 6.94 sec, reward 212.103, steps 1050
1318120: done 4191 episodes, mean reward -89.561, speed 30.10 f/s
1318257: done 4192 episodes, mean reward -89.614, speed 83.94 f/s
1318590: done 4193 episodes, mean reward -89.171, speed 83.29 f/s
1318795: done 4194 episodes, mean reward -89.213, speed 83.78 f/s
1318927: done 4195 episodes, mean reward -89.207, speed 81.28 f/s
Test done in 6.39 sec, reward 194.741, steps 950
1319136: done 4196 episodes, mean reward -89.397, speed 23.60 f/s
1319294: done 4197 episodes, mean reward -89.427, speed 82.15 f/s
1319559: done 4198 episodes, mean reward -88.947, speed 84.04 f/s
1319721: done 4199 episodes, mean reward -89.100, speed 83.15 f/s
1319869: done 4200 episodes, mean reward -89.068, speed 78.19 f/s
Test done in 5.73 sec, reward 135.890, steps 876
1320341: done 4201 episodes, mean reward -88.033, speed 41.42 f/s
1320496: done 4202 episodes, mean reward -87.896, speed 81.02 f/s
1320741: done 4203 episodes, mean reward -87.787, speed 84.30 f/s
1320977: done 4204 episodes, mean reward -87.789, speed 81.11 f/s
Test done in 7.93 sec, reward 259.380, steps 1108
1321160: done 4205 episodes, mean reward -88.167, speed 18.14 f/s
1321283: done 4206 episodes, mean reward -88.268, speed 80.54 f/s
1321496: done 4207 episodes, mean reward -88.123, speed 82.37 f/s
1321682: done 4208 episodes, mean reward -88.182, speed 85.24 f/s
Test done in 8.14 sec, reward 268.709, steps 1147
1322109: done 4209 episodes, mean reward -87.620, speed 31.93 f/s
1322422: done 4210 episodes, mean reward -87.632, speed 81.65 f/s
1322649: done 4211 episodes, mean reward -87.585, speed 82.05 f/s
1322862: done 4212 episodes, mean reward -88.232, speed 80.33 f/s
1322977: done 4213 episodes, mean reward -88.632, speed 82.59 f/s
Test done in 6.66 sec, reward 227.495, steps 995
1323144: done 4214 episodes, mean reward -88.709, speed 19.31 f/s
1323433: done 4215 episodes, mean reward -88.493, speed 83.08 f/s
1323883: done 4216 episodes, mean reward -87.783, speed 82.74 f/s
Test done in 6.71 sec, reward 229.828, steps 982
1324030: done 4217 episodes, mean reward -88.291, speed 17.38 f/s
1324122: done 4218 episodes, mean reward -88.170, speed 85.70 f/s
1324416: done 4219 episodes, mean reward -88.605, speed 83.65 f/s
1324595: done 4220 episodes, mean reward -88.615, speed 82.92 f/s
1324966: done 4221 episodes, mean reward -88.331, speed 84.53 f/s
Test done in 6.30 sec, reward 188.747, steps 931
1325208: done 4222 episodes, mean reward -88.732, speed 26.37 f/s
1325584: done 4223 episodes, mean reward -88.142, speed 84.28 f/s
1325710: done 4224 episodes, mean reward -88.187, speed 80.52 f/s
1325889: done 4225 episodes, mean reward -88.240, speed 82.11 f/s
Test done in 7.30 sec, reward 275.078, steps 1068
Best reward updated: 273.159 -> 275.078
1326105: done 4226 episodes, mean reward -88.552, speed 21.43 f/s
1326198: done 4227 episodes, mean reward -88.702, speed 81.59 f/s
1326355: done 4228 episodes, mean reward -88.615, speed 84.89 f/s
1326464: done 4229 episodes, mean reward -88.617, speed 84.96 f/s
1326555: done 4230 episodes, mean reward -88.533, speed 81.96 f/s
1326659: done 4231 episodes, mean reward -88.740, speed 81.97 f/s
1326792: done 4232 episodes, mean reward -88.929, speed 79.63 f/s
1326970: done 4233 episodes, mean reward -89.377, speed 82.58 f/s
Test done in 5.51 sec, reward 146.889, steps 799
1327082: done 4234 episodes, mean reward -89.649, speed 16.36 f/s
1327268: done 4235 episodes, mean reward -89.561, speed 86.44 f/s
1327473: done 4236 episodes, mean reward -89.514, speed 84.44 f/s
1327642: done 4237 episodes, mean reward -89.616, speed 85.79 f/s
1327857: done 4238 episodes, mean reward -89.833, speed 82.32 f/s
1327973: done 4239 episodes, mean reward -89.840, speed 86.50 f/s
Test done in 6.56 sec, reward 218.368, steps 938
1328159: done 4240 episodes, mean reward -89.581, speed 21.08 f/s
1328572: done 4241 episodes, mean reward -88.834, speed 83.21 f/s
1328955: done 4242 episodes, mean reward -88.320, speed 84.42 f/s
Test done in 5.48 sec, reward 146.462, steps 799
1329073: done 4243 episodes, mean reward -88.556, speed 16.87 f/s
1329221: done 4244 episodes, mean reward -88.825, speed 81.07 f/s
1329560: done 4245 episodes, mean reward -89.091, speed 82.59 f/s
1329706: done 4246 episodes, mean reward -89.384, speed 83.71 f/s
1329838: done 4247 episodes, mean reward -89.138, speed 86.95 f/s
Test done in 5.85 sec, reward 166.318, steps 835
1330036: done 4248 episodes, mean reward -88.892, speed 24.05 f/s
1330330: done 4249 episodes, mean reward -88.485, speed 86.35 f/s
1330584: done 4251 episodes, mean reward -88.805, speed 83.32 f/s
1330774: done 4252 episodes, mean reward -88.746, speed 85.29 f/s
Test done in 7.30 sec, reward 251.965, steps 1066
1331076: done 4253 episodes, mean reward -88.938, speed 27.74 f/s
1331293: done 4254 episodes, mean reward -88.884, speed 83.18 f/s
1331598: done 4255 episodes, mean reward -88.200, speed 85.15 f/s
1331761: done 4256 episodes, mean reward -88.008, speed 82.38 f/s
1331995: done 4257 episodes, mean reward -87.844, speed 83.70 f/s
Test done in 6.92 sec, reward 256.484, steps 1024
1332266: done 4258 episodes, mean reward -87.101, speed 26.83 f/s
1332529: done 4259 episodes, mean reward -86.659, speed 81.96 f/s
1332765: done 4260 episodes, mean reward -86.406, speed 81.91 f/s
1332967: done 4261 episodes, mean reward -86.402, speed 86.28 f/s
Test done in 5.82 sec, reward 169.203, steps 819
1333048: done 4262 episodes, mean reward -86.639, speed 11.95 f/s
1333226: done 4263 episodes, mean reward -86.747, speed 81.28 f/s
1333371: done 4264 episodes, mean reward -86.524, speed 86.36 f/s
1333675: done 4265 episodes, mean reward -85.991, speed 85.02 f/s
1333820: done 4266 episodes, mean reward -85.856, speed 82.32 f/s
Test done in 4.47 sec, reward 51.382, steps 646
1334023: done 4267 episodes, mean reward -85.556, speed 29.52 f/s
1334188: done 4268 episodes, mean reward -85.664, speed 81.85 f/s
1334359: done 4269 episodes, mean reward -85.499, speed 82.16 f/s
1334551: done 4270 episodes, mean reward -85.389, speed 80.10 f/s
1334700: done 4271 episodes, mean reward -85.429, speed 81.77 f/s
1334798: done 4272 episodes, mean reward -85.689, speed 79.88 f/s
1334936: done 4273 episodes, mean reward -85.640, speed 83.74 f/s
Test done in 1.96 sec, reward -50.570, steps 275
1335010: done 4274 episodes, mean reward -85.752, speed 26.31 f/s
1335107: done 4275 episodes, mean reward -85.684, speed 82.93 f/s
1335329: done 4276 episodes, mean reward -85.234, speed 84.41 f/s
1335412: done 4277 episodes, mean reward -85.287, speed 81.04 f/s
1335544: done 4278 episodes, mean reward -85.745, speed 88.64 f/s
1335719: done 4279 episodes, mean reward -85.555, speed 81.42 f/s
1335815: done 4280 episodes, mean reward -85.888, speed 83.49 f/s
1335973: done 4281 episodes, mean reward -85.823, speed 84.91 f/s
Test done in 3.64 sec, reward 40.155, steps 526
1336178: done 4282 episodes, mean reward -85.659, speed 33.59 f/s
1336372: done 4283 episodes, mean reward -85.558, speed 75.51 f/s
1336598: done 4284 episodes, mean reward -85.076, speed 81.43 f/s
1336739: done 4285 episodes, mean reward -84.849, speed 83.39 f/s
1336844: done 4286 episodes, mean reward -84.802, speed 82.01 f/s
Test done in 4.15 sec, reward 88.939, steps 617
1337047: done 4287 episodes, mean reward -84.766, speed 30.89 f/s
1337162: done 4288 episodes, mean reward -84.670, speed 80.61 f/s
1337256: done 4289 episodes, mean reward -84.712, speed 84.67 f/s
1337363: done 4290 episodes, mean reward -84.818, speed 80.19 f/s
1337523: done 4291 episodes, mean reward -85.054, speed 85.63 f/s
1337764: done 4292 episodes, mean reward -84.873, speed 82.38 f/s
1337956: done 4293 episodes, mean reward -85.332, speed 84.06 f/s
Test done in 2.11 sec, reward -47.480, steps 298
1338064: done 4294 episodes, mean reward -85.567, speed 31.80 f/s
1338203: done 4295 episodes, mean reward -85.575, speed 84.39 f/s
1338398: done 4296 episodes, mean reward -85.461, speed 80.96 f/s
1338542: done 4297 episodes, mean reward -85.551, speed 81.99 f/s
1338633: done 4298 episodes, mean reward -86.023, speed 83.56 f/s
1338798: done 4299 episodes, mean reward -85.817, speed 83.59 f/s
1338944: done 4300 episodes, mean reward -85.816, speed 86.58 f/s
Test done in 2.66 sec, reward -5.573, steps 387
1339157: done 4301 episodes, mean reward -86.547, speed 40.89 f/s
1339285: done 4302 episodes, mean reward -86.674, speed 83.10 f/s
1339536: done 4303 episodes, mean reward -86.423, speed 86.73 f/s
1339828: done 4304 episodes, mean reward -86.541, speed 84.97 f/s
1339971: done 4305 episodes, mean reward -86.102, speed 85.24 f/s
Test done in 5.06 sec, reward 160.375, steps 741
1340093: done 4306 episodes, mean reward -86.106, speed 18.63 f/s
1340183: done 4307 episodes, mean reward -86.245, speed 72.70 f/s
1340350: done 4308 episodes, mean reward -85.939, speed 85.13 f/s
1340792: done 4309 episodes, mean reward -85.715, speed 84.66 f/s
Test done in 5.87 sec, reward 168.061, steps 881
1341085: done 4310 episodes, mean reward -85.161, speed 31.53 f/s
1341241: done 4311 episodes, mean reward -85.074, speed 85.18 f/s
1341425: done 4312 episodes, mean reward -84.835, speed 84.64 f/s
1341725: done 4313 episodes, mean reward -84.134, speed 83.31 f/s
1341921: done 4314 episodes, mean reward -83.722, speed 84.52 f/s
Test done in 5.44 sec, reward 142.575, steps 782
1342034: done 4315 episodes, mean reward -84.095, speed 16.53 f/s
1342139: done 4316 episodes, mean reward -84.863, speed 83.16 f/s
1342327: done 4317 episodes, mean reward -84.550, speed 80.89 f/s
1342524: done 4318 episodes, mean reward -84.245, speed 82.96 f/s
1342614: done 4319 episodes, mean reward -84.137, speed 83.01 f/s
1342768: done 4320 episodes, mean reward -83.990, speed 86.30 f/s
1342892: done 4321 episodes, mean reward -84.511, speed 86.14 f/s
1342996: done 4322 episodes, mean reward -84.361, speed 84.65 f/s
Test done in 5.77 sec, reward 185.101, steps 853
1343301: done 4323 episodes, mean reward -84.102, speed 32.92 f/s
1343587: done 4324 episodes, mean reward -83.552, speed 85.49 f/s
1343786: done 4325 episodes, mean reward -83.197, speed 80.06 f/s
Test done in 4.62 sec, reward 107.794, steps 670
1344045: done 4326 episodes, mean reward -82.501, speed 33.59 f/s
1344209: done 4327 episodes, mean reward -82.255, speed 84.13 f/s
1344640: done 4328 episodes, mean reward -81.450, speed 82.18 f/s
1344777: done 4329 episodes, mean reward -81.444, speed 83.81 f/s
1344914: done 4330 episodes, mean reward -81.334, speed 83.63 f/s
Test done in 5.91 sec, reward 203.357, steps 869
1345517: done 4331 episodes, mean reward -80.081, speed 45.81 f/s
1345698: done 4332 episodes, mean reward -79.883, speed 82.96 f/s
1345880: done 4333 episodes, mean reward -79.597, speed 77.10 f/s
Test done in 4.54 sec, reward 107.884, steps 636
1346036: done 4335 episodes, mean reward -79.619, speed 23.93 f/s
1346186: done 4336 episodes, mean reward -79.447, speed 83.74 f/s
1346341: done 4337 episodes, mean reward -79.011, speed 82.76 f/s
1346511: done 4338 episodes, mean reward -78.850, speed 85.63 f/s
1346658: done 4339 episodes, mean reward -78.707, speed 81.95 f/s
1346850: done 4340 episodes, mean reward -78.560, speed 84.93 f/s
1346987: done 4341 episodes, mean reward -79.079, speed 81.70 f/s
Test done in 4.99 sec, reward 127.249, steps 729
1347065: done 4342 episodes, mean reward -79.596, speed 13.12 f/s
1347194: done 4343 episodes, mean reward -79.414, speed 84.92 f/s
1347311: done 4344 episodes, mean reward -79.667, speed 83.88 f/s
1347519: done 4345 episodes, mean reward -79.522, speed 83.02 f/s
1347650: done 4346 episodes, mean reward -79.525, speed 80.84 f/s
Test done in 4.35 sec, reward 101.942, steps 642
1348190: done 4347 episodes, mean reward -78.440, speed 50.01 f/s
1348308: done 4348 episodes, mean reward -78.545, speed 80.64 f/s
1348500: done 4350 episodes, mean reward -78.888, speed 82.82 f/s
1348601: done 4351 episodes, mean reward -78.654, speed 82.20 f/s
1348693: done 4352 episodes, mean reward -78.811, speed 84.19 f/s
1348953: done 4353 episodes, mean reward -78.897, speed 84.55 f/s
Test done in 5.16 sec, reward 164.607, steps 757
1349104: done 4354 episodes, mean reward -79.144, speed 21.67 f/s
1349201: done 4355 episodes, mean reward -79.718, speed 84.81 f/s
1349462: done 4356 episodes, mean reward -79.389, speed 77.30 f/s
1349612: done 4357 episodes, mean reward -79.521, speed 79.56 f/s
1349866: done 4358 episodes, mean reward -79.522, speed 83.39 f/s
Test done in 5.67 sec, reward 188.152, steps 820
1350098: done 4359 episodes, mean reward -79.630, speed 27.32 f/s
1350250: done 4360 episodes, mean reward -79.797, speed 82.93 f/s
1350374: done 4361 episodes, mean reward -79.763, speed 85.28 f/s
1350634: done 4363 episodes, mean reward -79.349, speed 82.73 f/s
1350941: done 4364 episodes, mean reward -78.920, speed 84.71 f/s
Test done in 5.65 sec, reward 160.679, steps 826
1351038: done 4365 episodes, mean reward -79.575, speed 14.27 f/s
1351264: done 4367 episodes, mean reward -79.904, speed 83.76 f/s
1351384: done 4368 episodes, mean reward -80.049, speed 80.25 f/s
1351480: done 4369 episodes, mean reward -80.193, speed 80.81 f/s
1351639: done 4371 episodes, mean reward -80.586, speed 85.28 f/s
1351723: done 4372 episodes, mean reward -80.758, speed 81.87 f/s
1351827: done 4373 episodes, mean reward -80.939, speed 84.63 f/s
Test done in 6.81 sec, reward 219.165, steps 978
1352035: done 4374 episodes, mean reward -80.633, speed 22.48 f/s
1352316: done 4375 episodes, mean reward -80.029, speed 84.15 f/s
1352584: done 4376 episodes, mean reward -79.936, speed 83.05 f/s
1352736: done 4377 episodes, mean reward -79.563, speed 84.00 f/s
1352995: done 4379 episodes, mean reward -79.612, speed 80.12 f/s
Test done in 7.09 sec, reward 265.519, steps 1029
1353121: done 4380 episodes, mean reward -79.433, speed 14.59 f/s
1353400: done 4381 episodes, mean reward -79.002, speed 84.42 f/s
1353507: done 4382 episodes, mean reward -79.369, speed 85.43 f/s
1353614: done 4383 episodes, mean reward -79.587, speed 81.77 f/s
1353739: done 4384 episodes, mean reward -79.873, speed 87.40 f/s
Test done in 6.98 sec, reward 250.295, steps 979
1354043: done 4385 episodes, mean reward -80.050, speed 28.56 f/s
1354212: done 4386 episodes, mean reward -80.004, speed 82.79 f/s
1354340: done 4387 episodes, mean reward -80.147, speed 82.46 f/s
1354750: done 4388 episodes, mean reward -79.339, speed 81.77 f/s
1354913: done 4389 episodes, mean reward -78.982, speed 83.69 f/s
Test done in 6.83 sec, reward 259.704, steps 1021
1355052: done 4390 episodes, mean reward -78.512, speed 16.35 f/s
1355428: done 4391 episodes, mean reward -78.277, speed 85.22 f/s
1355622: done 4392 episodes, mean reward -78.169, speed 82.39 f/s
1355729: done 4393 episodes, mean reward -78.270, speed 83.31 f/s
Test done in 6.61 sec, reward 229.935, steps 936
1356053: done 4394 episodes, mean reward -77.334, speed 30.90 f/s
1356145: done 4395 episodes, mean reward -77.383, speed 83.66 f/s
1356375: done 4397 episodes, mean reward -77.476, speed 82.24 f/s
1356545: done 4398 episodes, mean reward -77.072, speed 84.90 f/s
1356648: done 4399 episodes, mean reward -77.318, speed 80.71 f/s
1356863: done 4400 episodes, mean reward -76.952, speed 83.30 f/s
Test done in 6.24 sec, reward 208.005, steps 909
1357061: done 4401 episodes, mean reward -76.834, speed 23.15 f/s
1357176: done 4402 episodes, mean reward -76.831, speed 84.44 f/s
1357304: done 4403 episodes, mean reward -77.254, speed 84.93 f/s
1357448: done 4404 episodes, mean reward -77.142, speed 83.86 f/s
1357647: done 4405 episodes, mean reward -77.389, speed 83.65 f/s
1357874: done 4406 episodes, mean reward -76.993, speed 84.51 f/s
1357966: done 4407 episodes, mean reward -77.117, speed 78.65 f/s
Test done in 5.43 sec, reward 161.922, steps 785
1358125: done 4408 episodes, mean reward -77.249, speed 21.82 f/s
1358375: done 4410 episodes, mean reward -78.428, speed 85.17 f/s
1358660: done 4411 episodes, mean reward -78.058, speed 84.74 f/s
1358875: done 4412 episodes, mean reward -78.010, speed 84.67 f/s
Test done in 6.66 sec, reward 256.528, steps 971
1359165: done 4413 episodes, mean reward -77.951, speed 28.52 f/s
1359300: done 4414 episodes, mean reward -78.208, speed 83.07 f/s
1359423: done 4415 episodes, mean reward -77.986, speed 83.24 f/s
1359690: done 4416 episodes, mean reward -77.490, speed 84.42 f/s
1359966: done 4417 episodes, mean reward -77.056, speed 82.50 f/s
Test done in 5.86 sec, reward 198.511, steps 878
1360090: done 4418 episodes, mean reward -77.393, speed 16.87 f/s
1360360: done 4419 episodes, mean reward -77.040, speed 82.23 f/s
1360627: done 4420 episodes, mean reward -76.708, speed 83.44 f/s
1360746: done 4421 episodes, mean reward -76.874, speed 86.02 f/s
1360865: done 4422 episodes, mean reward -76.816, speed 84.75 f/s
1360991: done 4423 episodes, mean reward -77.475, speed 85.26 f/s
Test done in 6.75 sec, reward 253.506, steps 996
1361148: done 4424 episodes, mean reward -77.915, speed 18.16 f/s
1361301: done 4425 episodes, mean reward -78.069, speed 85.22 f/s
1361496: done 4426 episodes, mean reward -78.438, speed 85.54 f/s
1361641: done 4427 episodes, mean reward -78.570, speed 82.20 f/s
1361829: done 4429 episodes, mean reward -79.885, speed 84.16 f/s
1361939: done 4430 episodes, mean reward -79.953, speed 82.08 f/s
Test done in 6.95 sec, reward 254.660, steps 1020
1362106: done 4431 episodes, mean reward -81.042, speed 18.63 f/s
1362334: done 4433 episodes, mean reward -81.546, speed 86.35 f/s
1362545: done 4434 episodes, mean reward -81.169, speed 83.52 f/s
1362658: done 4435 episodes, mean reward -81.124, speed 84.73 f/s
1362810: done 4436 episodes, mean reward -81.179, speed 82.34 f/s
1362985: done 4437 episodes, mean reward -81.256, speed 84.52 f/s
Test done in 7.40 sec, reward 256.223, steps 1011
1363187: done 4438 episodes, mean reward -81.076, speed 20.63 f/s
1363298: done 4439 episodes, mean reward -81.567, speed 84.14 f/s
1363393: done 4440 episodes, mean reward -81.935, speed 83.83 f/s
1363527: done 4441 episodes, mean reward -82.099, speed 83.63 f/s
1363815: done 4443 episodes, mean reward -81.972, speed 83.01 f/s
Test done in 5.44 sec, reward 138.310, steps 792
1364065: done 4445 episodes, mean reward -81.932, speed 29.64 f/s
1364232: done 4446 episodes, mean reward -81.749, speed 82.04 f/s
1364521: done 4447 episodes, mean reward -82.877, speed 84.80 f/s
1364673: done 4448 episodes, mean reward -82.951, speed 84.83 f/s
1364782: done 4449 episodes, mean reward -82.957, speed 86.78 f/s
Test done in 4.43 sec, reward 115.301, steps 675
1365063: done 4450 episodes, mean reward -82.465, speed 36.10 f/s
1365369: done 4452 episodes, mean reward -82.689, speed 82.37 f/s
1365467: done 4453 episodes, mean reward -83.214, speed 83.93 f/s
1365607: done 4454 episodes, mean reward -83.225, speed 87.33 f/s
1365736: done 4455 episodes, mean reward -83.435, speed 84.50 f/s
1365892: done 4456 episodes, mean reward -83.823, speed 86.02 f/s
1365998: done 4457 episodes, mean reward -84.056, speed 82.44 f/s
Test done in 7.78 sec, reward 229.572, steps 1134
1366267: done 4458 episodes, mean reward -84.714, speed 24.56 f/s
1366627: done 4459 episodes, mean reward -84.356, speed 84.87 f/s
1366784: done 4460 episodes, mean reward -84.365, speed 83.23 f/s
1366993: done 4461 episodes, mean reward -84.357, speed 84.56 f/s
Test done in 9.03 sec, reward 216.796, steps 1287
1367222: done 4462 episodes, mean reward -84.023, speed 19.59 f/s
1367416: done 4463 episodes, mean reward -84.052, speed 82.00 f/s
1367716: done 4464 episodes, mean reward -84.196, speed 83.30 f/s
1367808: done 4465 episodes, mean reward -84.354, speed 84.84 f/s
1367953: done 4466 episodes, mean reward -84.215, speed 83.93 f/s
Test done in 7.50 sec, reward 270.715, steps 1109
1368365: done 4467 episodes, mean reward -83.494, speed 33.59 f/s
1368549: done 4468 episodes, mean reward -83.489, speed 81.28 f/s
1368830: done 4469 episodes, mean reward -83.114, speed 84.57 f/s
Test done in 7.73 sec, reward 228.788, steps 1101
1369724: done 4470 episodes, mean reward -80.846, speed 47.75 f/s
1369855: done 4471 episodes, mean reward -80.928, speed 83.90 f/s
1369995: done 4472 episodes, mean reward -80.799, speed 79.61 f/s
Test done in 8.31 sec, reward 178.702, steps 1197
1370088: done 4473 episodes, mean reward -80.839, speed 9.92 f/s
Test done in 7.80 sec, reward 250.745, steps 1130
1371688: done 4474 episodes, mean reward -81.434, speed 58.97 f/s
1371870: done 4475 episodes, mean reward -81.870, speed 84.19 f/s
Test done in 7.16 sec, reward 230.952, steps 1062
Test done in 7.81 sec, reward 271.413, steps 1132
1373470: done 4476 episodes, mean reward -82.913, speed 46.57 f/s
1373653: done 4477 episodes, mean reward -82.839, speed 82.42 f/s
1373736: done 4478 episodes, mean reward -82.882, speed 80.29 f/s
Test done in 8.67 sec, reward 210.928, steps 1259
Test done in 7.84 sec, reward 243.191, steps 1146
1375336: done 4479 episodes, mean reward -83.648, speed 45.11 f/s
Test done in 10.72 sec, reward -90.473, steps 1507
1376936: done 4480 episodes, mean reward -84.329, speed 52.99 f/s
Test done in 11.44 sec, reward -186.870, steps 1600
1377178: done 4481 episodes, mean reward -84.650, speed 16.77 f/s
1377272: done 4482 episodes, mean reward -84.607, speed 83.69 f/s
Test done in 11.19 sec, reward -185.482, steps 1600
1378872: done 4483 episodes, mean reward -85.095, speed 51.82 f/s
Test done in 11.36 sec, reward -181.071, steps 1600
Test done in 11.22 sec, reward -180.878, steps 1600
1380472: done 4484 episodes, mean reward -85.914, speed 38.21 f/s
Test done in 11.28 sec, reward -180.642, steps 1600
Test done in 11.10 sec, reward -177.012, steps 1600
1382072: done 4485 episodes, mean reward -86.320, speed 38.60 f/s
Test done in 11.44 sec, reward -161.555, steps 1600
1383672: done 4486 episodes, mean reward -86.957, speed 52.39 f/s
Test done in 11.38 sec, reward -168.391, steps 1600
Test done in 11.43 sec, reward -167.111, steps 1600
1385272: done 4487 episodes, mean reward -87.621, speed 38.19 f/s
Test done in 11.37 sec, reward -164.266, steps 1600
1386872: done 4488 episodes, mean reward -89.016, speed 52.24 f/s
Test done in 11.35 sec, reward -172.655, steps 1600
Test done in 10.39 sec, reward -32.194, steps 1504
1388472: done 4489 episodes, mean reward -89.854, speed 38.77 f/s
Test done in 9.04 sec, reward 142.294, steps 1285
Test done in 9.72 sec, reward 165.052, steps 1430
1390072: done 4490 episodes, mean reward -90.515, speed 42.06 f/s
1390915: done 4491 episodes, mean reward -90.703, speed 81.75 f/s
Test done in 9.82 sec, reward 139.619, steps 1380
1391387: done 4492 episodes, mean reward -90.135, speed 30.63 f/s
1391600: done 4493 episodes, mean reward -90.107, speed 82.42 f/s
Test done in 8.61 sec, reward 247.795, steps 1237
1392425: done 4494 episodes, mean reward -89.956, speed 44.81 f/s
1392793: done 4495 episodes, mean reward -89.497, speed 85.28 f/s
Test done in 10.10 sec, reward -31.506, steps 1471
1393022: done 4496 episodes, mean reward -89.416, speed 17.85 f/s
Test done in 9.30 sec, reward 113.536, steps 1368
1394622: done 4497 episodes, mean reward -89.201, speed 56.41 f/s
1394753: done 4498 episodes, mean reward -89.312, speed 84.14 f/s
Test done in 5.70 sec, reward 141.309, steps 828
1395159: done 4499 episodes, mean reward -88.683, speed 38.75 f/s
1395546: done 4500 episodes, mean reward -88.572, speed 86.05 f/s
1395954: done 4501 episodes, mean reward -88.787, speed 85.81 f/s
Test done in 3.19 sec, reward -0.904, steps 467
1396006: done 4502 episodes, mean reward -88.796, speed 13.60 f/s
1396229: done 4504 episodes, mean reward -88.917, speed 85.97 f/s
1396451: done 4505 episodes, mean reward -88.598, speed 85.68 f/s
1396613: done 4506 episodes, mean reward -88.702, speed 85.31 f/s
1396711: done 4507 episodes, mean reward -88.634, speed 80.83 f/s
1396959: done 4509 episodes, mean reward -88.918, speed 76.90 f/s
Test done in 7.66 sec, reward 156.266, steps 1107
1397125: done 4510 episodes, mean reward -88.922, speed 17.24 f/s
1397228: done 4512 episodes, mean reward -89.746, speed 88.09 f/s
1397568: done 4514 episodes, mean reward -89.964, speed 85.57 f/s
Test done in 4.30 sec, reward 50.955, steps 604
1398045: done 4515 episodes, mean reward -89.255, speed 47.68 f/s
1398353: done 4516 episodes, mean reward -89.415, speed 81.31 f/s
1398674: done 4517 episodes, mean reward -89.513, speed 81.98 f/s
1398898: done 4518 episodes, mean reward -89.146, speed 84.75 f/s
1398992: done 4519 episodes, mean reward -89.616, speed 83.56 f/s
Test done in 2.03 sec, reward -26.634, steps 295
1399067: done 4520 episodes, mean reward -90.254, speed 25.87 f/s
1399290: done 4522 episodes, mean reward -90.437, speed 84.83 f/s
1399391: done 4523 episodes, mean reward -90.363, speed 86.11 f/s
Test done in 5.57 sec, reward 127.342, steps 812
1400066: done 4524 episodes, mean reward -89.181, speed 49.71 f/s
1400772: done 4525 episodes, mean reward -87.645, speed 84.26 f/s
Test done in 8.10 sec, reward 251.564, steps 1166
1401022: done 4526 episodes, mean reward -87.563, speed 22.09 f/s
1401265: done 4527 episodes, mean reward -87.147, speed 82.38 f/s
1401377: done 4528 episodes, mean reward -86.893, speed 74.67 f/s
1401708: done 4529 episodes, mean reward -86.224, speed 83.03 f/s
1401949: done 4530 episodes, mean reward -86.101, speed 81.82 f/s
Test done in 5.85 sec, reward 80.634, steps 878
1402142: done 4531 episodes, mean reward -86.266, speed 23.59 f/s
1402244: done 4532 episodes, mean reward -86.223, speed 81.97 f/s
1402428: done 4533 episodes, mean reward -86.307, speed 83.89 f/s
1402720: done 4534 episodes, mean reward -86.312, speed 85.39 f/s
1402839: done 4535 episodes, mean reward -86.360, speed 84.48 f/s
Test done in 7.61 sec, reward 253.383, steps 1081
1403024: done 4536 episodes, mean reward -86.567, speed 18.86 f/s
1403110: done 4537 episodes, mean reward -86.608, speed 82.40 f/s
1403420: done 4538 episodes, mean reward -86.568, speed 81.93 f/s
1403863: done 4539 episodes, mean reward -85.451, speed 79.03 f/s
Test done in 8.07 sec, reward 272.560, steps 1205
1404188: done 4540 episodes, mean reward -84.957, speed 26.97 f/s
1404334: done 4541 episodes, mean reward -84.920, speed 85.07 f/s
1404448: done 4543 episodes, mean reward -85.287, speed 81.61 f/s
1404613: done 4544 episodes, mean reward -85.065, speed 83.46 f/s
1404754: done 4546 episodes, mean reward -85.498, speed 85.57 f/s
Test done in 7.29 sec, reward 232.647, steps 1071
1405089: done 4547 episodes, mean reward -84.790, speed 29.89 f/s
1405227: done 4548 episodes, mean reward -84.600, speed 86.10 f/s
1405392: done 4549 episodes, mean reward -84.458, speed 81.54 f/s
1405520: done 4551 episodes, mean reward -84.940, speed 82.48 f/s
1405869: done 4552 episodes, mean reward -84.052, speed 81.82 f/s
1405995: done 4553 episodes, mean reward -83.881, speed 81.94 f/s
Test done in 5.74 sec, reward 171.830, steps 839
1406064: done 4554 episodes, mean reward -83.935, speed 10.54 f/s
1406479: done 4555 episodes, mean reward -83.307, speed 83.33 f/s
1406718: done 4556 episodes, mean reward -82.993, speed 84.05 f/s
Test done in 7.11 sec, reward 242.096, steps 987
1407113: done 4557 episodes, mean reward -82.067, speed 33.49 f/s
1407384: done 4558 episodes, mean reward -81.553, speed 83.64 f/s
1407644: done 4559 episodes, mean reward -81.730, speed 83.56 f/s
1407850: done 4560 episodes, mean reward -81.562, speed 81.03 f/s
Test done in 7.36 sec, reward 242.933, steps 1093
1408336: done 4561 episodes, mean reward -81.049, speed 36.68 f/s
1408436: done 4562 episodes, mean reward -81.448, speed 81.68 f/s
1408641: done 4563 episodes, mean reward -81.424, speed 81.85 f/s
1408831: done 4564 episodes, mean reward -81.637, speed 85.66 f/s
1408990: done 4565 episodes, mean reward -81.229, speed 86.68 f/s
Test done in 7.36 sec, reward 259.942, steps 1070
1409119: done 4566 episodes, mean reward -81.298, speed 14.57 f/s
1409269: done 4567 episodes, mean reward -82.056, speed 84.29 f/s
1409432: done 4568 episodes, mean reward -82.114, speed 82.07 f/s
1409516: done 4569 episodes, mean reward -82.568, speed 82.58 f/s
1409662: done 4570 episodes, mean reward -84.628, speed 83.74 f/s
1409809: done 4572 episodes, mean reward -84.807, speed 83.08 f/s
Test done in 7.27 sec, reward 221.235, steps 1063
1410209: done 4574 episodes, mean reward -83.877, speed 32.83 f/s
1410356: done 4575 episodes, mean reward -84.145, speed 88.82 f/s
1410580: done 4576 episodes, mean reward -83.295, speed 90.37 f/s
1410872: done 4577 episodes, mean reward -83.321, speed 85.04 f/s
1410964: done 4578 episodes, mean reward -83.232, speed 85.35 f/s
Test done in 7.73 sec, reward 281.342, steps 1123
Best reward updated: 275.078 -> 281.342
1411154: done 4579 episodes, mean reward -82.408, speed 18.72 f/s
1411419: done 4580 episodes, mean reward -81.389, speed 83.12 f/s
1411622: done 4581 episodes, mean reward -81.218, speed 82.36 f/s
Test done in 8.18 sec, reward 273.016, steps 1264
1412104: done 4582 episodes, mean reward -80.169, speed 34.68 f/s
1412260: done 4583 episodes, mean reward -79.384, speed 83.93 f/s
1412366: done 4584 episodes, mean reward -78.694, speed 85.45 f/s
1412453: done 4585 episodes, mean reward -78.318, speed 79.60 f/s
Test done in 8.53 sec, reward 247.682, steps 1278
1413089: done 4586 episodes, mean reward -76.397, speed 39.19 f/s
1413277: done 4587 episodes, mean reward -75.824, speed 81.76 f/s
1413752: done 4588 episodes, mean reward -74.349, speed 85.73 f/s
Test done in 7.10 sec, reward 194.969, steps 1060
1414275: done 4589 episodes, mean reward -72.717, speed 38.82 f/s
1414399: done 4590 episodes, mean reward -72.167, speed 81.90 f/s
1414913: done 4591 episodes, mean reward -71.405, speed 84.93 f/s
Test done in 8.06 sec, reward 276.066, steps 1204
1415073: done 4592 episodes, mean reward -72.160, speed 16.05 f/s
1415343: done 4593 episodes, mean reward -71.839, speed 86.11 f/s
1415500: done 4594 episodes, mean reward -72.728, speed 83.52 f/s
1415696: done 4595 episodes, mean reward -73.085, speed 83.79 f/s
1415901: done 4596 episodes, mean reward -72.846, speed 85.02 f/s
Test done in 9.70 sec, reward 260.568, steps 1433
1416041: done 4597 episodes, mean reward -73.180, speed 12.30 f/s
1416118: done 4598 episodes, mean reward -73.468, speed 73.68 f/s
1416440: done 4599 episodes, mean reward -73.583, speed 79.84 f/s
1416824: done 4600 episodes, mean reward -73.284, speed 85.40 f/s
1416934: done 4601 episodes, mean reward -73.512, speed 80.07 f/s
Test done in 9.72 sec, reward 258.094, steps 1450
1417100: done 4602 episodes, mean reward -73.421, speed 14.16 f/s
1417483: done 4603 episodes, mean reward -72.775, speed 82.99 f/s
1417782: done 4604 episodes, mean reward -72.385, speed 84.20 f/s
Test done in 9.01 sec, reward 263.401, steps 1343
1418116: done 4605 episodes, mean reward -71.983, speed 25.72 f/s
1418355: done 4606 episodes, mean reward -71.954, speed 81.79 f/s
1418458: done 4607 episodes, mean reward -71.918, speed 83.82 f/s
1418569: done 4608 episodes, mean reward -71.712, speed 86.04 f/s
1418786: done 4609 episodes, mean reward -71.353, speed 87.91 f/s
Test done in 10.51 sec, reward 252.831, steps 1519
1419008: done 4610 episodes, mean reward -71.382, speed 16.81 f/s
1419272: done 4611 episodes, mean reward -70.795, speed 80.79 f/s
1419403: done 4612 episodes, mean reward -70.606, speed 83.73 f/s
1419487: done 4613 episodes, mean reward -70.574, speed 71.33 f/s
1419740: done 4614 episodes, mean reward -70.980, speed 85.71 f/s
1419865: done 4616 episodes, mean reward -72.327, speed 83.46 f/s
Test done in 7.12 sec, reward 148.230, steps 1038
1420174: done 4617 episodes, mean reward -72.243, speed 28.65 f/s
1420415: done 4619 episodes, mean reward -72.639, speed 86.41 f/s
1420712: done 4620 episodes, mean reward -72.154, speed 83.68 f/s
1420941: done 4622 episodes, mean reward -72.075, speed 84.88 f/s
Test done in 5.89 sec, reward 42.956, steps 878
1421066: done 4623 episodes, mean reward -72.265, speed 17.05 f/s
1421190: done 4624 episodes, mean reward -73.485, speed 87.31 f/s
1421494: done 4625 episodes, mean reward -74.818, speed 83.13 f/s
1421845: done 4627 episodes, mean reward -75.451, speed 82.16 f/s
1421975: done 4628 episodes, mean reward -75.564, speed 85.03 f/s
Test done in 0.61 sec, reward -118.443, steps 86
1422231: done 4629 episodes, mean reward -75.815, speed 69.91 f/s
1422388: done 4630 episodes, mean reward -76.278, speed 83.68 f/s
1422480: done 4631 episodes, mean reward -76.297, speed 82.41 f/s
1422875: done 4632 episodes, mean reward -75.546, speed 84.01 f/s
1422998: done 4633 episodes, mean reward -75.493, speed 88.26 f/s
Test done in 8.76 sec, reward 269.593, steps 1300
1423174: done 4634 episodes, mean reward -75.844, speed 16.21 f/s
1423361: done 4636 episodes, mean reward -75.915, speed 83.81 f/s
1423629: done 4637 episodes, mean reward -75.540, speed 81.60 f/s
1423714: done 4638 episodes, mean reward -75.992, speed 80.51 f/s
1423872: done 4639 episodes, mean reward -76.604, speed 85.30 f/s
Test done in 6.01 sec, reward 123.389, steps 909
1424140: done 4640 episodes, mean reward -76.432, speed 29.28 f/s
1424276: done 4641 episodes, mean reward -76.436, speed 86.47 f/s
1424404: done 4642 episodes, mean reward -76.470, speed 83.96 f/s
1424499: done 4643 episodes, mean reward -76.485, speed 85.52 f/s
1424620: done 4644 episodes, mean reward -76.750, speed 84.98 f/s
1424859: done 4645 episodes, mean reward -76.280, speed 83.89 f/s
Test done in 9.13 sec, reward 265.900, steps 1345
1425075: done 4646 episodes, mean reward -75.965, speed 18.38 f/s
1425234: done 4647 episodes, mean reward -76.452, speed 87.56 f/s
1425563: done 4648 episodes, mean reward -76.129, speed 86.05 f/s
1425931: done 4649 episodes, mean reward -75.605, speed 85.90 f/s
Test done in 8.55 sec, reward 239.417, steps 1286
1426158: done 4650 episodes, mean reward -75.396, speed 20.14 f/s
1426305: done 4651 episodes, mean reward -75.247, speed 84.47 f/s
1426544: done 4652 episodes, mean reward -75.686, speed 81.66 f/s
1426887: done 4653 episodes, mean reward -75.076, speed 84.35 f/s
1426996: done 4654 episodes, mean reward -75.137, speed 85.70 f/s
Test done in 6.59 sec, reward 178.557, steps 997
1427232: done 4655 episodes, mean reward -75.257, speed 25.45 f/s
1427414: done 4656 episodes, mean reward -75.581, speed 85.25 f/s
1427515: done 4657 episodes, mean reward -76.420, speed 87.06 f/s
1427749: done 4658 episodes, mean reward -76.449, speed 83.12 f/s
1427901: done 4659 episodes, mean reward -76.957, speed 84.57 f/s
Test done in 8.26 sec, reward 272.200, steps 1255
1428061: done 4660 episodes, mean reward -77.085, speed 15.76 f/s
1428284: done 4661 episodes, mean reward -77.516, speed 84.78 f/s
1428628: done 4662 episodes, mean reward -76.928, speed 85.31 f/s
1428822: done 4663 episodes, mean reward -77.016, speed 83.18 f/s
1428929: done 4664 episodes, mean reward -77.378, speed 85.96 f/s
Test done in 1.89 sec, reward -135.352, steps 268
1429047: done 4665 episodes, mean reward -77.575, speed 35.77 f/s
1429151: done 4666 episodes, mean reward -77.620, speed 84.68 f/s
1429403: done 4667 episodes, mean reward -77.497, speed 74.76 f/s
1429528: done 4668 episodes, mean reward -77.608, speed 81.48 f/s
1429649: done 4669 episodes, mean reward -77.662, speed 87.95 f/s
1429760: done 4670 episodes, mean reward -77.678, speed 85.12 f/s
Test done in 9.37 sec, reward 264.171, steps 1394
1430044: done 4671 episodes, mean reward -77.188, speed 22.35 f/s
1430173: done 4672 episodes, mean reward -77.112, speed 86.20 f/s
1430408: done 4673 episodes, mean reward -76.896, speed 83.49 f/s
1430847: done 4674 episodes, mean reward -76.585, speed 84.09 f/s
Test done in 8.51 sec, reward 276.803, steps 1260
1431150: done 4675 episodes, mean reward -76.122, speed 25.08 f/s
1431641: done 4676 episodes, mean reward -75.370, speed 85.54 f/s
1431894: done 4677 episodes, mean reward -75.400, speed 86.80 f/s
Test done in 8.10 sec, reward 275.818, steps 1245
1432335: done 4678 episodes, mean reward -74.539, speed 31.80 f/s
1432511: done 4679 episodes, mean reward -74.908, speed 78.54 f/s
1432897: done 4680 episodes, mean reward -74.758, speed 82.01 f/s
Test done in 8.35 sec, reward 271.158, steps 1273
1433033: done 4681 episodes, mean reward -75.001, speed 13.60 f/s
1433138: done 4682 episodes, mean reward -76.041, speed 85.45 f/s
1433280: done 4683 episodes, mean reward -76.419, speed 86.28 f/s
1433574: done 4684 episodes, mean reward -75.940, speed 83.72 f/s
1433778: done 4685 episodes, mean reward -75.693, speed 84.36 f/s
Test done in 8.26 sec, reward 275.245, steps 1223
1434008: done 4686 episodes, mean reward -76.739, speed 20.89 f/s
1434184: done 4687 episodes, mean reward -76.622, speed 82.48 f/s
1434486: done 4689 episodes, mean reward -78.387, speed 84.18 f/s
1434588: done 4690 episodes, mean reward -78.588, speed 82.38 f/s
Test done in 9.53 sec, reward 261.697, steps 1407
1435221: done 4691 episodes, mean reward -78.386, speed 37.18 f/s
1435311: done 4692 episodes, mean reward -78.582, speed 85.73 f/s
1435426: done 4693 episodes, mean reward -78.831, speed 84.26 f/s
1435578: done 4694 episodes, mean reward -78.720, speed 86.50 f/s
1435677: done 4695 episodes, mean reward -78.761, speed 86.37 f/s
Test done in 7.62 sec, reward 253.623, steps 1146
1436031: done 4697 episodes, mean reward -78.839, speed 29.75 f/s
1436298: done 4698 episodes, mean reward -78.300, speed 86.89 f/s
1436711: done 4699 episodes, mean reward -78.025, speed 85.13 f/s
Test done in 10.64 sec, reward 57.663, steps 1494
1437225: done 4700 episodes, mean reward -77.717, speed 30.71 f/s
1437490: done 4701 episodes, mean reward -77.218, speed 85.21 f/s
1437761: done 4702 episodes, mean reward -76.814, speed 82.95 f/s
1437934: done 4703 episodes, mean reward -77.140, speed 81.22 f/s
Test done in 8.09 sec, reward 276.555, steps 1218
1438174: done 4704 episodes, mean reward -77.173, speed 21.98 f/s
1438371: done 4706 episodes, mean reward -78.189, speed 86.38 f/s
1438562: done 4707 episodes, mean reward -78.202, speed 87.20 f/s
1438698: done 4708 episodes, mean reward -78.297, speed 82.00 f/s
1438872: done 4709 episodes, mean reward -78.577, speed 83.24 f/s
1438988: done 4710 episodes, mean reward -78.767, speed 82.87 f/s
Test done in 8.79 sec, reward 245.635, steps 1315
1439067: done 4711 episodes, mean reward -79.468, speed 8.15 f/s
1439229: done 4712 episodes, mean reward -79.363, speed 85.24 f/s
1439396: done 4713 episodes, mean reward -79.285, speed 81.63 f/s
1439514: done 4714 episodes, mean reward -79.417, speed 87.50 f/s
1439680: done 4715 episodes, mean reward -79.213, speed 86.76 f/s
1439867: done 4716 episodes, mean reward -78.915, speed 82.00 f/s
Test done in 8.74 sec, reward 235.480, steps 1269
1440107: done 4717 episodes, mean reward -79.003, speed 20.66 f/s
1440215: done 4718 episodes, mean reward -78.751, speed 81.76 f/s
1440455: done 4719 episodes, mean reward -78.748, speed 86.00 f/s
1440899: done 4720 episodes, mean reward -78.166, speed 85.25 f/s
Test done in 8.65 sec, reward 271.395, steps 1295
1441242: done 4721 episodes, mean reward -77.339, speed 27.01 f/s
1441388: done 4722 episodes, mean reward -77.286, speed 81.29 f/s
1441708: done 4723 episodes, mean reward -76.573, speed 82.77 f/s
1441812: done 4724 episodes, mean reward -76.799, speed 83.64 f/s
1441984: done 4725 episodes, mean reward -76.967, speed 84.89 f/s
Test done in 8.17 sec, reward 275.270, steps 1205
Test done in 8.67 sec, reward 257.379, steps 1206
1443584: done 4726 episodes, mean reward -77.088, speed 44.32 f/s
1443716: done 4727 episodes, mean reward -77.443, speed 81.55 f/s
1443822: done 4728 episodes, mean reward -77.422, speed 86.43 f/s
Test done in 8.31 sec, reward 232.319, steps 1228
1444060: done 4729 episodes, mean reward -77.126, speed 21.46 f/s
1444304: done 4730 episodes, mean reward -76.571, speed 82.51 f/s
1444401: done 4731 episodes, mean reward -76.617, speed 80.66 f/s
1444663: done 4732 episodes, mean reward -76.875, speed 82.36 f/s
1444901: done 4733 episodes, mean reward -76.659, speed 85.88 f/s
Test done in 8.28 sec, reward 233.266, steps 1220
1445063: done 4734 episodes, mean reward -76.506, speed 15.95 f/s
1445337: done 4735 episodes, mean reward -76.104, speed 86.55 f/s
1445594: done 4736 episodes, mean reward -75.638, speed 84.98 f/s
Test done in 8.14 sec, reward 255.967, steps 1178
1446127: done 4737 episodes, mean reward -74.957, speed 36.96 f/s
Test done in 7.71 sec, reward 242.789, steps 1114
1447727: done 4738 episodes, mean reward -75.254, speed 59.78 f/s
1447953: done 4739 episodes, mean reward -75.075, speed 82.80 f/s
Test done in 8.33 sec, reward 274.441, steps 1244
1448287: done 4740 episodes, mean reward -75.102, speed 27.38 f/s
1448416: done 4741 episodes, mean reward -75.042, speed 82.68 f/s
1448575: done 4742 episodes, mean reward -74.690, speed 83.99 f/s
1448843: done 4743 episodes, mean reward -73.965, speed 85.27 f/s
Test done in 7.71 sec, reward 115.248, steps 1113
1449116: done 4745 episodes, mean reward -74.225, speed 24.71 f/s
1449206: done 4746 episodes, mean reward -74.438, speed 83.32 f/s
1449437: done 4748 episodes, mean reward -75.179, speed 87.14 f/s
1449835: done 4749 episodes, mean reward -75.072, speed 84.54 f/s
Test done in 8.44 sec, reward 270.557, steps 1251
1450073: done 4750 episodes, mean reward -74.848, speed 21.11 f/s
1450454: done 4751 episodes, mean reward -74.300, speed 83.11 f/s
1450601: done 4752 episodes, mean reward -74.615, speed 80.84 f/s
Test done in 7.09 sec, reward 235.810, steps 1097
1451160: done 4753 episodes, mean reward -73.996, speed 40.89 f/s
1451281: done 4754 episodes, mean reward -73.893, speed 84.57 f/s
1451408: done 4755 episodes, mean reward -74.169, speed 80.77 f/s
1451908: done 4756 episodes, mean reward -73.445, speed 85.50 f/s
Test done in 7.86 sec, reward 241.047, steps 1154
1452044: done 4757 episodes, mean reward -73.431, speed 14.28 f/s
1452206: done 4758 episodes, mean reward -73.766, speed 78.26 f/s
1452454: done 4759 episodes, mean reward -73.678, speed 82.52 f/s
1452753: done 4760 episodes, mean reward -73.260, speed 82.74 f/s
1452984: done 4761 episodes, mean reward -73.100, speed 84.01 f/s
Test done in 7.24 sec, reward 213.525, steps 1046
1453162: done 4762 episodes, mean reward -73.413, speed 19.09 f/s
1453406: done 4763 episodes, mean reward -73.114, speed 83.04 f/s
Test done in 7.75 sec, reward 217.710, steps 1154
1454043: done 4764 episodes, mean reward -71.544, speed 41.08 f/s
1454149: done 4765 episodes, mean reward -71.458, speed 83.03 f/s
1454806: done 4766 episodes, mean reward -69.994, speed 85.67 f/s
Test done in 7.44 sec, reward 240.170, steps 1105
1455130: done 4768 episodes, mean reward -69.771, speed 28.96 f/s
Test done in 6.18 sec, reward 151.101, steps 925
1456367: done 4769 episodes, mean reward -68.461, speed 59.29 f/s
1456677: done 4770 episodes, mean reward -68.061, speed 85.21 f/s
Test done in 7.44 sec, reward 208.076, steps 1075
1457039: done 4771 episodes, mean reward -67.605, speed 30.71 f/s
1457237: done 4772 episodes, mean reward -67.270, speed 85.21 f/s
1457751: done 4773 episodes, mean reward -66.388, speed 83.68 f/s
Test done in 7.81 sec, reward 227.888, steps 1123
1458297: done 4774 episodes, mean reward -66.000, speed 38.28 f/s
1458519: done 4775 episodes, mean reward -66.094, speed 82.99 f/s
1458850: done 4777 episodes, mean reward -66.856, speed 79.66 f/s
1458956: done 4778 episodes, mean reward -67.758, speed 79.85 f/s
Test done in 7.60 sec, reward 231.657, steps 1106
1459278: done 4779 episodes, mean reward -67.116, speed 28.01 f/s
1459383: done 4780 episodes, mean reward -67.728, speed 83.20 f/s
1459527: done 4781 episodes, mean reward -67.721, speed 83.25 f/s
1459644: done 4782 episodes, mean reward -67.655, speed 86.01 f/s
1459760: done 4783 episodes, mean reward -67.382, speed 84.98 f/s
1459857: done 4784 episodes, mean reward -67.985, speed 80.83 f/s
Test done in 8.18 sec, reward 259.504, steps 1212
1460486: done 4786 episodes, mean reward -67.400, speed 40.34 f/s
1460664: done 4787 episodes, mean reward -67.302, speed 84.74 f/s
1460794: done 4788 episodes, mean reward -67.225, speed 83.15 f/s
1460908: done 4789 episodes, mean reward -67.469, speed 86.00 f/s
Test done in 8.35 sec, reward 241.727, steps 1219
1461416: done 4790 episodes, mean reward -65.973, speed 35.62 f/s
Test done in 8.48 sec, reward 186.433, steps 1228
1462497: done 4791 episodes, mean reward -63.387, speed 49.36 f/s
1462764: done 4793 episodes, mean reward -63.113, speed 83.41 f/s
Test done in 8.08 sec, reward 221.893, steps 1208
1463304: done 4795 episodes, mean reward -62.453, speed 37.23 f/s
1463397: done 4796 episodes, mean reward -62.178, speed 82.67 f/s
1463502: done 4797 episodes, mean reward -62.593, speed 81.41 f/s
1463975: done 4798 episodes, mean reward -62.200, speed 83.94 f/s
Test done in 7.29 sec, reward 226.477, steps 1074
1464054: done 4799 episodes, mean reward -62.954, speed 9.57 f/s
1464354: done 4800 episodes, mean reward -63.416, speed 83.93 f/s
1464493: done 4801 episodes, mean reward -63.689, speed 81.82 f/s
1464686: done 4802 episodes, mean reward -64.011, speed 84.08 f/s
Test done in 10.06 sec, reward 214.168, steps 1422
1465008: done 4803 episodes, mean reward -63.535, speed 23.33 f/s
1465419: done 4804 episodes, mean reward -63.219, speed 84.82 f/s
1465639: done 4805 episodes, mean reward -62.729, speed 85.68 f/s
Test done in 8.57 sec, reward 212.134, steps 1262
1466292: done 4806 episodes, mean reward -61.099, speed 39.63 f/s
1466428: done 4807 episodes, mean reward -61.100, speed 84.62 f/s
1466612: done 4808 episodes, mean reward -60.808, speed 81.92 f/s
1466774: done 4809 episodes, mean reward -60.671, speed 83.25 f/s
1466897: done 4810 episodes, mean reward -60.689, speed 88.23 f/s
Test done in 9.84 sec, reward 196.257, steps 1470
1467447: done 4811 episodes, mean reward -59.748, speed 33.66 f/s
1467997: done 4812 episodes, mean reward -59.104, speed 80.96 f/s
Test done in 9.38 sec, reward 186.285, steps 1328
1468218: done 4813 episodes, mean reward -58.709, speed 18.52 f/s
1468745: done 4814 episodes, mean reward -57.875, speed 81.83 f/s
1468854: done 4815 episodes, mean reward -58.069, speed 86.08 f/s
Test done in 8.44 sec, reward 233.032, steps 1221
1469654: done 4816 episodes, mean reward -56.441, speed 44.69 f/s
1469816: done 4817 episodes, mean reward -56.622, speed 83.87 f/s
Test done in 8.07 sec, reward 120.443, steps 1169
1470058: done 4818 episodes, mean reward -56.159, speed 21.90 f/s
Test done in 9.00 sec, reward 197.274, steps 1272
1471316: done 4819 episodes, mean reward -53.974, speed 52.15 f/s
Test done in 9.70 sec, reward 209.489, steps 1403
1472446: done 4821 episodes, mean reward -53.600, speed 48.48 f/s
Test done in 6.70 sec, reward 115.246, steps 981
1473066: done 4822 episodes, mean reward -52.310, speed 43.80 f/s
1473182: done 4823 episodes, mean reward -52.818, speed 84.41 f/s
1473408: done 4824 episodes, mean reward -52.328, speed 81.57 f/s
1473531: done 4825 episodes, mean reward -52.527, speed 86.29 f/s
1473645: done 4826 episodes, mean reward -52.055, speed 83.46 f/s
1473757: done 4827 episodes, mean reward -52.099, speed 86.46 f/s
1473885: done 4828 episodes, mean reward -51.973, speed 79.38 f/s
Test done in 7.70 sec, reward 165.086, steps 1114
1474080: done 4830 episodes, mean reward -52.708, speed 19.14 f/s
1474209: done 4831 episodes, mean reward -52.616, speed 82.30 f/s
1474727: done 4833 episodes, mean reward -53.092, speed 83.68 f/s
1474857: done 4834 episodes, mean reward -53.115, speed 85.99 f/s
1474986: done 4835 episodes, mean reward -53.390, speed 83.09 f/s
Test done in 9.00 sec, reward 244.392, steps 1300
1475072: done 4836 episodes, mean reward -53.948, speed 8.53 f/s
1475169: done 4837 episodes, mean reward -55.145, speed 79.71 f/s
1475303: done 4838 episodes, mean reward -54.751, speed 84.78 f/s
1475588: done 4839 episodes, mean reward -54.432, speed 86.08 f/s
Test done in 6.96 sec, reward 235.094, steps 998
1476133: done 4840 episodes, mean reward -54.049, speed 40.47 f/s
1476230: done 4841 episodes, mean reward -54.113, speed 80.78 f/s
1476333: done 4842 episodes, mean reward -54.466, speed 85.58 f/s
1476895: done 4843 episodes, mean reward -54.124, speed 84.00 f/s
1476981: done 4844 episodes, mean reward -54.051, speed 84.55 f/s
Test done in 7.07 sec, reward 178.533, steps 947
1477058: done 4845 episodes, mean reward -54.273, speed 9.62 f/s
1477243: done 4846 episodes, mean reward -54.034, speed 84.86 f/s
1477873: done 4847 episodes, mean reward -52.305, speed 81.54 f/s
Test done in 6.47 sec, reward 156.932, steps 912
1478241: done 4848 episodes, mean reward -51.863, speed 33.80 f/s
1478349: done 4849 episodes, mean reward -52.717, speed 81.74 f/s
1478720: done 4850 episodes, mean reward -52.221, speed 83.92 f/s
1478881: done 4852 episodes, mean reward -52.937, speed 82.94 f/s
Test done in 6.51 sec, reward 193.467, steps 976
1479023: done 4853 episodes, mean reward -54.059, speed 17.35 f/s
1479235: done 4854 episodes, mean reward -53.687, speed 81.32 f/s
1479349: done 4855 episodes, mean reward -53.786, speed 85.70 f/s
1479568: done 4856 episodes, mean reward -54.163, speed 82.74 f/s
1479737: done 4857 episodes, mean reward -54.078, speed 84.25 f/s
Test done in 6.96 sec, reward 193.557, steps 979
1480008: done 4858 episodes, mean reward -53.539, speed 26.67 f/s
1480286: done 4859 episodes, mean reward -53.156, speed 83.02 f/s
1480415: done 4860 episodes, mean reward -53.502, speed 85.79 f/s
1480707: done 4861 episodes, mean reward -53.281, speed 84.68 f/s
1480852: done 4862 episodes, mean reward -53.293, speed 80.49 f/s
Test done in 7.32 sec, reward 125.472, steps 1055
1481181: done 4864 episodes, mean reward -55.212, speed 29.02 f/s
1481381: done 4865 episodes, mean reward -54.966, speed 82.56 f/s
Test done in 6.52 sec, reward 144.032, steps 879
1482022: done 4866 episodes, mean reward -55.066, speed 44.99 f/s
1482453: done 4867 episodes, mean reward -55.272, speed 83.85 f/s
1482877: done 4868 episodes, mean reward -54.862, speed 84.44 f/s
Test done in 7.01 sec, reward 156.739, steps 986
1483486: done 4869 episodes, mean reward -55.291, speed 42.71 f/s
1483638: done 4870 episodes, mean reward -55.694, speed 85.67 f/s
1483774: done 4871 episodes, mean reward -56.327, speed 81.24 f/s
Test done in 5.64 sec, reward 36.162, steps 821
1484171: done 4872 episodes, mean reward -56.250, speed 38.65 f/s
1484459: done 4873 episodes, mean reward -56.868, speed 82.08 f/s
1484621: done 4874 episodes, mean reward -58.056, speed 83.85 f/s
Test done in 4.77 sec, reward 11.005, steps 651
1485031: done 4875 episodes, mean reward -57.595, speed 42.22 f/s
1485355: done 4876 episodes, mean reward -57.104, speed 77.32 f/s
1485736: done 4877 episodes, mean reward -56.899, speed 81.97 f/s
Test done in 6.09 sec, reward 132.020, steps 868
1486164: done 4878 episodes, mean reward -56.264, speed 38.61 f/s
1486392: done 4879 episodes, mean reward -56.533, speed 81.77 f/s
1486783: done 4880 episodes, mean reward -56.001, speed 83.12 f/s
Test done in 5.82 sec, reward 176.226, steps 833
1487240: done 4881 episodes, mean reward -55.112, speed 40.22 f/s
1487583: done 4882 episodes, mean reward -54.631, speed 84.14 f/s
1487743: done 4883 episodes, mean reward -54.810, speed 78.58 f/s
Test done in 6.75 sec, reward 241.821, steps 993
1488585: done 4884 episodes, mean reward -52.925, speed 49.99 f/s
Test done in 7.13 sec, reward 199.736, steps 1006
1489010: done 4886 episodes, mean reward -53.522, speed 33.55 f/s
1489185: done 4887 episodes, mean reward -53.681, speed 84.29 f/s
1489639: done 4888 episodes, mean reward -52.964, speed 82.46 f/s
1489861: done 4889 episodes, mean reward -52.788, speed 86.25 f/s
Test done in 5.49 sec, reward 96.604, steps 769
1490177: done 4890 episodes, mean reward -53.976, speed 33.93 f/s
1490896: done 4891 episodes, mean reward -56.413, speed 81.55 f/s
Test done in 8.08 sec, reward 263.755, steps 1164
1491106: done 4892 episodes, mean reward -56.087, speed 19.72 f/s
1491489: done 4893 episodes, mean reward -55.882, speed 83.90 f/s
1491882: done 4894 episodes, mean reward -55.016, speed 83.73 f/s
Test done in 7.93 sec, reward 218.784, steps 1063
1492206: done 4895 episodes, mean reward -55.357, speed 26.95 f/s
1492341: done 4896 episodes, mean reward -55.323, speed 80.10 f/s
1492599: done 4897 episodes, mean reward -55.010, speed 79.23 f/s
Test done in 7.33 sec, reward 217.525, steps 1063
1493520: done 4898 episodes, mean reward -53.738, speed 49.56 f/s
Test done in 7.23 sec, reward 277.135, steps 1037
1494022: done 4900 episodes, mean reward -53.758, speed 37.77 f/s
1494232: done 4901 episodes, mean reward -53.800, speed 85.43 f/s
1494541: done 4902 episodes, mean reward -53.416, speed 82.30 f/s
1494808: done 4903 episodes, mean reward -53.555, speed 85.52 f/s
1494973: done 4904 episodes, mean reward -54.262, speed 82.56 f/s
Test done in 7.43 sec, reward 229.236, steps 1059
1495451: done 4905 episodes, mean reward -53.954, speed 36.62 f/s
1495632: done 4906 episodes, mean reward -55.361, speed 83.52 f/s
Test done in 7.30 sec, reward 258.930, steps 1066
1496003: done 4907 episodes, mean reward -54.779, speed 31.81 f/s
1496355: done 4908 episodes, mean reward -54.447, speed 82.51 f/s
1496635: done 4909 episodes, mean reward -54.002, speed 81.88 f/s
1496959: done 4910 episodes, mean reward -53.536, speed 79.47 f/s
Test done in 6.78 sec, reward 231.931, steps 985
1497232: done 4911 episodes, mean reward -53.898, speed 27.11 f/s
1497522: done 4912 episodes, mean reward -54.366, speed 84.65 f/s
Test done in 7.48 sec, reward 253.997, steps 1064
1498040: done 4913 episodes, mean reward -53.756, speed 37.78 f/s
1498419: done 4914 episodes, mean reward -53.832, speed 75.64 f/s
1498571: done 4915 episodes, mean reward -53.730, speed 82.99 f/s
1498750: done 4916 episodes, mean reward -55.241, speed 82.37 f/s
1498848: done 4917 episodes, mean reward -55.366, speed 81.38 f/s
Test done in 5.02 sec, reward 153.842, steps 736
1499177: done 4918 episodes, mean reward -55.210, speed 36.42 f/s
1499281: done 4919 episodes, mean reward -57.677, speed 79.51 f/s
1499464: done 4920 episodes, mean reward -57.420, speed 81.70 f/s
1499673: done 4921 episodes, mean reward -59.241, speed 84.66 f/s
Test done in 8.05 sec, reward 238.628, steps 1144
1500126: done 4922 episodes, mean reward -59.636, speed 33.72 f/s
1500356: done 4923 episodes, mean reward -59.448, speed 83.62 f/s
1500485: done 4924 episodes, mean reward -59.682, speed 85.37 f/s
1500907: done 4925 episodes, mean reward -58.731, speed 83.35 f/s
Test done in 6.79 sec, reward 268.841, steps 1005
1501053: done 4926 episodes, mean reward -58.629, speed 17.14 f/s
1501345: done 4927 episodes, mean reward -58.142, speed 86.57 f/s
1501437: done 4928 episodes, mean reward -58.364, speed 83.37 f/s
1501698: done 4929 episodes, mean reward -57.907, speed 78.06 f/s
1501835: done 4930 episodes, mean reward -57.874, speed 79.76 f/s
1501947: done 4931 episodes, mean reward -57.821, speed 83.70 f/s
Test done in 6.18 sec, reward 220.449, steps 895
1502077: done 4932 episodes, mean reward -57.604, speed 16.79 f/s
1502471: done 4933 episodes, mean reward -57.225, speed 82.36 f/s
1502578: done 4934 episodes, mean reward -57.345, speed 86.67 f/s
1502683: done 4935 episodes, mean reward -57.370, speed 82.49 f/s
Test done in 7.12 sec, reward 218.116, steps 1002
1503073: done 4936 episodes, mean reward -56.462, speed 33.20 f/s
1503433: done 4937 episodes, mean reward -55.599, speed 85.49 f/s
1503538: done 4938 episodes, mean reward -55.751, speed 78.85 f/s
1503683: done 4939 episodes, mean reward -56.314, speed 83.71 f/s
1503787: done 4940 episodes, mean reward -57.323, speed 88.44 f/s
Test done in 7.84 sec, reward 255.065, steps 1129
1504076: done 4941 episodes, mean reward -56.850, speed 25.53 f/s
1504219: done 4942 episodes, mean reward -56.543, speed 83.66 f/s
1504327: done 4943 episodes, mean reward -57.420, speed 82.00 f/s
1504482: done 4944 episodes, mean reward -57.273, speed 84.63 f/s
1504582: done 4945 episodes, mean reward -57.316, speed 79.63 f/s
Test done in 7.97 sec, reward 241.914, steps 1121
1505062: done 4946 episodes, mean reward -56.587, speed 34.73 f/s
1505602: done 4947 episodes, mean reward -56.891, speed 83.76 f/s
1505821: done 4948 episodes, mean reward -57.053, speed 78.76 f/s
Test done in 6.15 sec, reward 162.612, steps 912
1506788: done 4949 episodes, mean reward -54.642, speed 54.67 f/s
1506919: done 4950 episodes, mean reward -55.422, speed 89.00 f/s
Test done in 8.22 sec, reward 243.627, steps 1176
1507040: done 4951 episodes, mean reward -55.415, speed 12.52 f/s
1507193: done 4952 episodes, mean reward -55.432, speed 85.57 f/s
1507410: done 4953 episodes, mean reward -55.163, speed 79.41 f/s
1507585: done 4954 episodes, mean reward -55.420, speed 81.99 f/s
1507777: done 4955 episodes, mean reward -55.131, speed 78.86 f/s
1507929: done 4956 episodes, mean reward -55.461, speed 83.88 f/s
Test done in 7.36 sec, reward 240.591, steps 1024
1508097: done 4957 episodes, mean reward -55.501, speed 17.86 f/s
1508420: done 4958 episodes, mean reward -55.578, speed 83.03 f/s
Test done in 7.33 sec, reward 274.884, steps 1084
1509037: done 4959 episodes, mean reward -54.660, speed 41.73 f/s
1509260: done 4960 episodes, mean reward -54.695, speed 82.82 f/s
1509455: done 4961 episodes, mean reward -55.109, speed 83.72 f/s
1509703: done 4962 episodes, mean reward -54.834, speed 81.59 f/s
1509877: done 4963 episodes, mean reward -54.618, speed 83.86 f/s
Test done in 7.10 sec, reward 246.897, steps 1041
1510144: done 4964 episodes, mean reward -54.224, speed 26.00 f/s
1510426: done 4965 episodes, mean reward -53.931, speed 82.94 f/s
1510907: done 4966 episodes, mean reward -54.255, speed 83.01 f/s
Test done in 8.05 sec, reward 269.059, steps 1111
1511025: done 4967 episodes, mean reward -53.903, speed 12.46 f/s
1511267: done 4968 episodes, mean reward -54.291, speed 79.67 f/s
1511436: done 4970 episodes, mean reward -55.240, speed 82.44 f/s
1511908: done 4971 episodes, mean reward -54.347, speed 80.55 f/s
Test done in 5.43 sec, reward 164.103, steps 761
1512038: done 4972 episodes, mean reward -54.591, speed 18.63 f/s
1512174: done 4973 episodes, mean reward -55.294, speed 82.52 f/s
1512629: done 4974 episodes, mean reward -54.376, speed 82.51 f/s
1512724: done 4975 episodes, mean reward -55.227, speed 81.70 f/s
1512869: done 4976 episodes, mean reward -55.622, speed 83.91 f/s
Test done in 7.40 sec, reward 276.711, steps 1082
1513544: done 4977 episodes, mean reward -54.831, speed 43.77 f/s
1513998: done 4978 episodes, mean reward -54.449, speed 83.45 f/s
Test done in 8.19 sec, reward 224.808, steps 1206
1514393: done 4979 episodes, mean reward -54.007, speed 30.23 f/s
1514503: done 4980 episodes, mean reward -54.451, speed 79.09 f/s
Test done in 6.47 sec, reward 183.030, steps 946
1515134: done 4981 episodes, mean reward -53.983, speed 44.78 f/s
1515579: done 4982 episodes, mean reward -53.448, speed 81.42 f/s
Test done in 7.54 sec, reward 221.212, steps 1119
1516007: done 4983 episodes, mean reward -52.464, speed 33.90 f/s
Test done in 7.62 sec, reward 244.221, steps 1126
1517607: done 4984 episodes, mean reward -54.559, speed 59.70 f/s
1517718: done 4985 episodes, mean reward -54.491, speed 81.36 f/s
1517850: done 4986 episodes, mean reward -54.904, speed 83.42 f/s
1517992: done 4987 episodes, mean reward -54.951, speed 85.51 f/s
Test done in 5.62 sec, reward 153.746, steps 832
1518060: done 4988 episodes, mean reward -55.855, speed 10.57 f/s
1518148: done 4989 episodes, mean reward -56.130, speed 81.02 f/s
1518385: done 4990 episodes, mean reward -55.895, speed 82.92 f/s
1518487: done 4991 episodes, mean reward -57.225, speed 85.40 f/s
1518641: done 4992 episodes, mean reward -57.354, speed 79.97 f/s
1518848: done 4993 episodes, mean reward -57.719, speed 80.89 f/s
Test done in 6.89 sec, reward 221.007, steps 1029
1519072: done 4994 episodes, mean reward -58.228, speed 23.33 f/s
1519198: done 4995 episodes, mean reward -58.712, speed 79.72 f/s
1519386: done 4996 episodes, mean reward -58.475, speed 85.62 f/s
1519507: done 4997 episodes, mean reward -58.807, speed 84.50 f/s
1519791: done 4998 episodes, mean reward -60.474, speed 84.99 f/s
1519959: done 4999 episodes, mean reward -60.112, speed 85.24 f/s
Test done in 6.35 sec, reward 201.785, steps 944
1520090: done 5000 episodes, mean reward -60.758, speed 16.51 f/s
1520212: done 5001 episodes, mean reward -60.996, speed 85.57 f/s
1520432: done 5002 episodes, mean reward -61.192, speed 82.86 f/s
1520582: done 5004 episodes, mean reward -62.021, speed 85.78 f/s
1520667: done 5005 episodes, mean reward -62.910, speed 82.45 f/s
1520909: done 5006 episodes, mean reward -62.716, speed 84.61 f/s
Test done in 7.15 sec, reward 223.741, steps 973
1521040: done 5007 episodes, mean reward -63.309, speed 14.98 f/s
1521216: done 5008 episodes, mean reward -63.891, speed 85.73 f/s
1521347: done 5009 episodes, mean reward -64.495, speed 82.87 f/s
1521610: done 5010 episodes, mean reward -64.545, speed 82.70 f/s
1521739: done 5011 episodes, mean reward -64.916, speed 81.36 f/s
1521895: done 5012 episodes, mean reward -65.206, speed 83.32 f/s
Test done in 6.76 sec, reward 259.824, steps 1003
1522129: done 5013 episodes, mean reward -65.964, speed 24.20 f/s
1522226: done 5014 episodes, mean reward -66.758, speed 83.31 f/s
1522401: done 5016 episodes, mean reward -67.149, speed 85.79 f/s
1522997: done 5018 episodes, mean reward -66.735, speed 83.31 f/s
Test done in 7.25 sec, reward 251.973, steps 1055
1523112: done 5019 episodes, mean reward -66.714, speed 13.38 f/s
1523365: done 5020 episodes, mean reward -66.399, speed 83.97 f/s
1523558: done 5021 episodes, mean reward -66.356, speed 83.35 f/s
1523746: done 5022 episodes, mean reward -67.585, speed 85.28 f/s
1523862: done 5023 episodes, mean reward -67.909, speed 85.89 f/s
Test done in 5.77 sec, reward 201.069, steps 872
1524015: done 5024 episodes, mean reward -67.757, speed 20.16 f/s
1524157: done 5025 episodes, mean reward -68.545, speed 79.91 f/s
1524340: done 5026 episodes, mean reward -68.491, speed 72.56 f/s
1524478: done 5027 episodes, mean reward -68.866, speed 80.82 f/s
1524602: done 5028 episodes, mean reward -68.736, speed 82.03 f/s
1524701: done 5029 episodes, mean reward -69.164, speed 79.98 f/s
1524807: done 5030 episodes, mean reward -69.271, speed 82.42 f/s
1524948: done 5031 episodes, mean reward -69.595, speed 84.93 f/s
Test done in 7.33 sec, reward 282.316, steps 1069
Best reward updated: 281.342 -> 282.316
1525135: done 5032 episodes, mean reward -69.450, speed 19.56 f/s
1525420: done 5033 episodes, mean reward -69.821, speed 81.51 f/s
1525896: done 5035 episodes, mean reward -69.033, speed 83.01 f/s
Test done in 6.97 sec, reward 270.907, steps 1042
1526203: done 5037 episodes, mean reward -70.267, speed 29.07 f/s
1526512: done 5038 episodes, mean reward -69.504, speed 83.48 f/s
1526790: done 5039 episodes, mean reward -68.957, speed 83.27 f/s
1526877: done 5040 episodes, mean reward -68.929, speed 85.76 f/s
Test done in 6.19 sec, reward 198.221, steps 915
1527047: done 5042 episodes, mean reward -69.808, speed 20.70 f/s
1527162: done 5043 episodes, mean reward -69.940, speed 85.27 f/s
1527292: done 5044 episodes, mean reward -70.023, speed 82.67 f/s
1527419: done 5045 episodes, mean reward -70.010, speed 86.51 f/s
1527586: done 5046 episodes, mean reward -70.879, speed 81.78 f/s
1527671: done 5047 episodes, mean reward -72.379, speed 72.30 f/s
1527918: done 5049 episodes, mean reward -74.975, speed 80.44 f/s
Test done in 6.78 sec, reward 221.868, steps 994
1528295: done 5051 episodes, mean reward -74.769, speed 33.19 f/s
1528415: done 5052 episodes, mean reward -74.856, speed 84.57 f/s
1528592: done 5053 episodes, mean reward -75.200, speed 82.38 f/s
1528874: done 5054 episodes, mean reward -74.888, speed 83.05 f/s
Test done in 7.12 sec, reward 263.705, steps 1030
1529060: done 5056 episodes, mean reward -75.288, speed 19.98 f/s
1529149: done 5057 episodes, mean reward -75.421, speed 81.25 f/s
1529387: done 5058 episodes, mean reward -75.619, speed 85.28 f/s
1529596: done 5059 episodes, mean reward -76.845, speed 81.89 f/s
1529762: done 5060 episodes, mean reward -76.802, speed 87.01 f/s
Test done in 6.29 sec, reward 230.225, steps 948
1530003: done 5061 episodes, mean reward -76.628, speed 26.37 f/s
1530256: done 5063 episodes, mean reward -76.835, speed 86.61 f/s
1530580: done 5064 episodes, mean reward -76.492, speed 82.95 f/s
1530885: done 5065 episodes, mean reward -76.416, speed 79.50 f/s
1530983: done 5066 episodes, mean reward -77.457, speed 87.99 f/s
Test done in 7.11 sec, reward 240.165, steps 1073
1531125: done 5067 episodes, mean reward -77.382, speed 16.15 f/s
1531291: done 5068 episodes, mean reward -77.524, speed 82.16 f/s
1531442: done 5069 episodes, mean reward -77.460, speed 83.37 f/s
1531621: done 5071 episodes, mean reward -78.578, speed 86.63 f/s
1531820: done 5072 episodes, mean reward -78.336, speed 82.91 f/s
1531932: done 5073 episodes, mean reward -78.128, speed 82.05 f/s
Test done in 6.92 sec, reward 268.388, steps 1009
1532030: done 5074 episodes, mean reward -78.990, speed 12.14 f/s
1532127: done 5075 episodes, mean reward -78.853, speed 83.82 f/s
1532268: done 5076 episodes, mean reward -78.690, speed 81.44 f/s
1532874: done 5077 episodes, mean reward -78.829, speed 84.10 f/s
Test done in 6.35 sec, reward 225.623, steps 967
1533006: done 5078 episodes, mean reward -79.964, speed 16.70 f/s
1533176: done 5079 episodes, mean reward -80.638, speed 84.61 f/s
1533286: done 5080 episodes, mean reward -80.639, speed 82.31 f/s
1533418: done 5081 episodes, mean reward -82.093, speed 84.92 f/s
1533644: done 5082 episodes, mean reward -82.768, speed 85.15 f/s
1533733: done 5083 episodes, mean reward -83.841, speed 81.75 f/s
1533839: done 5084 episodes, mean reward -83.572, speed 85.91 f/s
1533942: done 5085 episodes, mean reward -83.481, speed 82.02 f/s
Test done in 6.68 sec, reward 205.253, steps 970
1534133: done 5086 episodes, mean reward -83.516, speed 21.28 f/s
1534246: done 5087 episodes, mean reward -83.639, speed 84.17 f/s
1534391: done 5088 episodes, mean reward -83.684, speed 86.69 f/s
1534539: done 5089 episodes, mean reward -83.703, speed 83.55 f/s
1534749: done 5090 episodes, mean reward -83.987, speed 80.46 f/s
Test done in 6.43 sec, reward 205.668, steps 962
1535068: done 5092 episodes, mean reward -84.067, speed 31.28 f/s
1535209: done 5093 episodes, mean reward -84.451, speed 81.01 f/s
1535321: done 5094 episodes, mean reward -84.678, speed 88.81 f/s
1535479: done 5095 episodes, mean reward -84.651, speed 83.80 f/s
1535608: done 5096 episodes, mean reward -84.840, speed 81.14 f/s
1535752: done 5097 episodes, mean reward -84.882, speed 87.07 f/s
1535935: done 5098 episodes, mean reward -85.049, speed 83.36 f/s
Test done in 7.53 sec, reward 223.758, steps 1048
1536090: done 5099 episodes, mean reward -85.049, speed 16.46 f/s
1536258: done 5100 episodes, mean reward -85.005, speed 81.07 f/s
1536532: done 5102 episodes, mean reward -85.158, speed 83.07 f/s
1536772: done 5103 episodes, mean reward -84.782, speed 84.67 f/s
1536933: done 5104 episodes, mean reward -84.414, speed 81.62 f/s
Test done in 7.12 sec, reward 235.665, steps 1051
1537166: done 5105 episodes, mean reward -83.952, speed 23.54 f/s
1537346: done 5106 episodes, mean reward -84.235, speed 82.61 f/s
1537585: done 5107 episodes, mean reward -83.980, speed 84.06 f/s
1537887: done 5109 episodes, mean reward -83.625, speed 82.86 f/s
Test done in 5.83 sec, reward 186.537, steps 888
1538003: done 5110 episodes, mean reward -84.014, speed 15.86 f/s
1538178: done 5111 episodes, mean reward -83.869, speed 83.25 f/s
1538293: done 5112 episodes, mean reward -84.102, speed 85.21 f/s
1538563: done 5113 episodes, mean reward -84.210, speed 83.32 f/s
1538841: done 5114 episodes, mean reward -83.676, speed 86.19 f/s
1538951: done 5115 episodes, mean reward -83.547, speed 84.65 f/s
Test done in 7.52 sec, reward 283.541, steps 1089
Best reward updated: 282.316 -> 283.541
1539058: done 5116 episodes, mean reward -83.600, speed 12.12 f/s
1539249: done 5117 episodes, mean reward -83.350, speed 81.69 f/s
1539378: done 5118 episodes, mean reward -84.659, speed 83.50 f/s
1539570: done 5119 episodes, mean reward -84.383, speed 82.14 f/s
1539722: done 5120 episodes, mean reward -84.863, speed 83.15 f/s
1539946: done 5121 episodes, mean reward -84.880, speed 83.24 f/s
Test done in 7.39 sec, reward 279.959, steps 1100
1540065: done 5122 episodes, mean reward -84.776, speed 13.44 f/s
1540243: done 5123 episodes, mean reward -84.643, speed 81.14 f/s
1540446: done 5124 episodes, mean reward -84.609, speed 80.65 f/s
1540725: done 5125 episodes, mean reward -84.312, speed 82.66 f/s
1540894: done 5126 episodes, mean reward -84.283, speed 84.72 f/s
Test done in 6.59 sec, reward 247.608, steps 1002
1541205: done 5127 episodes, mean reward -83.774, speed 30.31 f/s
1541547: done 5128 episodes, mean reward -83.154, speed 83.32 f/s
1541694: done 5129 episodes, mean reward -83.037, speed 84.65 f/s
1541886: done 5130 episodes, mean reward -82.874, speed 85.98 f/s
Test done in 5.11 sec, reward 142.759, steps 762
1542078: done 5131 episodes, mean reward -82.473, speed 26.05 f/s
1542256: done 5132 episodes, mean reward -82.580, speed 79.60 f/s
1542431: done 5134 episodes, mean reward -82.839, speed 79.28 f/s
1542627: done 5135 episodes, mean reward -83.690, speed 78.80 f/s
1542838: done 5136 episodes, mean reward -83.241, speed 80.47 f/s
Test done in 6.81 sec, reward 258.093, steps 1021
1543149: done 5137 episodes, mean reward -83.164, speed 29.65 f/s
1543276: done 5138 episodes, mean reward -83.851, speed 83.04 f/s
1543471: done 5140 episodes, mean reward -84.309, speed 82.26 f/s
1543720: done 5141 episodes, mean reward -84.015, speed 83.30 f/s
1543853: done 5142 episodes, mean reward -83.998, speed 80.70 f/s
Test done in 5.48 sec, reward 189.725, steps 813
1544288: done 5143 episodes, mean reward -82.991, speed 40.41 f/s
1544400: done 5144 episodes, mean reward -83.108, speed 82.23 f/s
1544604: done 5146 episodes, mean reward -83.238, speed 86.85 f/s
Test done in 6.22 sec, reward 230.640, steps 935
1545036: done 5148 episodes, mean reward -82.347, speed 37.96 f/s
1545189: done 5149 episodes, mean reward -82.580, speed 80.91 f/s
1545355: done 5150 episodes, mean reward -82.353, speed 85.82 f/s
1545476: done 5151 episodes, mean reward -82.566, speed 81.34 f/s
1545645: done 5153 episodes, mean reward -82.859, speed 82.92 f/s
1545851: done 5155 episodes, mean reward -83.460, speed 82.85 f/s
1545955: done 5156 episodes, mean reward -83.526, speed 79.97 f/s
Test done in 7.03 sec, reward 263.364, steps 989
1546079: done 5157 episodes, mean reward -83.514, speed 14.48 f/s
1546189: done 5158 episodes, mean reward -83.814, speed 81.57 f/s
1546320: done 5159 episodes, mean reward -84.007, speed 83.89 f/s
1546542: done 5160 episodes, mean reward -83.930, speed 85.55 f/s
Test done in 6.77 sec, reward 236.052, steps 993
1547092: done 5161 episodes, mean reward -83.677, speed 40.95 f/s
1547443: done 5162 episodes, mean reward -83.088, speed 84.16 f/s
1547691: done 5163 episodes, mean reward -83.188, speed 80.14 f/s
1547794: done 5164 episodes, mean reward -84.031, speed 82.29 f/s
1547949: done 5165 episodes, mean reward -84.542, speed 85.00 f/s
Test done in 6.38 sec, reward 185.176, steps 902
1548079: done 5166 episodes, mean reward -84.587, speed 16.38 f/s
1548462: done 5167 episodes, mean reward -83.953, speed 84.02 f/s
1548585: done 5168 episodes, mean reward -84.326, speed 82.93 f/s
1548786: done 5169 episodes, mean reward -84.028, speed 83.48 f/s
1548909: done 5170 episodes, mean reward -84.012, speed 84.75 f/s
Test done in 5.47 sec, reward 171.155, steps 800
1549077: done 5172 episodes, mean reward -84.417, speed 22.45 f/s
1549223: done 5173 episodes, mean reward -84.237, speed 82.82 f/s
1549737: done 5174 episodes, mean reward -83.194, speed 83.84 f/s
1549887: done 5175 episodes, mean reward -83.201, speed 81.99 f/s
1549999: done 5176 episodes, mean reward -83.445, speed 84.81 f/s
Test done in 6.76 sec, reward 241.982, steps 1008
1550142: done 5177 episodes, mean reward -84.810, speed 16.88 f/s
1550247: done 5178 episodes, mean reward -84.650, speed 85.85 f/s
1550426: done 5179 episodes, mean reward -84.441, speed 80.66 f/s
1550591: done 5180 episodes, mean reward -84.420, speed 81.51 f/s
1550736: done 5181 episodes, mean reward -84.330, speed 81.33 f/s
1550883: done 5182 episodes, mean reward -84.522, speed 79.62 f/s
Test done in 5.16 sec, reward 132.084, steps 726
1551028: done 5183 episodes, mean reward -84.153, speed 20.77 f/s
1551246: done 5184 episodes, mean reward -83.942, speed 78.27 f/s
1551384: done 5185 episodes, mean reward -84.121, speed 80.03 f/s
1551573: done 5186 episodes, mean reward -84.156, speed 80.23 f/s
1551831: done 5187 episodes, mean reward -83.595, speed 81.30 f/s
1551973: done 5188 episodes, mean reward -83.417, speed 78.64 f/s
Test done in 5.63 sec, reward 180.461, steps 799
1552141: done 5189 episodes, mean reward -83.104, speed 21.96 f/s
1552596: done 5190 episodes, mean reward -82.453, speed 81.46 f/s
1552747: done 5191 episodes, mean reward -82.315, speed 75.74 f/s
1552932: done 5192 episodes, mean reward -82.134, speed 78.48 f/s
Test done in 5.84 sec, reward 160.263, steps 811
1553076: done 5193 episodes, mean reward -82.030, speed 18.90 f/s
1553184: done 5194 episodes, mean reward -82.112, speed 81.66 f/s
1553363: done 5195 episodes, mean reward -82.076, speed 84.83 f/s
1553485: done 5196 episodes, mean reward -82.333, speed 83.74 f/s
1553638: done 5197 episodes, mean reward -82.161, speed 83.10 f/s
1553730: done 5198 episodes, mean reward -82.521, speed 77.85 f/s
1553863: done 5199 episodes, mean reward -82.619, speed 81.52 f/s
1554000: done 5200 episodes, mean reward -82.671, speed 83.66 f/s
Test done in 5.93 sec, reward 204.701, steps 864
1554105: done 5201 episodes, mean reward -82.701, speed 14.56 f/s
1554221: done 5202 episodes, mean reward -82.773, speed 82.20 f/s
1554470: done 5203 episodes, mean reward -82.689, speed 82.06 f/s
1554717: done 5204 episodes, mean reward -82.509, speed 84.33 f/s
1554993: done 5205 episodes, mean reward -82.403, speed 86.06 f/s
Test done in 6.25 sec, reward 249.144, steps 936
1555136: done 5206 episodes, mean reward -82.227, speed 17.91 f/s
1555221: done 5207 episodes, mean reward -82.552, speed 83.41 f/s
1555347: done 5208 episodes, mean reward -82.355, speed 84.20 f/s
1555519: done 5210 episodes, mean reward -82.832, speed 81.55 f/s
1555651: done 5211 episodes, mean reward -82.955, speed 80.54 f/s
1555798: done 5212 episodes, mean reward -82.837, speed 83.29 f/s
1555967: done 5213 episodes, mean reward -82.819, speed 83.77 f/s
Test done in 6.67 sec, reward 234.607, steps 933
1556085: done 5214 episodes, mean reward -83.262, speed 14.54 f/s
1556312: done 5216 episodes, mean reward -83.162, speed 82.57 f/s
1556468: done 5217 episodes, mean reward -83.175, speed 83.73 f/s
1556648: done 5219 episodes, mean reward -83.374, speed 83.23 f/s
1556849: done 5220 episodes, mean reward -83.094, speed 82.81 f/s
Test done in 6.46 sec, reward 259.497, steps 961
1557447: done 5222 episodes, mean reward -82.045, speed 43.60 f/s
1557572: done 5223 episodes, mean reward -82.053, speed 83.73 f/s
1557813: done 5224 episodes, mean reward -81.858, speed 81.17 f/s
Test done in 6.40 sec, reward 259.932, steps 962
1558072: done 5225 episodes, mean reward -81.827, speed 27.51 f/s
1558328: done 5226 episodes, mean reward -81.583, speed 81.79 f/s
1558593: done 5228 episodes, mean reward -82.656, speed 86.23 f/s
1558715: done 5229 episodes, mean reward -82.732, speed 83.93 f/s
Test done in 5.77 sec, reward 229.800, steps 880
1559021: done 5230 episodes, mean reward -82.221, speed 32.55 f/s
1559181: done 5232 episodes, mean reward -82.838, speed 82.72 f/s
1559334: done 5233 episodes, mean reward -82.774, speed 82.57 f/s
1559486: done 5234 episodes, mean reward -82.874, speed 75.42 f/s
1559660: done 5236 episodes, mean reward -83.334, speed 79.79 f/s
1559869: done 5238 episodes, mean reward -83.808, speed 82.74 f/s
Test done in 6.87 sec, reward 255.760, steps 1004
1560012: done 5239 episodes, mean reward -83.527, speed 16.64 f/s
1560305: done 5240 episodes, mean reward -83.169, speed 82.87 f/s
1560769: done 5241 episodes, mean reward -82.069, speed 81.92 f/s
1560930: done 5242 episodes, mean reward -81.998, speed 83.38 f/s
Test done in 6.52 sec, reward 258.500, steps 947
1561163: done 5243 episodes, mean reward -82.647, speed 25.17 f/s
1561577: done 5245 episodes, mean reward -81.940, speed 85.51 f/s
1562000: done 5246 episodes, mean reward -80.831, speed 83.84 f/s
Test done in 6.53 sec, reward 263.298, steps 943
1562100: done 5247 episodes, mean reward -80.659, speed 12.99 f/s
1562210: done 5248 episodes, mean reward -81.281, speed 83.71 f/s
1562402: done 5250 episodes, mean reward -81.442, speed 82.53 f/s
1562564: done 5252 episodes, mean reward -81.469, speed 82.59 f/s
1562678: done 5253 episodes, mean reward -81.414, speed 83.61 f/s
Test done in 5.45 sec, reward 175.680, steps 794
1563042: done 5254 episodes, mean reward -80.355, speed 36.30 f/s
1563628: done 5255 episodes, mean reward -78.821, speed 83.54 f/s
1563727: done 5256 episodes, mean reward -78.781, speed 84.14 f/s
1563904: done 5257 episodes, mean reward -78.509, speed 81.58 f/s
Test done in 5.75 sec, reward 215.899, steps 873
1564001: done 5258 episodes, mean reward -78.522, speed 13.97 f/s
1564170: done 5259 episodes, mean reward -78.448, speed 83.32 f/s
1564269: done 5260 episodes, mean reward -78.813, speed 86.14 f/s
1564679: done 5261 episodes, mean reward -78.483, speed 83.39 f/s
1564836: done 5262 episodes, mean reward -78.857, speed 82.90 f/s
1564943: done 5263 episodes, mean reward -79.176, speed 86.63 f/s
Test done in 5.95 sec, reward 203.498, steps 866
1565093: done 5264 episodes, mean reward -78.948, speed 19.23 f/s
1565245: done 5266 episodes, mean reward -79.136, speed 85.50 f/s
1565332: done 5267 episodes, mean reward -80.004, speed 82.84 f/s
1565438: done 5268 episodes, mean reward -79.906, speed 87.02 f/s
1565548: done 5269 episodes, mean reward -80.259, speed 84.23 f/s
1565888: done 5270 episodes, mean reward -79.606, speed 84.11 f/s
1565983: done 5271 episodes, mean reward -79.605, speed 85.19 f/s
Test done in 5.99 sec, reward 231.687, steps 899
1566206: done 5272 episodes, mean reward -79.388, speed 25.20 f/s
1566340: done 5273 episodes, mean reward -79.368, speed 81.59 f/s
1566466: done 5275 episodes, mean reward -80.635, speed 83.85 f/s
1566617: done 5276 episodes, mean reward -80.427, speed 82.39 f/s
1566831: done 5278 episodes, mean reward -80.249, speed 84.30 f/s
1566978: done 5279 episodes, mean reward -80.441, speed 85.51 f/s
Test done in 6.20 sec, reward 243.944, steps 918
1567158: done 5280 episodes, mean reward -80.427, speed 21.41 f/s
1567296: done 5281 episodes, mean reward -80.407, speed 82.04 f/s
1567396: done 5282 episodes, mean reward -80.488, speed 83.47 f/s
1567589: done 5284 episodes, mean reward -80.846, speed 84.07 f/s
1567734: done 5285 episodes, mean reward -80.534, speed 87.39 f/s
1567899: done 5287 episodes, mean reward -81.212, speed 83.06 f/s
Test done in 6.40 sec, reward 260.715, steps 964
1568208: done 5288 episodes, mean reward -80.557, speed 30.50 f/s
1568299: done 5289 episodes, mean reward -80.874, speed 82.70 f/s
1568591: done 5290 episodes, mean reward -80.937, speed 85.08 f/s
1568744: done 5291 episodes, mean reward -80.934, speed 81.92 f/s
1568845: done 5292 episodes, mean reward -81.095, speed 81.44 f/s
1568975: done 5293 episodes, mean reward -80.977, speed 85.74 f/s
Test done in 6.62 sec, reward 277.670, steps 964
1569123: done 5294 episodes, mean reward -80.777, speed 17.67 f/s
1569355: done 5295 episodes, mean reward -80.441, speed 85.02 f/s
1569486: done 5296 episodes, mean reward -80.317, speed 81.25 f/s
1569619: done 5297 episodes, mean reward -80.460, speed 82.39 f/s
1569695: done 5298 episodes, mean reward -80.471, speed 72.56 f/s
1569998: done 5299 episodes, mean reward -79.938, speed 79.82 f/s
Test done in 5.80 sec, reward 215.399, steps 815
1570100: done 5300 episodes, mean reward -80.086, speed 14.39 f/s
1570211: done 5301 episodes, mean reward -79.974, speed 86.68 f/s
1570316: done 5302 episodes, mean reward -79.909, speed 83.36 f/s
1570516: done 5303 episodes, mean reward -79.844, speed 81.20 f/s
1570667: done 5305 episodes, mean reward -80.906, speed 80.70 f/s
1570762: done 5306 episodes, mean reward -81.024, speed 83.12 f/s
Test done in 6.70 sec, reward 287.526, steps 997
Best reward updated: 283.541 -> 287.526
1571259: done 5307 episodes, mean reward -79.690, speed 39.03 f/s
1571350: done 5308 episodes, mean reward -79.845, speed 85.67 f/s
1571771: done 5309 episodes, mean reward -78.830, speed 82.89 f/s
1571914: done 5310 episodes, mean reward -78.599, speed 86.12 f/s
Test done in 6.97 sec, reward 285.896, steps 1036
1572048: done 5311 episodes, mean reward -78.390, speed 15.61 f/s
1572195: done 5312 episodes, mean reward -78.372, speed 82.52 f/s
1572405: done 5313 episodes, mean reward -78.263, speed 86.14 f/s
1572577: done 5314 episodes, mean reward -78.032, speed 81.63 f/s
1572696: done 5315 episodes, mean reward -77.904, speed 84.57 f/s
1572801: done 5316 episodes, mean reward -78.041, speed 83.92 f/s
Test done in 6.95 sec, reward 287.219, steps 990
1573241: done 5317 episodes, mean reward -77.121, speed 36.13 f/s
1573620: done 5318 episodes, mean reward -76.249, speed 84.94 f/s
1573789: done 5319 episodes, mean reward -76.043, speed 83.67 f/s
1573919: done 5320 episodes, mean reward -76.321, speed 80.92 f/s
Test done in 6.09 sec, reward 228.953, steps 888
1574339: done 5321 episodes, mean reward -75.296, speed 37.53 f/s
1574472: done 5322 episodes, mean reward -76.499, speed 80.28 f/s
1574595: done 5323 episodes, mean reward -76.577, speed 81.45 f/s
1574942: done 5324 episodes, mean reward -76.262, speed 82.87 f/s
Test done in 6.50 sec, reward 263.593, steps 951
1575205: done 5325 episodes, mean reward -76.350, speed 27.24 f/s
1575312: done 5326 episodes, mean reward -76.756, speed 84.39 f/s
1575434: done 5327 episodes, mean reward -76.828, speed 81.27 f/s
1575552: done 5328 episodes, mean reward -76.909, speed 84.26 f/s
1575885: done 5329 episodes, mean reward -76.066, speed 82.61 f/s
Test done in 6.45 sec, reward 263.844, steps 934
1576001: done 5330 episodes, mean reward -76.701, speed 14.90 f/s
1576185: done 5331 episodes, mean reward -76.238, speed 77.86 f/s
1576289: done 5332 episodes, mean reward -76.137, speed 79.10 f/s
1576468: done 5333 episodes, mean reward -76.038, speed 81.04 f/s
1576571: done 5334 episodes, mean reward -76.030, speed 80.22 f/s
1576787: done 5336 episodes, mean reward -75.799, speed 82.59 f/s
Test done in 6.46 sec, reward 264.379, steps 943
1577075: done 5337 episodes, mean reward -75.083, speed 28.79 f/s
1577245: done 5339 episodes, mean reward -75.569, speed 84.50 f/s
1577896: done 5340 episodes, mean reward -74.386, speed 82.89 f/s
Test done in 5.28 sec, reward 171.449, steps 791
1578131: done 5342 episodes, mean reward -75.574, speed 29.14 f/s
1578238: done 5343 episodes, mean reward -75.849, speed 84.96 f/s
1578401: done 5345 episodes, mean reward -76.626, speed 83.95 f/s
1578682: done 5346 episodes, mean reward -77.134, speed 82.30 f/s
1578768: done 5347 episodes, mean reward -77.327, speed 80.35 f/s
Test done in 6.83 sec, reward 285.498, steps 1009
1579004: done 5349 episodes, mean reward -77.284, speed 24.46 f/s
1579118: done 5350 episodes, mean reward -77.177, speed 85.92 f/s
1579217: done 5351 episodes, mean reward -77.185, speed 85.06 f/s
1579309: done 5352 episodes, mean reward -77.058, speed 82.36 f/s
1579405: done 5353 episodes, mean reward -77.057, speed 84.74 f/s
1579582: done 5355 episodes, mean reward -79.628, speed 88.02 f/s
1579762: done 5357 episodes, mean reward -80.086, speed 82.02 f/s
1579928: done 5358 episodes, mean reward -79.889, speed 85.41 f/s
Test done in 5.92 sec, reward 220.291, steps 872
1580007: done 5359 episodes, mean reward -80.141, speed 11.52 f/s
1580104: done 5360 episodes, mean reward -80.074, speed 83.41 f/s
1580219: done 5361 episodes, mean reward -81.075, speed 79.00 f/s
1580536: done 5362 episodes, mean reward -80.648, speed 80.95 f/s
1580862: done 5363 episodes, mean reward -79.875, speed 84.43 f/s
Test done in 5.77 sec, reward 193.985, steps 847
1581002: done 5364 episodes, mean reward -79.913, speed 18.60 f/s
1581183: done 5365 episodes, mean reward -79.629, speed 81.40 f/s
1581360: done 5367 episodes, mean reward -79.518, speed 78.15 f/s
1581549: done 5369 episodes, mean reward -79.479, speed 76.76 f/s
1581755: done 5370 episodes, mean reward -79.809, speed 81.23 f/s
1581873: done 5371 episodes, mean reward -79.742, speed 82.31 f/s
1581983: done 5372 episodes, mean reward -79.950, speed 80.10 f/s
Test done in 6.62 sec, reward 254.892, steps 940
1582521: done 5373 episodes, mean reward -78.633, speed 41.30 f/s
1582733: done 5375 episodes, mean reward -78.454, speed 84.41 f/s
1582943: done 5376 episodes, mean reward -78.617, speed 83.54 f/s
Test done in 6.29 sec, reward 253.343, steps 919
1583173: done 5377 episodes, mean reward -78.132, speed 25.41 f/s
1583547: done 5378 episodes, mean reward -77.527, speed 84.31 f/s
1583818: done 5379 episodes, mean reward -77.069, speed 82.25 f/s
Test done in 6.96 sec, reward 285.537, steps 1021
1584033: done 5380 episodes, mean reward -76.758, speed 22.62 f/s
1584369: done 5381 episodes, mean reward -76.076, speed 84.26 f/s
1584461: done 5382 episodes, mean reward -76.072, speed 82.55 f/s
1584710: done 5384 episodes, mean reward -76.068, speed 82.75 f/s
1584849: done 5385 episodes, mean reward -76.073, speed 79.12 f/s
Test done in 6.92 sec, reward 285.478, steps 1012
1585121: done 5386 episodes, mean reward -75.358, speed 26.60 f/s
1585267: done 5387 episodes, mean reward -75.242, speed 82.36 f/s
1585529: done 5388 episodes, mean reward -75.357, speed 82.19 f/s
1585792: done 5389 episodes, mean reward -74.833, speed 86.48 f/s
1585975: done 5390 episodes, mean reward -75.238, speed 82.90 f/s
Test done in 7.03 sec, reward 252.858, steps 999
1586213: done 5391 episodes, mean reward -74.941, speed 23.98 f/s
1586532: done 5392 episodes, mean reward -74.386, speed 84.56 f/s
1586909: done 5394 episodes, mean reward -74.072, speed 80.03 f/s
Test done in 6.89 sec, reward 282.520, steps 1041
1587012: done 5395 episodes, mean reward -74.547, speed 12.71 f/s
1587181: done 5396 episodes, mean reward -74.332, speed 82.72 f/s
1587299: done 5397 episodes, mean reward -74.380, speed 86.20 f/s
1587553: done 5399 episodes, mean reward -74.728, speed 82.07 f/s
1587675: done 5400 episodes, mean reward -74.598, speed 82.23 f/s
1587783: done 5401 episodes, mean reward -74.509, speed 81.78 f/s
Test done in 6.83 sec, reward 246.823, steps 967
1588114: done 5402 episodes, mean reward -73.953, speed 30.63 f/s
1588247: done 5403 episodes, mean reward -74.380, speed 83.91 f/s
1588356: done 5404 episodes, mean reward -74.299, speed 82.67 f/s
1588678: done 5406 episodes, mean reward -73.917, speed 78.64 f/s
1588808: done 5407 episodes, mean reward -75.098, speed 79.68 f/s
1588929: done 5408 episodes, mean reward -74.979, speed 81.19 f/s
Test done in 6.47 sec, reward 238.474, steps 956
1589059: done 5409 episodes, mean reward -75.869, speed 16.06 f/s
1589290: done 5410 episodes, mean reward -75.651, speed 83.79 f/s
1589397: done 5411 episodes, mean reward -75.816, speed 83.40 f/s
1589540: done 5413 episodes, mean reward -76.185, speed 80.56 f/s
Test done in 5.89 sec, reward 208.876, steps 878
1590409: done 5414 episodes, mean reward -74.092, speed 52.85 f/s
1590556: done 5416 episodes, mean reward -74.307, speed 84.89 f/s
1590675: done 5417 episodes, mean reward -75.226, speed 82.81 f/s
1590794: done 5418 episodes, mean reward -75.906, speed 86.25 f/s
1590980: done 5420 episodes, mean reward -76.138, speed 84.99 f/s
Test done in 6.52 sec, reward 243.424, steps 944
1591143: done 5421 episodes, mean reward -76.908, speed 19.25 f/s
1591315: done 5422 episodes, mean reward -76.855, speed 74.32 f/s
1591400: done 5423 episodes, mean reward -76.882, speed 83.88 f/s
1591823: done 5424 episodes, mean reward -76.665, speed 78.72 f/s
1591960: done 5425 episodes, mean reward -77.016, speed 81.67 f/s
Test done in 6.06 sec, reward 185.614, steps 892
1592116: done 5426 episodes, mean reward -76.904, speed 19.68 f/s
1592251: done 5427 episodes, mean reward -76.756, speed 82.84 f/s
1592345: done 5428 episodes, mean reward -76.895, speed 83.82 f/s
1592490: done 5429 episodes, mean reward -77.747, speed 80.24 f/s
1592692: done 5430 episodes, mean reward -77.435, speed 82.47 f/s
1592958: done 5431 episodes, mean reward -77.202, speed 82.69 f/s
Test done in 7.01 sec, reward 261.823, steps 1030
1593008: done 5432 episodes, mean reward -77.272, speed 6.57 f/s
1593161: done 5433 episodes, mean reward -77.212, speed 83.85 f/s
1593345: done 5434 episodes, mean reward -77.058, speed 85.25 f/s
1593516: done 5435 episodes, mean reward -76.742, speed 79.11 f/s
Test done in 6.15 sec, reward 202.870, steps 909
1594351: done 5436 episodes, mean reward -74.708, speed 51.24 f/s
1594501: done 5437 episodes, mean reward -75.079, speed 81.92 f/s
1594595: done 5438 episodes, mean reward -74.976, speed 79.26 f/s
1594896: done 5439 episodes, mean reward -74.143, speed 76.89 f/s
Test done in 6.85 sec, reward 255.093, steps 976
1595168: done 5440 episodes, mean reward -75.267, speed 26.97 f/s
1595710: done 5442 episodes, mean reward -74.325, speed 82.13 f/s
1595816: done 5443 episodes, mean reward -74.299, speed 78.30 f/s
1595958: done 5444 episodes, mean reward -73.919, speed 83.10 f/s
Test done in 6.14 sec, reward 206.663, steps 887
1596125: done 5445 episodes, mean reward -73.607, speed 20.16 f/s
1596377: done 5446 episodes, mean reward -73.741, speed 75.98 f/s
1596490: done 5447 episodes, mean reward -73.505, speed 83.23 f/s
1596596: done 5448 episodes, mean reward -73.475, speed 83.93 f/s
1596929: done 5449 episodes, mean reward -72.911, speed 82.32 f/s
Test done in 6.52 sec, reward 242.651, steps 966
1597023: done 5450 episodes, mean reward -72.970, speed 12.19 f/s
1597315: done 5451 episodes, mean reward -72.262, speed 82.70 f/s
1597506: done 5452 episodes, mean reward -71.971, speed 83.18 f/s
1597661: done 5453 episodes, mean reward -71.638, speed 82.52 f/s
1597828: done 5454 episodes, mean reward -71.311, speed 86.14 f/s
1597975: done 5455 episodes, mean reward -70.923, speed 82.80 f/s
Test done in 7.54 sec, reward 282.625, steps 1073
1598069: done 5456 episodes, mean reward -70.787, speed 10.80 f/s
1598181: done 5457 episodes, mean reward -70.760, speed 80.41 f/s
1598280: done 5458 episodes, mean reward -70.897, speed 82.82 f/s
1598448: done 5459 episodes, mean reward -70.466, speed 82.28 f/s
1598546: done 5460 episodes, mean reward -70.382, speed 83.67 f/s
1598686: done 5461 episodes, mean reward -70.179, speed 83.16 f/s
1598876: done 5462 episodes, mean reward -70.472, speed 81.65 f/s
Test done in 7.14 sec, reward 259.842, steps 1022
1599037: done 5464 episodes, mean reward -71.321, speed 17.53 f/s
1599179: done 5465 episodes, mean reward -71.365, speed 81.79 f/s
1599265: done 5466 episodes, mean reward -71.320, speed 83.95 f/s
1599728: done 5467 episodes, mean reward -70.262, speed 81.42 f/s
1599856: done 5468 episodes, mean reward -69.948, speed 80.96 f/s
Test done in 6.39 sec, reward 234.824, steps 949
1600514: done 5470 episodes, mean reward -68.789, speed 45.66 f/s
1600750: done 5471 episodes, mean reward -68.351, speed 83.74 f/s
1600914: done 5472 episodes, mean reward -68.148, speed 82.47 f/s
Test done in 6.64 sec, reward 228.188, steps 978
1601105: done 5474 episodes, mean reward -69.441, speed 21.24 f/s
1601194: done 5475 episodes, mean reward -69.623, speed 83.18 f/s
1601445: done 5476 episodes, mean reward -69.301, speed 81.78 f/s
1601620: done 5478 episodes, mean reward -70.624, speed 84.27 f/s
1601739: done 5479 episodes, mean reward -71.090, speed 82.06 f/s
1601927: done 5480 episodes, mean reward -71.238, speed 82.03 f/s
Test done in 6.77 sec, reward 234.511, steps 972
1602034: done 5481 episodes, mean reward -71.968, speed 13.21 f/s
1602269: done 5482 episodes, mean reward -71.456, speed 81.24 f/s
1602497: done 5484 episodes, mean reward -71.324, speed 81.47 f/s
1602607: done 5485 episodes, mean reward -71.522, speed 80.18 f/s
1602779: done 5486 episodes, mean reward -71.832, speed 79.26 f/s
1602941: done 5488 episodes, mean reward -72.496, speed 84.85 f/s
Test done in 6.04 sec, reward 226.556, steps 917
1603348: done 5489 episodes, mean reward -71.951, speed 37.61 f/s
1603479: done 5490 episodes, mean reward -72.131, speed 88.07 f/s
1603663: done 5491 episodes, mean reward -72.125, speed 83.46 f/s
1603766: done 5492 episodes, mean reward -72.725, speed 80.90 f/s
1603877: done 5493 episodes, mean reward -72.540, speed 86.81 f/s
Test done in 6.01 sec, reward 198.434, steps 843
1604013: done 5494 episodes, mean reward -73.102, speed 17.87 f/s
1604101: done 5495 episodes, mean reward -73.056, speed 81.29 f/s
1604250: done 5497 episodes, mean reward -73.387, speed 81.37 f/s
1604374: done 5498 episodes, mean reward -73.265, speed 82.91 f/s
1604490: done 5499 episodes, mean reward -73.416, speed 80.14 f/s
1604655: done 5500 episodes, mean reward -73.305, speed 74.16 f/s
1604937: done 5501 episodes, mean reward -72.805, speed 81.73 f/s
Test done in 6.15 sec, reward 219.819, steps 908
1605007: done 5502 episodes, mean reward -73.544, speed 10.02 f/s
1605239: done 5503 episodes, mean reward -73.064, speed 79.73 f/s
1605366: done 5504 episodes, mean reward -72.935, speed 80.70 f/s
1605453: done 5505 episodes, mean reward -72.951, speed 83.07 f/s
1605718: done 5506 episodes, mean reward -73.132, speed 83.36 f/s
1605819: done 5507 episodes, mean reward -73.260, speed 77.86 f/s
1605965: done 5508 episodes, mean reward -73.188, speed 82.06 f/s
Test done in 6.71 sec, reward 259.350, steps 996
1606109: done 5509 episodes, mean reward -73.031, speed 17.03 f/s
1606200: done 5510 episodes, mean reward -73.456, speed 83.77 f/s
1606355: done 5511 episodes, mean reward -73.317, speed 82.38 f/s
1606516: done 5512 episodes, mean reward -73.176, speed 82.35 f/s
1606771: done 5513 episodes, mean reward -72.585, speed 83.04 f/s
1606877: done 5514 episodes, mean reward -74.860, speed 84.46 f/s
1606982: done 5515 episodes, mean reward -74.669, speed 80.36 f/s
Test done in 6.30 sec, reward 236.320, steps 958
1607081: done 5516 episodes, mean reward -74.613, speed 13.13 f/s
1607264: done 5518 episodes, mean reward -74.705, speed 78.58 f/s
1607510: done 5519 episodes, mean reward -74.152, speed 84.36 f/s
1607640: done 5520 episodes, mean reward -74.117, speed 84.57 f/s
1607741: done 5521 episodes, mean reward -74.268, speed 86.63 f/s
1607899: done 5522 episodes, mean reward -74.188, speed 81.53 f/s
Test done in 7.39 sec, reward 272.330, steps 1027
1608012: done 5523 episodes, mean reward -74.021, speed 12.76 f/s
1608166: done 5525 episodes, mean reward -75.294, speed 79.92 f/s
1608375: done 5526 episodes, mean reward -75.062, speed 81.84 f/s
1608505: done 5527 episodes, mean reward -75.033, speed 82.45 f/s
1608661: done 5528 episodes, mean reward -74.729, speed 84.14 f/s
1608864: done 5529 episodes, mean reward -74.380, speed 79.62 f/s
Test done in 7.02 sec, reward 272.440, steps 1020
1609056: done 5531 episodes, mean reward -75.225, speed 20.64 f/s
1609231: done 5532 episodes, mean reward -74.943, speed 83.19 f/s
1609386: done 5534 episodes, mean reward -75.325, speed 81.01 f/s
1609527: done 5535 episodes, mean reward -75.414, speed 82.68 f/s
1609618: done 5536 episodes, mean reward -77.550, speed 81.80 f/s
1609783: done 5537 episodes, mean reward -77.550, speed 83.88 f/s
1609886: done 5538 episodes, mean reward -77.531, speed 82.53 f/s
Test done in 6.95 sec, reward 272.585, steps 1038
1610243: done 5539 episodes, mean reward -77.577, speed 31.73 f/s
1610342: done 5540 episodes, mean reward -78.138, speed 83.87 f/s
1610556: done 5541 episodes, mean reward -77.798, speed 80.57 f/s
1610754: done 5542 episodes, mean reward -78.569, speed 81.07 f/s
Test done in 7.39 sec, reward 285.652, steps 1043
1611166: done 5543 episodes, mean reward -77.550, speed 33.35 f/s
1611277: done 5544 episodes, mean reward -77.721, speed 83.07 f/s
1611466: done 5545 episodes, mean reward -77.636, speed 81.75 f/s
1611739: done 5546 episodes, mean reward -77.648, speed 82.31 f/s
1611843: done 5547 episodes, mean reward -77.902, speed 85.35 f/s
1611976: done 5548 episodes, mean reward -77.715, speed 82.51 f/s
Test done in 6.48 sec, reward 258.399, steps 980
1612048: done 5549 episodes, mean reward -78.463, speed 9.73 f/s
1612133: done 5550 episodes, mean reward -78.498, speed 81.28 f/s
1612397: done 5551 episodes, mean reward -78.696, speed 81.38 f/s
1612577: done 5552 episodes, mean reward -78.949, speed 85.22 f/s
1612704: done 5553 episodes, mean reward -79.036, speed 85.39 f/s
1612814: done 5554 episodes, mean reward -79.195, speed 78.92 f/s
1612932: done 5555 episodes, mean reward -79.426, speed 81.14 f/s
Test done in 6.76 sec, reward 255.106, steps 976
1613060: done 5556 episodes, mean reward -79.365, speed 15.32 f/s
1613182: done 5557 episodes, mean reward -79.319, speed 85.21 f/s
1613649: done 5558 episodes, mean reward -78.132, speed 82.06 f/s
1613756: done 5559 episodes, mean reward -78.390, speed 80.81 f/s
1613857: done 5560 episodes, mean reward -78.415, speed 81.18 f/s
Test done in 6.92 sec, reward 284.681, steps 1027
1614010: done 5561 episodes, mean reward -78.387, speed 17.40 f/s
1614346: done 5562 episodes, mean reward -77.876, speed 81.48 f/s
1614426: done 5563 episodes, mean reward -77.856, speed 79.91 f/s
1614519: done 5564 episodes, mean reward -78.016, speed 84.27 f/s
1614753: done 5565 episodes, mean reward -77.756, speed 84.05 f/s
Test done in 7.17 sec, reward 284.615, steps 1037
1615174: done 5566 episodes, mean reward -76.750, speed 33.98 f/s
1615406: done 5568 episodes, mean reward -77.983, speed 78.65 f/s
1615602: done 5569 episodes, mean reward -77.624, speed 84.20 f/s
1615962: done 5571 episodes, mean reward -79.014, speed 81.65 f/s
Test done in 6.95 sec, reward 267.214, steps 1001
1616173: done 5572 episodes, mean reward -79.006, speed 22.17 f/s
1616266: done 5573 episodes, mean reward -78.933, speed 86.35 f/s
1616351: done 5574 episodes, mean reward -79.064, speed 83.98 f/s
1616484: done 5575 episodes, mean reward -78.951, speed 79.50 f/s
1616601: done 5577 episodes, mean reward -79.303, speed 84.78 f/s
1616830: done 5578 episodes, mean reward -78.821, speed 82.12 f/s
Test done in 5.26 sec, reward 175.358, steps 787
1617002: done 5579 episodes, mean reward -78.649, speed 23.35 f/s
1617125: done 5580 episodes, mean reward -78.817, speed 81.24 f/s
1617233: done 5581 episodes, mean reward -78.934, speed 82.33 f/s
1617446: done 5583 episodes, mean reward -79.364, speed 82.07 f/s
1617564: done 5584 episodes, mean reward -79.434, speed 82.95 f/s
1617682: done 5586 episodes, mean reward -79.903, speed 77.99 f/s
1617885: done 5588 episodes, mean reward -79.806, speed 75.96 f/s
Test done in 6.33 sec, reward 226.399, steps 889
1618035: done 5589 episodes, mean reward -80.551, speed 18.13 f/s
1618161: done 5590 episodes, mean reward -80.631, speed 83.39 f/s
1618291: done 5591 episodes, mean reward -80.942, speed 82.99 f/s
1618479: done 5593 episodes, mean reward -80.947, speed 81.49 f/s
1618614: done 5594 episodes, mean reward -81.051, speed 79.77 f/s
1618745: done 5595 episodes, mean reward -80.948, speed 81.40 f/s
1618869: done 5596 episodes, mean reward -80.814, speed 78.80 f/s
Test done in 6.89 sec, reward 271.539, steps 1008
1619074: done 5598 episodes, mean reward -80.636, speed 21.87 f/s
1619229: done 5599 episodes, mean reward -80.555, speed 82.54 f/s
1619668: done 5600 episodes, mean reward -79.686, speed 82.18 f/s
1619948: done 5601 episodes, mean reward -79.790, speed 81.48 f/s
Test done in 6.92 sec, reward 244.663, steps 1001
1620051: done 5602 episodes, mean reward -79.683, speed 12.68 f/s
1620147: done 5603 episodes, mean reward -80.139, speed 79.31 f/s
1620389: done 5604 episodes, mean reward -79.770, speed 82.98 f/s
1620488: done 5605 episodes, mean reward -79.647, speed 82.65 f/s
1620642: done 5607 episodes, mean reward -79.971, speed 78.30 f/s
1620810: done 5609 episodes, mean reward -80.338, speed 83.63 f/s
1620952: done 5610 episodes, mean reward -80.267, speed 82.04 f/s
Test done in 6.50 sec, reward 209.516, steps 885
1621127: done 5611 episodes, mean reward -80.141, speed 20.16 f/s
1621212: done 5612 episodes, mean reward -80.303, speed 79.06 f/s
1621433: done 5613 episodes, mean reward -80.374, speed 79.05 f/s
1621553: done 5614 episodes, mean reward -80.421, speed 81.87 f/s
1621662: done 5615 episodes, mean reward -80.487, speed 81.31 f/s
1621836: done 5616 episodes, mean reward -80.241, speed 79.85 f/s
1621960: done 5617 episodes, mean reward -80.167, speed 80.12 f/s
Test done in 5.90 sec, reward 198.733, steps 866
1622070: done 5618 episodes, mean reward -80.237, speed 15.28 f/s
1622251: done 5619 episodes, mean reward -80.332, speed 82.46 f/s
1622472: done 5621 episodes, mean reward -80.327, speed 84.74 f/s
1622561: done 5622 episodes, mean reward -80.577, speed 81.57 f/s
1622701: done 5623 episodes, mean reward -80.573, speed 83.09 f/s
1622807: done 5624 episodes, mean reward -80.459, speed 79.71 f/s
1622931: done 5625 episodes, mean reward -80.316, speed 79.00 f/s
Test done in 6.67 sec, reward 269.411, steps 996
1623065: done 5626 episodes, mean reward -80.726, speed 16.04 f/s
1623215: done 5628 episodes, mean reward -81.179, speed 80.65 f/s
1623573: done 5630 episodes, mean reward -80.918, speed 83.43 f/s
1623671: done 5631 episodes, mean reward -81.151, speed 85.95 f/s
Test done in 7.34 sec, reward 285.742, steps 1055
1624117: done 5633 episodes, mean reward -80.336, speed 34.32 f/s
1624211: done 5634 episodes, mean reward -80.379, speed 74.63 f/s
1624341: done 5635 episodes, mean reward -80.430, speed 80.68 f/s
1624433: done 5636 episodes, mean reward -80.445, speed 83.18 f/s
1624656: done 5637 episodes, mean reward -80.230, speed 79.95 f/s
1624835: done 5638 episodes, mean reward -79.944, speed 80.02 f/s
1624973: done 5639 episodes, mean reward -80.485, speed 82.51 f/s
Test done in 7.00 sec, reward 289.527, steps 1013
Best reward updated: 287.526 -> 289.527
1625063: done 5640 episodes, mean reward -80.491, speed 11.05 f/s
1625162: done 5641 episodes, mean reward -80.780, speed 81.71 f/s
1625297: done 5642 episodes, mean reward -81.003, speed 79.28 f/s
1625428: done 5643 episodes, mean reward -82.016, speed 81.48 f/s
1625587: done 5645 episodes, mean reward -82.588, speed 81.45 f/s
1625676: done 5646 episodes, mean reward -83.071, speed 83.26 f/s
1625832: done 5647 episodes, mean reward -82.740, speed 83.40 f/s
1625922: done 5648 episodes, mean reward -82.890, speed 81.98 f/s
Test done in 6.73 sec, reward 263.440, steps 979
1626147: done 5649 episodes, mean reward -82.439, speed 23.78 f/s
1626408: done 5651 episodes, mean reward -82.686, speed 82.33 f/s
1626502: done 5652 episodes, mean reward -82.871, speed 84.09 f/s
1626683: done 5653 episodes, mean reward -82.919, speed 85.92 f/s
Test done in 7.12 sec, reward 286.878, steps 1030
1627021: done 5654 episodes, mean reward -82.319, speed 29.93 f/s
1627124: done 5655 episodes, mean reward -82.425, speed 83.34 f/s
1627339: done 5657 episodes, mean reward -82.374, speed 77.44 f/s
1627441: done 5658 episodes, mean reward -83.631, speed 72.98 f/s
1627543: done 5659 episodes, mean reward -83.736, speed 78.92 f/s
1627694: done 5660 episodes, mean reward -83.529, speed 78.31 f/s
1627893: done 5661 episodes, mean reward -83.329, speed 82.82 f/s
Test done in 7.38 sec, reward 287.894, steps 1051
1628023: done 5663 episodes, mean reward -84.337, speed 14.45 f/s
1628258: done 5664 episodes, mean reward -83.768, speed 82.31 f/s
1628375: done 5665 episodes, mean reward -84.159, speed 84.15 f/s
1628575: done 5667 episodes, mean reward -85.090, speed 82.73 f/s
1628757: done 5669 episodes, mean reward -85.671, speed 83.50 f/s
1628849: done 5670 episodes, mean reward -85.635, speed 82.23 f/s
Test done in 6.58 sec, reward 260.661, steps 967
1629047: done 5671 episodes, mean reward -86.012, speed 21.95 f/s
1629133: done 5672 episodes, mean reward -86.432, speed 84.65 f/s
1629243: done 5673 episodes, mean reward -86.342, speed 80.31 f/s
1629327: done 5674 episodes, mean reward -86.523, speed 82.48 f/s
1629670: done 5675 episodes, mean reward -85.835, speed 83.05 f/s
1629867: done 5677 episodes, mean reward -85.702, speed 84.02 f/s
Test done in 6.28 sec, reward 260.059, steps 927
1630178: done 5679 episodes, mean reward -85.893, speed 30.99 f/s
1630359: done 5681 episodes, mean reward -85.922, speed 80.42 f/s
1630596: done 5683 episodes, mean reward -85.971, speed 81.40 f/s
1630684: done 5684 episodes, mean reward -86.163, speed 81.27 f/s
1630924: done 5685 episodes, mean reward -85.593, speed 83.21 f/s
Test done in 6.75 sec, reward 250.713, steps 986
1631205: done 5686 episodes, mean reward -84.887, speed 27.56 f/s
1631341: done 5687 episodes, mean reward -84.725, speed 84.00 f/s
1631500: done 5688 episodes, mean reward -84.634, speed 82.63 f/s
1631637: done 5689 episodes, mean reward -84.799, speed 83.56 f/s
1631744: done 5690 episodes, mean reward -84.775, speed 80.10 f/s
1631850: done 5691 episodes, mean reward -84.893, speed 81.25 f/s
1631942: done 5692 episodes, mean reward -84.928, speed 85.02 f/s
Test done in 6.67 sec, reward 264.761, steps 992
1632045: done 5693 episodes, mean reward -84.941, speed 12.96 f/s
1632179: done 5694 episodes, mean reward -84.813, speed 77.33 f/s
1632519: done 5695 episodes, mean reward -84.228, speed 75.78 f/s
1632620: done 5696 episodes, mean reward -84.529, speed 73.11 f/s
1632792: done 5697 episodes, mean reward -84.249, speed 82.09 f/s
1632918: done 5698 episodes, mean reward -84.242, speed 83.10 f/s
Test done in 6.62 sec, reward 266.928, steps 993
1633137: done 5699 episodes, mean reward -83.987, speed 23.65 f/s
1633387: done 5700 episodes, mean reward -84.506, speed 83.12 f/s
1633486: done 5701 episodes, mean reward -85.041, speed 81.76 f/s
1633902: done 5702 episodes, mean reward -84.158, speed 81.87 f/s
Test done in 7.10 sec, reward 274.603, steps 1034
1634038: done 5703 episodes, mean reward -84.170, speed 15.58 f/s
1634185: done 5704 episodes, mean reward -84.458, speed 84.69 f/s
1634297: done 5705 episodes, mean reward -84.539, speed 74.67 f/s
1634546: done 5706 episodes, mean reward -83.955, speed 83.55 f/s
1634812: done 5707 episodes, mean reward -83.456, speed 80.60 f/s
1634984: done 5709 episodes, mean reward -83.483, speed 81.70 f/s
Test done in 6.25 sec, reward 217.295, steps 926
1635133: done 5710 episodes, mean reward -83.578, speed 18.46 f/s
1635237: done 5711 episodes, mean reward -84.009, speed 81.19 f/s
1635368: done 5712 episodes, mean reward -83.857, speed 85.19 f/s
1635547: done 5714 episodes, mean reward -84.423, speed 81.96 f/s
1635689: done 5715 episodes, mean reward -84.357, speed 82.63 f/s
1635975: done 5717 episodes, mean reward -84.257, speed 81.71 f/s
Test done in 6.98 sec, reward 288.729, steps 1030
1636062: done 5718 episodes, mean reward -84.510, speed 10.82 f/s
1636200: done 5719 episodes, mean reward -84.793, speed 81.69 f/s
1636358: done 5720 episodes, mean reward -84.615, speed 83.43 f/s
1636642: done 5721 episodes, mean reward -84.291, speed 80.64 f/s
1636743: done 5722 episodes, mean reward -84.232, speed 78.36 f/s
1636857: done 5723 episodes, mean reward -84.441, speed 84.97 f/s
1636989: done 5724 episodes, mean reward -84.319, speed 80.91 f/s
Test done in 6.33 sec, reward 249.906, steps 928
1637144: done 5725 episodes, mean reward -84.264, speed 18.81 f/s
1637265: done 5727 episodes, mean reward -84.423, speed 80.94 f/s
1637460: done 5728 episodes, mean reward -84.240, speed 81.24 f/s
1637569: done 5729 episodes, mean reward -84.045, speed 76.43 f/s
1637772: done 5731 episodes, mean reward -84.710, speed 83.02 f/s
1637959: done 5732 episodes, mean reward -84.337, speed 83.66 f/s
Test done in 6.46 sec, reward 249.610, steps 962
1638065: done 5733 episodes, mean reward -85.382, speed 13.61 f/s
1638327: done 5734 episodes, mean reward -84.831, speed 82.06 f/s
1638483: done 5735 episodes, mean reward -84.742, speed 80.87 f/s
1638680: done 5736 episodes, mean reward -84.484, speed 82.65 f/s
1638834: done 5737 episodes, mean reward -84.850, speed 84.28 f/s
1638937: done 5738 episodes, mean reward -85.144, speed 82.30 f/s
Test done in 6.57 sec, reward 261.755, steps 938
1639064: done 5739 episodes, mean reward -85.310, speed 15.59 f/s
1639353: done 5741 episodes, mean reward -84.944, speed 82.94 f/s
1639757: done 5743 episodes, mean reward -84.403, speed 82.80 f/s
1639886: done 5744 episodes, mean reward -84.389, speed 83.13 f/s
Test done in 5.99 sec, reward 225.850, steps 860
1640007: done 5745 episodes, mean reward -84.155, speed 16.36 f/s
1640089: done 5746 episodes, mean reward -84.316, speed 80.74 f/s
1640226: done 5747 episodes, mean reward -84.445, speed 81.35 f/s
1640359: done 5748 episodes, mean reward -84.268, speed 78.98 f/s
1640481: done 5749 episodes, mean reward -84.626, speed 84.74 f/s
1640594: done 5750 episodes, mean reward -84.582, speed 81.98 f/s
1640719: done 5751 episodes, mean reward -84.763, speed 83.47 f/s
1640828: done 5752 episodes, mean reward -84.634, speed 80.56 f/s
1640928: done 5753 episodes, mean reward -84.684, speed 78.37 f/s
Test done in 6.91 sec, reward 261.196, steps 996
1641067: done 5754 episodes, mean reward -85.212, speed 15.95 f/s
1641161: done 5755 episodes, mean reward -85.094, speed 82.17 f/s
1641302: done 5757 episodes, mean reward -85.429, speed 83.97 f/s
1641491: done 5758 episodes, mean reward -85.040, speed 79.95 f/s
1641759: done 5759 episodes, mean reward -84.457, speed 84.48 f/s
1641879: done 5760 episodes, mean reward -84.678, speed 81.34 f/s
Test done in 6.51 sec, reward 254.001, steps 960
1642075: done 5762 episodes, mean reward -84.945, speed 21.87 f/s
1642220: done 5763 episodes, mean reward -84.648, speed 76.28 f/s
1642488: done 5765 episodes, mean reward -84.967, speed 81.65 f/s
1642619: done 5766 episodes, mean reward -84.798, speed 82.43 f/s
1642741: done 5767 episodes, mean reward -84.704, speed 82.08 f/s
1642823: done 5768 episodes, mean reward -84.798, speed 79.00 f/s
1642991: done 5769 episodes, mean reward -84.549, speed 81.15 f/s
Test done in 6.43 sec, reward 264.073, steps 957
1643131: done 5770 episodes, mean reward -84.337, speed 17.15 f/s
1643281: done 5771 episodes, mean reward -84.500, speed 80.64 f/s
1643419: done 5772 episodes, mean reward -84.155, speed 82.08 f/s
1643558: done 5773 episodes, mean reward -84.118, speed 80.47 f/s
1643671: done 5774 episodes, mean reward -83.891, speed 83.70 f/s
1643839: done 5775 episodes, mean reward -84.627, speed 83.77 f/s
1643969: done 5776 episodes, mean reward -84.585, speed 82.35 f/s
Test done in 6.03 sec, reward 219.226, steps 882
1644045: done 5777 episodes, mean reward -84.670, speed 10.89 f/s
1644149: done 5778 episodes, mean reward -84.552, speed 82.91 f/s
1644293: done 5779 episodes, mean reward -84.978, speed 84.02 f/s
1644406: done 5780 episodes, mean reward -84.892, speed 82.75 f/s
1644542: done 5781 episodes, mean reward -84.812, speed 78.16 f/s
1644715: done 5782 episodes, mean reward -84.563, speed 81.96 f/s
1644937: done 5783 episodes, mean reward -84.240, speed 82.96 f/s
Test done in 6.48 sec, reward 238.500, steps 945
1645148: done 5784 episodes, mean reward -83.674, speed 23.34 f/s
1645275: done 5785 episodes, mean reward -84.061, speed 80.85 f/s
1645441: done 5786 episodes, mean reward -84.364, speed 82.37 f/s
1645551: done 5787 episodes, mean reward -84.315, speed 83.33 f/s
1645652: done 5788 episodes, mean reward -84.555, speed 84.89 f/s
1645795: done 5790 episodes, mean reward -84.809, speed 85.80 f/s
1645913: done 5791 episodes, mean reward -84.680, speed 81.69 f/s
Test done in 6.39 sec, reward 232.069, steps 920
1646130: done 5792 episodes, mean reward -84.187, speed 24.04 f/s
1646242: done 5793 episodes, mean reward -84.247, speed 79.63 f/s
1646469: done 5794 episodes, mean reward -83.984, speed 82.44 f/s
1646636: done 5796 episodes, mean reward -84.615, speed 82.32 f/s
1646736: done 5797 episodes, mean reward -84.845, speed 83.42 f/s
1646927: done 5798 episodes, mean reward -84.601, speed 79.32 f/s
Test done in 5.95 sec, reward 216.270, steps 875
1647048: done 5799 episodes, mean reward -84.917, speed 16.13 f/s
1647171: done 5800 episodes, mean reward -85.390, speed 81.15 f/s
1647381: done 5801 episodes, mean reward -84.893, speed 82.22 f/s
1647504: done 5802 episodes, mean reward -85.686, speed 80.65 f/s
1647707: done 5803 episodes, mean reward -85.352, speed 82.95 f/s
1647941: done 5805 episodes, mean reward -85.287, speed 81.97 f/s
Test done in 6.83 sec, reward 290.781, steps 1017
Best reward updated: 289.527 -> 290.781
1648031: done 5806 episodes, mean reward -85.865, speed 11.34 f/s
1648175: done 5807 episodes, mean reward -86.165, speed 83.29 f/s
1648357: done 5808 episodes, mean reward -85.896, speed 81.85 f/s
1648563: done 5809 episodes, mean reward -85.668, speed 80.75 f/s
1648765: done 5810 episodes, mean reward -85.259, speed 80.67 f/s
1648913: done 5811 episodes, mean reward -85.143, speed 83.56 f/s
Test done in 4.84 sec, reward 113.390, steps 667
1649180: done 5812 episodes, mean reward -84.848, speed 32.80 f/s
1649309: done 5813 episodes, mean reward -84.633, speed 81.81 f/s
1649426: done 5814 episodes, mean reward -84.548, speed 80.72 f/s
1649701: done 5816 episodes, mean reward -84.268, speed 80.79 f/s
1649786: done 5817 episodes, mean reward -84.727, speed 81.68 f/s
1649873: done 5818 episodes, mean reward -84.666, speed 83.54 f/s
Test done in 6.12 sec, reward 243.075, steps 900
1650013: done 5819 episodes, mean reward -84.723, speed 17.55 f/s
1650133: done 5820 episodes, mean reward -84.753, speed 82.04 f/s
1650264: done 5821 episodes, mean reward -85.114, speed 81.24 f/s
1650350: done 5822 episodes, mean reward -85.182, speed 78.69 f/s
1650633: done 5823 episodes, mean reward -84.485, speed 78.50 f/s
1650761: done 5824 episodes, mean reward -84.463, speed 83.26 f/s
1650901: done 5825 episodes, mean reward -84.415, speed 82.40 f/s
Test done in 5.99 sec, reward 205.014, steps 865
1651025: done 5826 episodes, mean reward -84.183, speed 16.56 f/s
1651129: done 5827 episodes, mean reward -84.089, speed 82.24 f/s
1651463: done 5829 episodes, mean reward -83.796, speed 80.11 f/s
1651611: done 5830 episodes, mean reward -83.395, speed 82.70 f/s
1651725: done 5831 episodes, mean reward -83.342, speed 80.71 f/s
1651900: done 5833 episodes, mean reward -83.730, speed 82.22 f/s
Test done in 6.22 sec, reward 233.454, steps 904
1652050: done 5834 episodes, mean reward -84.062, speed 18.60 f/s
1652151: done 5835 episodes, mean reward -84.277, speed 76.57 f/s
1652245: done 5836 episodes, mean reward -84.565, speed 76.42 f/s
1652337: done 5837 episodes, mean reward -84.540, speed 77.83 f/s
1652609: done 5839 episodes, mean reward -84.396, speed 79.24 f/s
1652789: done 5841 episodes, mean reward -84.685, speed 79.26 f/s
1652887: done 5842 episodes, mean reward -84.706, speed 83.26 f/s
Test done in 6.74 sec, reward 245.696, steps 968
1653088: done 5843 episodes, mean reward -85.120, speed 21.99 f/s
1653342: done 5844 episodes, mean reward -84.526, speed 82.07 f/s
1653476: done 5845 episodes, mean reward -84.501, speed 81.84 f/s
1653661: done 5847 episodes, mean reward -84.352, speed 82.49 f/s
1653746: done 5848 episodes, mean reward -84.471, speed 81.22 f/s
1653864: done 5849 episodes, mean reward -84.449, speed 76.23 f/s
Test done in 6.53 sec, reward 259.057, steps 948
1654003: done 5850 episodes, mean reward -84.217, speed 16.97 f/s
1654203: done 5852 episodes, mean reward -84.395, speed 83.26 f/s
1654291: done 5853 episodes, mean reward -84.465, speed 79.68 f/s
1654410: done 5854 episodes, mean reward -84.549, speed 85.94 f/s
1654710: done 5856 episodes, mean reward -84.170, speed 81.96 f/s
1654921: done 5858 episodes, mean reward -84.509, speed 82.52 f/s
Test done in 6.26 sec, reward 259.688, steps 937
1655119: done 5860 episodes, mean reward -85.039, speed 22.94 f/s
1655211: done 5861 episodes, mean reward -85.004, speed 81.06 f/s
1655462: done 5862 episodes, mean reward -84.671, speed 82.94 f/s
1655672: done 5864 episodes, mean reward -84.710, speed 74.42 f/s
1655789: done 5865 episodes, mean reward -84.925, speed 79.34 f/s
1655925: done 5867 episodes, mean reward -85.218, speed 79.86 f/s
Test done in 5.93 sec, reward 192.180, steps 860
1656170: done 5869 episodes, mean reward -84.953, speed 27.39 f/s
1656267: done 5870 episodes, mean reward -85.086, speed 83.79 f/s
1656368: done 5871 episodes, mean reward -85.207, speed 81.16 f/s
1656499: done 5872 episodes, mean reward -85.266, speed 81.63 f/s
1656643: done 5873 episodes, mean reward -85.333, speed 81.07 f/s
1656818: done 5874 episodes, mean reward -85.240, speed 83.10 f/s
Test done in 6.89 sec, reward 291.785, steps 986
Best reward updated: 290.781 -> 291.785
1657050: done 5876 episodes, mean reward -85.151, speed 23.95 f/s
1657219: done 5877 episodes, mean reward -85.046, speed 82.12 f/s
1657548: done 5879 episodes, mean reward -84.829, speed 82.20 f/s
1657796: done 5880 episodes, mean reward -84.382, speed 83.44 f/s
Test done in 5.67 sec, reward 210.487, steps 845
1658107: done 5881 episodes, mean reward -83.979, speed 32.51 f/s
1658229: done 5882 episodes, mean reward -84.039, speed 80.30 f/s
1658573: done 5883 episodes, mean reward -83.668, speed 82.22 f/s
1658693: done 5884 episodes, mean reward -84.148, speed 83.66 f/s
1658856: done 5885 episodes, mean reward -83.973, speed 80.44 f/s
1658973: done 5886 episodes, mean reward -84.156, speed 81.22 f/s
Test done in 6.10 sec, reward 205.988, steps 839
1659285: done 5887 episodes, mean reward -83.627, speed 31.27 f/s
1659372: done 5888 episodes, mean reward -83.680, speed 77.90 f/s
1659729: done 5889 episodes, mean reward -82.738, speed 79.21 f/s
1659843: done 5890 episodes, mean reward -82.613, speed 82.97 f/s
1659979: done 5891 episodes, mean reward -82.579, speed 79.53 f/s
Test done in 4.27 sec, reward 128.211, steps 636
1660068: done 5892 episodes, mean reward -82.985, speed 16.63 f/s
1660253: done 5893 episodes, mean reward -82.593, speed 79.50 f/s
1660457: done 5895 episodes, mean reward -82.764, speed 83.69 f/s
1660555: done 5896 episodes, mean reward -82.683, speed 82.02 f/s
1660707: done 5897 episodes, mean reward -82.605, speed 83.68 f/s
1660873: done 5898 episodes, mean reward -82.747, speed 80.30 f/s
Test done in 3.76 sec, reward 102.614, steps 568
1661017: done 5900 episodes, mean reward -83.029, speed 26.07 f/s
1661118: done 5901 episodes, mean reward -83.319, speed 80.10 f/s
1661239: done 5902 episodes, mean reward -83.330, speed 81.89 f/s
1661462: done 5903 episodes, mean reward -83.277, speed 83.22 f/s
1661579: done 5904 episodes, mean reward -83.114, speed 83.91 f/s
1661806: done 5906 episodes, mean reward -83.133, speed 82.95 f/s
1661896: done 5907 episodes, mean reward -83.476, speed 82.87 f/s
Test done in 5.36 sec, reward 183.404, steps 784
1662066: done 5909 episodes, mean reward -84.242, speed 22.74 f/s
1662305: done 5911 episodes, mean reward -84.597, speed 83.17 f/s
1662433: done 5912 episodes, mean reward -84.844, speed 82.10 f/s
1662524: done 5913 episodes, mean reward -85.027, speed 84.58 f/s
1662686: done 5915 episodes, mean reward -85.113, speed 74.22 f/s
1662773: done 5916 episodes, mean reward -85.525, speed 79.15 f/s
1662877: done 5917 episodes, mean reward -85.438, speed 76.61 f/s
Test done in 6.06 sec, reward 240.777, steps 900
1663033: done 5918 episodes, mean reward -85.060, speed 19.63 f/s
1663282: done 5920 episodes, mean reward -84.885, speed 83.13 f/s
1663372: done 5921 episodes, mean reward -85.079, speed 81.74 f/s
1663489: done 5922 episodes, mean reward -84.938, speed 82.72 f/s
1663595: done 5923 episodes, mean reward -85.616, speed 81.92 f/s
Test done in 0.60 sec, reward -98.422, steps 85
1664002: done 5924 episodes, mean reward -84.685, speed 72.62 f/s
1664086: done 5925 episodes, mean reward -84.916, speed 83.78 f/s
1664209: done 5926 episodes, mean reward -84.956, speed 81.76 f/s
1664302: done 5927 episodes, mean reward -84.975, speed 82.90 f/s
1664471: done 5929 episodes, mean reward -85.703, speed 80.09 f/s
1664688: done 5930 episodes, mean reward -85.589, speed 82.22 f/s
1664768: done 5931 episodes, mean reward -85.722, speed 76.77 f/s
1664916: done 5933 episodes, mean reward -85.707, speed 83.95 f/s
Test done in 1.09 sec, reward -76.796, steps 156
1665057: done 5934 episodes, mean reward -85.774, speed 50.43 f/s
1665216: done 5936 episodes, mean reward -85.849, speed 81.10 f/s
1665378: done 5938 episodes, mean reward -85.941, speed 83.09 f/s
1665463: done 5939 episodes, mean reward -86.360, speed 84.11 f/s
1665647: done 5941 episodes, mean reward -86.440, speed 81.85 f/s
1665750: done 5942 episodes, mean reward -86.454, speed 81.62 f/s
1665903: done 5944 episodes, mean reward -87.513, speed 83.48 f/s
Test done in 0.60 sec, reward -101.214, steps 76
1666053: done 5946 episodes, mean reward -87.732, speed 63.54 f/s
1666143: done 5947 episodes, mean reward -87.851, speed 82.50 f/s
1666292: done 5949 episodes, mean reward -88.051, speed 82.24 f/s
1666444: done 5951 episodes, mean reward -88.260, speed 80.18 f/s
1666600: done 5952 episodes, mean reward -88.122, speed 80.85 f/s
1666682: done 5953 episodes, mean reward -88.138, speed 80.37 f/s
1666778: done 5954 episodes, mean reward -88.148, speed 82.63 f/s
Test done in 6.18 sec, reward 254.709, steps 869
1667069: done 5956 episodes, mean reward -88.023, speed 29.51 f/s
1667261: done 5957 episodes, mean reward -87.654, speed 82.52 f/s
1667660: done 5959 episodes, mean reward -86.952, speed 82.67 f/s
1667889: done 5960 episodes, mean reward -86.679, speed 81.14 f/s
1667989: done 5961 episodes, mean reward -86.687, speed 82.40 f/s
Test done in 4.16 sec, reward 105.862, steps 594
1668070: done 5962 episodes, mean reward -87.229, speed 15.75 f/s
1668420: done 5963 episodes, mean reward -86.156, speed 83.54 f/s
1668807: done 5964 episodes, mean reward -85.242, speed 82.23 f/s
1668989: done 5966 episodes, mean reward -85.296, speed 81.13 f/s
Test done in 1.46 sec, reward -55.310, steps 203
1669099: done 5967 episodes, mean reward -85.367, speed 39.33 f/s
1669183: done 5968 episodes, mean reward -85.277, speed 83.13 f/s
1669390: done 5969 episodes, mean reward -85.249, speed 81.98 f/s
1669720: done 5970 episodes, mean reward -84.560, speed 83.22 f/s
1669808: done 5971 episodes, mean reward -84.577, speed 79.56 f/s
1669894: done 5972 episodes, mean reward -84.669, speed 84.34 f/s
Test done in 5.98 sec, reward 261.177, steps 886
1670043: done 5974 episodes, mean reward -85.110, speed 19.02 f/s
1670125: done 5975 episodes, mean reward -85.064, speed 77.23 f/s
1670377: done 5976 episodes, mean reward -84.718, speed 83.46 f/s
1670519: done 5977 episodes, mean reward -84.600, speed 82.02 f/s
1670690: done 5979 episodes, mean reward -85.260, speed 83.62 f/s
1670836: done 5980 episodes, mean reward -85.655, speed 83.03 f/s
Test done in 5.91 sec, reward 239.283, steps 896
1671037: done 5982 episodes, mean reward -86.341, speed 23.95 f/s
1671163: done 5983 episodes, mean reward -87.070, speed 85.32 f/s
1671302: done 5984 episodes, mean reward -86.867, speed 81.49 f/s
1671458: done 5985 episodes, mean reward -87.033, speed 82.36 f/s
1671560: done 5986 episodes, mean reward -87.070, speed 76.18 f/s
1671716: done 5987 episodes, mean reward -87.519, speed 80.89 f/s
1671839: done 5989 episodes, mean reward -88.681, speed 81.87 f/s
1671949: done 5991 episodes, mean reward -89.096, speed 82.45 f/s
Test done in 5.74 sec, reward 229.760, steps 841
1672002: done 5992 episodes, mean reward -89.193, speed 8.29 f/s
1672176: done 5993 episodes, mean reward -89.304, speed 82.63 f/s
1672321: done 5994 episodes, mean reward -89.067, speed 81.22 f/s
1672448: done 5995 episodes, mean reward -89.032, speed 82.18 f/s
1672539: done 5996 episodes, mean reward -89.014, speed 80.56 f/s
1672728: done 5997 episodes, mean reward -88.846, speed 84.71 f/s
1672935: done 5999 episodes, mean reward -89.042, speed 82.34 f/s
Test done in 6.20 sec, reward 269.142, steps 910
1673029: done 6000 episodes, mean reward -88.964, speed 12.79 f/s
1673166: done 6001 episodes, mean reward -88.843, speed 83.64 f/s
1673264: done 6002 episodes, mean reward -88.902, speed 81.59 f/s
1673528: done 6003 episodes, mean reward -88.697, speed 81.21 f/s
1673611: done 6004 episodes, mean reward -88.892, speed 81.21 f/s
1673707: done 6005 episodes, mean reward -88.882, speed 81.90 f/s
1673836: done 6006 episodes, mean reward -89.041, speed 81.16 f/s
1673950: done 6007 episodes, mean reward -88.938, speed 83.78 f/s
Test done in 6.52 sec, reward 252.502, steps 891
1674076: done 6008 episodes, mean reward -88.611, speed 15.64 f/s
1674246: done 6009 episodes, mean reward -88.207, speed 83.14 f/s
1674393: done 6011 episodes, mean reward -88.397, speed 79.66 f/s
1674600: done 6012 episodes, mean reward -88.120, speed 83.86 f/s
1674685: done 6013 episodes, mean reward -88.176, speed 81.86 f/s
1674822: done 6014 episodes, mean reward -87.869, speed 82.41 f/s
1674947: done 6015 episodes, mean reward -87.849, speed 79.67 f/s
Test done in 5.89 sec, reward 261.548, steps 885
1675147: done 6016 episodes, mean reward -87.386, speed 23.96 f/s
1675269: done 6017 episodes, mean reward -87.282, speed 78.81 f/s
1675348: done 6018 episodes, mean reward -87.666, speed 78.45 f/s
1675514: done 6020 episodes, mean reward -88.221, speed 81.28 f/s
1675667: done 6021 episodes, mean reward -87.914, speed 81.31 f/s
1675766: done 6022 episodes, mean reward -88.019, speed 79.33 f/s
1675860: done 6023 episodes, mean reward -88.007, speed 85.41 f/s
Test done in 6.29 sec, reward 275.248, steps 940
1676023: done 6024 episodes, mean reward -88.789, speed 19.79 f/s
1676127: done 6025 episodes, mean reward -88.626, speed 83.08 f/s
1676276: done 6027 episodes, mean reward -88.779, speed 80.78 f/s
1676615: done 6028 episodes, mean reward -87.853, speed 82.92 f/s
1676761: done 6030 episodes, mean reward -88.205, speed 79.73 f/s
1676867: done 6031 episodes, mean reward -88.169, speed 81.43 f/s
1676957: done 6032 episodes, mean reward -88.125, speed 78.98 f/s
Test done in 6.45 sec, reward 273.427, steps 931
1677094: done 6033 episodes, mean reward -87.893, speed 16.88 f/s
1677176: done 6034 episodes, mean reward -88.184, speed 81.06 f/s
1677285: done 6035 episodes, mean reward -88.181, speed 75.80 f/s
1677457: done 6036 episodes, mean reward -87.765, speed 74.11 f/s
1677540: done 6037 episodes, mean reward -87.805, speed 79.30 f/s
1677659: done 6038 episodes, mean reward -87.587, speed 80.34 f/s
1677746: done 6039 episodes, mean reward -87.571, speed 79.37 f/s
1677833: done 6040 episodes, mean reward -87.532, speed 79.68 f/s
Test done in 5.55 sec, reward 201.084, steps 802
1678034: done 6042 episodes, mean reward -87.677, speed 25.26 f/s
1678189: done 6043 episodes, mean reward -87.474, speed 84.29 f/s
1678345: done 6044 episodes, mean reward -87.263, speed 83.86 f/s
1678452: done 6045 episodes, mean reward -87.197, speed 83.30 f/s
1678558: done 6046 episodes, mean reward -87.174, speed 80.40 f/s
1678640: done 6047 episodes, mean reward -87.059, speed 79.33 f/s
1678817: done 6048 episodes, mean reward -86.700, speed 83.18 f/s
1678957: done 6049 episodes, mean reward -86.611, speed 85.89 f/s
Test done in 5.79 sec, reward 241.229, steps 874
1679065: done 6050 episodes, mean reward -86.642, speed 15.12 f/s
1679155: done 6051 episodes, mean reward -86.670, speed 81.10 f/s
1679250: done 6052 episodes, mean reward -86.902, speed 79.06 f/s
1679432: done 6054 episodes, mean reward -87.015, speed 81.52 f/s
1679625: done 6055 episodes, mean reward -86.642, speed 82.08 f/s
1679850: done 6056 episodes, mean reward -86.672, speed 81.16 f/s
1679973: done 6057 episodes, mean reward -86.758, speed 84.04 f/s
Test done in 6.53 sec, reward 291.114, steps 988
1680068: done 6058 episodes, mean reward -86.640, speed 12.36 f/s
1680163: done 6059 episodes, mean reward -87.438, speed 81.40 f/s
1680274: done 6060 episodes, mean reward -87.766, speed 82.17 f/s
1680419: done 6061 episodes, mean reward -87.555, speed 78.98 f/s
1680621: done 6062 episodes, mean reward -87.104, speed 80.15 f/s
1680817: done 6064 episodes, mean reward -88.930, speed 81.02 f/s
Test done in 6.23 sec, reward 236.081, steps 913
1681129: done 6065 episodes, mean reward -88.290, speed 31.30 f/s
1681239: done 6066 episodes, mean reward -88.264, speed 80.10 f/s
1681325: done 6067 episodes, mean reward -88.134, speed 80.81 f/s
1681472: done 6068 episodes, mean reward -87.911, speed 83.00 f/s
1681846: done 6069 episodes, mean reward -87.370, speed 82.56 f/s
1681931: done 6070 episodes, mean reward -88.162, speed 82.86 f/s
Test done in 4.25 sec, reward 122.327, steps 616
1682144: done 6071 episodes, mean reward -87.869, speed 30.26 f/s
1682312: done 6072 episodes, mean reward -87.674, speed 82.45 f/s
1682425: done 6073 episodes, mean reward -87.660, speed 81.78 f/s
1682556: done 6074 episodes, mean reward -87.556, speed 75.67 f/s
1682679: done 6075 episodes, mean reward -87.350, speed 82.09 f/s
1682988: done 6076 episodes, mean reward -87.199, speed 77.16 f/s
Test done in 5.95 sec, reward 257.400, steps 885
1683264: done 6077 episodes, mean reward -86.791, speed 29.57 f/s
1683351: done 6078 episodes, mean reward -86.771, speed 81.52 f/s
1683456: done 6079 episodes, mean reward -86.829, speed 82.27 f/s
1683581: done 6080 episodes, mean reward -86.752, speed 84.17 f/s
1683704: done 6081 episodes, mean reward -86.628, speed 82.22 f/s
1683868: done 6082 episodes, mean reward -86.473, speed 81.99 f/s
Test done in 5.84 sec, reward 237.947, steps 865
1684274: done 6083 episodes, mean reward -85.662, speed 37.65 f/s
1684369: done 6084 episodes, mean reward -85.829, speed 85.94 f/s
1684528: done 6085 episodes, mean reward -85.875, speed 83.38 f/s
1684636: done 6086 episodes, mean reward -85.902, speed 80.05 f/s
1684718: done 6087 episodes, mean reward -86.211, speed 80.76 f/s
1684823: done 6088 episodes, mean reward -85.938, speed 75.09 f/s
1684966: done 6090 episodes, mean reward -86.002, speed 81.43 f/s
Test done in 5.71 sec, reward 210.317, steps 865
1685164: done 6091 episodes, mean reward -85.590, speed 24.32 f/s
1685274: done 6092 episodes, mean reward -85.545, speed 83.48 f/s
1685510: done 6094 episodes, mean reward -86.026, speed 82.41 f/s
1685613: done 6095 episodes, mean reward -86.156, speed 80.01 f/s
1685817: done 6096 episodes, mean reward -85.892, speed 83.19 f/s
Test done in 6.49 sec, reward 199.001, steps 881
1686034: done 6097 episodes, mean reward -85.734, speed 23.77 f/s
1686142: done 6098 episodes, mean reward -85.650, speed 78.90 f/s
1686401: done 6099 episodes, mean reward -85.344, speed 80.39 f/s
1686834: done 6100 episodes, mean reward -84.192, speed 78.51 f/s
1686957: done 6101 episodes, mean reward -84.377, speed 78.54 f/s
Test done in 6.73 sec, reward 275.555, steps 988
1687101: done 6102 episodes, mean reward -84.249, speed 16.58 f/s
1687199: done 6103 episodes, mean reward -84.834, speed 80.40 f/s
1687333: done 6104 episodes, mean reward -84.565, speed 80.53 f/s
1687430: done 6105 episodes, mean reward -84.656, speed 77.50 f/s
1687717: done 6106 episodes, mean reward -84.146, speed 83.49 f/s
1687946: done 6107 episodes, mean reward -83.716, speed 78.46 f/s
Test done in 6.79 sec, reward 281.575, steps 987
1688097: done 6108 episodes, mean reward -83.664, speed 17.56 f/s
1688202: done 6109 episodes, mean reward -84.131, speed 83.19 f/s
1688317: done 6110 episodes, mean reward -83.992, speed 81.91 f/s
1688800: done 6111 episodes, mean reward -82.729, speed 82.64 f/s
1688911: done 6112 episodes, mean reward -83.254, speed 79.57 f/s
Test done in 6.15 sec, reward 166.436, steps 896
1689127: done 6113 episodes, mean reward -83.126, speed 24.25 f/s
1689295: done 6115 episodes, mean reward -83.541, speed 80.26 f/s
1689593: done 6116 episodes, mean reward -83.491, speed 81.38 f/s
1689732: done 6117 episodes, mean reward -83.500, speed 84.61 f/s
Test done in 6.11 sec, reward 183.664, steps 884
1690137: done 6118 episodes, mean reward -82.842, speed 36.38 f/s
1690453: done 6119 episodes, mean reward -82.156, speed 77.61 f/s
1690641: done 6120 episodes, mean reward -81.710, speed 81.59 f/s
1690872: done 6121 episodes, mean reward -81.516, speed 82.72 f/s
Test done in 7.30 sec, reward 197.054, steps 1045
1691148: done 6122 episodes, mean reward -81.064, speed 26.14 f/s
1691414: done 6123 episodes, mean reward -80.484, speed 81.87 f/s
1691699: done 6124 episodes, mean reward -80.424, speed 82.44 f/s
Test done in 7.08 sec, reward 268.754, steps 1054
1692098: done 6125 episodes, mean reward -79.613, speed 33.55 f/s
Test done in 7.02 sec, reward 186.933, steps 1040
1693005: done 6126 episodes, mean reward -77.384, speed 50.39 f/s
1693175: done 6127 episodes, mean reward -77.120, speed 82.49 f/s
1693521: done 6128 episodes, mean reward -77.207, speed 81.36 f/s
1693647: done 6129 episodes, mean reward -76.978, speed 79.66 f/s
1693779: done 6130 episodes, mean reward -76.902, speed 82.14 f/s
1693885: done 6131 episodes, mean reward -76.826, speed 82.55 f/s
Test done in 6.41 sec, reward 203.505, steps 943
1694173: done 6132 episodes, mean reward -76.153, speed 28.80 f/s
1694380: done 6133 episodes, mean reward -76.056, speed 84.49 f/s
1694660: done 6134 episodes, mean reward -75.311, speed 82.25 f/s
Test done in 6.16 sec, reward 218.917, steps 911
1695050: done 6136 episodes, mean reward -74.895, speed 35.63 f/s
1695143: done 6137 episodes, mean reward -74.793, speed 82.37 f/s
Test done in 6.13 sec, reward 191.113, steps 874
1696089: done 6139 episodes, mean reward -72.376, speed 53.06 f/s
1696200: done 6140 episodes, mean reward -72.192, speed 81.39 f/s
1696648: done 6141 episodes, mean reward -71.191, speed 82.98 f/s
1696744: done 6142 episodes, mean reward -71.135, speed 82.84 f/s
Test done in 2.51 sec, reward 3.721, steps 366
1697146: done 6143 episodes, mean reward -70.236, speed 55.08 f/s
1697681: done 6144 episodes, mean reward -68.960, speed 81.84 f/s
1697762: done 6145 episodes, mean reward -69.101, speed 76.71 f/s
Test done in 6.01 sec, reward 242.103, steps 873
1698058: done 6146 episodes, mean reward -68.316, speed 30.87 f/s
1698336: done 6147 episodes, mean reward -67.766, speed 83.95 f/s
1698555: done 6149 episodes, mean reward -67.910, speed 80.68 f/s
1698639: done 6150 episodes, mean reward -67.823, speed 79.90 f/s
1698741: done 6151 episodes, mean reward -67.783, speed 82.39 f/s
Test done in 6.72 sec, reward 251.636, steps 956
1699040: done 6152 episodes, mean reward -67.038, speed 28.78 f/s
1699277: done 6153 episodes, mean reward -66.433, speed 78.20 f/s
1699438: done 6155 episodes, mean reward -67.015, speed 76.64 f/s
1699836: done 6156 episodes, mean reward -66.444, speed 82.42 f/s
1699929: done 6157 episodes, mean reward -66.526, speed 81.81 f/s
Test done in 6.44 sec, reward 242.702, steps 944
1700080: done 6159 episodes, mean reward -66.760, speed 17.87 f/s
1700297: done 6160 episodes, mean reward -66.395, speed 81.71 f/s
1700383: done 6161 episodes, mean reward -66.571, speed 83.79 f/s
1700464: done 6162 episodes, mean reward -66.943, speed 77.33 f/s
1700863: done 6163 episodes, mean reward -65.932, speed 80.70 f/s
1700992: done 6164 episodes, mean reward -65.939, speed 79.06 f/s
Test done in 5.48 sec, reward 184.954, steps 783
1701219: done 6165 episodes, mean reward -66.087, speed 27.70 f/s
1701324: done 6166 episodes, mean reward -66.041, speed 83.48 f/s
1701413: done 6167 episodes, mean reward -66.020, speed 79.14 f/s
1701511: done 6168 episodes, mean reward -66.199, speed 86.16 f/s
1701610: done 6169 episodes, mean reward -67.089, speed 85.71 f/s
Test done in 6.37 sec, reward 241.388, steps 924
1702066: done 6170 episodes, mean reward -65.786, speed 38.22 f/s
1702159: done 6171 episodes, mean reward -66.044, speed 82.72 f/s
1702258: done 6172 episodes, mean reward -66.224, speed 76.88 f/s
1702406: done 6173 episodes, mean reward -65.906, speed 79.97 f/s
1702518: done 6174 episodes, mean reward -65.725, speed 76.09 f/s
1702741: done 6175 episodes, mean reward -65.468, speed 80.88 f/s
1702861: done 6176 episodes, mean reward -65.969, speed 83.06 f/s
1702959: done 6177 episodes, mean reward -66.486, speed 81.50 f/s
Test done in 6.60 sec, reward 247.838, steps 952
1703206: done 6178 episodes, mean reward -65.723, speed 25.67 f/s
1703436: done 6179 episodes, mean reward -65.124, speed 82.54 f/s
1703635: done 6180 episodes, mean reward -64.966, speed 83.09 f/s
1703879: done 6181 episodes, mean reward -64.439, speed 82.50 f/s
Test done in 6.62 sec, reward 276.216, steps 973
1704045: done 6183 episodes, mean reward -65.833, speed 19.20 f/s
1704253: done 6184 episodes, mean reward -65.479, speed 79.96 f/s
1704414: done 6185 episodes, mean reward -65.195, speed 82.23 f/s
1704672: done 6186 episodes, mean reward -64.689, speed 80.55 f/s
1704841: done 6187 episodes, mean reward -64.328, speed 81.96 f/s
Test done in 6.14 sec, reward 227.053, steps 881
1705683: done 6188 episodes, mean reward -61.900, speed 51.12 f/s
1705775: done 6189 episodes, mean reward -61.669, speed 80.43 f/s
1705924: done 6190 episodes, mean reward -61.392, speed 82.81 f/s
Test done in 6.94 sec, reward 236.379, steps 970
1706069: done 6192 episodes, mean reward -61.753, speed 16.52 f/s
1706279: done 6193 episodes, mean reward -61.180, speed 82.04 f/s
1706522: done 6194 episodes, mean reward -60.774, speed 82.81 f/s
1706611: done 6195 episodes, mean reward -60.936, speed 86.67 f/s
Test done in 6.56 sec, reward 266.307, steps 968
1707334: done 6196 episodes, mean reward -59.391, speed 46.76 f/s
1707575: done 6198 episodes, mean reward -59.515, speed 81.85 f/s
1707674: done 6199 episodes, mean reward -60.026, speed 82.07 f/s
1707906: done 6200 episodes, mean reward -60.561, speed 82.74 f/s
Test done in 6.87 sec, reward 288.753, steps 1023
1708059: done 6201 episodes, mean reward -60.428, speed 17.37 f/s
1708478: done 6203 episodes, mean reward -59.842, speed 81.21 f/s
1708823: done 6204 episodes, mean reward -59.162, speed 84.02 f/s
1708908: done 6205 episodes, mean reward -59.070, speed 81.14 f/s
Test done in 6.78 sec, reward 283.295, steps 1015
1709071: done 6206 episodes, mean reward -59.431, speed 18.32 f/s
1709168: done 6208 episodes, mean reward -60.168, speed 74.34 f/s
1709293: done 6209 episodes, mean reward -59.881, speed 77.39 f/s
1709481: done 6210 episodes, mean reward -59.645, speed 83.51 f/s
1709598: done 6211 episodes, mean reward -60.798, speed 82.68 f/s
1709813: done 6212 episodes, mean reward -60.274, speed 81.55 f/s
Test done in 6.79 sec, reward 278.375, steps 1006
1710035: done 6213 episodes, mean reward -59.766, speed 23.48 f/s
1710158: done 6214 episodes, mean reward -59.530, speed 83.49 f/s
1710246: done 6215 episodes, mean reward -59.530, speed 83.11 f/s
1710375: done 6216 episodes, mean reward -59.899, speed 68.98 f/s
1710483: done 6217 episodes, mean reward -59.946, speed 75.27 f/s
Test done in 6.78 sec, reward 277.614, steps 960
1711059: done 6219 episodes, mean reward -59.696, speed 41.73 f/s
1711170: done 6220 episodes, mean reward -59.910, speed 83.64 f/s
1711304: done 6221 episodes, mean reward -60.185, speed 81.52 f/s
1711575: done 6223 episodes, mean reward -60.880, speed 81.33 f/s
1711838: done 6224 episodes, mean reward -60.536, speed 82.73 f/s
Test done in 6.39 sec, reward 255.791, steps 947
1712691: done 6225 episodes, mean reward -59.183, speed 51.18 f/s
1712959: done 6226 episodes, mean reward -60.744, speed 83.18 f/s
Test done in 6.56 sec, reward 260.296, steps 936
1713295: done 6227 episodes, mean reward -60.127, speed 31.59 f/s
1713389: done 6228 episodes, mean reward -60.907, speed 80.53 f/s
1713630: done 6229 episodes, mean reward -60.504, speed 80.70 f/s
Test done in 6.63 sec, reward 279.499, steps 964
1714096: done 6230 episodes, mean reward -59.608, speed 37.80 f/s
1714227: done 6231 episodes, mean reward -59.534, speed 84.66 f/s
1714336: done 6232 episodes, mean reward -60.175, speed 82.12 f/s
1714593: done 6234 episodes, mean reward -60.740, speed 81.45 f/s
Test done in 6.05 sec, reward 235.442, steps 906
1715404: done 6235 episodes, mean reward -58.744, speed 50.84 f/s
1715936: done 6237 episodes, mean reward -58.581, speed 83.00 f/s
Test done in 6.43 sec, reward 254.402, steps 960
1716348: done 6239 episodes, mean reward -60.373, speed 35.51 f/s
1716619: done 6240 episodes, mean reward -59.921, speed 81.81 f/s
1716711: done 6241 episodes, mean reward -60.756, speed 81.86 f/s
1716992: done 6242 episodes, mean reward -60.116, speed 81.71 f/s
Test done in 5.86 sec, reward 217.773, steps 867
1717047: done 6243 episodes, mean reward -61.231, speed 8.35 f/s
1717137: done 6244 episodes, mean reward -62.666, speed 84.28 f/s
1717523: done 6245 episodes, mean reward -61.703, speed 80.99 f/s
1717961: done 6246 episodes, mean reward -61.349, speed 83.23 f/s
Test done in 4.19 sec, reward 88.839, steps 612
1718207: done 6247 episodes, mean reward -61.378, speed 33.96 f/s
1718343: done 6248 episodes, mean reward -61.222, speed 81.38 f/s
1718428: done 6249 episodes, mean reward -61.448, speed 83.93 f/s
1718575: done 6250 episodes, mean reward -61.169, speed 79.92 f/s
1718663: done 6251 episodes, mean reward -61.115, speed 86.39 f/s
1718905: done 6252 episodes, mean reward -61.390, speed 84.64 f/s
Test done in 5.54 sec, reward 217.543, steps 846
1719022: done 6253 episodes, mean reward -61.693, speed 16.77 f/s
1719451: done 6254 episodes, mean reward -60.490, speed 77.08 f/s
1719582: done 6255 episodes, mean reward -60.408, speed 85.09 f/s
1719747: done 6256 episodes, mean reward -61.138, speed 74.35 f/s
1719947: done 6257 episodes, mean reward -60.777, speed 83.79 f/s
Test done in 5.68 sec, reward 208.801, steps 831
1720005: done 6258 episodes, mean reward -60.617, speed 8.98 f/s
1720433: done 6259 episodes, mean reward -59.320, speed 79.48 f/s
1720596: done 6261 episodes, mean reward -59.875, speed 84.22 f/s
1720938: done 6262 episodes, mean reward -59.085, speed 81.94 f/s
Test done in 5.61 sec, reward 208.925, steps 838
1721055: done 6263 episodes, mean reward -59.934, speed 16.63 f/s
1721146: done 6264 episodes, mean reward -60.045, speed 81.07 f/s
1721251: done 6265 episodes, mean reward -60.429, speed 82.27 f/s
1721363: done 6266 episodes, mean reward -60.446, speed 84.40 f/s
1721477: done 6267 episodes, mean reward -60.407, speed 82.00 f/s
1721565: done 6268 episodes, mean reward -60.390, speed 84.33 f/s
1721663: done 6269 episodes, mean reward -60.335, speed 82.87 f/s
1721989: done 6270 episodes, mean reward -60.769, speed 81.41 f/s
Test done in 6.28 sec, reward 229.331, steps 910
1722051: done 6271 episodes, mean reward -60.837, speed 8.87 f/s
1722225: done 6272 episodes, mean reward -60.587, speed 82.46 f/s
1722321: done 6273 episodes, mean reward -60.852, speed 83.43 f/s
1722497: done 6274 episodes, mean reward -60.722, speed 83.29 f/s
1722587: done 6275 episodes, mean reward -61.143, speed 79.19 f/s
1722711: done 6276 episodes, mean reward -61.211, speed 84.64 f/s
1722846: done 6278 episodes, mean reward -62.078, speed 78.54 f/s
Test done in 5.69 sec, reward 164.792, steps 828
1723285: done 6280 episodes, mean reward -61.939, speed 39.38 f/s
1723549: done 6281 episodes, mean reward -61.885, speed 81.40 f/s
Test done in 6.24 sec, reward 246.297, steps 923
1724387: done 6282 episodes, mean reward -59.575, speed 50.77 f/s
1724553: done 6283 episodes, mean reward -59.312, speed 78.85 f/s
1724661: done 6284 episodes, mean reward -59.799, speed 79.05 f/s
1724820: done 6286 episodes, mean reward -60.902, speed 85.09 f/s
Test done in 6.36 sec, reward 252.357, steps 939
1725019: done 6287 episodes, mean reward -60.668, speed 22.58 f/s
1725423: done 6288 episodes, mean reward -62.187, speed 84.42 f/s
1725575: done 6289 episodes, mean reward -61.952, speed 83.02 f/s
1725677: done 6290 episodes, mean reward -62.080, speed 80.34 f/s
1725819: done 6291 episodes, mean reward -61.846, speed 82.44 f/s
1725992: done 6292 episodes, mean reward -61.466, speed 82.65 f/s
Test done in 4.91 sec, reward 132.513, steps 700
1726158: done 6293 episodes, mean reward -61.633, speed 23.42 f/s
1726534: done 6294 episodes, mean reward -61.353, speed 79.98 f/s
1726691: done 6295 episodes, mean reward -61.060, speed 83.04 f/s
1726826: done 6296 episodes, mean reward -62.747, speed 82.64 f/s
Test done in 5.33 sec, reward 186.387, steps 794
1727219: done 6297 episodes, mean reward -61.834, speed 38.75 f/s
1727308: done 6298 episodes, mean reward -62.185, speed 81.36 f/s
1727394: done 6299 episodes, mean reward -62.201, speed 80.09 f/s
1727525: done 6300 episodes, mean reward -62.520, speed 85.06 f/s
1727635: done 6302 episodes, mean reward -62.713, speed 82.15 f/s
1727897: done 6303 episodes, mean reward -62.852, speed 81.13 f/s
Test done in 5.28 sec, reward 206.032, steps 801
1728125: done 6304 episodes, mean reward -63.276, speed 28.55 f/s
1728227: done 6305 episodes, mean reward -63.177, speed 82.74 f/s
1728420: done 6306 episodes, mean reward -62.833, speed 81.47 f/s
1728682: done 6307 episodes, mean reward -62.198, speed 83.50 f/s
1728910: done 6308 episodes, mean reward -61.654, speed 84.58 f/s
Test done in 6.73 sec, reward 281.197, steps 1006
1729115: done 6310 episodes, mean reward -61.888, speed 22.13 f/s
1729470: done 6311 episodes, mean reward -61.046, speed 83.11 f/s
1729549: done 6312 episodes, mean reward -61.481, speed 77.95 f/s
1729819: done 6314 episodes, mean reward -61.864, speed 76.30 f/s
Test done in 4.99 sec, reward 168.638, steps 743
1730056: done 6316 episodes, mean reward -61.872, speed 29.25 f/s
1730186: done 6317 episodes, mean reward -61.833, speed 80.07 f/s
1730486: done 6318 episodes, mean reward -61.081, speed 79.61 f/s
1730594: done 6319 episodes, mean reward -62.325, speed 78.66 f/s
1730690: done 6320 episodes, mean reward -62.518, speed 82.25 f/s
1730802: done 6321 episodes, mean reward -62.709, speed 80.15 f/s
Test done in 6.01 sec, reward 244.625, steps 922
1731363: done 6322 episodes, mean reward -61.084, speed 43.54 f/s
1731472: done 6323 episodes, mean reward -61.284, speed 83.06 f/s
1731564: done 6324 episodes, mean reward -61.840, speed 80.54 f/s
1731691: done 6325 episodes, mean reward -63.945, speed 83.96 f/s
1731787: done 6326 episodes, mean reward -64.509, speed 84.78 f/s
1731880: done 6327 episodes, mean reward -65.253, speed 72.45 f/s
Test done in 5.73 sec, reward 212.050, steps 862
1732010: done 6328 episodes, mean reward -65.087, speed 17.64 f/s
1732211: done 6329 episodes, mean reward -65.263, speed 81.10 f/s
1732486: done 6331 episodes, mean reward -65.994, speed 83.46 f/s
1732669: done 6332 episodes, mean reward -65.608, speed 81.90 f/s
1732837: done 6333 episodes, mean reward -65.199, speed 81.64 f/s
1732973: done 6334 episodes, mean reward -65.472, speed 84.12 f/s
Test done in 4.62 sec, reward 118.711, steps 679
1733070: done 6335 episodes, mean reward -67.365, speed 16.76 f/s
1733241: done 6337 episodes, mean reward -68.220, speed 78.84 f/s
1733361: done 6338 episodes, mean reward -68.046, speed 79.67 f/s
1733547: done 6339 episodes, mean reward -68.410, speed 82.20 f/s
1733710: done 6341 episodes, mean reward -69.169, speed 83.64 f/s
1733899: done 6343 episodes, mean reward -69.694, speed 75.90 f/s
Test done in 6.65 sec, reward 255.991, steps 990
1734256: done 6344 episodes, mean reward -68.891, speed 32.45 f/s
1734369: done 6345 episodes, mean reward -69.686, speed 85.27 f/s
1734613: done 6346 episodes, mean reward -70.483, speed 83.93 f/s
1734739: done 6347 episodes, mean reward -70.908, speed 79.89 f/s
1734846: done 6348 episodes, mean reward -70.930, speed 81.81 f/s
1734934: done 6349 episodes, mean reward -70.888, speed 80.41 f/s
Test done in 6.01 sec, reward 228.704, steps 908
1735001: done 6350 episodes, mean reward -71.243, speed 9.87 f/s
1735136: done 6352 episodes, mean reward -71.795, speed 81.28 f/s
1735289: done 6354 episodes, mean reward -73.007, speed 82.75 f/s
1735381: done 6355 episodes, mean reward -73.011, speed 79.22 f/s
1735513: done 6356 episodes, mean reward -73.139, speed 82.97 f/s
1735601: done 6357 episodes, mean reward -73.665, speed 79.19 f/s
1735782: done 6359 episodes, mean reward -74.844, speed 83.47 f/s
Test done in 6.39 sec, reward 275.977, steps 975
1736052: done 6360 episodes, mean reward -74.184, speed 28.18 f/s
1736148: done 6361 episodes, mean reward -74.512, speed 82.26 f/s
1736309: done 6363 episodes, mean reward -75.578, speed 82.54 f/s
1736560: done 6365 episodes, mean reward -75.306, speed 83.15 f/s
1736652: done 6366 episodes, mean reward -75.369, speed 82.94 f/s
1736853: done 6368 episodes, mean reward -75.340, speed 80.99 f/s
Test done in 4.79 sec, reward 148.325, steps 697
1737010: done 6370 episodes, mean reward -76.245, speed 23.40 f/s
1737116: done 6371 episodes, mean reward -76.259, speed 83.00 f/s
1737228: done 6372 episodes, mean reward -76.505, speed 84.80 f/s
1737346: done 6373 episodes, mean reward -76.309, speed 84.71 f/s
1737511: done 6374 episodes, mean reward -76.240, speed 83.61 f/s
1737882: done 6376 episodes, mean reward -75.612, speed 84.25 f/s
1737974: done 6377 episodes, mean reward -75.657, speed 83.52 f/s
Test done in 5.92 sec, reward 228.430, steps 896
1738230: done 6378 episodes, mean reward -74.948, speed 27.81 f/s
1738462: done 6380 episodes, mean reward -75.705, speed 77.97 f/s
1738703: done 6382 episodes, mean reward -78.344, speed 82.33 f/s
1738834: done 6383 episodes, mean reward -78.283, speed 83.55 f/s
Test done in 6.30 sec, reward 261.807, steps 929
1739029: done 6385 episodes, mean reward -77.889, speed 22.51 f/s
1739259: done 6387 episodes, mean reward -78.206, speed 80.52 f/s
1739376: done 6388 episodes, mean reward -79.083, speed 78.56 f/s
1739505: done 6389 episodes, mean reward -79.212, speed 82.60 f/s
1739611: done 6390 episodes, mean reward -79.267, speed 75.90 f/s
1739717: done 6391 episodes, mean reward -79.448, speed 86.16 f/s
Test done in 6.79 sec, reward 297.410, steps 1024
Best reward updated: 291.785 -> 297.410
1740077: done 6392 episodes, mean reward -78.968, speed 32.01 f/s
1740260: done 6394 episodes, mean reward -80.085, speed 82.27 f/s
1740365: done 6395 episodes, mean reward -80.164, speed 84.00 f/s
1740481: done 6397 episodes, mean reward -81.565, speed 84.80 f/s
1740599: done 6398 episodes, mean reward -81.376, speed 82.59 f/s
1740740: done 6399 episodes, mean reward -81.140, speed 81.42 f/s
1740911: done 6401 episodes, mean reward -81.273, speed 80.41 f/s
Test done in 6.24 sec, reward 246.647, steps 943
1741085: done 6403 episodes, mean reward -81.965, speed 21.05 f/s
1741232: done 6405 episodes, mean reward -82.472, speed 83.38 f/s
1741383: done 6407 episodes, mean reward -83.481, speed 82.73 f/s
1741524: done 6408 episodes, mean reward -83.768, speed 82.92 f/s
1741718: done 6410 episodes, mean reward -83.858, speed 76.17 f/s
1741815: done 6411 episodes, mean reward -84.803, speed 81.57 f/s
1741937: done 6412 episodes, mean reward -84.623, speed 79.79 f/s
Test done in 6.58 sec, reward 248.132, steps 991
1742037: done 6413 episodes, mean reward -84.505, speed 12.89 f/s
1742158: done 6414 episodes, mean reward -84.709, speed 85.91 f/s
1742320: done 6416 episodes, mean reward -84.770, speed 82.25 f/s
1742540: done 6418 episodes, mean reward -85.565, speed 82.97 f/s
1742771: done 6420 episodes, mean reward -85.629, speed 80.74 f/s
1742905: done 6421 episodes, mean reward -85.482, speed 85.43 f/s
Test done in 7.05 sec, reward 281.830, steps 1039
1743043: done 6422 episodes, mean reward -86.989, speed 15.92 f/s
1743210: done 6423 episodes, mean reward -86.863, speed 83.68 f/s
1743569: done 6425 episodes, mean reward -86.437, speed 81.00 f/s
1743730: done 6427 episodes, mean reward -86.581, speed 83.85 f/s
1743838: done 6428 episodes, mean reward -86.684, speed 82.03 f/s
1743953: done 6430 episodes, mean reward -87.151, speed 82.63 f/s
Test done in 5.84 sec, reward 224.469, steps 908
1744066: done 6431 episodes, mean reward -87.337, speed 15.70 f/s
1744203: done 6432 episodes, mean reward -87.544, speed 84.71 f/s
1744285: done 6433 episodes, mean reward -87.954, speed 79.95 f/s
1744418: done 6434 episodes, mean reward -87.958, speed 82.82 f/s
1744505: done 6435 episodes, mean reward -88.098, speed 83.43 f/s
1744632: done 6436 episodes, mean reward -87.992, speed 80.63 f/s
1744971: done 6437 episodes, mean reward -87.181, speed 82.81 f/s
Test done in 6.53 sec, reward 253.656, steps 961
1745239: done 6438 episodes, mean reward -86.683, speed 27.52 f/s
1745354: done 6439 episodes, mean reward -86.948, speed 81.09 f/s
1745568: done 6440 episodes, mean reward -86.496, speed 81.28 f/s
1745803: done 6441 episodes, mean reward -85.878, speed 79.51 f/s
Test done in 7.01 sec, reward 294.026, steps 1029
1746094: done 6443 episodes, mean reward -85.693, speed 27.87 f/s
1746249: done 6445 episodes, mean reward -86.688, speed 80.67 f/s
1746404: done 6446 episodes, mean reward -86.707, speed 88.24 f/s
1746509: done 6447 episodes, mean reward -86.942, speed 80.08 f/s
1746690: done 6448 episodes, mean reward -86.739, speed 80.96 f/s
1746894: done 6449 episodes, mean reward -86.499, speed 79.52 f/s
Test done in 6.43 sec, reward 253.132, steps 985
1747003: done 6450 episodes, mean reward -86.402, speed 13.98 f/s
1747187: done 6452 episodes, mean reward -86.394, speed 80.39 f/s
1747302: done 6453 episodes, mean reward -86.159, speed 80.38 f/s
1747450: done 6454 episodes, mean reward -85.976, speed 80.31 f/s
1747587: done 6456 episodes, mean reward -86.360, speed 82.83 f/s
1747689: done 6457 episodes, mean reward -86.135, speed 88.37 f/s
1747788: done 6458 episodes, mean reward -85.962, speed 81.77 f/s
1747929: done 6459 episodes, mean reward -85.991, speed 85.83 f/s
Test done in 6.08 sec, reward 252.276, steps 939
1748233: done 6460 episodes, mean reward -86.022, speed 30.65 f/s
1748433: done 6462 episodes, mean reward -85.594, speed 80.20 f/s
1748685: done 6463 episodes, mean reward -84.953, speed 84.22 f/s
1748832: done 6465 episodes, mean reward -85.368, speed 81.60 f/s
Test done in 6.54 sec, reward 270.457, steps 976
1749075: done 6467 episodes, mean reward -85.189, speed 25.84 f/s
1749191: done 6468 episodes, mean reward -85.267, speed 83.60 f/s
1749407: done 6470 episodes, mean reward -85.124, speed 81.77 f/s
1749566: done 6471 episodes, mean reward -84.865, speed 82.41 f/s
1749784: done 6473 episodes, mean reward -85.068, speed 83.78 f/s
1749878: done 6474 episodes, mean reward -85.331, speed 83.15 f/s
1749999: done 6475 episodes, mean reward -85.158, speed 81.26 f/s
Test done in 5.59 sec, reward 195.538, steps 811
1750200: done 6476 episodes, mean reward -85.507, speed 25.06 f/s
1750316: done 6477 episodes, mean reward -85.254, speed 87.62 f/s
1750412: done 6478 episodes, mean reward -85.749, speed 81.76 f/s
1750544: done 6479 episodes, mean reward -85.555, speed 84.25 f/s
1750657: done 6481 episodes, mean reward -85.628, speed 83.69 f/s
1750799: done 6483 episodes, mean reward -86.334, speed 81.60 f/s
1750944: done 6484 episodes, mean reward -86.206, speed 82.80 f/s
Test done in 4.79 sec, reward 139.207, steps 724
1751061: done 6485 episodes, mean reward -86.227, speed 18.85 f/s
1751283: done 6487 episodes, mean reward -86.106, speed 80.50 f/s
1751507: done 6488 episodes, mean reward -85.712, speed 82.73 f/s
1751753: done 6489 episodes, mean reward -85.356, speed 82.52 f/s
1751867: done 6490 episodes, mean reward -85.232, speed 72.07 f/s
1751984: done 6491 episodes, mean reward -85.161, speed 76.50 f/s
Test done in 5.42 sec, reward 175.798, steps 773
1752104: done 6492 episodes, mean reward -85.943, speed 17.46 f/s
1752194: done 6493 episodes, mean reward -85.729, speed 83.10 f/s
1752310: done 6495 episodes, mean reward -86.141, speed 80.28 f/s
1752413: done 6496 episodes, mean reward -85.963, speed 83.98 f/s
1752552: done 6498 episodes, mean reward -86.029, speed 81.16 f/s
1752656: done 6499 episodes, mean reward -86.035, speed 83.48 f/s
1752886: done 6500 episodes, mean reward -85.483, speed 79.94 f/s
Test done in 5.77 sec, reward 192.308, steps 875
1753054: done 6501 episodes, mean reward -85.218, speed 21.42 f/s
1753299: done 6502 episodes, mean reward -84.699, speed 81.24 f/s
1753387: done 6503 episodes, mean reward -84.608, speed 78.95 f/s
1753474: done 6504 episodes, mean reward -84.548, speed 84.74 f/s
1753591: done 6505 episodes, mean reward -84.481, speed 83.35 f/s
1753715: done 6506 episodes, mean reward -84.424, speed 81.69 f/s
1753822: done 6507 episodes, mean reward -84.349, speed 84.09 f/s
Test done in 5.19 sec, reward 170.616, steps 757
1754028: done 6509 episodes, mean reward -84.264, speed 26.73 f/s
1754225: done 6511 episodes, mean reward -84.437, speed 84.15 f/s
1754369: done 6512 episodes, mean reward -84.411, speed 83.94 f/s
1754458: done 6513 episodes, mean reward -84.625, speed 85.15 f/s
1754643: done 6514 episodes, mean reward -84.326, speed 81.41 f/s
1754836: done 6515 episodes, mean reward -83.888, speed 82.76 f/s
1754972: done 6516 episodes, mean reward -83.904, speed 82.01 f/s
Test done in 5.97 sec, reward 244.238, steps 908
1755187: done 6517 episodes, mean reward -83.356, speed 24.83 f/s
1755496: done 6519 episodes, mean reward -83.010, speed 80.16 f/s
1755642: done 6521 episodes, mean reward -83.213, speed 84.87 f/s
1755730: done 6522 episodes, mean reward -83.258, speed 81.46 f/s
1755932: done 6523 episodes, mean reward -83.076, speed 82.98 f/s
Test done in 5.05 sec, reward 167.153, steps 786
1756034: done 6524 episodes, mean reward -83.091, speed 16.31 f/s
1756161: done 6525 episodes, mean reward -83.565, speed 82.16 f/s
1756282: done 6526 episodes, mean reward -83.435, speed 80.24 f/s
1756467: done 6527 episodes, mean reward -83.201, speed 85.08 f/s
1756581: done 6528 episodes, mean reward -83.268, speed 84.77 f/s
1756670: done 6529 episodes, mean reward -83.132, speed 80.52 f/s
1756839: done 6531 episodes, mean reward -83.073, speed 80.74 f/s
Test done in 6.11 sec, reward 225.692, steps 932
1757039: done 6532 episodes, mean reward -82.766, speed 23.53 f/s
1757230: done 6533 episodes, mean reward -82.286, speed 82.50 f/s
1757430: done 6534 episodes, mean reward -82.075, speed 82.60 f/s
1757553: done 6535 episodes, mean reward -82.013, speed 82.81 f/s
1757663: done 6536 episodes, mean reward -82.052, speed 85.54 f/s
1757867: done 6538 episodes, mean reward -83.502, speed 81.76 f/s
Test done in 6.37 sec, reward 282.116, steps 970
1758154: done 6540 episodes, mean reward -83.862, speed 28.98 f/s
1758516: done 6541 episodes, mean reward -83.451, speed 83.77 f/s
1758647: done 6542 episodes, mean reward -83.120, speed 79.38 f/s
1758725: done 6543 episodes, mean reward -83.551, speed 74.67 f/s
1758835: done 6544 episodes, mean reward -83.247, speed 77.19 f/s
1758940: done 6545 episodes, mean reward -83.251, speed 78.15 f/s
Test done in 5.11 sec, reward 170.811, steps 777
1759012: done 6546 episodes, mean reward -83.548, speed 12.03 f/s
1759168: done 6547 episodes, mean reward -83.218, speed 83.62 f/s
1759281: done 6548 episodes, mean reward -83.415, speed 75.45 f/s
1759424: done 6549 episodes, mean reward -83.555, speed 82.73 f/s
1759602: done 6550 episodes, mean reward -83.292, speed 85.39 f/s
1759946: done 6551 episodes, mean reward -82.786, speed 82.85 f/s
Test done in 6.36 sec, reward 277.891, steps 958
1760057: done 6552 episodes, mean reward -82.661, speed 14.43 f/s
1760208: done 6554 episodes, mean reward -83.155, speed 81.74 f/s
1760332: done 6555 episodes, mean reward -83.222, speed 83.25 f/s
1760448: done 6556 episodes, mean reward -83.046, speed 81.78 f/s
1760609: done 6557 episodes, mean reward -82.857, speed 81.99 f/s
1760789: done 6559 episodes, mean reward -82.852, speed 83.50 f/s
1760922: done 6561 episodes, mean reward -83.452, speed 82.63 f/s
Test done in 5.75 sec, reward 247.039, steps 893
1761075: done 6563 episodes, mean reward -84.174, speed 20.02 f/s
1761216: done 6564 episodes, mean reward -83.934, speed 86.43 f/s
1761303: done 6565 episodes, mean reward -83.903, speed 84.84 f/s
1761581: done 6566 episodes, mean reward -83.333, speed 83.71 f/s
1761738: done 6568 episodes, mean reward -83.586, speed 80.62 f/s
1761846: done 6569 episodes, mean reward -83.412, speed 82.96 f/s
1761990: done 6571 episodes, mean reward -84.049, speed 86.61 f/s
Test done in 4.06 sec, reward 95.972, steps 607
1762133: done 6572 episodes, mean reward -83.932, speed 24.65 f/s
1762304: done 6573 episodes, mean reward -83.791, speed 84.54 f/s
1762426: done 6574 episodes, mean reward -83.682, speed 82.17 f/s
1762552: done 6575 episodes, mean reward -83.725, speed 80.83 f/s
1762831: done 6576 episodes, mean reward -84.005, speed 81.21 f/s
1762950: done 6577 episodes, mean reward -84.189, speed 81.99 f/s
Test done in 6.37 sec, reward 280.704, steps 993
1763016: done 6578 episodes, mean reward -84.329, speed 9.22 f/s
1763191: done 6579 episodes, mean reward -84.185, speed 83.42 f/s
1763292: done 6580 episodes, mean reward -84.054, speed 82.04 f/s
1763518: done 6581 episodes, mean reward -83.669, speed 81.99 f/s
1763658: done 6582 episodes, mean reward -83.435, speed 83.77 f/s
1763787: done 6583 episodes, mean reward -83.198, speed 83.02 f/s
Test done in 5.80 sec, reward 232.571, steps 851
Test done in 5.75 sec, reward 213.573, steps 865
1765387: done 6584 episodes, mean reward -82.990, speed 51.72 f/s
1765491: done 6585 episodes, mean reward -83.021, speed 81.60 f/s
1765644: done 6586 episodes, mean reward -82.832, speed 79.26 f/s
1765839: done 6587 episodes, mean reward -82.669, speed 78.19 f/s
1765932: done 6588 episodes, mean reward -83.164, speed 78.15 f/s
Test done in 5.46 sec, reward 211.408, steps 807
1766016: done 6589 episodes, mean reward -83.783, speed 13.10 f/s
1766138: done 6590 episodes, mean reward -83.813, speed 79.97 f/s
1766226: done 6591 episodes, mean reward -83.792, speed 87.97 f/s
1766327: done 6592 episodes, mean reward -83.907, speed 82.80 f/s
1766471: done 6594 episodes, mean reward -83.847, speed 78.21 f/s
1766576: done 6595 episodes, mean reward -83.796, speed 83.47 f/s
1766696: done 6596 episodes, mean reward -83.681, speed 79.80 f/s
1766784: done 6597 episodes, mean reward -83.701, speed 82.00 f/s
1766916: done 6598 episodes, mean reward -83.491, speed 76.79 f/s
Test done in 5.41 sec, reward 190.707, steps 792
1767034: done 6600 episodes, mean reward -84.209, speed 17.24 f/s
1767154: done 6602 episodes, mean reward -85.205, speed 80.62 f/s
1767243: done 6603 episodes, mean reward -85.191, speed 83.83 f/s
1767367: done 6604 episodes, mean reward -85.048, speed 84.65 f/s
1767546: done 6605 episodes, mean reward -84.988, speed 85.35 f/s
1767731: done 6607 episodes, mean reward -85.245, speed 83.87 f/s
1767855: done 6609 episodes, mean reward -85.725, speed 83.31 f/s
Test done in 6.73 sec, reward 298.423, steps 995
Best reward updated: 297.410 -> 298.423
1768027: done 6610 episodes, mean reward -85.224, speed 19.58 f/s
1768300: done 6612 episodes, mean reward -85.287, speed 83.14 f/s
1768432: done 6613 episodes, mean reward -85.146, speed 85.40 f/s
1768604: done 6614 episodes, mean reward -85.441, speed 82.68 f/s
1768718: done 6615 episodes, mean reward -85.727, speed 83.22 f/s
1768956: done 6617 episodes, mean reward -86.540, speed 82.16 f/s
Test done in 6.65 sec, reward 281.247, steps 988
1769155: done 6618 episodes, mean reward -86.081, speed 21.87 f/s
1769287: done 6619 episodes, mean reward -86.550, speed 83.14 f/s
1769381: done 6620 episodes, mean reward -86.468, speed 77.02 f/s
1769538: done 6621 episodes, mean reward -86.191, speed 83.98 f/s
1769676: done 6622 episodes, mean reward -86.239, speed 84.43 f/s
1769822: done 6623 episodes, mean reward -86.458, speed 81.44 f/s
1769968: done 6625 episodes, mean reward -86.963, speed 82.90 f/s
Test done in 5.68 sec, reward 206.883, steps 826
1770052: done 6626 episodes, mean reward -87.037, speed 12.55 f/s
1770210: done 6627 episodes, mean reward -87.190, speed 80.53 f/s
1770347: done 6628 episodes, mean reward -86.973, speed 84.07 f/s
1770513: done 6629 episodes, mean reward -86.843, speed 83.97 f/s
1770611: done 6630 episodes, mean reward -86.740, speed 82.86 f/s
1770706: done 6631 episodes, mean reward -86.803, speed 82.41 f/s
1770843: done 6633 episodes, mean reward -87.828, speed 80.48 f/s
1770980: done 6634 episodes, mean reward -88.005, speed 81.39 f/s
Test done in 4.91 sec, reward 169.526, steps 742
1771141: done 6635 episodes, mean reward -87.774, speed 23.34 f/s
1771278: done 6637 episodes, mean reward -87.805, speed 85.67 f/s
1771405: done 6638 episodes, mean reward -87.809, speed 85.02 f/s
1771528: done 6639 episodes, mean reward -87.557, speed 85.18 f/s
1771610: done 6640 episodes, mean reward -87.984, speed 79.15 f/s
1771745: done 6641 episodes, mean reward -88.630, speed 85.23 f/s
1771889: done 6642 episodes, mean reward -88.724, speed 81.41 f/s
Test done in 6.24 sec, reward 262.939, steps 955
1772072: done 6644 episodes, mean reward -88.815, speed 21.85 f/s
1772230: done 6645 episodes, mean reward -88.674, speed 84.17 f/s
1772325: done 6646 episodes, mean reward -88.601, speed 82.22 f/s
1772523: done 6647 episodes, mean reward -88.462, speed 79.49 f/s
1772639: done 6648 episodes, mean reward -88.476, speed 76.13 f/s
1772851: done 6649 episodes, mean reward -88.235, speed 75.27 f/s
1772947: done 6650 episodes, mean reward -88.421, speed 76.73 f/s
Test done in 7.04 sec, reward 299.953, steps 1030
Best reward updated: 298.423 -> 299.953
1773029: done 6651 episodes, mean reward -88.834, speed 10.19 f/s
1773143: done 6652 episodes, mean reward -88.832, speed 87.98 f/s
Test done in 7.00 sec, reward 270.966, steps 1031
1774007: done 6653 episodes, mean reward -89.828, speed 49.33 f/s
1774252: done 6654 episodes, mean reward -89.134, speed 84.26 f/s
1774375: done 6655 episodes, mean reward -89.099, speed 80.87 f/s
1774512: done 6656 episodes, mean reward -88.910, speed 84.89 f/s
1774804: done 6658 episodes, mean reward -88.951, speed 84.45 f/s
1774926: done 6659 episodes, mean reward -88.973, speed 84.48 f/s
Test done in 7.03 sec, reward 274.392, steps 1052
1775111: done 6660 episodes, mean reward -88.626, speed 20.12 f/s
1775198: done 6661 episodes, mean reward -88.536, speed 82.17 f/s
1775305: done 6662 episodes, mean reward -88.551, speed 81.60 f/s
1775417: done 6663 episodes, mean reward -88.335, speed 81.65 f/s
1775538: done 6664 episodes, mean reward -88.811, speed 80.67 f/s
1775665: done 6665 episodes, mean reward -88.767, speed 83.73 f/s
1775771: done 6666 episodes, mean reward -89.286, speed 84.50 f/s
1775877: done 6667 episodes, mean reward -89.251, speed 78.74 f/s
Test done in 5.51 sec, reward 200.360, steps 825
1776026: done 6668 episodes, mean reward -89.139, speed 19.97 f/s
1776145: done 6669 episodes, mean reward -89.231, speed 82.49 f/s
1776335: done 6670 episodes, mean reward -88.764, speed 85.00 f/s
1776426: done 6671 episodes, mean reward -88.593, speed 84.10 f/s
1776541: done 6672 episodes, mean reward -88.402, speed 85.49 f/s
1776633: done 6673 episodes, mean reward -88.653, speed 81.34 f/s
1776734: done 6674 episodes, mean reward -88.889, speed 78.57 f/s
1776889: done 6675 episodes, mean reward -88.840, speed 81.96 f/s
Test done in 5.21 sec, reward 195.603, steps 790
1777092: done 6676 episodes, mean reward -88.582, speed 26.35 f/s
1777192: done 6677 episodes, mean reward -88.638, speed 81.55 f/s
1777346: done 6678 episodes, mean reward -88.365, speed 85.61 f/s
1777479: done 6679 episodes, mean reward -88.518, speed 82.79 f/s
1777580: done 6680 episodes, mean reward -88.531, speed 81.75 f/s
1777692: done 6681 episodes, mean reward -88.837, speed 80.89 f/s
1777889: done 6683 episodes, mean reward -88.966, speed 83.01 f/s
Test done in 6.32 sec, reward 243.008, steps 954
1778043: done 6684 episodes, mean reward -88.956, speed 18.66 f/s
1778127: done 6685 episodes, mean reward -89.043, speed 82.95 f/s
1778300: done 6687 episodes, mean reward -89.668, speed 84.18 f/s
1778423: done 6688 episodes, mean reward -89.551, speed 82.50 f/s
1778514: done 6689 episodes, mean reward -89.469, speed 82.91 f/s
1778646: done 6690 episodes, mean reward -89.357, speed 84.23 f/s
1778855: done 6692 episodes, mean reward -89.173, speed 83.79 f/s
1778943: done 6693 episodes, mean reward -89.042, speed 85.03 f/s
Test done in 6.46 sec, reward 261.809, steps 973
1779079: done 6695 episodes, mean reward -89.324, speed 16.64 f/s
1779224: done 6697 episodes, mean reward -89.520, speed 81.18 f/s
1779367: done 6699 episodes, mean reward -89.918, speed 82.19 f/s
1779499: done 6700 episodes, mean reward -89.619, speed 82.83 f/s
1779705: done 6702 episodes, mean reward -89.535, speed 80.81 f/s
1779989: done 6704 episodes, mean reward -89.366, speed 81.58 f/s
Test done in 6.85 sec, reward 299.037, steps 1001
1780150: done 6705 episodes, mean reward -89.211, speed 18.26 f/s
1780454: done 6706 episodes, mean reward -88.552, speed 85.01 f/s
1780546: done 6707 episodes, mean reward -88.581, speed 81.10 f/s
1780658: done 6708 episodes, mean reward -88.332, speed 83.85 f/s
1780804: done 6709 episodes, mean reward -88.131, speed 83.04 f/s
1780925: done 6710 episodes, mean reward -88.545, speed 84.83 f/s
Test done in 5.86 sec, reward 214.207, steps 869
1781069: done 6711 episodes, mean reward -88.230, speed 18.97 f/s
1781223: done 6712 episodes, mean reward -88.656, speed 77.83 f/s
1781459: done 6714 episodes, mean reward -88.520, speed 79.98 f/s
1781686: done 6716 episodes, mean reward -88.170, speed 87.49 f/s
1781769: done 6717 episodes, mean reward -88.018, speed 81.91 f/s
1781854: done 6718 episodes, mean reward -88.260, speed 84.64 f/s
1781944: done 6719 episodes, mean reward -88.459, speed 83.33 f/s
Test done in 6.32 sec, reward 247.154, steps 966
1782036: done 6720 episodes, mean reward -88.624, speed 12.49 f/s
1782127: done 6721 episodes, mean reward -89.107, speed 84.48 f/s
1782283: done 6723 episodes, mean reward -89.602, speed 82.41 f/s
1782367: done 6724 episodes, mean reward -89.508, speed 83.98 f/s
1782618: done 6725 episodes, mean reward -88.898, speed 83.62 f/s
1782709: done 6726 episodes, mean reward -88.990, speed 82.73 f/s
1782880: done 6728 episodes, mean reward -89.394, speed 84.24 f/s
Test done in 5.11 sec, reward 168.909, steps 781
1783033: done 6730 episodes, mean reward -89.769, speed 21.93 f/s
1783181: done 6731 episodes, mean reward -89.605, speed 81.27 f/s
1783313: done 6732 episodes, mean reward -89.339, speed 82.48 f/s
1783407: done 6733 episodes, mean reward -89.232, speed 80.18 f/s
1783621: done 6734 episodes, mean reward -88.995, speed 82.08 f/s
Test done in 5.78 sec, reward 195.020, steps 859
1784894: done 6735 episodes, mean reward -89.438, speed 59.62 f/s
Test done in 6.23 sec, reward 235.920, steps 902
1785078: done 6736 episodes, mean reward -89.008, speed 21.62 f/s
1785196: done 6737 episodes, mean reward -88.937, speed 76.65 f/s
1785292: done 6738 episodes, mean reward -88.979, speed 81.29 f/s
1785396: done 6739 episodes, mean reward -89.141, speed 82.96 f/s
1785561: done 6740 episodes, mean reward -88.705, speed 81.76 f/s
1785678: done 6741 episodes, mean reward -88.926, speed 80.53 f/s
1785812: done 6742 episodes, mean reward -88.842, speed 80.90 f/s
1785983: done 6743 episodes, mean reward -88.536, speed 78.11 f/s
Test done in 6.75 sec, reward 300.675, steps 1036
Best reward updated: 299.953 -> 300.675
1786076: done 6744 episodes, mean reward -88.483, speed 11.83 f/s
1786260: done 6745 episodes, mean reward -88.335, speed 83.49 f/s
1786408: done 6746 episodes, mean reward -88.141, speed 78.81 f/s
1786552: done 6748 episodes, mean reward -88.774, speed 80.57 f/s
1786810: done 6749 episodes, mean reward -88.611, speed 82.64 f/s
1786977: done 6750 episodes, mean reward -88.455, speed 84.21 f/s
Test done in 7.08 sec, reward 296.107, steps 1080
1787139: done 6751 episodes, mean reward -88.201, speed 18.04 f/s
1787314: done 6753 episodes, mean reward -87.398, speed 83.41 f/s
1787406: done 6754 episodes, mean reward -87.925, speed 82.61 f/s
1787498: done 6755 episodes, mean reward -87.915, speed 81.98 f/s
1787614: done 6756 episodes, mean reward -88.167, speed 83.86 f/s
1787739: done 6757 episodes, mean reward -87.985, speed 73.42 f/s
1787868: done 6758 episodes, mean reward -88.021, speed 77.36 f/s
Test done in 7.01 sec, reward 273.185, steps 1038
1788046: done 6760 episodes, mean reward -88.373, speed 19.41 f/s
1788266: done 6762 episodes, mean reward -88.524, speed 81.44 f/s
1788360: done 6763 episodes, mean reward -88.511, speed 81.86 f/s
1788486: done 6764 episodes, mean reward -88.019, speed 81.39 f/s
1788645: done 6765 episodes, mean reward -87.892, speed 81.56 f/s
1788761: done 6766 episodes, mean reward -87.958, speed 83.92 f/s
1788965: done 6768 episodes, mean reward -88.066, speed 82.85 f/s
Test done in 6.61 sec, reward 295.101, steps 1013
1789072: done 6769 episodes, mean reward -88.089, speed 13.58 f/s
1789371: done 6771 episodes, mean reward -88.300, speed 84.41 f/s
1789491: done 6772 episodes, mean reward -88.338, speed 80.65 f/s
1789626: done 6773 episodes, mean reward -88.333, speed 81.14 f/s
1789867: done 6775 episodes, mean reward -88.358, speed 81.13 f/s
1789974: done 6776 episodes, mean reward -88.630, speed 84.25 f/s
Test done in 6.52 sec, reward 264.497, steps 972
1790055: done 6777 episodes, mean reward -88.731, speed 10.68 f/s
1790151: done 6778 episodes, mean reward -88.903, speed 77.94 f/s
1790400: done 6779 episodes, mean reward -88.811, speed 83.70 f/s
1790548: done 6780 episodes, mean reward -88.615, speed 81.84 f/s
1790680: done 6781 episodes, mean reward -88.555, speed 83.23 f/s
1790858: done 6782 episodes, mean reward -88.262, speed 85.99 f/s
Test done in 5.52 sec, reward 179.449, steps 794
1791064: done 6784 episodes, mean reward -88.734, speed 25.85 f/s
1791200: done 6785 episodes, mean reward -88.528, speed 78.80 f/s
1791299: done 6786 episodes, mean reward -88.370, speed 86.77 f/s
1791526: done 6787 episodes, mean reward -87.923, speed 82.48 f/s
1791698: done 6788 episodes, mean reward -87.811, speed 85.58 f/s
1791794: done 6789 episodes, mean reward -87.686, speed 80.99 f/s
1791963: done 6790 episodes, mean reward -87.582, speed 83.44 f/s
Test done in 6.17 sec, reward 236.410, steps 926
1792091: done 6791 episodes, mean reward -87.700, speed 16.62 f/s
1792193: done 6792 episodes, mean reward -87.915, speed 80.98 f/s
1792286: done 6793 episodes, mean reward -87.924, speed 85.20 f/s
1792470: done 6795 episodes, mean reward -87.572, speed 83.58 f/s
1792702: done 6796 episodes, mean reward -87.050, speed 83.51 f/s
1792937: done 6798 episodes, mean reward -86.684, speed 84.21 f/s
Test done in 6.78 sec, reward 297.524, steps 1026
1793014: done 6799 episodes, mean reward -86.500, speed 10.08 f/s
1793121: done 6800 episodes, mean reward -86.685, speed 80.36 f/s
1793209: done 6801 episodes, mean reward -86.659, speed 82.12 f/s
1793341: done 6802 episodes, mean reward -86.663, speed 80.96 f/s
1793439: done 6803 episodes, mean reward -86.627, speed 81.30 f/s
1793650: done 6804 episodes, mean reward -86.635, speed 84.20 f/s
1793803: done 6805 episodes, mean reward -86.689, speed 80.40 f/s
1793970: done 6806 episodes, mean reward -86.939, speed 82.88 f/s
Test done in 5.97 sec, reward 231.687, steps 891
1794056: done 6807 episodes, mean reward -86.982, speed 12.35 f/s
1794143: done 6808 episodes, mean reward -87.104, speed 82.27 f/s
1794283: done 6809 episodes, mean reward -87.090, speed 74.83 f/s
1794410: done 6810 episodes, mean reward -86.881, speed 76.91 f/s
1794520: done 6811 episodes, mean reward -87.054, speed 79.36 f/s
1794614: done 6812 episodes, mean reward -86.973, speed 84.15 f/s
1794719: done 6813 episodes, mean reward -86.932, speed 85.21 f/s
1794916: done 6814 episodes, mean reward -86.944, speed 82.27 f/s
Test done in 5.82 sec, reward 224.275, steps 906
1795050: done 6815 episodes, mean reward -86.869, speed 18.04 f/s
1795210: done 6816 episodes, mean reward -86.985, speed 85.45 f/s
1795365: done 6818 episodes, mean reward -87.163, speed 84.85 f/s
1795448: done 6819 episodes, mean reward -87.032, speed 82.81 f/s
1795621: done 6820 episodes, mean reward -86.650, speed 77.99 f/s
1795725: done 6821 episodes, mean reward -86.452, speed 80.73 f/s
1795911: done 6823 episodes, mean reward -86.177, speed 82.16 f/s
Test done in 6.65 sec, reward 258.141, steps 986
1796121: done 6824 episodes, mean reward -85.703, speed 22.92 f/s
1796294: done 6825 episodes, mean reward -86.102, speed 80.89 f/s
1796505: done 6826 episodes, mean reward -85.610, speed 86.20 f/s
1796626: done 6827 episodes, mean reward -85.595, speed 84.18 f/s
Test done in 6.77 sec, reward 295.862, steps 1033
1797036: done 6828 episodes, mean reward -84.541, speed 34.93 f/s
1797425: done 6829 episodes, mean reward -83.756, speed 84.87 f/s
1797548: done 6830 episodes, mean reward -83.740, speed 83.03 f/s
1797766: done 6832 episodes, mean reward -84.088, speed 78.97 f/s
1797868: done 6833 episodes, mean reward -84.066, speed 75.47 f/s
Test done in 6.43 sec, reward 256.484, steps 968
1798006: done 6834 episodes, mean reward -84.303, speed 16.94 f/s
1798117: done 6835 episodes, mean reward -83.977, speed 82.21 f/s
1798219: done 6836 episodes, mean reward -84.386, speed 81.42 f/s
Test done in 6.54 sec, reward 245.483, steps 969
1799819: done 6837 episodes, mean reward -84.376, speed 61.64 f/s
1799970: done 6838 episodes, mean reward -84.244, speed 84.74 f/s
Test done in 5.94 sec, reward 208.237, steps 894
1800190: done 6839 episodes, mean reward -83.777, speed 25.64 f/s
1800427: done 6840 episodes, mean reward -83.542, speed 82.92 f/s
1800653: done 6841 episodes, mean reward -83.056, speed 82.75 f/s
1800804: done 6843 episodes, mean reward -83.557, speed 85.27 f/s
1800965: done 6844 episodes, mean reward -83.363, speed 84.38 f/s
Test done in 6.89 sec, reward 294.434, steps 1033
1801048: done 6845 episodes, mean reward -83.813, speed 10.52 f/s
1801176: done 6846 episodes, mean reward -83.989, speed 84.58 f/s
1801282: done 6847 episodes, mean reward -83.765, speed 84.78 f/s
1801423: done 6848 episodes, mean reward -83.593, speed 86.00 f/s
1801520: done 6849 episodes, mean reward -84.199, speed 81.13 f/s
1801696: done 6850 episodes, mean reward -84.187, speed 81.02 f/s
1801834: done 6851 episodes, mean reward -84.262, speed 85.57 f/s
1801985: done 6852 episodes, mean reward -83.895, speed 79.76 f/s
Test done in 6.43 sec, reward 267.517, steps 984
1802129: done 6853 episodes, mean reward -83.851, speed 17.65 f/s
1802232: done 6854 episodes, mean reward -83.832, speed 86.27 f/s
1802407: done 6856 episodes, mean reward -84.002, speed 83.56 f/s
1802592: done 6857 episodes, mean reward -83.897, speed 83.76 f/s
1802708: done 6858 episodes, mean reward -84.010, speed 79.21 f/s
1802833: done 6859 episodes, mean reward -83.849, speed 83.87 f/s
Test done in 5.88 sec, reward 223.207, steps 867
1803007: done 6861 episodes, mean reward -84.006, speed 22.01 f/s
1803121: done 6862 episodes, mean reward -83.954, speed 80.20 f/s
1803374: done 6864 episodes, mean reward -84.027, speed 83.13 f/s
1803505: done 6865 episodes, mean reward -84.065, speed 81.07 f/s
1803693: done 6867 episodes, mean reward -83.963, speed 84.91 f/s
1803835: done 6868 episodes, mean reward -84.091, speed 83.00 f/s
1803936: done 6869 episodes, mean reward -84.165, speed 87.38 f/s
Test done in 6.90 sec, reward 278.984, steps 1058
1804090: done 6870 episodes, mean reward -83.922, speed 17.46 f/s
1804261: done 6871 episodes, mean reward -83.898, speed 81.26 f/s
1804364: done 6872 episodes, mean reward -83.849, speed 76.33 f/s
1804497: done 6873 episodes, mean reward -83.735, speed 80.95 f/s
1804671: done 6874 episodes, mean reward -83.435, speed 83.91 f/s
1804846: done 6875 episodes, mean reward -83.445, speed 83.42 f/s
1804957: done 6876 episodes, mean reward -83.486, speed 80.17 f/s
Test done in 7.19 sec, reward 291.921, steps 1077
1805021: done 6877 episodes, mean reward -83.488, speed 8.02 f/s
1805204: done 6878 episodes, mean reward -83.450, speed 83.38 f/s
1805326: done 6879 episodes, mean reward -83.598, speed 82.02 f/s
1805584: done 6880 episodes, mean reward -83.310, speed 84.72 f/s
1805672: done 6881 episodes, mean reward -83.468, speed 83.20 f/s
1805759: done 6882 episodes, mean reward -83.697, speed 81.43 f/s
1805863: done 6883 episodes, mean reward -83.486, speed 85.50 f/s
Test done in 6.91 sec, reward 293.759, steps 1062
1806068: done 6885 episodes, mean reward -83.592, speed 21.71 f/s
1806246: done 6886 episodes, mean reward -83.429, speed 82.48 f/s
1806377: done 6887 episodes, mean reward -83.749, speed 81.87 f/s
1806461: done 6888 episodes, mean reward -83.980, speed 78.61 f/s
1806573: done 6889 episodes, mean reward -84.020, speed 81.98 f/s
1806701: done 6890 episodes, mean reward -84.260, speed 81.37 f/s
1806813: done 6891 episodes, mean reward -84.026, speed 85.34 f/s
1806904: done 6892 episodes, mean reward -84.169, speed 85.39 f/s
Test done in 5.64 sec, reward 193.647, steps 855
1807091: done 6893 episodes, mean reward -83.809, speed 23.70 f/s
1807232: done 6894 episodes, mean reward -83.644, speed 81.84 f/s
1807421: done 6895 episodes, mean reward -83.473, speed 79.49 f/s
1807546: done 6896 episodes, mean reward -83.869, speed 82.78 f/s
1807740: done 6897 episodes, mean reward -83.475, speed 82.37 f/s
1807949: done 6898 episodes, mean reward -83.231, speed 78.58 f/s
Test done in 6.78 sec, reward 266.299, steps 1028
1808049: done 6899 episodes, mean reward -83.152, speed 12.57 f/s
1808168: done 6900 episodes, mean reward -83.135, speed 86.72 f/s
1808282: done 6901 episodes, mean reward -83.016, speed 80.01 f/s
1808443: done 6902 episodes, mean reward -83.100, speed 80.29 f/s
1808536: done 6903 episodes, mean reward -83.047, speed 83.70 f/s
1808760: done 6904 episodes, mean reward -83.053, speed 83.36 f/s
1808892: done 6905 episodes, mean reward -83.172, speed 86.23 f/s
Test done in 6.49 sec, reward 269.620, steps 985
1809061: done 6907 episodes, mean reward -83.540, speed 19.87 f/s
1809156: done 6908 episodes, mean reward -83.410, speed 79.04 f/s
1809302: done 6910 episodes, mean reward -83.760, speed 80.43 f/s
1809467: done 6911 episodes, mean reward -83.512, speed 84.68 f/s
1809584: done 6912 episodes, mean reward -83.478, speed 82.02 f/s
1809743: done 6913 episodes, mean reward -83.306, speed 82.75 f/s
1809946: done 6914 episodes, mean reward -83.258, speed 81.22 f/s
Test done in 6.79 sec, reward 299.253, steps 1035
1810007: done 6915 episodes, mean reward -83.371, speed 8.09 f/s
1810140: done 6916 episodes, mean reward -83.570, speed 84.26 f/s
1810336: done 6917 episodes, mean reward -83.203, speed 82.98 f/s
1810540: done 6918 episodes, mean reward -82.865, speed 83.90 f/s
1810822: done 6920 episodes, mean reward -82.889, speed 82.77 f/s
Test done in 7.83 sec, reward 238.415, steps 1117
1811029: done 6921 episodes, mean reward -82.578, speed 20.21 f/s
1811216: done 6923 episodes, mean reward -82.506, speed 83.36 f/s
1811445: done 6924 episodes, mean reward -82.475, speed 78.85 f/s
1811563: done 6925 episodes, mean reward -82.552, speed 81.35 f/s
1811657: done 6926 episodes, mean reward -82.939, speed 79.67 f/s
1811812: done 6927 episodes, mean reward -82.692, speed 81.52 f/s
1811968: done 6928 episodes, mean reward -83.619, speed 83.41 f/s
Test done in 5.80 sec, reward 198.172, steps 846
1812078: done 6929 episodes, mean reward -84.277, speed 15.31 f/s
1812244: done 6931 episodes, mean reward -84.363, speed 84.24 f/s
1812376: done 6933 episodes, mean reward -84.568, speed 85.56 f/s
1812485: done 6934 episodes, mean reward -84.659, speed 81.81 f/s
1812669: done 6935 episodes, mean reward -84.495, speed 82.95 f/s
1812792: done 6936 episodes, mean reward -84.332, speed 81.93 f/s
1812887: done 6937 episodes, mean reward -84.339, speed 84.88 f/s
1812999: done 6938 episodes, mean reward -84.533, speed 85.23 f/s
Test done in 6.55 sec, reward 258.836, steps 976
1813095: done 6939 episodes, mean reward -84.996, speed 12.48 f/s
1813385: done 6940 episodes, mean reward -84.733, speed 83.57 f/s
1813521: done 6941 episodes, mean reward -85.598, speed 80.57 f/s
1813612: done 6942 episodes, mean reward -85.598, speed 83.60 f/s
1813916: done 6944 episodes, mean reward -85.484, speed 85.82 f/s
Test done in 6.87 sec, reward 275.711, steps 1020
1814014: done 6945 episodes, mean reward -85.257, speed 12.14 f/s
1814138: done 6946 episodes, mean reward -85.148, speed 76.76 f/s
1814285: done 6947 episodes, mean reward -84.968, speed 78.17 f/s
1814414: done 6948 episodes, mean reward -84.962, speed 82.43 f/s
1814520: done 6949 episodes, mean reward -84.841, speed 78.00 f/s
1814612: done 6950 episodes, mean reward -85.116, speed 82.30 f/s
1814796: done 6951 episodes, mean reward -85.291, speed 81.71 f/s
1814926: done 6952 episodes, mean reward -85.358, speed 79.45 f/s
Test done in 6.33 sec, reward 256.751, steps 991
1815093: done 6953 episodes, mean reward -85.239, speed 20.03 f/s
1815807: done 6954 episodes, mean reward -85.753, speed 83.88 f/s
1815924: done 6955 episodes, mean reward -85.602, speed 81.77 f/s
Test done in 5.97 sec, reward 248.162, steps 918
1816034: done 6956 episodes, mean reward -85.371, speed 15.18 f/s
1816241: done 6957 episodes, mean reward -85.634, speed 83.12 f/s
1816461: done 6958 episodes, mean reward -85.320, speed 88.79 f/s
1816548: done 6959 episodes, mean reward -85.412, speed 83.64 f/s
1816769: done 6961 episodes, mean reward -85.154, speed 81.95 f/s
Test done in 6.30 sec, reward 247.512, steps 961
1817025: done 6962 episodes, mean reward -84.772, speed 27.19 f/s
1817146: done 6963 episodes, mean reward -84.612, speed 82.64 f/s
1817325: done 6964 episodes, mean reward -84.740, speed 83.80 f/s
1817520: done 6966 episodes, mean reward -84.923, speed 82.68 f/s
1817603: done 6967 episodes, mean reward -85.028, speed 82.55 f/s
1817722: done 6969 episodes, mean reward -85.041, speed 76.06 f/s
1817878: done 6970 episodes, mean reward -85.046, speed 74.08 f/s
Test done in 7.50 sec, reward 265.123, steps 1087
1818358: done 6972 episodes, mean reward -84.405, speed 35.84 f/s
1818517: done 6973 episodes, mean reward -84.357, speed 82.02 f/s
1818661: done 6974 episodes, mean reward -84.499, speed 81.32 f/s
1818770: done 6975 episodes, mean reward -84.557, speed 81.61 f/s
1818927: done 6976 episodes, mean reward -84.399, speed 82.05 f/s
Test done in 7.83 sec, reward 281.352, steps 1009
1819012: done 6977 episodes, mean reward -84.306, speed 9.60 f/s
1819183: done 6979 episodes, mean reward -84.443, speed 80.79 f/s
1819297: done 6980 episodes, mean reward -84.802, speed 78.42 f/s
1819406: done 6981 episodes, mean reward -84.761, speed 70.29 f/s
1819514: done 6982 episodes, mean reward -84.874, speed 63.87 f/s
1819593: done 6983 episodes, mean reward -84.906, speed 64.38 f/s
1819692: done 6984 episodes, mean reward -84.824, speed 69.28 f/s
1819886: done 6985 episodes, mean reward -84.646, speed 74.54 f/s
Test done in 6.89 sec, reward 268.938, steps 973
1820089: done 6986 episodes, mean reward -84.694, speed 21.49 f/s
1820730: done 6987 episodes, mean reward -85.625, speed 76.49 f/s
1820923: done 6989 episodes, mean reward -85.701, speed 82.17 f/s
Test done in 7.07 sec, reward 297.862, steps 1015
1821037: done 6990 episodes, mean reward -85.827, speed 13.29 f/s
1821338: done 6991 episodes, mean reward -86.425, speed 80.75 f/s
1821462: done 6992 episodes, mean reward -86.227, speed 77.21 f/s
1821566: done 6993 episodes, mean reward -86.671, speed 77.91 f/s
1821754: done 6994 episodes, mean reward -86.538, speed 78.87 f/s
1821939: done 6995 episodes, mean reward -86.528, speed 80.93 f/s
Traceback (most recent call last):
  File "K:\Projects\python\DQN\learning\BipedalWalker\train_ddpg.py", line 161, in <module>
    print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
  File "K:\Projects\python\DQN\learning\BipedalWalker\train_ddpg.py", line 51, in test_net
    if done or truncated:
  File "C:\Users\admin\.conda\envs\pytorch-gym\lib\site-packages\torch\nn\modules\module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "C:\Users\admin\.conda\envs\pytorch-gym\lib\site-packages\torch\nn\modules\module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "K:\Projects\python\DQN\learning\BipedalWalker\lib\model.py", line 92, in forward
    return self.net(x)
  File "C:\Users\admin\.conda\envs\pytorch-gym\lib\site-packages\torch\nn\modules\module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "C:\Users\admin\.conda\envs\pytorch-gym\lib\site-packages\torch\nn\modules\module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "C:\Users\admin\.conda\envs\pytorch-gym\lib\site-packages\torch\nn\modules\container.py", line 217, in forward
    input = module(input)
  File "C:\Users\admin\.conda\envs\pytorch-gym\lib\site-packages\torch\nn\modules\module.py", line 1511, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
  File "C:\Users\admin\.conda\envs\pytorch-gym\lib\site-packages\torch\nn\modules\module.py", line 1520, in _call_impl
    return forward_call(*args, **kwargs)
  File "C:\Users\admin\.conda\envs\pytorch-gym\lib\site-packages\torch\nn\modules\linear.py", line 116, in forward
    return F.linear(input, self.weight, self.bias)
KeyboardInterrupt

Process finished with exit code -1
