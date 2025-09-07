# Vapoursynth-Temporal-Frequency-Denoise-CUDA
時間軸で動く高／中／低　周波数をデノイズします。

```

clip= core.cuda_TMP.TemporalDenoiseCUDA( clip,      radius=6,     
    alphaLow=1, 
    alphaMid=3,  
    alphaHigh=1.0, 
 strength=1
)
 ```
