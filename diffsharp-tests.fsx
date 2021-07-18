//#r "nuget: DiffSharp-lite, 1.0.0-preview-987646120"
#r "nuget: DiffSharp-lite, 1.0.0-preview-783523654"
#I @"C:\Users\cybernetic\anaconda3\Lib\site-packages\torch\lib" 

open DiffSharp

dsharp.config(dtype=Dtype.Float32, device=Device.CPU, backend=Backend.Torch)

dsharp.ones(10)
let t = dsharp.tensor [ 0 .. 10 ]

t + t 

t * 2f
2 * t.[0..2]

open DiffSharp.Compose

dsharp.tensor [[1..4];[1..4]]
|> dsharp.unsqueeze 0
   
let t2 = dsharp.tensor [[1..4]; [15;6;7;8]]
t2.mean(-1)

let t3 = dsharp.tensor [[[1..4]; [15;6;7;8]];[[11;2;3;4]; [151;6;7;8]]]

t3.mean(-1)

let lnorm (x:Tensor) = 
    let mean = x.mean(-1)
    let std = x.stddev(-1)

    let a = dsharp.ones(4)
    let b = dsharp.zeros(4)
    let eps = 1e-6

    a * (x - mean) / (std + eps) + b

lnorm (dsharp.tensor [1..4])