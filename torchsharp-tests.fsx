//#r "nuget: TorchSharp, 0.91.52681" 
#r "nuget: TorchSharp, 0.91.52604" 
#I @"C:\Users\cybernetic\anaconda3\Lib\site-packages\torch\lib" 

open System.Runtime.InteropServices
open TorchSharp
open TorchSharp.Tensor

Torch.LoadNativeBackend(true)

module NumericLiteralR =
    let FromZero() = 0f.ToScalar()
    let FromOne() = 1f.ToScalar()
    let FromInt32 (n:int) = n.ToScalar()

type TorchTensor with 
    member t.To(d:Device) = t.``to``(d) 
    member t.GetSlice(startIdx1:int64 option, endIdx1 : int64 option) =
        let dim = t.shape
        if dim.Length <> 1 then failwith "Dimensions must be = 1"

        let sidx1 = defaultArg startIdx1 0L 
        let endidx1 = defaultArg endIdx1 dim.[0]

        t.[TorchTensorIndex.Slice(sidx1, endidx1)]        

    member t.GetSlice(startIdx1:int64 option, endIdx1 : int64 option,startIdx2:int64 option, endIdx2 : int64 option) =
        let dim = t.shape
        if dim.Length <> 2 then failwith "Dimensions must be = 2"

        let sidx1 = defaultArg startIdx1 0L
        let sidx2 = defaultArg startIdx2 0L
        let endidx1 = defaultArg endIdx1 dim.[0] 
        let endidx2 = defaultArg endIdx2 dim.[1]  
        t.[TorchTensorIndex.Slice(sidx1, endidx1),TorchTensorIndex.Slice(sidx2, endidx2 + 1L)]
        
module Tensor =
    let toArray (toType) (t : TorchTensor) = 
        match t.shape with
        | [| w |] ->
            [| for i in 0L..w - 1L ->  t.[i] |> toType|]
        | _ -> failwith "Incompatible dimensions" 

    let toJaggedArray2D (toType) (t : TorchTensor) = 
        match t.shape with
        | [| w; h|] ->
            [| for i in 0L..w - 1L ->
                [| for j in 0L..h - 1L -> t.[i, j] |> toType|] |]
        
        | [| 1L; w; h |] ->
            [| for i in 0L..w - 1L ->
                [| for j in 0L..h - 1L -> t.[0L, i, j] |> toType|] |]
        
        | [| 1L; 1L; w; h |] ->
            [| for i in 0L..w - 1L ->
                [| for j in 0L..h - 1L -> t.[0L, 0L, i, j] |> toType|] |]
        | _ -> failwith "Incompatible dimensions"

    let toArray2D toType t = array2D (toJaggedArray2D toType t)
     
TorchSharp.Torch.IsCudaAvailable()
  
let t = Tensor.Float32Tensor.from([|1f..10f|])//.To(Device.CUDA)

t.[0L] <- 5L.ToTorchTensor()

t.[0L] |> float32

(t + t).unsqueeze(0L) |> Tensor.toArray2D float32

let ts = t.[..3L]

t.[..3L] |> Tensor.toArray float32

(2R * t + t).To(Device.CPU) |> Tensor.toArray float32

open TorchSharp.NN
open type TorchSharp.NN.Modules

Float32Tensor.arange(0R, 10R, 2R) |> Tensor.toArray float32

 
let m =
    Sequential(("A", Linear(2L, 2L) :> Module), ("B", Linear(2L, 2L) :> Module))

t   --> Linear(10L, 2L)
    --> Linear(2L, 2L)
