#### Test 
###### Train a MLP on mnist dataset with two concurrent processes and one shared model

#### Parameters
batch_size: 10 <br>
loss: cross_entropy <br>
optimiser: SGD <br>
activation: leaky_relu <br>
 
#### Architecture
MLP <br>
[784] - [512] - [512] - [10]
 
#### Hardware
Mac - Intel Core i7


#### Result
```Process : 46680 | Epoch 5 Batch : 4600 Loss: 0.00761935580521822
Process : 46679 | Epoch 5 Batch : 4600 Loss: 0.0011240231106057763
Process : 46680 | Epoch 5 Batch : 4800 Loss: 0.0025705136358737946
Process : 46679 | Epoch 5 Batch : 4800 Loss: 0.0190707016736269
Process : 46680 | Epoch 5 Batch : 5000 Loss: 0.002591111697256565
Process : 46679 | Epoch 5 Batch : 5000 Loss: 2.118285556207411e-05
Process : 46680 | Epoch 5 Batch : 5200 Loss: 0.002402252983301878
Process : 46679 | Epoch 5 Batch : 5200 Loss: 0.016756201162934303
Process : 46680 | Epoch 5 Batch : 5400 Loss: 0.00024294503964483738
Process : 46679 | Epoch 5 Batch : 5400 Loss: 6.198154005687684e-05
Process : 46680 | Epoch 5 Batch : 5600 Loss: 0.02597472071647644
Process : 46679 | Epoch 5 Batch : 5600 Loss: 0.00348040834069252
Process : 46680 | Epoch 5 Batch : 5800 Loss: 0.04024457186460495
Process : 46679 | Epoch 5 Batch : 5800 Loss: 0.2212938815355301
Test set: Average loss: 0.0058858621675564335 | Accuracy: 0.9825
```

#### Verification
