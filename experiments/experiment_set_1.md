#### Test 
###### Train a MLP on mnist dataset with concurrent processes and one shared model

#### Parameters
loss: cross_entropy <br>
optimiser: SGD <br> 
&nbsp;&nbsp;&nbsp;&nbsp;lr=0.001 <br> 
&nbsp;&nbsp;&nbsp;&nbsp;momentum=0.9 <br>
activation: leaky_relu <br>
epochs: 5

 
#### Architecture
MLP <br>
[784] - [512] - [512] - [10]
 
#### Hardware
Mac - Intel Core i7


#### Results

|No of Processes|Batch Size| Image Transformations | Accuracy | 
|:-------------:|:--------:|:---------------------:|:--------:|
|      1        |   5      |         None          |  0.9807  |
|      1        |  10      |         None          |  0.9760  |
|      5        |  10      |         None          |  0.9825  |
|      5        | 100      |         None          |  0.9581  |
|      5        |   5      |         None          |  0.9836  |
|      5        |   4      |         None          |  0.9854  |
|      5        |   6      |         None          |  0.9853  |
|      10       |   1      |         None          |  0.9795  |
|      2        |   2      |         None          |  0.9821  |
|      5        |   5      |   Affine -15 +15      |  0.9891  |
|      7        |   5      |   Affine -15 +15      |  0.9903  |
|      7        |   5      |Elastic(Alpha 40,Sigma 2), Affine -15 +15|  0.9924  |


#### Architecture
MLP <br>
[784] - [256] - [256] - [10]

|No of Processes|Batch Size| Image Transformations | Accuracy | 
|:-------------:|:--------:|:---------------------:|:--------:|
|      7        |   5      | Elastic(Alpha 40,Sigma 2), Affine -15 +15|  0.9898 -> 0.991  |
|      1        |   5      | Elastic(Alpha 40,Sigma 2), Affine -15 +15|   0.9833  |
|      1        |   5      | None |   0.9785  |
|      1        |   100      | None |   0.9376  |
|      1        |   200      | None |   0.9197  |



#### Architecture
MLP <br>
[784] - [10] - [10] - [10]

|No of Processes|Batch Size| Image Transformations | Accuracy | 
|:-------------:|:--------:|:---------------------:|:--------:|
|      7        |   5      | Elastic(Alpha 40,Sigma 2), Affine -15 +15|  0.9292  |
|      1        |   5      | Elastic(Alpha 40,Sigma 2), Affine -15 +15|  0.9254  |
|      1        |   5      | None|  0.9198  |
|      1        |   100    | None|  0.9078  |
|      1        |   200    | None|  0.8836  |
