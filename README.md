# The result of the experiments

## K-Fold only use train sets and validation sets
 The MUTAG dataset was used with 10-fold cross-validation. Using the method in sklearn, random_state is fixed to 1. The highest validation accuracy is extracted from each fold as the accuracy of that fold, and finally the model's accuracy is calculated by averaging.
## Update 2023/9/13
1. Do all the experiments on MUTAG, and create tables.  :fast_forward:
2. Using faster kernel. :fast_forward: 
3. Early stop
4. Trying different attentions.
5. Positional encoding
6. Multi head attention

## MUTAG 2023/9/13 
<font color=Yellow>Bold plus italic is the highest accuracy, bold is the second highest, and italic is the third highest.</font>

## train_val_test split
### SP
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 |0.8167+/-0.0334<br>4727s |0.8056+/-0.0354<br>5998s |0.7944+/-0.0324<br>7956s |0.8222+/-0.0292<br>10307s |0.8222+/-0.0270<br>12418s |
| 2 |0.8222+/-0.0270<br>4824s |0.8333+/-0.0192<br>6121s |0.7889+/-0.0322<br>8034s | 0.8086+/-0.0288<br>10338s|0.8333+/-0.0236<br>12535s |
| 3 |0.8444+/-0.0350<br>4829s |0.8056+/-0.0307<br>6157s |0.8278+/-0.0214<br>8150s |0.8000+/-0.0335<br>10602s |0.8444+/-0.0246<br>12677s |
| 4 |0.8278+/-0.0299<br>4933s |0.8167+/-0.0249<br>6299s |0.8000+/-0.0285<br>8297s |0.8000+/-0.0335<br>10610s | 0.8222+/-0.0258<br>12719s|
| 5 |0.7889+/-0.0375<br>5042s |0.8333+/-0.0283<br>6454s |0.8000+/-0.0335<br>8426s |0.8167+/-0.0249<br>10639s |0.8444+/-0.0246<br>12856s |
| 6 |0.8389+/-0.0254<br>5198s |**0.8556+/-0.0211<br>6476s** |0.8000+/-0.0285<br>8560s |0.8056+/-0.0345<br>10764s |0.8278+/-0.0214<br>12978s |


### WL_GPU
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 |0.8389+/-0.0288<br>5663s |0.8278+/-0.0299<br>6275s |0.8222+/-0.0281<br>6814s |0.8444+/-0.0219<br>7117s |0.8500+/-0.0177<br>7703s |
| 2 |0.8500+/-0.0261<br>5830s|0.8333+/-0.0248<br>6200s |0.8389+/-0.0266<br>6730s |0.8333+/-0.0272<br>7297s |0.8278+/-0.0266<br>7693s |
| 3 |0.8611+/-0.0226<br>5869s |0.8333+/-0.0248<br>6524s |0.8444+/-0.0246<br>6941s |0.8444+/-0.0281<br>7522s | 0.8444+/-0.0205<br>7854s|
| 4 |**0.8667+/-0.0238<br>6003s** |0.8389+/-0.0254<br>6482s |0.8333+/-0.0314<br>6968s |0.8111+/-0.0306<br>7508s |0.8278+/-0.0288<br>8023s |
| 5 |0.8500+/-0.0273<br>6032s |0.8222+/-0.0322<br>6516s|0.8556+/-0.0196<br>7073s |0.8333+/-0.0222<br>7703s |0.8333+/-0.0208<br>8200s |
| 6 |**0.8667+/-0.0196<br>6150s** |0.8611+/-0.0180<br>6648s |0.8444+/-0.0258<br>7149s | 0.8278+/-0.0277<br>7637s|0.8333+/-0.0208<br>8277s |
### WL_GPU Parallel and precomputing
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 |0.8722+/-0.0223<br>130s |0.8556+/-0.2108<br>183s |0.8500+/-0.0193<br>198s |0.8444+/-0.0246<br>198s |0.8500+/-0.0236<br>199s |
| 2 |0.8611+/-0.0239<br>167s |0.8444+/-0.0219<br>221s |0.8111+/-0.0316<br>235s |0.8444+/-0.0258<br>235s |0.8389+/-0.0299<br>237s |
| 3 |0.8611+/-0.0239<br>206s |0.8222+/-0.0233<br>258s |0.8333+/-0.0261<br>272s |0.8222+/-0.0331<br>271s |0.8389+/-0.0277<br>278s |
| 4 |0.8444+/-0.0292<br>246s |0.8444+/-0.0205<br>296s |0.8500+/-0.0261<br>309s |0.8167+/-0.0305<br>309s |0.8389+/-0.0214<br>315s |
| 5 |0.8667+/-0.0238<br>284s |0.8222+/-0.0258<br>330s |0.8278+/-0.0288<br>348s |0.8389+/-0.0328<br>345s |0.8333+/-0.0283<br>348s |
| 6 |***0.8722+/-0.0177<br>318s*** |0.8333+/-0.0236<br>367s |0.8167+/-0.0261<br>385s |0.8389+/-0.0266<br>384s |0.8389+/-0.0299<br>384s |
### Graphlet 
sample 3 nodes graphlets
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 |None | |0.8222+/-0.0270<br>12331s |0.8222+/-0.0246<br>15531s |0.8167+/-0.0294<br>18601s |
| 2 |None |0.8278+/-0.0228<br>8644s |0.8222+/-0.0292<br>12524s |0.8167+/-0.0305<br>15646s|0.8111+/-0.0326<br>18817s |
| 3 |None | |0.8389+/-0.0214<br>12589s |0.8500+/-0.0236<br>15976s |0.8056+/-0.0297<br>18679s |
| 4 |None |0.8389+/-0.0242<br>8828s |0.8278+/-0.0288<br>12620s |0.8278+/-0.0228<br>16023s |0.8333+/-0.0283<br>19165s |
| 5 |None |0.8333+/-0.0222<br>8911s |0.8111+/-0.0251<br>12567s |0.8222+/-0.0331<br>16020s |0.8389+/-0.0228<br>18577s |
| 6 |None |**0.8500+/-0.0223<br>9060s** |0.8500+/-0.0209<br>12914s |0.8222+/-0.0233<br>15893s |0.8278+/-0.0337<br>19063s |



## PTC
### WL_GPU Parallel and precomputing
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 |0.5941+/-0.0275<br>383s |0.5882+/-0.0228<br>1799s |0.5971+/-0.0208<br>3753s |0.5676+/-0.0276<br>4176s |0.5882+/-0.0300<br>3462s |
| 2 |0.5735+/-0.0187<br>447s |0.6206+/-0.0237<br>1864s |0.5912+/-0.0244<br>3805s |0.5971+/-0.0253<br>4220s |0.5882+/-0.0191<br>3518s |
| 3 |0.5834+/-0.0199<br>510s |***0.6382+/-0.0220<br>1930s*** |0.6206+/-0.0214<br>3873s |0.5882+/-0.0212<br>4279s |0.5676+/-0.0243<br>3579s |
| 4 |0.5735+/-0.0196<br>576s |0.5941+/-0.0155<br>1988s |0.6118+/-0.0215<br>3941s |0.6029+/-0.0270<br>4356s |0.5824+/-0.0256<br>3643s |
| 5 |0.6088+/-0.0224<br>648s |0.6206+/-0.0121<br>2061s |0.5824+/-0.0242<br>4002s |0.5912+/-0.0258<br>4410s |0.5412+/-0.0349<br>3709s |
| 6 |0.5676+/-0.0250<br>704s |0.6235+/-0.0215<br>2125s |0.6147+/-0.0169<br>4071s |0.5824+/-0.0199<br>4472s |0.5647+/-0.0269<br>3766s |
### WL_GPU Parallel and precomputing(k=3)
batch_size = 64

LR:0.001  StepLR(optimizer, step_size=25, gamma=0.5)

epochs:300 and get the best model from the last 50 epochs with the highest validation accuracy.

| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 |0.5706+/-0.0280<br>465s |0.5794+/-0.0212<br>708s |0.6088+/-0.0232<br>762s |0.6206+/-0.0339<br>772s |0.5912+/-0.0344<br>752s |
| 2 |0.5676+/-0.0351<br>604s |0.6206+/-0.0204<br>835s |0.5941+/-0.0256<br>886s |0.5765+/-0.0300<br>895s |0.5647+/-0.0316<br>876s |
| 3 |0.6088+/-0.0266<br>733s |0.5912+/-0.0323<br>969s |0.5588+/-0.0224<br>1009s |0.6059+/-0.0209<br>1018s |0.5588+/-0.0208<br>1003s |
| 4 |0.5853+/-0.0265<br>849s |***0.6235+/-0.0275<br>1087s ***|0.5382+/-0.0294<br>1129s |0.5471+/-0.0236<br>1144s |0.5412+/-0.0280<br>1132s |
| 5 |0.5735+/-0.0229<br>985s |0.6029+/-0.0298<br>1204s |0.5824+/-0.0180<br>1268s |0.5471+/-0.0264<br>1272s |0.5676+/-0.0228<br>1250s |
| 6 |0.5559+/-0.0339<br>1116s |0.6029+/-0.0247<br>1326s |0.5882+/-0.0288<br>1390s |0.5471+/-0.0325<br>1385s |0.5676 +/- 0.0333<br>1371s |

## NCI1

### WL_GPU Parallel and precomputing
batch_size = 64

LR:0.001  StepLR(optimizer, step_size=25, gamma=0.5)

epochs:300 and get the best model from the last 50 epochs with the highest validation accuracy.

| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |


## PROTEINS
batch_size = 32

LR:0.001  StepLR(optimizer, step_size=25, gamma=0.5)

epochs:300 and get the best model from the last 50 epochs with the highest validation accuracy.


### WL_GPU Parallel and precomputing
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 |0.7468+/-0.0120<br>6762s |0.7459+/-0.0118<br>30317s | | | |
| 2 |0.7441+/-0.0130<br>6635s |0.7459+/-0.0112<br>30937s | | | |
| 3 |0.7423+/-0.0122<br>6964s | | | | |
| 4 |0.7414+/-0.0113<br>7300s | | | | |
| 5 |0.7450+/-0.0110<br>7625s | | | | |
| 6 |0.7468+/-0.0115<br>8003s | | | | |
## GraphiT method test 
SP MUTAG 0.8444 +/- 0.0233 9850s

SP PTC 0.6059 +/- 0.0217 10845s

However, comparing with the GraphiT results, although our results for validation and test accuracy are not very bad, the train loss is still very high and it is difficult to reach near 0 case in GraphiT. And the validation loss curve stabilizes and does not move at the end.


