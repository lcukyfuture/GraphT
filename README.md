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
### WL_GPU Parallel and precomputing(k=3)
batch_size = 64

LR:0.0003 warm_up 10 epochs 

epochs:250 and get the best model from the last 70 epochs with the highest validation accuracy.
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 |**0.8722+/-0.0223<br>130s** |0.8556+/-0.2108<br>183s |0.8500+/-0.0193<br>198s |0.8444+/-0.0246<br>198s |0.8500+/-0.0236<br>199s |
| 2 |0.8611+/-0.0239<br>167s |0.8444+/-0.0219<br>221s |0.8111+/-0.0316<br>235s |0.8444+/-0.0258<br>235s |0.8389+/-0.0299<br>237s |
| 3 |0.8611+/-0.0239<br>206s |0.8222+/-0.0233<br>258s |0.8333+/-0.0261<br>272s |0.8222+/-0.0331<br>271s |0.8389+/-0.0277<br>278s |
| 4 |0.8444+/-0.0292<br>246s |0.8444+/-0.0205<br>296s |0.8500+/-0.0261<br>309s |0.8167+/-0.0305<br>309s |0.8389+/-0.0214<br>315s |
| 5 |0.8667+/-0.0238<br>284s |0.8222+/-0.0258<br>330s |0.8278+/-0.0288<br>348s |0.8389+/-0.0328<br>345s |0.8333+/-0.0283<br>348s |
| 6 |***0.8722+/-0.0177<br>318s*** |0.8333+/-0.0236<br>367s |0.8167+/-0.0261<br>385s |0.8389+/-0.0266<br>384s |0.8389+/-0.0299<br>384s |
### WL_GPU Parallel and precomputing(k=3)
batch_size = 64

LR:0.001  StepLR(optimizer, step_size=25, gamma=0.5)

epochs:300 and get the best model from the last 50 epochs with the highest validation accuracy.
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 |**0.8611+/-0.0196<br>241s** |0.8444+/-0.0205<br>292s |0.8556+/-0.0179<br>315s |0.8611+/-0.018<br>314s |0.8278+/-0.0242<315s> |
| 2 |0.8556+/-0.0196<br>307s |0.8389+/-0.0242<br>376s |0.8500+/-0.0158<br>379s |0.8500+/-0.0158<br>383s |0.8444+/-0.0189<br>385s |
| 3 |0.8611+/-0.0196<br>379s |0.8333+/-0.0248<br>447s |0.8167+/-0.0209<br>456s |0.8500+/-0.0193<br>459s |0.8500+/-0.0236<br>462s |
| 4 |0.8500+/-0.0177<br>451s |0.8389+/-0.0242<br>513s |0.8333+/-0.0236<br>532s |0.8444+/-0.0153<br>538s |0.8389+/-0.0214<br>541s |
| 5 |***0.8667+/-0.0161<br>515s*** |0.8278+/-0.0254<br>582s |0.8333+/-0.0272<br>606s |0.8333+/-0.0222<br>607s |0.8444+/-0.0153<br>611s |
| 6 |0.8500+/-0.0158<br>583s |0.8444+/-0.0233<br>655s |0.8222+/-0.0189<br>677s |0.8333+/-0.0208<br>682s |0.8500+/-0.0158<br>678s |
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
### WL_GPU Parallel and precomputing(k=5)
batch_size = 128

LR = 0.0003 warm up 10 epochs

epochs: 250 and get the best model from the last 70 epochs with the highest validation accuracy.
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
| 4 |0.5853+/-0.0265<br>849s |***0.6235+/-0.0275<br>1087s***|0.5382+/-0.0294<br>1129s |0.5471+/-0.0236<br>1144s |0.5412+/-0.0280<br>1132s |
| 5 |0.5735+/-0.0229<br>985s |0.6029+/-0.0298<br>1204s |0.5824+/-0.0180<br>1268s |0.5471+/-0.0264<br>1272s |0.5676+/-0.0228<br>1250s |
| 6 |0.5559+/-0.0339<br>1116s |0.6029+/-0.0247<br>1326s |0.5882+/-0.0288<br>1390s |0.5471+/-0.0325<br>1385s |0.5676 +/- 0.0333<br>1371s |

## NCI1

### WL_GPU Parallel and precomputing(k=3)
batch_size = 64

LR:0.001  StepLR(optimizer, step_size=25, gamma=0.5)

epochs:300 and get the best model from the last 50 epochs with the highest validation accuracy.

| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 |0.7109+/-0.0089<br>5505s |***0.7139+/-0.0097<br>15770s*** | | | |
| 2 |0.7078+/-0.0101<br>6993s |0.7058+/-0.0086<br>17310s | | | |
| 3 |0.6900+/-0.0119<br>8553s |0.7039+/-0.01221<br>18979s | | | |
| 4 |0.6889+/-0.0111<br>10172s |0.6949+/-0.0116<br>20422s | | | |
| 5 |0.6701+/-0.0114<br>11678s |0.6813+/-0.0145<br>21393s | | | |
| 6 |0.6783+/-0.0142<br>13052s |0.6540+/-0.0127<br>22849s | | | |


## PROTEINS
batch_size = 32

LR:0.001  StepLR(optimizer, step_size=25, gamma=0.5)

epochs:300 and get the best model from the last 50 epochs with the highest validation accuracy.


### WL_GPU Parallel and precomputing(k=3)
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 |0.7468+/-0.0120<br>6762s |0.7459+/-0.0118<br>30317s | | | |
| 2 |0.7441+/-0.0130<br>6635s |0.7459+/-0.0112<br>30937s | | | |
| 3 |0.7423+/-0.0122<br>6964s |0.7432+/-0.0131<br>31408s | | | |
| 4 |0.7414+/-0.0113<br>7300s | | | | |
| 5 |0.7450+/-0.0110<br>7625s | | | | |
| 6 |0.7468+/-0.0115<br>8003s | | | | |
### WL_GPU Parallel and precomputing(batch_size=16, k=3)
batch_size = 16
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 |***0.7486+/-0.0146<br>7107s*** |0.7468+/-0.0118<br>21671s | | | |
| 2 |0.7414+/-0.0125<br>7808s |0.7432+/-0.0131<br>22337s | | | |
| 3 |0.7378+/-0.0113<br>8467s |0.7477+/-0.0140<br>23017s | | | |
| 4 |0.7396+/-0.0136<br>9138s |0.7477+/-0.0132<br>23578s | | | |
| 5 |0.7432+/-0.0116<br>9816s |0.7477+/-0.0123<br>24318s | | | |
| 6 |0.7387+/-0.0122<br>10500s |0.7333+/-0.0151<br>24943s | | | |
## GraphiT method test 
### Reuslt template
| Layers/Hops | 1 | 2 | 3 | 4 | 5 |
| :-----:| :------: | :------: | :----: | :---:| :---:|
| 1 | | | | | |
| 2 | | | | | |
| 3 | | | | | |
| 4 | | | | | |
| 5 | | | | | |
| 6 | | | | | |

SP MUTAG 0.8444 +/- 0.0233 9850s

SP PTC 0.6059 +/- 0.0217 10845s

However, comparing with the GraphiT results, although our results for validation and test accuracy are not very bad, the train loss is still very high and it is difficult to reach near 0 case in GraphiT. And the validation loss curve stabilizes and does not move at the end.


## WL_GPU document
The WL_GPU framework is divided into WL_Conv and WL. WL_Conv handles specific WL convolutions, while WL manages the entire multi-layered WL hashing process. When compared with the CPU-based method (GraKeL), enhancements in the GPU method improve computational efficiency. Datasets are batch-processed and integrated into the dataloader during training. In each WL iteration, Unique labels or colors within subgraphs are enumerated, and color histograms are extracted. Using torch_scatter, node labels aggregate neighboring labels. Nodes then receive a new "hash code" — a numerical representation of inherent and neighboring features, contrasting the string-based hashing in CPU methods. After multiple iterations, node labels produce subgraph-wide histograms, representing the subgraph's structure. By comparing these histograms, structural similarity between nodes is assessed. Despite the CPU's parallel capabilities, the GPU method offers superior memory and computational performance.

