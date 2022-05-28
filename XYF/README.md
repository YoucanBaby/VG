# GTR

## Framework

![image-20220529022814827](https://xyf-image.oss-cn-beijing.aliyuncs.com/img/202205290228913.png)

![image-20220529021018433](https://xyf-image.oss-cn-beijing.aliyuncs.com/img/202205290210499.png)

## Download Datasets

Please download the data from [box](https://rochester.box.com/s/swu6rlqcdlebvwml8dyescmi7ra0owc5) or [baidu](https://pan.baidu.com/s/1pwo6lK71_ebit_hWykvgqQ?pwd=1mw4) and save it to the `data` folder.



## Training

```sh
$ python moment_localization/run.py --cfg experiments/tacos/GTR.yaml --verbose --tag base
```



# XYF

## Framework

![image-20220529020152050](https://xyf-image.oss-cn-beijing.aliyuncs.com/img/202205290201230.png)

因为复现 GTR 出现了问题。bug 蛮多的。比如，在 MSMA 模块，模型几乎学不到任何有用的信息。损失函数也有点问题。

所以我替换掉了 MSMA 模块，重新设计一个模型来验证之前的模型是有问题的。

模型结构如下：

![image-20220529020300957](https://xyf-image.oss-cn-beijing.aliyuncs.com/img/202205290203018.png)

- [ ] TODO 损失函数还有待优化



## Training

```sh
$ python moment_localization/run.py --cfg experiments/tacos/XYF.yaml --verbose --tag base
```



# Code path

It is recommended to do experiments on the TACOS dataset, because this dataset is small and results will be available soon.

```sh
--experiments
	--tacos
		--GTR.yaml		# 超参数配置
		--XYF.yaml		# 超参数配置

--lib
	--core
		--config.py		# 超参数配置
	--dataset
		--dataset.py	# 数据集配置
	--models
		--utils
			--loss.py	
		--GTR.py
		--XYF.py
		
--moment_localization
	--run.py
	--eval.py
```



























