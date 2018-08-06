# char-rnn-writer
基于Char RNN实现的作诗Web应用,效果看起来还`o**k`

## 训练结果：

**君**

君王安死处，教柯得仙容。
雾露老寒浦，风吹古角声

**山**

山色寒不高，松连气含渠。
暂行不见江，自属寒露冷。

**夜**

夜皆凤义花，浮云引艳切。
红进双柳赠，风雨万声和。


上面只是理想的测试结果，大家输入一些字之后会发现很多不理想的情况(～￣▽￣)～

## 参考：
代码主要来自大神的：https://github.com/jinfagang/tensorflow_poems

自己主要是将char rnn实现了一遍，学习和总结了下其原理，然后将训练模型做成一个小应用

**测试效果如下**：

![](https://github.com/yanqiangmiffy/char-rnn-writer/blob/master/assets/result1.png)
![](https://github.com/yanqiangmiffy/char-rnn-writer/blob/master/assets/result2.png)

## 生成名字

自己从[名字大全](http://xm.99166.com/mzdq/)这个网站爬取所有形式组成的名字，爬取代码在`data\spider`目录下
![](https://github.com/yanqiangmiffy/char-rnn-writer/blob/master/assets/name.png)

这个模型也可以用来生成名字,训练命令如下:

`python train.py --batch_size 64 --result_dir 'result/name' --model_prefix 'names' --epochs 50 --file_path 'data/names.txt'`

**测试效果如下**：

![](https://github.com/yanqiangmiffy/char-rnn-writer/blob/master/assets/result3.png)
