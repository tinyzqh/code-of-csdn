
&emsp;&emsp;红绿灯分为导向灯和圆形灯。一般圆形灯在路口只有一盏灯，红灯亮时禁止直行和左转，可以右转弯。导向灯市带有箭头的，可以有两个或三个，分别指示不同方向的行车和停车。按指示的灯即可，没有右转向导向灯的情况下可以视为可以右转。

&emsp;&emsp;RGB颜色空间以R(Red:红）、G(Green:绿色)、 B(Blue:蓝）三种基本色为基础，进行不同程度的叠加，产生丰富而广泛的颜色，所以俗称三基色模式。在大自然中有无穷多种不同的颜色，而人眼只能分辨有限种不同的颜色，RGB棋式可表示一千六百多万种不同的颜色，在人跟看起来它非常接近大自然的颜色，故又称为自然色彩模式。红绿蓝代表可见光谱中的三种基木颜色或称为三原色，每一种颜色按其亮度的不同分为256个等级。当色光三原色重叠时，由于不同的混色比例能产生各种中间色。

&emsp;&emsp;RGB颜色空间最大的优点就是直观，容易理解。缺点是R、G、B这三个分量是高度相关的，即如果一个颜色的某一个分量发生了一定程度的改变，那么这个相色很可能要发生改变；人眼对于常见的 红绿蓝三色的敏感程度是不一样的，因此RGB颜色空间的均匀性非常差，且两种颜色之间的知觉差异色差不能表示为改颜色空间中两点间的距离，但是利用线性或非线性变换，则可以从RGB颜色空间推导出其他的颜色特征空间。

&emsp;&emsp;而在HSV颜色空间中，颜色的参数分别是：色调(H)，饱和度(S)，明度(V)。色调H，用角度度量，取值范围为$0^{o}~360^{o}$，从红色开始按逆时针方向计算，红色为$0^{o}$，绿色为$120^{o}$，蓝色为$240^{o}$。它们的补色是：黄色为$60^{o}$，青色为$180^{o}$，品红为$360^{o}$。饱和度S表示颜色接近光谱色的程度。一种颜色，可以看成是某种光谱色于白色混合的结果。其中光谱色所占的比例愈大，颜色接近光谱色的程度就愈高，颜色的饱和度也就愈高。饱和度高，颜色则深而艳。光谱色的白光成分为0，饱和度达到最高。通常取值范围为$0\%-100\%$，值越大，颜色越饱和。明度V表示颜色明亮的程度，对于光源色，明度值与发光体的光亮度有关；对于物体色，此值和物体的透射比或反射比有关。通常取值范围为$0\%$(黑)到$100\%$(白)。相对于RGB空间，HSV空间能够非常直观的表达色彩的明暗，色调，以及鲜艳程度，方便进行颜色之间的对比。将图片从传统的RGB颜色空间转换到HSV模型空间，能够大大提高目标识别与检测的抗干扰能力，使得检测结果更为精确。

&emsp;&emsp;本设计中红绿灯检测程序主要有detectColor.py文件和TLState.py两个文件。

&emsp;&emsp;在detectColor.py文件中主要是检测被TLState.py分割出来的灯的颜色，首先利用OpenCV中的cv2.cvtColor(image,cv2.COLOR_BGR2HSV)函数，将图片从BGR格式转换为HSV格式。之后利用cv2.inRange()函数设阈值，去除背景部分，再进行中值滤波，最后计算非零像素点数，取其像素点最多的那个对应的结果作为最终结果。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106092859736.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTA1OTAzMQ==,size_1,color_FFFFFF,t_70)

![颜色检测模块代码示意图](https://img-blog.csdnimg.cn/20191106092922451.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTA1OTAzMQ==,size_1,color_FFFFFF,t_70)

&emsp;&emsp;在TLState.py文件中，进行灰度处理，之后利用cv2.HoughCircles()函数进行霍夫圆环检测。将检测到的圆环送入detectColor.py文件中的detectColor()函数中进行颜色检测。

![霍夫圆环检测及颜色识别模块代码示意图](https://img-blog.csdnimg.cn/20191106093017774.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTA1OTAzMQ==,size_1,color_FFFFFF,t_70)

&emsp;&emsp;红绿灯检测得到的结果如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106093103898.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTA1OTAzMQ==,size_1,color_FFFFFF,t_70)

&emsp;&emsp;红绿灯带箭头检测得到的结果如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106093311898.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zOTA1OTAzMQ==,size_1,color_FFFFFF,t_70)
