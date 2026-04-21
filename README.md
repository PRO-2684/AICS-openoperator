# AICS-OpenOperator

## 🔗 Related Links

- [OpenOperator](https://openoperator.cn) | [Monitor](http://152.136.18.42:13000)
- [BangC Tutorial](https://gitservice.cstcloud.cn/kcxain/BangcTutorial/)
- [南京智能计算中心算力平台](https://paas.extrotec.com:30443/)
- [Start Kit](https://github.com/kernel-competition-bot/openoperator-start-kit)

## 🚀 Quick Start

1. Clone this repo
2. Navigate to the root directory of the repo and run `setup.sh` to:
    - Set up git hooks: This will automatically update `config` file to the list of changed files in each commit

## 提交说明

提交时仓库根目录需要包含`config`文件和题目的`mlu`代码文件。文件组织结构如下

```bash
.
├── config			# 配置文件，用于指定要评估的题目
├── LeakyReLU.mlu	# bangc代码文件，必须包含kernel函数定义和用于外部程序调用的函数定义
├── ...				# 其他题目的bangc代码文件
└── README.md		# 可选的代码说明
```

> [!NOTE]
>
> 通过`config`文件可以指定本次提交想要评估的题目范围
> config文件的每行代表一个题目，应按照题目序号的三位数字给出
> 例如，LeakyReLU的序号是001，为了评估LeakyReLU题目，config中必须包含一行`001`

> [!TIP]
>
> 每道题目的评估耗时预计不少于30s，评估系统评估完所有题目后才会返回结果，请合理安排评估请求，尽量不要一次性评估太多题目。

> [!CAUTION]
>
> 如果提交中不包含config文件，则会默认评估所有题目！

### 代码要求

1. 代码文件必须以题目名称命名，这是评估脚本能找到你代码的关键要求。
2. 代码中要覆盖头文件引用，核函数定义和用于外部调用的函数定义。
3. 用于外部调用的函数名必须设置为bang_func，bang_func的返回值为`torch::Tensor`，输入参数包含`torch::Tensor input`和参考代码中`__init__`部分定义的其他参数，请参考LeakyReLU示例进行理解。

## 题目&打分

题目按照类别分为`basic`，`easy`，`medium`，`hard`。其中`basic`是必做题，其他类为挑战题。

打分有两个指标：

- 算子结果必须与参考结果误差不大于1e-2，精度达标后性能评估结果才有效
- 性能分数按照`bangc`代码硬件时间相对于`torch`的执行时间赋值

## 最佳实践

1. 每次只评估少量题目
2. 尽量在调试服务器debug，远程评估时通过阅读commit评论中的报错进行debug
3. 系统只接收`main`分支的提交，所以请分时开发或者做好分支管理
4. github评论是执行结束第一时间更新的，排行榜是周期性更新的，且只会记录团队历史最好成绩
