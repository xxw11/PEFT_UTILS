# PEFT_UTILS

PEFT_UTILS 是Parameter-Efficient Fine-Tuning (PEFT) 方法的一个工具库。目前收集了LoRA、GLoRA、SSF 和 RepAdapter等PEFT方法。

## 使用示例

以下是一个使用repadapter加载和训练模型的简单示例：


0. **导入模块**
    可以从repadapter.py，或peft_utils包中导入需要的模块。


1. **插入PEFT层**

   在模型中插入PEFT层。

   ```python
   set_repadapter(model=model)
   ```

   若需要只训练特定线性层，可修改set_repadapter使用正则表示式匹配特定的name

   ```python
   import re
   import torch.nn as nn
   def set_repadapter(model, pattern):
   # 编译正则表达式模式
   regex = re.compile(pattern)
   for name, module in model.named_modules():
   # 检查模块是否是线性层并且名称匹配正则表达式
   if isinstance(module, nn.Linear) and regex.match(name):
   ```


2. **设置模型参数的`requires_grad`属性**

   根据需要, 设置模型参数的`requires_grad`属性来确定哪些参数需要训练。在使用不同方法的时候'SSF'需要随所使用的方法调整，

   ```python
   trainable = []
   for n, p in model.named_parameters():
       if any([x in n for x in ['adapter']]):
           trainable.append(p)
           p.requires_grad = True
       else:
           p.requires_grad = False
   ```

3. **保存模型repadapter部分**

   训练完成后，一般仅保存模型的repadapter参数，这样可以节省非常多的硬盘空间占用，是repadapter的优势之一。

   ```python
   import os
   save_repadapter(os.path.join(output_dir,"final.pt"), model=model)
   ```

4. **加载模型repadapter部分**

   若需要加载训练完成后保存的模型。model需要set_repadapter后再加载。

   ```python
   load_repadapter(load_path, model=model)
   ```

5. **重参数化模型**
   merge_repadapter在模型训练后用于简化模型结构，减少模型大小和推理时间。
   merge_repadapter传入模型，和repadapter的保存路径，会进行重参数化。
   ```python
   merge_repadapter(model,load_path=None,has_loaded=False)
   ```

## Acknowledgements

This project uses methods and code from multiple sources:

- The method "Towards Efficient Visual Adaption via Structural Re-parameterization" by Luo et al. is incorporated, with the implementation found at the [RepAdapter GitHub repository](https://github.com/luogen1996/RepAdapter/tree/main). We are grateful to Luo, Gen and the co-authors for their contributions.

- Additionally, we leverage "One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning" by Arnav Chavan et al. The corresponding code is available at the [GLoRA GitHub repository](https://github.com/Arnav0400/ViT-Slim/tree/master/GLoRA). Our thanks go to Arnav Chavan and his colleagues for making their work accessible.

## Citations

If you use the methods from Luo et al. or Chavan et al. in your work, please consider citing their papers:

```bibtex
@article{luo2023towards,
  title={Towards Efficient Visual Adaption via Structural Re-parameterization},
  author={Luo, Gen and Huang, Minglang and Zhou, Yiyi  and Sun, Xiaoshuai and Jiang, Guangnan and Wang, Zhiyu and Ji, Rongrong},
  journal={arXiv preprint arXiv:2302.08106},
  year={2023}
}

@misc{chavan2023oneforall,
  title={One-for-All: Generalized LoRA for Parameter-Efficient Fine-tuning},
  author={Arnav Chavan and Zhuang Liu and Deepak Gupta and Eric Xing and Zhiqiang Shen},
  year={2023},
  eprint={2306.07967},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}