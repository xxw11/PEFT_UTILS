# PEFT_UTIL

PEFT_UTILS 是Parameter-Efficient Fine-Tuning (PEFT) 方法的一个工具库。目前收集了LoRA、GLoRA、SSF 和 RepAdapter等PEFT方法。

## 使用示例

以下是一个使用SSF加载和训练模型的简单示例：

1. **插入PEFT层**

   在模型中插入PEFT层。

   ```python
   set_ssf(model=model)
   ```

2. **设置模型参数的`requires_grad`属性**

   根据需要, 设置模型参数的`requires_grad`属性来确定哪些参数需要训练。在使用不同方法的时候'SSF'需要随所使用的方法调整，

   ```python
   trainable = []
   for n, p in model.named_parameters():
       if any([x in n for x in ['SSF']]):
           trainable.append(p)
           p.requires_grad = True
       else:
           p.requires_grad = False
   ```

3. **保存模型**

   训练完成后保存模型。

   ```python
   save_ssf(output_dir+"final.pt", model=model)
   ```

4. **加载模型**

   若需要加载训练完成后保存的模型。

   ```python
   load_ssf(load_path, model=model)
   ```

