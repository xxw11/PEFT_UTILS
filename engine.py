import os
import torch
from torch.nn import functional as F
from tqdm import tqdm
from utils import save, log
from avalanche.evaluation.metrics.accuracy import Accuracy
from timm.scheduler.cosine_lr import CosineLRScheduler

def train_dist(args, model,loss_fn, train_dl, test_dl, opt, scheduler, logger, epoch):
    gpu_id=int(os.environ["LOCAL_RANK"])
    logger.info("begin training...")
    # model = model.cuda()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        for i, batch in enumerate(train_dl):
            x, y = batch[0].to(gpu_id), batch[1].to(gpu_id)
            out = model(x)
            loss = loss_fn(out, y, model)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 50 == 49:
            acc = test_dist(model, test_dl)[1]
            if acc > args.best_acc:
                args.best_acc = acc
                save(args=args,model=model)
            if gpu_id == 0 :
                pbar.set_description(str(acc) + '|' + str(args.best_acc))
                logger.info("epoch:"+str(ep))
                logger.info("accuracy:"+str(acc))


    model = model.cpu()
    return model

@torch.no_grad()
def test_dist(model, dl):
    gpu_id=int(os.environ["LOCAL_RANK"])
    model.eval()
    acc = Accuracy()
    for batch in dl:
        x, y = batch[0].to(gpu_id), batch[1].to(gpu_id)
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 1)
    return acc.result()

def train(args, model, loss_fn, train_dl, test_dl, opt, scheduler, logger, epoch):
    logger.info("begin training...")
    model = model.cuda()
    pbar = tqdm(range(epoch))
    for ep in pbar:
        model.train()
        for i, batch in enumerate(train_dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            out = model(x)
            loss = loss_fn(out, y, model,logger)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 50 == 49:
            acc = test(model, test_dl)[1]
            if acc > args.best_acc:
                args.best_acc = acc
                save(args=args,model=model)
                model = model.cuda()
            pbar.set_description(str(acc) + '|' + str(args.best_acc))
            logger.info("epoch:"+str(ep))
            logger.info("accuracy:"+str(acc))
    model = model.cpu()
    return model



@torch.no_grad()
def test(model, dl):
    model.eval()
    acc = Accuracy()
    for batch in dl:
        x, y = batch[0].cuda(), batch[1].cuda()
        out = model(x).data
        acc.update(out.argmax(dim=1).view(-1), y, 1)
    return acc.result()


def train_clm(args, model, loss_fn, train_dl, opt, scheduler, logger, epoch=1):
    logger.info("begin training...")
    model = model.bfloat16().cuda()
    total_steps = epoch * len(train_dl)  # 计算总的迭代步数
    pbar = tqdm(total=total_steps)  # 进度条基于总的迭代步数
    current_step = 0
    scheduler = CosineLRScheduler(opt, t_initial=total_steps, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6)
    for ep in range(epoch):
        model.train()
        for i, batch in enumerate(train_dl):
            input_ids, labels, attention_mask = (batch['input_ids'].cuda(), batch['labels'].cuda(), batch['attention_mask'].cuda())
            results = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = results['loss']
            opt.zero_grad()
            loss.backward()
            opt.step()
            # 每个step都更新scheduler
            if scheduler is not None:
                scheduler.step(current_step)
            # 更新进度条
            pbar.set_description(f"Epoch {ep+1}, Loss: {loss.item()}")
            pbar.update(1)  # 更新一个step
            # 更新当前的步数
            current_step += 1
    model = model.cpu()
    pbar.close()  # 关闭进度条
    return model

def train_clm_dist(args, model, loss_fn, train_dl, opt, scheduler, logger, epoch=1):
    gpu_id=int(os.environ["LOCAL_RANK"])
    if gpu_id == 0:
        logger.info("begin training...")
        # save(args,model)
        # logger.info("saved")
    model = model.bfloat16().to(gpu_id)
    total_steps = epoch * len(train_dl)  # 计算总的迭代步数
    pbar = tqdm(total=total_steps)  # 进度条基于总的迭代步数
    current_step = 0
    scheduler = CosineLRScheduler(opt, t_initial=total_steps, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6)
    for ep in range(epoch):
        model.train()
        for i, batch in enumerate(train_dl):
            input_ids, labels, attention_mask = (batch['input_ids'].to(gpu_id), batch['labels'].to(gpu_id), batch['attention_mask'].to(gpu_id))
            results = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
            loss = results['loss']
            opt.zero_grad()
            loss.backward()
            opt.step()
            # 每个step都更新scheduler
            if scheduler is not None:
                scheduler.step(current_step)
            # 更新进度条
            pbar.set_description(f"current_step {current_step+1}, Loss: {loss.item()}")
            if gpu_id == 0:
                logger.info(f"current_step {current_step+1}, Loss: {loss.item()}")
                if current_step % 1000 == 0:
                    save(args,model,in_string=f'step{current_step}')
                    model = model.bfloat16().to(gpu_id)
                    torch.cuda.empty_cache()
            pbar.update(1)  # 更新一个step
            # 更新当前的步数
            current_step += 1
    pbar.close()  # 关闭进度条
    if gpu_id == 0:
        save(args,model)
    return model

def train_clm_ds(args, engine, train_dl, opt, scheduler, logger, epoch=1):
    logger.info("begin training...")
    total_steps = epoch * len(train_dl)  # 计算总的迭代步数
    pbar = tqdm(total=total_steps)  # 进度条基于总的迭代步数
    current_step = 0
    scheduler = CosineLRScheduler(opt, t_initial=total_steps, warmup_t=10, lr_min=1e-5, warmup_lr_init=1e-6)
    for ep in range(epoch):
        for i, batch in enumerate(train_dl):
            input_ids, labels = (batch['input_ids'].to(args.local_rank), batch['labels'].to(args.local_rank) )
            results = engine(input_ids=input_ids, labels=labels)
            loss = results['loss']
            opt.zero_grad()
            engine.backward(loss)
            engine.step()
            # 每个step都更新scheduler
            if scheduler is not None:
                scheduler.step(current_step)
            # 更新进度条
            pbar.set_description(f"current_step {current_step+1}, Loss: {loss.item()}")

            logger.info(f"current_step {current_step+1}, Loss: {loss.item()}")
            if current_step % 1000 == 0:
                save(args,engine,in_string=f'step{current_step}')
                # model = model.bfloat16().to(gpu_id)
            if current_step % 5 == 0:
                torch.cuda.empty_cache()
            pbar.update(1)  # 更新一个step
            # 更新当前的步数
            current_step += 1
    pbar.close()  # 关闭进度条
    save(args,engine,in_string=f'step{current_step}')