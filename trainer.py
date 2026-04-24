import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm

from utils import (
    DiceLoss,
    calculate_dice_ce_loss,
    calculate_vessel_loss,
    test_single_volume, 
    test_single_image,
    test_single_image_tiler,
)


def trainer_acdc(args, model, snapshot_path):
    from datasets.dataset_acdc import BaseDataSets, RandomGenerator
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=transforms.Compose([
        RandomGenerator([args.img_size, args.img_size])]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(ignore_index=4)
    dice_loss = DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))
    logging.info("{} val iterations per epoch".format(len(valloader)))
    # logging.info("{} test iterations per epoch".format(len(testloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            outputs = model(volume_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = volume_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 500 == 0:  # 500
                model.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    image, label = sampled_batch["image"], sampled_batch["label"]
                    metric_i = test_single_volume(image, label, model, classes=num_classes,
                                                  patch_size=[args.img_size, args.img_size])
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]

                mean_hd95 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)
                writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

                if performance > best_performance:
                    best_iteration, best_performance, best_hd95 = iter_num, performance, mean_hd95
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_best)
                    logging.info('Best model | iteration %d : mean_dice : %f mean_hd95 : %f' % (
                    iter_num, performance, mean_hd95))

                logging.info('iteration %d : mean_dice : %f mean_hd95 : %f' % (iter_num, performance, mean_hd95))
                model.train()

            if iter_num >= max_iterations:
                break

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        save_interval = 50  # int(max_epoch/6)
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_drive(args, model, snapshot_path):
    from datasets.dataset_drive import RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    test_fn = test_single_image_tiler if args.tile else test_single_image
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = args.Dataset(
        base_dir=args.root_path, 
        split="train",
        transform=transforms.Compose(
            [RandomGenerator(output_size=[args.img_size, args.img_size])]
        ))
    
    # Use 'test' split for validation if 'val' is not robust enough or use 'val' as intended
    db_val = args.Dataset(base_dir=args.root_path, split="val")
    
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    start_epoch = 0
    best_performance = -1.0

    if hasattr(args, 'resume') and args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        iter_num = checkpoint['iter_num']
        best_performance = checkpoint.get('best_performance', -1.0)
        logging.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            fov_mask = sampled_batch.get('fov_mask', None)
            if fov_mask is not None:
                fov_mask = fov_mask.cuda()
                
            outputs = model(image_batch)

            # Loss calculation based on specified loss type in config
            if args.loss_name == 'dice_ce':
                loss_dict = calculate_dice_ce_loss(outputs, label_batch, num_classes=num_classes)
            elif args.loss_name.startswith('vessel'):
                ce_weights = torch.tensor([0.2, 0.8]).cuda()
                loss_dict = calculate_vessel_loss(outputs, label_batch, num_classes=num_classes, fov_mask=fov_mask, ce_weight=ce_weights)
            else:
                raise ValueError(f"Unsupported loss name: {args.loss_name}")
            
            total_loss = loss_dict['total_loss']
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)

            loss_log = f'iteration {iter_num} :'

            for key, value in loss_dict.items():
                writer.add_scalar(f'info/{key}', value, iter_num)
                loss_log += f' {key}: {value.item():.4f},'

            logging.info(loss_log.rstrip(','))

            if iter_num % 100 == 0 or iter_num >= max_iterations:                
                # import cv2
                # Log visualization
                image = image_batch[0, 0:3, :, :] # RGB
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                
                outputs_argmax = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                outputs_argmax = outputs_argmax[0, ...]*255
                # cv2.imwrite(snapshot_path + "/prediction.png", outputs_argmax.cpu().detach().numpy().astype(np.uint8).transpose(1,2,0))
                writer.add_image('train/Prediction', outputs_argmax, iter_num)
                
                labs = label_batch[0,...]*255
                # cv2.imwrite(snapshot_path + "/groundtruth.png", labs.cpu().detach().numpy().astype(np.uint8))
                writer.add_image('train/GroundTruth', labs.unsqueeze(0), iter_num)
                
        # Run validation on every epoch
        if (epoch_num + 1) % 1 == 0:
            model.eval()
            metric_list = (0.0,)  # shape (1,) for broadcasting 

            for i_batch, sampled_batch in enumerate(valloader):
                if "fov_mask" in sampled_batch:
                    for image, label, mask in zip(sampled_batch["image"], sampled_batch["label"], sampled_batch["fov_mask"]):
                        metric_i = test_fn(image, label, model, classes=num_classes, fov_mask=mask)
                        metric_list += np.array(metric_i)
                else:
                    for image, label in zip(sampled_batch["image"], sampled_batch["label"]):
                        metric_i = test_fn(image, label, model, classes=num_classes)
                        metric_list += np.array(metric_i)
            
            metric_list = np.array(metric_list)
            # logging.info(metric_list)  # shape: (1, 2)

            metric_list = metric_list / len(db_val)
            performance = metric_list[0, 0] # Dice
            mean_hd95 = metric_list[0, 1]  # HD95
            
            writer.add_scalar('info/val_mean_dice', performance, iter_num)
            writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

            if performance > best_performance:
                best_performance = performance
                save_best = os.path.join(snapshot_path, 'best_model.pth')
                state = {
                    'epoch': epoch_num + 1,
                    'iter_num': iter_num,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_performance': best_performance
                }
                torch.save(state, save_best)
                logging.info('Best model | epoch %d : mean_dice : %f mean_hd95 : %f' % (epoch_num, performance, mean_hd95))

            model.train()

        # Regular checkpoint saving for resuming (latest model)
        save_latest_path = os.path.join(snapshot_path, 'latest_model.pth')
        state = {
            'epoch': epoch_num + 1,
            'iter_num': iter_num,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_performance': best_performance
        }
        torch.save(state, save_latest_path)

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(state, save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_chasedb(args, model, snapshot_path):
    from datasets.dataset_chasedb import ChaseDB_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = ChaseDB_dataset(base_dir=args.root_path, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    db_val = ChaseDB_dataset(base_dir=args.root_path, split="val", transform=None)
    
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    best_performance = -1.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 100 == 0:
                image = image_batch[0, 0:3, :, :]
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                
                outputs_argmax = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_argmax[0, ...] * 255 if num_classes==2 else outputs_argmax[0,...]*50, iter_num)
                
                labs = label_batch[0, ...].unsqueeze(0) * 255 if num_classes==2 else label_batch[0,...].unsqueeze(0)*50
                writer.add_image('train/GroundTruth', labs, iter_num)

        if (epoch_num + 1) % 1 == 0:
            model.eval()
            metric_list = 0.0
            for i_batch, sampled_batch in enumerate(valloader):
                # image, label = sampled_batch["image"], sampled_batch["label"]
                for image, label in zip(sampled_batch["image"], sampled_batch["label"]):
                    metric_i = test_single_image_tiler(image, label, model, classes=num_classes)
                    metric_list += np.array(metric_i)
            
            metric_list = metric_list / len(db_val)
            performance = metric_list[0, 0]
            mean_hd95 = metric_list[0, 1]
            
            writer.add_scalar('info/val_mean_dice', performance, iter_num)
            writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

            if performance > best_performance:
                best_performance = performance
                save_best = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_best)
                logging.info('Best model | epoch %d : mean_dice : %f mean_hd95 : %f' % (epoch_num, performance, mean_hd95))

            model.train()

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


def trainer_hrf(args, model, snapshot_path):
    from datasets.dataset_hrf import HRF_dataset, RandomGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = HRF_dataset(base_dir=args.root_path, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    db_val = HRF_dataset(base_dir=args.root_path, split="val", transform=None)
    
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    
    best_performance = -1.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 100 == 0:
                image = image_batch[0, 0:3, :, :]
                if image.shape[0] == 1:
                    image = image.repeat(3, 1, 1)
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                
                outputs_argmax = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_argmax[0, ...] * 255 if num_classes==2 else outputs_argmax[0,...]*50, iter_num)
                
                labs = label_batch[0, ...].unsqueeze(0) * 255 if num_classes==2 else label_batch[0,...].unsqueeze(0)*50
                writer.add_image('train/GroundTruth', labs, iter_num)

        if (epoch_num + 1) % 1 == 0:
            model.eval()
            metric_list = 0.0
            for i_batch, sampled_batch in enumerate(valloader):
                for image, label in zip(sampled_batch["image"], sampled_batch["label"]):
                    metric_i = test_single_image_tiler(image, label, model, classes=num_classes)
                    metric_list += np.array(metric_i)
            
            metric_list = metric_list / len(db_val)
            performance = metric_list[0, 0]
            mean_hd95 = metric_list[0, 1]
            
            writer.add_scalar('info/val_mean_dice', performance, iter_num)
            writer.add_scalar('info/val_mean_hd95', mean_hd95, iter_num)

            if performance > best_performance:
                best_performance = performance
                save_best = os.path.join(snapshot_path, 'best_model.pth')
                torch.save(model.state_dict(), save_best)
                logging.info('Best model | epoch %d : mean_dice : %f mean_hd95 : %f' % (epoch_num, performance, mean_hd95))

            model.train()

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
