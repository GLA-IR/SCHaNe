import math
import os.path
import sys
import json
import time
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.utils import accuracy, ModelEma
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from beit3_tools.beit3_datasets import get_sentencepiece_model_for_beit3
import numpy as np
from beit3_tools import utils
from tqdm import tqdm
import os
import CL_tools
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

class TaskHandler(object):
    def __init__(self) -> None:
        self.metric_logger = None
        self.split = None

    def train_batch(self, model, **kwargs):
        raise NotImplementedError()

    def eval_batch(self, model, **kwargs):
        raise NotImplementedError()

    def before_eval(self, metric_logger, data_loader, **kwargs):
        self.metric_logger = metric_logger
        # self.split = data_loader.dataset.split

    def after_eval(self, **kwargs):
        raise NotImplementedError()


class ImageClassificationHandler(TaskHandler):
    def __init__(self, args) -> None:
        super().__init__()
        # mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        # if mixup_active:
        #     # smoothing is handled with mixup label transform
        #     self.criterion = SoftTargetCrossEntropy()
        # elif args.label_smoothing > 0.:
        #     self.criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
        # else:

        self.args = args
        if args.loss == 'CE':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif args.loss == 'SupCon' or args.loss == 'SimCLR':
            self.ce_criterion = torch.nn.CrossEntropyLoss()
            self.cl_criterion = CL_tools.losses.SupConLoss(temperature=args.temp)
        elif args.loss == 'SCHaNe':
            self.cl_criterion = CL_tools.losses.SCHaNe
            self.ce_criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('Unknown loss function: {}'.format(args.loss))

    def train_batch(self, model, images, label):
        bsz = label.shape[0]

        if self.args.loss == 'CE':
            images = torch.stack([image[0] for image in images])
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
            cls, features,_ = model(images)
            loss = self.criterion(cls, label)

        elif self.args.loss == 'SupCon':
            images0 = torch.stack([image[0] for image in images])
            images1 = torch.stack([image[1] for image in images])
            images = torch.cat([images0, images1], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
            cls, features, _ = model(images)
            # normalize the feature
            features = F.normalize(features, dim=-1)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            cl_loss = self.cl_criterion(features, label)

            label = torch.cat([label, label], dim=0)
            ce_loss = self.ce_criterion(cls, label)

            loss = self.args.alpha * cl_loss + (1 - self.args.alpha) * ce_loss

        elif self.args.loss == 'SimCLR':
            images0 = torch.stack([image[0] for image in images])
            images1 = torch.stack([image[1] for image in images])
            images = torch.cat([images0, images1], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

            cls, features, _ = model(images)


            if not self.args.alpha == 1.0:
                double_label = torch.cat([label, label], dim=0)
                ce_loss = self.ce_criterion(cls, double_label)
            else:
                ce_loss = 0.0

            features = F.normalize(features, dim=-1)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            cl_loss = self.cl_criterion(features)
            loss = self.args.alpha * cl_loss + (1 - self.args.alpha) * ce_loss

        elif self.args.loss == 'SCHaNe':
            images0 = torch.stack([image[0] for image in images])
            images1 = torch.stack([image[1] for image in images])
            images = torch.cat([images0, images1], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

            # start_time = time.time()
            cls, features, _ = model(images)
            # end_time = time.time()
            # print('Time for forward pass: {}'.format(end_time-start_time))
            # cls = F.normalize(cls, dim=-1)

            # normalize the feature
            # features = F.normalize(features, dim=-1)
            # f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            if not self.args.alpha == 1.0:
                double_label = torch.cat([label, label], dim=0)
                ce_loss = self.ce_criterion(cls, double_label)
            else:
                ce_loss = 0.0

            features = F.normalize(features, dim=-1)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            cl_loss = self.cl_criterion(features, self.args.tau_plus, self.args.batch_size, self.args.beta,
                                        self.args.estimator, self.args.temp, labels=label)

            loss = self.args.alpha * cl_loss + (1 - self.args.alpha) * ce_loss


        return {
            "loss": loss,
        }

    def eval_batch(self, model, images, label):
        images = torch.stack(images)
        label = torch.as_tensor(label, dtype=torch.long, device='cuda')
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
        logits, _, _ = model(image=images)
        batch_size = images.shape[0]
        acc1, acc5 = accuracy(logits, label, topk=(1, 5))
        self.metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        self.metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


    def fewshot_eval(self, model, data):
        support_xs, support_ys, query_xs, query_ys = data[0]
        support_xs = support_xs.cuda()
        query_xs = query_xs.cuda()
        batch_size, channel, height, width = support_xs.size()
        support_xs = support_xs.view(-1, channel, height, width)
        query_xs = query_xs.view(-1, channel, height, width)

        if torch.cuda.is_available():
            support_xs = support_xs.cuda(non_blocking=True)
            query_xs = query_xs.cuda(non_blocking=True)
            # support_ys = support_ys.cuda(non_blocking=True)
            # query_ys = query_ys.cuda(non_blocking=True)

        _, _, support_features = model(support_xs)
        support_features = support_features.view(support_xs.size(0), -1)
        _,  _, query_features = model(query_xs)
        query_features = query_features.view(query_xs.size(0), -1)

        support_features = support_features.detach().cpu().numpy()
        query_features = query_features.detach().cpu().numpy()

        clf = LogisticRegression(penalty='l2',
                                 random_state=0,
                                 C=1.0,
                                 solver='lbfgs',
                                 max_iter=1000,
                                 multi_class='multinomial')
        clf.fit(support_features, support_ys)
        query_ys_pred = clf.predict(query_features)

        # batch_size = support_features.shape[0]

        acc1 = metrics.accuracy_score(query_ys, query_ys_pred) * 100
        self.metric_logger.meters['fewshot_acc1'].update(acc1.item(), n=batch_size)

        return acc1

    def after_eval(self, **kwargs):
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
              .format(top1=self.metric_logger.acc1, top5=self.metric_logger.acc5))
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "acc1"

    def after_fewshot_eval(self, **kwargs):
        print('* Acc@1 {top1.global_fewshot_avg:.3f} Acc@5 {top5.global_fewshot_avg:.3f}'
              .format(top1=self.metric_logger.fewshot_acc1, top5=self.metric_logger.fewshot_acc5))
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "acc1"

class FewshotImageClassificationHandler(TaskHandler):
    def __init__(self, args) -> None:
        super().__init__()

        self.args = args
        if args.loss == 'CE':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif args.loss == 'SupCon' or args.loss == 'SimCLR':
            self.ce_criterion = torch.nn.CrossEntropyLoss()
            self.cl_criterion = CL_tools.losses.SupConLoss(temperature=args.temp)
        elif args.loss == 'SCHaNe':
            self.cl_criterion = CL_tools.losses.SCHaNe
            self.ce_criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError('Unknown loss function: {}'.format(args.loss))
    def train_batch(self, model, images, label):
        bsz = label.shape[0]

        if self.args.loss == 'CE':
            images = torch.stack([image[0] for image in images])
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
            cls, features = model(images)
            loss = self.criterion(cls, label)

        elif self.args.loss == 'SupCon':
            images0 = torch.stack([image[0] for image in images])
            images1 = torch.stack([image[1] for image in images])
            images = torch.cat([images0, images1], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
            cls, features = model(images)
            # normalize the feature
            features = F.normalize(features, dim=-1)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            cl_loss = self.cl_criterion(features, label)

            label = torch.cat([label, label], dim=0)
            ce_loss = self.ce_criterion(cls, label)
            loss = self.args.alpha * cl_loss + (1 - self.args.alpha) * ce_loss
        elif self.args.loss == 'SimCLR':
            images0 = torch.stack([image[0] for image in images])
            images1 = torch.stack([image[1] for image in images])
            images = torch.cat([images0, images1], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
            features = model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            loss = self.criterion(features)

        elif self.args.loss == 'SCHaNe':
            images0 = torch.stack([image[0] for image in images])
            images1 = torch.stack([image[1] for image in images])
            images = torch.cat([images0, images1], dim=0)
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

            # start_time = time.time()
            cls, features = model(images)
            # end_time = time.time()
            # print('Time for forward pass: {}'.format(end_time-start_time))
            # cls = F.normalize(cls, dim=-1)

            # normalize the feature
            # features = F.normalize(features, dim=-1)
            # f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            # features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            double_label = torch.cat([label, label], dim=0)
            ce_loss = self.ce_criterion(cls, double_label)

            features = F.normalize(features, dim=-1)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            cl_loss = self.cl_criterion(features, self.args.tau_plus, self.args.batch_size, self.args.beta,
                                        self.args.estimator, self.args.temp, labels=label)

            loss = self.args.alpha * cl_loss + (1 - self.args.alpha) * ce_loss


        return {
            "loss": loss,
        }

    def fewshot_eval(self, model, data):
        support_xs, support_ys, query_xs, query_ys = data[0]
        support_xs = support_xs.cuda()
        query_xs = query_xs.cuda()
        batch_size, channel, height, width = support_xs.size()
        support_xs = support_xs.view(-1, channel, height, width)
        query_xs = query_xs.view(-1, channel, height, width)

        if torch.cuda.is_available():
            support_xs = support_xs.cuda(non_blocking=True)
            query_xs = query_xs.cuda(non_blocking=True)
            # support_ys = support_ys.cuda(non_blocking=True)
            # query_ys = query_ys.cuda(non_blocking=True)

        _, support_features = model(support_xs)
        support_features = support_features.view(support_xs.size(0), -1)
        _, query_features = model(query_xs)
        query_features = query_features.view(query_xs.size(0), -1)

        support_features = support_features.detach().cpu().numpy()
        query_features = query_features.detach().cpu().numpy()

        # support_ys = support_ys.view(-1).numpy()
        # query_ys = query_ys.view(-1).numpy()

        # images = torch.stack(images)
        # label = torch.as_tensor(label, dtype=torch.long, device='cuda')
        # if torch.cuda.is_available():
        #     images = images.cuda(non_blocking=True)
        #     label = label.cuda(non_blocking=True)
        # logits, _ = model(image=images)
        clf = LogisticRegression(penalty='l2',
                                 random_state=0,
                                 C=1.0,
                                 solver='lbfgs',
                                 max_iter=1000,
                                 multi_class='multinomial')
        clf.fit(support_features, support_ys)
        query_ys_pred = clf.predict(query_features)

        # batch_size = support_features.shape[0]

        acc1 = metrics.accuracy_score(query_ys, query_ys_pred)
        self.metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # self.metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    def after_eval(self, **kwargs):
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
              .format(top1=self.metric_logger.acc1, top5=self.metric_logger.acc5))
        return {k: meter.global_avg for k, meter in self.metric_logger.meters.items()}, "acc1"



def get_handler(args):
    return ImageClassificationHandler(args)
    if args.task not in ['miniImageNet', 'tieredImageNet','CIFAR-FS', 'FC100']:
        return ImageClassificationHandler(args)
    else:
        return FewshotImageClassificationHandler(args)
    # if args.task
    #
    # else:
    #     raise NotImplementedError("Sorry, %s is not support." % args.task)




def train_one_epoch(
        model: torch.nn.Module, data_loader: Iterable,
        optimizer: torch.optim.Optimizer, device: torch.device,
        handler: TaskHandler, epoch: int, start_steps: int,
        lr_schedule_values: list, loss_scaler, max_norm: float = 0,
        update_freq: int = 1, model_ema: Optional[ModelEma] = None,
        log_writer: Optional[utils.TensorboardLogger] = None,
        task=None, mixup_fn=None,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        global_step = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[global_step] * param_group["lr_scale"]
        # put input data into cuda
        # for tensor_key in data.keys():
        #     if not isinstance(data[tensor_key], list):
        #         # data[tensor_key] = [t.to(device, non_blocking=True) for t in data[tensor_key]]
        #         data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
        #     # print("input %s = %s" % (tensor_key, data[tensor_key]))
        #     if loss_scaler is None and 'image' in tensor_key:
        images, label = [sample[0] for sample in data], [sample[1] for sample in data]
        label = torch.as_tensor(label, dtype=torch.long, device=device)
        images = [[sample[0].half(), sample[0].half()] for sample in images]

        if loss_scaler is None:
            results = handler.train_batch(model, images=images, label=label )
        else:
            with torch.cuda.amp.autocast():
                results = handler.train_batch(model,images=images, label=label )

        loss = results.pop("loss")
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = utils.get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            kwargs = {
                "loss": loss_value,
            }
            for key in results:
                kwargs[key] = results[key]
            log_writer.update(head="train", **kwargs)

            kwargs = {
                "loss_scale": loss_scale_value,
                "lr": max_lr,
                "min_lr": min_lr,
                "weight_decay": weight_decay_value,
                "grad_norm": grad_norm,
            }
            log_writer.update(head="opt", **kwargs)
            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




@torch.no_grad()
def evaluate(data_loader, model, device, handler, build_ranking=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    for data in metric_logger.log_every(data_loader, 50, header):
        # for tensor_key in data.keys():
        #     data[tensor_key] = data[tensor_key].to(device, non_blocking=True)

        images, label = [sample[0] for sample in data], [sample[1] for sample in data]

        with torch.cuda.amp.autocast():
            handler.eval_batch(model, images=images, label=label)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return handler.after_eval()

import scipy
from scipy.stats import t

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

@torch.no_grad()
def fewshot_evaluate(data_loader, model, device, handler, build_ranking=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    handler.before_eval(metric_logger=metric_logger, data_loader=data_loader)

    acc = []
    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 50, header):
            # for tensor_key in data.keys():
            #     data[tensor_key] = data[tensor_key].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                acc1 = handler.fewshot_eval(model, data)
                acc.append(acc1)

    acc_feat, std_feat = mean_confidence_interval(acc)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    return acc_feat, std_feat