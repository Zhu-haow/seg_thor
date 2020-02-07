'''
    用于测试模型分割效果，保存输出结果: SavePath/SM/ResNetC1/0/result_batch1
'''


import logging
import time
import numpy as np
import data_utils.transforms as tr
from torchvision import transforms
from data_utils.torch_data import THOR_Data, get_cross_validation_paths, get_global_alpha
import torch
from torch.utils.data import DataLoader
from utils import setgpu, get_threshold, metric, segmentation_metrics
import os
from torch.backends import cudnn
from importlib import import_module
from torch.nn import DataParallel
from PIL import Image
from torchvision.utils import save_image

DEVICE = torch.device("cuda" if True else "cpu")

def main(args):
  torch.manual_seed(123)
  cudnn.benchmark = True
  setgpu("all")
  data_path = args["data_path"]
  _, test_files = get_cross_validation_paths(0)
  model = import_module('models.model_loader')
  net, loss = model.get_full_model(
    args["model_name"],
    "CombinedLoss",
    n_classes=5,
    alpha=None,
    if_closs=0)
  start_epoch = 0
  save_dir = args["save_dir"]
  checkpoint = torch.load(args["pre"])
  net.load_state_dict(checkpoint['state_dict'])
  net = net.to(DEVICE)
  loss = loss.to(DEVICE)
  net = DataParallel(net)

  eval_dice, eval_precision = evaluation(args, net, loss, save_dir, test_files, None)
  logging.info("img all saved ! mean_eval_dice = %.6f , eval_precision = %.6f"
                 %(eval_dice,eval_precision))


def joint_img_vertical(imgs):
  # print(imgs.ndim)
  if imgs.ndim ==4:     #RGB img
    width, height = imgs[0][0].shape
    result = Image.new('L', (width, height*imgs.shape[0]))
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    imgs = imgs.mul_(255).add_(0.5).clamp_(0, 255).cpu().numpy().transpose((0, 2, 3, 1))
  elif imgs.ndim ==3:   #single channel img
    width, height = imgs[0].shape
    result = Image.new('L', (width, height*imgs.shape[0]))
    imgs =imgs*255
  for i in range(imgs.shape[0]):
    im = Image.fromarray((imgs[i]).astype(np.uint8))
    result.paste(im, box=(0, i * height))

  # im = Image.fromarray((img * 255).astype(np.uint8))
  # im.save(filename)
  # result.save(filename)
  return result

def joint_img_horizontal(img,output,mask):
  # print(img.size, output.size, mask.size)
  if img.size == output.size == mask.size:
    width, height = img.size
    result = Image.new('L', (width * 3, height))
    result.paste(img, box=(0, 0))
    result.paste(output, box=(width, 0))
    result.paste(mask, box=(2 * width, 0))
  return result

def save_img(img,filename):
  '''
    it is used to save an img in filename directory
  :param img: the img should be Image type .
  :param filename: like "**.png"
  '''
  img.save(filename)


def evaluation(args, net, loss, save_dir, test_files, saved_thresholds):
  start_time = time.time()
  net.eval()
  eval_loss = []
  eval_s_loss = []
  total_precision = []
  total_recall = []

  composed_transforms_tr = transforms.Compose([
    tr.CenterCrop(384),
    tr.Normalize(mean=(0.12, 0.12, 0.12), std=(0.018, 0.018, 0.018)),
    tr.ToTensor2(args["n_class"])
  ])
  eval_dataset = THOR_Data(
    transform=composed_transforms_tr,
    path=args["data_path"],
    file_list=test_files
  )
  evalloader = DataLoader(
    eval_dataset,
    batch_size=args["batch_size"],
    shuffle=False,
    num_workers=4)
  cur_target = []
  cur_predict = []
  class_predict = []
  class_target = []
  for i, sample in enumerate(evalloader):
    data = sample['image']
    target_c = sample['label_c']
    target_s = sample['label_s']
    data = data.to(DEVICE)
    target_c = target_c.to(DEVICE)
    target_s = target_s.to(DEVICE)
    with torch.no_grad():
      output_s, output_c = net(data)
      cur_loss, c_loss, s_loss, c_p = loss(output_s, output_c, target_s, target_c)

    eval_loss.append(cur_loss.item())
    eval_s_loss.append(s_loss.item())
    cur_target.append(torch.argmax(target_s, 1).cpu().numpy())  #( n,batch ,h,w)

    cur_predict.append(torch.argmax(output_s, 1).cpu().numpy())
    class_target.append(target_c.cpu().numpy())
    class_predict.append(c_p.cpu().numpy())
    cur_precision, cur_recall = metric(
      np.concatenate(class_predict, 0), np.concatenate(class_target, 0), saved_thresholds)

    batch_img = joint_img_vertical(data)
    batch_output = joint_img_vertical(cur_target[i])
    batch_mask = joint_img_vertical(cur_predict[i])
    img_for_save = joint_img_horizontal(batch_img, batch_output,batch_mask)
    save_img(img_for_save, os.path.join(args["save_dir"],"img_out_mask_%d.png"% i))

    logging.info("id %d img have been saved successfully." % i)

    if i == args["test_num"]:
      break

  total_precision.append(np.array(cur_precision))
  total_recall.append(np.array(cur_recall))

  TPVFs, dices, PPVs, FPVFs = segmentation_metrics(
    np.concatenate(cur_predict, 0), np.concatenate(cur_target, 0))
  logging.info(
    '***************************************************************************'
  )
  return np.mean(dices), np.mean(total_precision)


if __name__ == '__main__':
    args = {
            "model_name":"Unet",
            "save_dir":'./SavePath/SM/ResNetC1/0/result_batch1',
            "test_flag": 0 ,
            "data_path" :'../data/data_npy/',
            "batch_size" :1,
            "pre":"./SavePath/SM/Unet/0/47.ckpt",
            "n_class":5,
            "test_num":100
            }
    if not os.path.exists(args["save_dir"]):
        os.makedirs(args["save_dir"])

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s,%(lineno)d: %(message)s\n',
        datefmt='%Y-%m-%d(%a)%H:%M:%S',
        filename=os.path.join(args["save_dir"], 'log.txt'),
        filemode='a')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    logging.info(args)

    try:
        main(args)
    except RuntimeError as exception:
        if "memory" in str(exception):
            print("WARNING: out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            raise exception