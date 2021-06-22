import argparse
import os
import pickle
from Sketchy import SketchyDataset
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import datetime
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from scipy.spatial.distance import cdist
import pretrainedmodels
import torch.nn.functional as F
from ResnetModel import CSEResnetModel_KDHashing
from test_cse_resnet_tuberlin_zeroshot import eval_precision, eval_AP_inner
from tool import compressITQ
# warnings.filterwarnings("error")

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch ResNet Model for Sketchy mAP Testing')
parser.add_argument('--arch', '-a', metavar='ARCH', default='cse_resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: se_resnet50)')
parser.add_argument('--num_classes', metavar='N', type=int, default=100,
                    help='number of classes (default: 100)')
parser.add_argument('--num_hashing', metavar='N', type=int, default=64,
                    help='number of hashing dimension (default: 64)')
parser.add_argument('--batch_size', default=50, type=int, metavar='N',
                    help='number of samples per batch')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')

parser.add_argument('--resume_dir',
                    default='../cse_resnet50/checkpoint',
                    type=str, metavar='PATH',
                    help='dir of model checkpoint (default: none)')
parser.add_argument('--resume_file',
                    default='model_best.pth.tar',
                    type=str, metavar='PATH',
                    help='file name of model checkpoint (default: none)')

parser.add_argument('--ems-loss', dest='ems_loss', action='store_true',
                    help='use ems loss for the training')
parser.add_argument('--precision', action='store_true', help='report precision@100')
parser.add_argument('--pretrained', action='store_true', help='use pretrained model')
parser.add_argument('--zero_version', metavar='VERSION', default='zeroshot1', type=str,
                    help='zeroshot version for training and testing (default: zeroshot1)')

parser.add_argument('--kd_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for kd loss (default: 1)')
parser.add_argument('--kdneg_lambda', metavar='LAMBDA', default='0.3', type=float,
                    help='lambda for semantic adjustment (default: 0.3)')
parser.add_argument('--sake_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for total SAKE loss (default: 1)')

parser.add_argument('--contrastive_lambda', metavar='LAMBDA', default='0.1', type=float,
                    help='lambda for contrastive loss')
parser.add_argument('--temperature', metavar='LAMBDA', default='0.07', type=float,
                    help='lambda for temperature in contrastive learning')
parser.add_argument('--contrastive_dim', metavar='N', type=int, default=128,
                    help='the dimension of contrastive feature (default: 128)')
parser.add_argument('--topk', metavar='N', type=int, default=10,
                    help='save topk embeddings in memory bank (default: 10)')
parser.add_argument('--memory_lambda', metavar='LAMBDA', default='1.0', type=float,
                    help='lambda for contrastive loss')


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    global args
    args = parser.parse_args()
    args.precision = True

    global savedir

    savedir = f'sketchy_kd({args.kd_lambda})_kdneg({args.kdneg_lambda})_sake({args.sake_lambda})_' \
              f'dim({args.num_hashing})_' \
              f'contrastive({args.contrastive_dim}-{args.contrastive_lambda})_T({args.temperature})_' \
              f'memory({args.topk}-{args.memory_lambda})'

    savedir = os.path.join(args.resume_dir, savedir)

    if args.zero_version == 'zeroshot2':
        args.num_classes = 104
        
    feature_file = os.path.join(savedir, 'features_zero.pickle')
    if os.path.isfile(feature_file):
        print('load saved SBIR features')
        with open(feature_file, 'rb') as fh:
            predicted_features_gallery, binary_features_gallery, gt_labels_gallery, \
            predicted_features_query, binary_features_query, gt_labels_query, \
            scores, binary_scores = pickle.load(fh)

        if scores is None:
            scores = - cdist(predicted_features_query, predicted_features_gallery)
            binary_scores = - cdist(binary_features_query, binary_features_gallery)

    else:
        print('prepare SBIR features using saved model')
        predicted_features_gallery, binary_features_gallery, gt_labels_gallery, \
        predicted_features_query, binary_features_query, gt_labels_query, \
        scores, binary_scores = prepare_features()


    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    mAP_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
    for fi in range(predicted_features_query.shape[0]):
        mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery)
        mAP_ls[gt_labels_query[fi]].append(mapi)

        mapi_binary = eval_AP_inner(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery)
        mAP_ls_binary[gt_labels_query[fi]].append(mapi_binary)
        
    # for mAPi,mAPs in enumerate(mAP_ls):
    #     print(str(mAPi)+' '+str(np.nanmean(mAPs))+' '+str(np.nanstd(mAPs)))
    mAP = np.array([np.nanmean(maps) for maps in mAP_ls]).mean()
    mAP_binary = np.array([np.nanmean(maps) for maps in mAP_ls_binary]).mean()
    print('mAP - real value: {:.4f}, hash: {:.4f}'.format(mAP, mAP_binary))

    if args.precision:
        prec_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
        prec_ls_binary = [[] for _ in range(len(np.unique(gt_labels_query)))]
        for fi in range(predicted_features_query.shape[0]):
            prec = eval_precision(gt_labels_query[fi], scores[fi], gt_labels_gallery)
            prec_ls[gt_labels_query[fi]].append(prec)

            prec_binary = eval_precision(gt_labels_query[fi], binary_scores[fi], gt_labels_gallery)
            prec_ls_binary[gt_labels_query[fi]].append(prec_binary)

        prec = np.array([np.nanmean(pre) for pre in prec_ls]).mean()
        prec_binary = np.array([np.nanmean(pre) for pre in prec_ls_binary]).mean()
        print('Precision - real value: {:.4f}, hash: {:.4f}'.format(prec, prec_binary))
        
    ################ PBIR
        
    # feature_file = os.path.join(args.resume_dir, 'features_photo.pickle')
    # if os.path.isfile(feature_file):
    #     print('load saved PBIR features')
    #     with open(feature_file, 'rb') as fh:
    #         predicted_features_gallery, gt_labels_gallery, \
    #         predicted_features_query, gt_labels_query, \
    #         scores = pickle.load(fh)
    #
    #     if scores is None:
    #         scores = - cdist(predicted_features_query, predicted_features_gallery)
    #
    # else:
    #     print('prepare PBIR features')
    #     predicted_features_gallery, gt_labels_gallery, \
    #     predicted_features_query, gt_labels_query, \
    #     scores = prepare_pbir_features(predicted_features_gallery, gt_labels_gallery)
    #
    #
    # mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    # for fi in range(predicted_features_query.shape[0]):
    #     mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery)
    #     mAP_ls[gt_labels_query[fi]].append(mapi)
    #
    # for mAPi,mAPs in enumerate(mAP_ls):
    #     print(str(mAPi)+' '+str(np.nanmean(mAPs))+' '+str(np.nanstd(mAPs)))
    #
    # prec_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    # for fi in range(predicted_features_query.shape[0]):
    #     prec = eval_precision(gt_labels_query[fi], scores[fi], gt_labels_gallery)
    #     prec_ls[gt_labels_query[fi]].append(prec)
    #
    # if args.precision:
    #     for preci,precs in enumerate(prec_ls):
    #         print(str(preci)+' '+str(np.nanmean(precs))+' '+str(np.nanstd(precs)))
    
def prepare_pbir_features(predicted_features_ext, gt_labels_ext):
    query_index = []
    for ll in np.unique(gt_labels_ext):
        query_index.append(np.where(gt_labels_ext==ll)[0][0:10])
        
    query_index = np.concatenate(query_index)
    
    query_index_bool = np.zeros(gt_labels_ext.shape[0]).astype(bool)
    query_index_bool[query_index] = True
    
    predicted_features_query = predicted_features_ext[query_index_bool]
    gt_labels_query = gt_labels_ext[query_index_bool]
    predicted_features_gallery = predicted_features_ext[np.logical_not(query_index_bool)]
    gt_labels_gallery = gt_labels_ext[np.logical_not(query_index_bool)]
    
    
    scores = - cdist(predicted_features_query, predicted_features_gallery)
    print('euclidean distance calculated')

    with open(os.path.join(args.resume_dir, 'features_photo.pickle'),'wb') as fh:
        pickle.dump([predicted_features_gallery, gt_labels_gallery, \
                    predicted_features_query, gt_labels_query, \
                    None],fh)
        
    return predicted_features_gallery, gt_labels_gallery, predicted_features_query, gt_labels_query, scores


def prepare_features():
    # create model
    # model = cse_resnet50(num_classes = args.num_classes, pretrained=None)
    # model = CSEResnetModel_KD(args.arch, args.num_classes, ems=args.ems_loss)
    model = CSEResnetModel_KDHashing(args.arch, args.num_hashing, args.num_classes)
    # model.cuda()
    model = nn.DataParallel(model).cuda()
    print(str(datetime.datetime.now()) + ' model inited.')

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # resume from a checkpoint
    if args.resume_file:
        resume = os.path.join(savedir, args.resume_file)
    else:
        resume = os.path.join(savedir, 'model_best.pth.tar')

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        args.start_epoch = checkpoint['epoch']
        
        save_dict = checkpoint['state_dict']
        model_dict = model.state_dict()

        trash_vars = [k for k in save_dict.keys() if k not in model_dict.keys()]
        print('trashed vars from resume dict:')
        print(trash_vars)

        resume_dict = {k: v for k, v in save_dict.items() if k in model_dict}
        # resume_dict['module.linear.cpars'] = save_dict['module.linear.weight']

        model_dict.update(resume_dict)
        model.load_state_dict(model_dict)

        # model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {} acc {:.4f})"
              .format(resume, checkpoint['epoch'], checkpoint['best_acc1'].item()))
    else:
        print("=> no checkpoint found at '{}'".format(resume))
        # return

    cudnn.benchmark = True

    # load data
    immean = [0.485, 0.456, 0.406] # RGB channel mean for imagenet
    imstd = [0.229, 0.224, 0.225]
    
    transformations = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize([224,224]),
                                          transforms.ToTensor(),
                                          transforms.Normalize(immean, imstd)])
    
    sketchy_zero_ext = SketchyDataset(split='zero', version='all_photo', zero_version=args.zero_version, \
                                         transform=transformations, aug=False)
    
    zero_loader_ext = DataLoader(dataset=sketchy_zero_ext, \
                                  batch_size=args.batch_size, shuffle=False, num_workers=3)
    
    
    sketchy_zero = SketchyDataset(split='zero', zero_version=args.zero_version, transform=transformations, aug=False)
    zero_loader = DataLoader(dataset=sketchy_zero, batch_size=args.batch_size, shuffle=False, num_workers=3)
    
    print(str(datetime.datetime.now()) + ' data loaded.')
    
    predicted_features_gallery, gt_labels_gallery = get_features(zero_loader_ext, model)
    
    predicted_features_query, gt_labels_query = get_features(zero_loader, model, 0)
    
    scores = - cdist(predicted_features_query, predicted_features_gallery)

    binary_features_query, binary_features_gallery = compressITQ(predicted_features_query, predicted_features_gallery)
    binary_scores = - cdist(binary_features_query, binary_features_gallery)
    print('euclidean distance calculated')
        
    

#     predicted_labels = validate(train_loader_ext, model, criterion)
    with open(os.path.join(savedir, 'features_zero.pickle'),'wb') as fh:
        pickle.dump([predicted_features_gallery, binary_features_gallery, gt_labels_gallery, \
                     predicted_features_query, binary_features_query, gt_labels_query, \
                     scores, binary_scores],fh)
        
    return predicted_features_gallery, binary_features_gallery, gt_labels_gallery, \
           predicted_features_query, binary_features_query, gt_labels_query, \
           scores, binary_scores


def get_features(data_loader, model, tag=1):
    # switch to evaluate mode
    model.eval()
    features_all = []
    targets_all = []
    # avgpool = nn.AvgPool2d(7, stride=1).cuda()
    avgpool = nn.AdaptiveAvgPool2d(1).cuda()
    for i, (input, target) in enumerate(data_loader):
        if i%10==0:
            print(i, end=' ', flush=True)
        
        
        tag_input = (torch.ones(input.size()[0],1)*tag).cuda()
        input = torch.autograd.Variable(input, requires_grad=False).cuda()
        
        
        # compute output
        # features = avgpool(model.module.features(input, tag_input)).cpu().detach().numpy()
        features = model.module.original_model.features(input, tag_input)
        if args.pretrained:
            features = model.module.original_model.avg_pool(features)
            features = features.view(features.size(0), -1)
        else:
            features, _ = model.module.original_model.hashing(features)
        
        features = F.normalize(features)
            
        features = features.cpu().detach().numpy()
        # features = features.reshape(input.size()[0],-1)
        
        # print(features.shape)
        # print(target.numpy().shape)
        # break
        
        
        features_all.append(features.reshape(input.size()[0],-1))
        targets_all.append(target.detach().numpy())
        
        
    print('')
        
    features_all = np.concatenate(features_all)
    targets_all = np.concatenate(targets_all)
    
    print('Features ready: {}, {}'.format(features_all.shape, targets_all.shape))
    
    return features_all, targets_all


if __name__ == '__main__':
    main()

