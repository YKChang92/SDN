import torch
import numpy as np
from pytorch_msssim import  SSIM
import os
import skimage.io as io
from densenet import SDNet, IFNet
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#log_dir='/home/cky/raid/log/'
str1='/data/im1/'
str2='/data/im2/'
str3='/data/im3/'
str4='/data/im4/'
str5='/data/im5/'
net_dict='/parameters/net_final.pth'
net2_dict='/parameters/net2_final.pth'
#result_dir='/home/changyakun_16/raid/result/'

#lis=np.load('/home/changyakun_16/ref/test_list3.npy')
#lis=np.load('/home/changyakun_16/ref/test_list1.npy')
lis=np.load('/home/changyakun_16/ref/test_list2.npy')
def trans(input):
    return np.float32(np.transpose(np.expand_dims(input,-1),(3,2,0,1)))
def read_image(s):
    s=str(s)
    im_a=trans(io.imread(str1+s+'.png'))
    im_f=trans(io.imread(str2+s+'.png'))
    g_a = trans(io.imread(str3 + s + '.png'))
    g_f = trans(io.imread(str4 + s + '.png'))
    G_f = trans(io.imread(str5 + s + '.png'))
    return im_a,im_f,g_a,g_f,G_f

net = SDNet()
net2=IFNet()
net=torch.nn.DataParallel(net)
net2=torch.nn.DataParallel(net2)
net.cuda()
net2.cuda()
net.load_state_dict(torch.load(net_dict))
net2.load_state_dict(torch.load(net2_dict))
ssim = SSIM(data_range=255, size_average=True, channel=3)
S_a=0
S_f=0
S_c=0
for i in range(0, np.shape(lis)[1]):
    im_a,im_f,g_a,g_f,G_f = read_image(lis[0,i])
    im_a,im_f,g_a,g_f,G_f=torch.tensor(im_a,dtype=torch.float32),torch.tensor(im_f),torch.tensor(g_a),torch.tensor(g_f),torch.tensor(G_f)
    im_a, im_f, g_a, g_f, G_f = im_a.cuda(), im_f.cuda(), g_a.cuda(), g_f.cuda(), G_f.cuda()
    net.eval()
    net2.eval()
    out_a, out_f = net(im_a - 90.5, im_f - 120.5)
    output = net2(out_a, out_f)
    output=torch.clamp(output,0,255)
    s1=torch.mean(ssim(out_a , g_a))
    s2=torch.mean(ssim(out_f , g_f))
    s3=torch.mean(ssim(output, G_f))
    S_a+=s1.item()
    S_f+=s2.item()
    S_c+=s3.item()

    # io.imsave(result_dir+str(lis[i])+'_1.png',np.transpose(np.squeeze(out_a.cpu().detach().numpy()),(1,2,0)))
    # io.imsave(result_dir + str(lis[i]) + '_2.png',np.transpose(np.squeeze(out_f.cpu().detach().numpy()),(1,2,0)))
    # io.imsave(result_dir + str(lis[i]) + '_3.png', np.transpose(np.squeeze(output.cpu().detach().numpy()),(1,2,0)))

    print('s1: %.6f,  s2: %.6f,   s3: %.6f' % (s1.item(),s2.item(),s3.item()))

print(S_c/np.shape(lis)[1])
