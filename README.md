# SDN
This is the code for paper "Siamese Dense Network for Reflection Removal with Flash and No-flash Image Pairs".  Before running the code, the following requirements should be installed:
Python 3.6.0
Pytorch 1.4.0 + cuda 10.0
pytorch-msssim 0.2.1
numpy 1.16.3 
skimage 0.15.0
Matlab (use FeatureSIM.m to calculate FSIM)
This is our environment, the versions of the tools do not have to be absolutely the same with ours. 

The trained parameters are in "parameters".

Dataset: the dataset can be download at: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx. The dataset contains six folders: im1(no flash glass images), im2(flash glass images), im3(no flash ground truth), im4(flash ground truth), im5(fusion ground truth), im6(detected over exposure regions). 

test_list1.npy(for synthetic1), test_list2.npy(for synthetic2),test_list3.npy(for real) indicate the indices of test images.

When you use it, change the data path and run "tests.py"

Note that this code is not the original code of the SDN paper. (For the reason that the old version is written by Tensorflow 1.x and Tensorlayer, we think now the old version must be difficult for users to use, and more and more researchers use pytorch for their works, so we release a pytorch version. )  And the SSIM,PSNR,FSIM  scores reported in SDN paper are calculated by Matlab. Thus, the performance of this version is a little different with old one (Results on Synthetic1,Synthetic2 are better; quantitative scores on real data are a little lower).  

If any comparison with SDN is made, you can take the performance of this version as a baseline.  

If the code does  provide some help to you, please cite the SDN paper, thank you very much:

@article{chang2020siamese,

  title={Siamese Dense Network for Reflection Removal with Flash and No-Flash Image Pairs},
  
  author={Chang, Yakun and Jung, Cheolkon and Sun, Jun and Wang, Fengqiao},
  
  journal={International Journal of Computer Vision},
  
  pages={1--26},
  
  year={2020},
  
  publisher={Springer}
  
}

If you use the released dataset, please cite the two works:

[1]Aksoy, Y., Kim, C., Kellnhofer, P., Paris, S., Elgharib, M., Pollefeys, M., Matusik, W.: A dataset of flash and ambient illumination pairs from the crowd. In: Proceedings of the European Conference on Computer Vision (ECCV), pp. 634-649 (2018)

[2]Song, S., Lichtenberg, S.P., Xiao, J.: Sun rgb-d: A rgb-d scene understanding benchmark suite. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 567-576 (2015)
