#生成验证码
import numpy as np
import captcha.image
import random
import string
import cv2
import tqdm
import time
import scipy.misc
import matplotlib as plb

#生成验证码
characters = string.digits + string.ascii_uppercase
#save_dir:保存文件夹路径
#num:生成验证码个数
def generateCaptcha(save_dir, num = 1):
    for i in range(0, num):
        randomStr = ''.join([random.choice(characters) for j in range(4)])
        width, height, nLen, nClass = 170, 80, 4, len(characters)
        #验证码图片大小：1070*80
        generator = captcha.image.ImageCaptcha(width, height)
        img = generator.generate_image(randomStr)
        filename = save_dir + randomStr + '.jpg'
        scipy.misc.toimage(img, cmin = 0.0, cmax = 1.0).save(filename)

if __name__ == '__main__':
    generateCaptcha(save_dir = '验证码图片/',num = 10)
