from captcha.image import ImageCaptcha  #pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
#验证码中的字符
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
            'O', 'P', 'Q', 'R', 'S', 'T', 'U','V', 'W', 'X', 'Y', 'Z']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u','v', 'w', 'x', 'y', 'z']

def random_captcha_lable(char_set = number+alphabet+ALPHABET,captcha_size=4):
    captcha_lable = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_lable.append(c)
    return captcha_lable
#生成字符对应的验证码
def gen_captcha_lable_and_image():
    image = ImageCaptcha()
    captcha_lable = random_captcha_lable()
    captcha_lable = ''.join(captcha_lable)
    captcha = image.generate(captcha_lable)
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_lable,captcha_image
if __name__ == '__main__':
    text,image = gen_captcha_lable_and_image()
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1,0.9,text,ha='center',va='center',transform=ax.transAxes)
    plt.imshow(image)
    plt.show()