import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn


model_path = './expr/netCRNN_99_100.pth'
img_path = '../text2font/imgs/x14y24pxHeadUpDaisy_2.png'
#alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
with open('alphabets.txt', 'r', encoding='utf8') as f:
    alphabet = f.read()

model = crnn.CRNN(32, 1, (len(alphabet)+1), 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
param_odict = torch.load(model_path)
for key in list(param_odict.keys()):
    param_odict[key.replace('module.', '')] = param_odict[key]
    param_odict.pop(key)
#crnn.load_state_dict(torch.load(opt.crnn))
model.load_state_dict(param_odict)
#model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

image = Image.open(img_path).convert('L')
transformer = dataset.resizeNormalize((int(image.size[0]/50.*32.), 32))
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
