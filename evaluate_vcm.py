import sys, os, os.path
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Net import Net
import _pickle as pickle
from HTK import HTKFile
import numpy as np

# Load model
net = Net(88, 50, 2) #.cuda()
net.load_state_dict(torch.load('model.pt', map_location = lambda storage, loc: storage))

#'/data/work2/aclew/structNN/data/VCMtemp/example2_1.htk'
# '/data/work2/aclew/structNN/data/VCMtemp/example2_1.rttm'
INPUT_FILE = sys.argv[1]        # Feature file containing 6,669-dim HTK-format features
OUTPUT_FILE = sys.argv[2]       # RTTM file to write the results to

# Load mean and variance for standadisation
with open('ling.eGeMAPS.func_utt.meanvar', 'rb') as f:
    mv = pickle.load(f, encoding='iso-8859-1')
    m, v = mv['mean'], mv['var']
std = lambda feat: (feat - m)/v

# Load input feature and predict
htk_reader = HTKFile()
htk_reader.load(INPUT_FILE)
feat = std(np.array(htk_reader.data))
input = Variable(torch.from_numpy(feat.astype('float32'))) #.cuda()
output = net(input).data.data.cpu().numpy()
# print(output)

# Print the predictions in RTTM format
class_names = ['NONL', 'LING']
nClasses = len(class_names)
cls = np.argmax(output) 
print('\n>>> Predicted class: {}\n'.format(cls))

key = os.path.splitext(os.path.basename(OUTPUT_FILE))[0]
with open(OUTPUT_FILE, 'w') as f:
    f.write('SPEAKER {} <NA> <NA> {} <NA> <NA>\n'.format(key, class_names[cls]))
