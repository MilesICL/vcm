import sys, os, os.path
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from Net import NetLing, NetSyll
import _pickle as pickle
from HTK import HTKFile
import numpy as np

# Load model
net_ling = NetLing(88, 1024, 2) #.cuda()
net_ling.load_state_dict(torch.load('modelLing.pt', map_location = lambda storage, loc: storage))

net_syll = NetSyll(88, 1024, 2) #.cuda()
net_syll.load_state_dict(torch.load('modelSyll.pt', map_location = lambda storage, loc: storage))


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
output_ling = net_ling(input).data.data.cpu().numpy()
output_syll = net_syll(input).data.data.cpu().numpy()
# print(output)

# # Print the predictions in RTTM format
# class_names = ['NONL', 'LING']
# nClasses = len(class_names)
# cls = np.argmax(output_ling)
# print('\n>>> Predicted class: {}\n'.format(cls))
#
# key = os.path.splitext(os.path.basename(OUTPUT_FILE))[0]
# with open(OUTPUT_FILE, 'w') as f:
#     f.write('SPEAKER {} <NA> <NA> {} <NA> <NA>\n'.format(key, class_names[cls]))



# Load input feature and predict
htk_reader = HTKFile()
htk_reader.load(INPUT_FILE)
feat = std(np.array(htk_reader.data))
input = Variable(torch.from_numpy(feat.astype('float32'))) #.cuda()
output_ling = net_ling(input).data.data.cpu().numpy()
output_syll = net_syll(input).data.data.cpu().numpy()

# Print the predictions in RTTM format
class_names_ling = ['NONL', 'LING']
class_names_syll = ['NONC', 'CANO', 'OTHE']
nClasses_ling = len(class_names_ling)
nClasses_syll = len(class_names_syll)
cls_ling = np.argmax(output_ling)
cls_syll = np.argmax(output_syll)
if cls_ling == 0:
    cls_syll = 2
print('\n>>> Predicted linguistic class: {}\t syllable class: {}\n'.format(class_names_ling[cls_ling], class_names_syll[cls_syll]))

key = os.path.splitext(os.path.basename(OUTPUT_FILE))[0]
with open(OUTPUT_FILE, 'w') as f:
    f.write('SPEAKER {} <NA> <NA> {} {} <NA> <NA>\n'.format(key, class_names_ling[cls_ling], class_names_syll[cls_syll]))
