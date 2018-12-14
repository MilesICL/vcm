import sys, os, os.path
import torch
from torch.autograd import Variable
from Net import NetVCM
try:
    import _pickle as pickle
except:
    import cPickle as pickle
from HTK import HTKFile
import numpy as np
import pandas as pd
import subprocess
import shutil


def seg_audio(input_audio, output_audio, onset, duration):
    assert os.path.isfile(input_audio)
    cmd_seg = 'sox ' + input_audio + " " + output_audio + ' trim ' + " " + onset + " " + duration
    subprocess.call(cmd_seg, shell=True)


def extract_feature(audio, feature):
    config = './config/gemaps/eGeMAPSv01a.conf'
    opensmile = '~/repos/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'
    # opensmile = '~/tools/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract'
    cmd = '{} -C {} -I {} -htkoutput {} >& /dev/null'.format(opensmile, config, audio, feature) #>& /dev/null
    subprocess.call(cmd, shell=True)


def predict_vcm(model, input, mean_var):
    ### read normalisation parameters
    assert os.path.exists(mean_var)
    with open(mean_var, 'rb') as f:
        mv = pickle.load(f)
        m, v = mv['mean'], mv['var']
    std = lambda feat: (feat - m) / v

    # Load input feature and predict
    htk_reader = HTKFile()
    htk_reader.load(input)
    feat = std(np.array(htk_reader.data))
    input = Variable(torch.from_numpy(feat.astype('float32')))  # .cuda()
    output_ling = model(input).data.data.cpu().numpy()
    prediction_confidence = output_ling.max()  # post propability

    class_names_ling = ['NCS', 'CNS', 'CRY', 'OTH']
    cls_ling = np.argmax(output_ling)
    predition_vcm = class_names_ling[cls_ling]  # prediction

    return predition_vcm, prediction_confidence


def main(audio_file, yun_rttm_file, vcm_rttm_file, mean_var, vcm_model):
    ### check the exist of the temporary folder
    tmpdir = os.path.dirname(audio_file) + '/VCMtemp'
    assert os.path.exists(tmpdir)

    with open(vcm_rttm_file, 'w+') as vf:
        # process each segment one by one. If it is infant vocalisation, do vcm
        with open(yun_rttm_file, 'r') as yf:
            for line in yf.readlines():
                els = line.split()
                file, onset, dur, cls, conf = els[1], els[3], els[4], els[7], els[8]
                if 'CHI' in els[7]:
                    audio_segment = '{}/{}_{}_{}.wav'.format(tmpdir, file.replace('.rttm', ''), onset, dur)
                    # print(audio_segment)
                    feature_file = audio_segment.replace('wav', 'htk')

                    ### segment audio file into small subsegments according to the yunitator output
                    try:
                        seg_audio(audio_file, audio_segment, onset, dur)
                    except:
                        print("Error: Cannot segment the auido: {}, from: {}, length: {}".format(audio_file, onset, dur))
                        exit()

                    ### extract acoustic feature
                    try:
                        extract_feature(audio_segment, feature_file)
                        if not os.path.isfile(feature_file):
                            print("Error: {} is not generated!".format(feature_file))
                            exit()
                    except:
                        print("Error: Cannot extract the acoustic features from: {}".format(audio_segment))
                        exit()

                    ### do vcm prediction
                    try:
                        vcm_prediction, vcm_confidence = predict_vcm(vcm_model, feature_file, mean_var)
                    except:
                        print("Error: Cannot proceed vcm prediction on: {}".format(audio_segment))
                        exit()

                    ### save prediction into rttm file
                    line = 'SPEAKER\t{}\t1\t{}\t{}\t<NA>\t<NA>\t{}\t{:.2f}\t<NA>\n'.format(file, onset, dur, vcm_prediction, float(vcm_confidence))
                    vf.write(line)


if __name__ == '__main__':
    ### global parameters
    audio_file = sys.argv[1]  # input audio file (daylong recording)
    yun_rttm_file = sys.argv[2] # input rttm file, results from yunitator
    # audio_file = '/data/work2/DiViMe/vcm/data/example.wav'
    # yun_rttm_file = '/data/work2/DiViMe/vcm/data/yunitator_example.rttm'
    vcm_rttm_file = yun_rttm_file.replace('yunitator', 'vcm') if len(sys.argv) < 4 else sys.argv[3]
    mean_var = './vcm.eGeMAPS.func_utt.meanvar'

    ### models
    vcm_net = NetVCM(88, 1024, 4)  # .cuda()
    vcm_net.load_state_dict(torch.load('vcm_model.pt', map_location=lambda storage, loc: storage))
    # net_syll = NetSyll(88, 1024, 2) #.cuda()
    # net_syll.load_state_dict(torch.load('modelSyll.pt', map_location = lambda storage, loc: storage))
    # vcm_model = vcm_net

    main(audio_file, yun_rttm_file, vcm_rttm_file, mean_var, vcm_net)


